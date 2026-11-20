"""The attention core (attend.py) on one chip with random q, key list and sink against
float64: decode tokens and prefill blocks, on the window alone and with selected
entries. One JSON line a case. README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi import formats  # noqa: E402
from projects.galaxy.attention_local import attend as at  # noqa: E402

STARTED = time.monotonic()
WINDOW = 128


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def bf16(values):
    bits = np.asarray(values, np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def errors(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    scale = np.sqrt((want**2).mean())
    return {
        "rms_rel": float(np.sqrt(((got - want) ** 2).mean()) / scale),
        "max_over_rms": float(np.abs(got - want).max() / scale),
    }


def case(rng, start, tokens, entries):
    """(q [64, 32, 512], the key list [22 x 32, 512] (window.py's layout: the sink's
    zero tile, the ring, the block's own tile, then `entries` selected entries),
    visible [32, 704], key tiles, the masks' variant, the entries' mask): random q rows
    and keys, an empty slot zero, a random quarter of the entries hidden from each row.
    """
    prefill = tokens > 1
    q = np.zeros((at.HEADS, 32, at.HEAD_DIM), np.float32)
    q[:, :tokens] = bf16(rng.standard_normal((at.HEADS, tokens, at.HEAD_DIM)))
    rows = at.KEY_TILES * 32
    kv = bf16(rng.standard_normal((rows, at.HEAD_DIM)).astype(np.float32))
    kv[:32] = 0.0
    seen = np.zeros((32, rows), bool)
    seen[:, :192] = at.visible(start, 32, prefill)
    kv[32 : 32 + WINDOW][~seen[:, 32 : 32 + WINDOW].any(0)] = 0.0
    if not prefill:
        kv[32 + WINDOW : 192] = 0.0
    chosen = rng.random((32, entries)) > 0.25
    seen[:, 192 : 192 + entries] = chosen
    kv[192 + entries :] = 0.0
    variant = at.prefill_variant(start) if prefill else at.decode_variant(start)
    key_tiles = 6 + (entries + 31) // 32
    return q, kv, seen, key_tiles, variant, at.entry_values(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeat", type=int, default=10)
    a = parser.parse_args()
    rng = np.random.default_rng(3)
    sink = (0.3 * rng.standard_normal(at.HEADS)).astype(np.float32)
    cases = {  # name: (start position, tokens, selected entries)
        "decode_full": (300, 1, 0),
        "decode_filling": (40, 1, 0),
        "prefill_0": (0, 32, 0),
        "prefill_160": (160, 32, 0),
        "decode_global": (300, 1, 512),
        "decode_global_200": (1000, 1, 200),
        "prefill_global": (0, 32, 16),
    }
    with km.Session(log=log) as s:
        made = {name: case(rng, *spec) for name, spec in cases.items()}
        shared = {
            "sinks": s.tensor(at.sink_values(sink), "bfloat16", "tile"),
            "masks": s.tensor(at.mask_values(), "bfloat16", "tile"),
        }
        tensors = {}
        for name, (q, kv, _, key_tiles, variant, entries) in made.items():
            if key_tiles not in shared:
                shared[key_tiles] = at.tables_tensor(s, key_tiles)
            tensors[name] = {
                "q": s.tensor(q.reshape(-1, at.HEAD_DIM), "bfloat16", "tile"),
                "kv": s.tensor(kv, "bfloat16", "tile"),
                "o": s.empty((at.HEADS * 32, at.HEAD_DIM), "bfloat16", "tile"),
                "tables": shared[key_tiles],
                "variant": variant,
                "entries": s.tensor(entries, "bfloat16", "tile"),
            }
        first = tensors["prefill_160"]
        program = at.AttendLocal(
            q=first["q"],
            kv=first["kv"],
            o=first["o"],
            tables=first["tables"],
            sinks=shared["sinks"],
            masks=shared["masks"],
            entries=first["entries"],
            session=s,
            log=log,
        )
        run = s.load(program)
        times = {name: [] for name in cases}
        try:
            for _ in range(a.repeat):
                for name, t in tensors.items():
                    words = at.words(
                        t["q"],
                        t["kv"],
                        t["o"],
                        t["tables"],
                        shared["sinks"],
                        shared["masks"],
                        t["variant"],
                        t["entries"],
                    )
                    event = run.wait(
                        run.submit(
                            at.AttendLocal.attend,
                            at.group(s),
                            args=words,
                            exclusive=True,
                        ),
                        timeout=30.0,
                    )
                    if event.kind != "done":
                        raise km.KomorebiError(f"{name}: {event}")
                    times[name].append(event.microseconds)
        finally:
            run.stop()
        for name, (start, tokens, _) in cases.items():
            q, kv, visible, key_tiles, variant, _ = made[name]
            (raw,) = s.read(tensors[name]["o"])
            got = np.asarray(
                formats.from_bytes(
                    np.asarray(raw).reshape(-1),
                    (at.HEADS * 32, at.HEAD_DIM),
                    "bfloat16",
                    "tile",
                ),
                np.float64,
            ).reshape(at.HEADS, 32, at.HEAD_DIM)
            sink_b = bf16(sink / np.float32(at.SCALE)) * np.float32(at.SCALE)
            want = at.reference(q, kv, sink_b, visible)
            rows = slice(0, tokens)
            report = {
                "case": name,
                "start": start,
                "tokens": tokens,
                "key_tiles": key_tiles,
                "variant": variant,
            }
            report.update(errors(got[:, rows], want[:, rows]))
            exact_sink = at.reference(q, kv, sink, visible)
            report["rms_rel_vs_f32_sink"] = errors(got[:, rows], exact_sink[:, rows])[
                "rms_rel"
            ]
            report["us_median"] = float(np.median(times[name]))
            report["us"] = [round(v, 1) for v in times[name]]
            log(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
