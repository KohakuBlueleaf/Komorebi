"""The rotary positions and the window's insert (window.py) on one chip with random
values, against float64 of the chip's own input: rope of q (64 heads) and kv at a start
position, the inverse of o, a decode token's row and a prefill block's tile into the
window (the insert: the window's words, exact). Prints one JSON line a case. README.md.
"""

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
from projects.galaxy.attention_local import window as wn  # noqa: E402

STARTED = time.monotonic()


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
        "bf16_equal": float((bf16(want.astype(np.float32)) == got).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--start", type=int, default=300, help="the rotation's position"
    )
    parser.add_argument("--compressed", type=int, default=0, help="1: YaRN's layer")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    a = parser.parse_args()
    rng = np.random.default_rng(a.seed)
    freqs = wn.frequencies(bool(a.compressed))
    positions = a.start + 32  # the tables' positions in this check
    cos, sin, sin_inverse = wn.tables(freqs, positions)
    keep, move = wn.insert_tables()
    heads_shape = (wn.HEADS * wn.TILE, wn.HEAD_DIM)
    values = {
        "q": bf16(rng.standard_normal(heads_shape, dtype=np.float32)),
        "o": bf16(rng.standard_normal(heads_shape, dtype=np.float32)),
        "kv": bf16(rng.standard_normal((wn.TILE, wn.HEAD_DIM), dtype=np.float32)),
        "window": bf16(
            rng.standard_normal((wn.WINDOW_TILES * wn.TILE, wn.HEAD_DIM), np.float32)
        ),
    }
    row, slot_row = wn.ring_row(a.start)
    with km.Session(log=log) as s:
        t = {name: s.tensor(v, "bfloat16", "tile") for name, v in values.items()}
        t["window2"] = s.tensor(values["window"], "bfloat16", "tile")
        t["y"] = s.empty((wn.HEADS * wn.TILE, wn.ROPE_DIM), "bfloat16", "tile")
        t["moved"] = s.empty((wn.TILE, wn.HEAD_DIM), "bfloat16", "tile")
        for name, v in (
            ("swap", wn.swap_tiles()),
            ("cos", cos),
            ("sin", sin),
            ("sin_inverse", sin_inverse),
            ("keep", keep),
            ("move", move),
        ):
            t[name] = s.tensor(v, "bfloat16", "tile")
        program = wn.Window(
            t=t["q"],
            swap=t["swap"],
            y=t["y"],
            cos=t["cos"],
            sin=t["sin"],
            window=t["window"],
            kv=t["kv"],
            keep=t["keep"],
            move=t["move"],
            moved=t["moved"],
            session=s,
            log=log,
        )
        run = s.load(program)
        columns = wn.HEAD_DIM // wn.TILE
        cases = [  # (case, [(handler, ranks, words)]); each case's commands once, timed
            (
                "rope_q",
                [
                    ("rope_swap", wn.HEADS, (t["q"], t["swap"], t["y"])),
                    ("rope", wn.HEADS, (t["q"], t["y"], t["cos"], t["sin"], a.start)),
                ],
            ),
            (
                "rope_o_inverse",
                [
                    ("rope_swap", wn.HEADS, (t["o"], t["swap"], t["y"])),
                    (
                        "rope",
                        wn.HEADS,
                        (t["o"], t["y"], t["cos"], t["sin_inverse"], a.start),
                    ),
                ],
            ),
            (
                "rope_kv",
                [
                    ("rope_swap", 1, (t["kv"], t["swap"], t["y"])),
                    ("rope", 1, (t["kv"], t["y"], t["cos"], t["sin"], a.start)),
                ],
            ),
            (
                "insert_row",
                [
                    (
                        "insert_move",
                        columns,
                        (t["kv"], t["move"], t["moved"], slot_row),
                    ),
                    (
                        "insert",
                        columns,
                        (t["window"], t["moved"], t["keep"], row, slot_row),
                    ),
                ],
            ),
            (
                "insert_block",
                [
                    (
                        "insert_move",
                        columns,
                        (t["kv"], t["move"], t["moved"], wn.WHOLE),
                    ),
                    (
                        "insert",
                        columns,
                        (t["window2"], t["moved"], t["keep"], wn.BLOCK_ROW, wn.WHOLE),
                    ),
                ],
            ),
        ]

        def submit(commands):
            spent = []
            for handler, count, words in commands:
                event = run.wait(
                    run.submit(getattr(wn.Window, handler), (0, count), args=words),
                    timeout=30.0,
                )
                if event.kind != "done":
                    raise km.KomorebiError(f"{handler}: {event}")
                spent.append(event.microseconds)
            return spent

        times = {}
        try:
            for name, commands in cases:
                times[name] = [submit(commands)]
            # the timed repeats: the inserts write the same words again
            for _ in range(a.repeat - 1):
                for name, commands in cases[3:]:
                    times[name].append(submit(commands))
        finally:
            run.stop()

        def read(name):
            (raw,) = s.read(t[name])
            shape = t[name].shape
            return np.asarray(
                formats.from_bytes(
                    np.asarray(raw).reshape(-1), shape, "bfloat16", "tile"
                ),
                np.float64,
            )

        got = {name: read(name) for name in ("q", "o", "kv", "window", "window2")}
    tokens = lambda v: v.reshape(-1, wn.TILE, wn.HEAD_DIM)  # noqa: E731
    want_window = values["window"].astype(np.float64).copy()
    want_window[row * wn.TILE + slot_row] = got["kv"][0]
    want_block = values["window"].astype(np.float64).copy()
    want_block[wn.BLOCK_ROW * wn.TILE : (wn.BLOCK_ROW + 1) * wn.TILE] = got["kv"]
    wants = {
        "rope_q": (got["q"], wn.rotate(tokens(values["q"]), freqs, a.start)),
        "rope_o_inverse": (
            got["o"],
            wn.rotate(tokens(values["o"]), freqs, a.start, inverse=True),
        ),
        "rope_kv": (got["kv"], wn.rotate(tokens(values["kv"]), freqs, a.start)),
    }
    ok = True
    for name, (g, want) in wants.items():
        g = tokens(g)
        report = {"case": name, "start": a.start, "compressed": a.compressed}
        report.update(errors(g[..., -wn.ROPE_DIM :], want[..., -wn.ROPE_DIM :]))
        report["first_448_equal"] = bool(
            (
                g[..., : -wn.ROPE_DIM]
                == tokens(values[name.split("_")[1]])[..., : -wn.ROPE_DIM]
            ).all()
        )
        report["us"] = [[round(v, 1) for v in each] for each in times[name]]
        ok &= report["first_448_equal"] and report["rms_rel"] < 1e-2
        log(json.dumps(report))
    for name, g, want in (
        ("insert_row", got["window"], want_window),
        ("insert_block", got["window2"], want_block),
    ):
        report = {
            "case": name,
            "row": row if name == "insert_row" else wn.BLOCK_ROW,
            "words_differ": int((g != want).sum()),
            "us_median": [float(v) for v in np.median(times[name], axis=0)],
            "us": [[round(v, 1) for v in each] for each in times[name]],
        }
        ok &= report["words_differ"] == 0
        log(json.dumps(report))
    log(json.dumps({"ok": bool(ok)}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
