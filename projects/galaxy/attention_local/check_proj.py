"""The projections chain on one chip with random weights, each step against float64 of
the values the chip holds (its own input, the weights as stored), and the whole chain
against float64 from x. Prints one JSON line a step and the chain's. README.md."""

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
from projects.galaxy.attention_local import layer as ly  # noqa: E402
from projects.galaxy.attention_local import proj as pj  # noqa: E402

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
        "bf16_units_max": int(units(got, want).max()),
    }


KERNEL_BYTES = 104_448  # the kernels' 103,408 B and the load's margin, aligned


def units(got, want):
    """|got - want| in bf16 steps of the float64 value rounded to bf16 (ordered bits)"""

    def ordered(values):
        bits = (np.asarray(values, np.float32).view(np.uint32) >> 16).astype(np.int64)
        return np.where(bits & 0x8000, 0x8000 - bits, bits)

    return np.abs(ordered(got) - ordered(bf16(np.asarray(want, np.float32))))


def rms_norm(x, g):
    x = np.asarray(x, np.float64)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + pj.EPS) * g


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--format", default="bfp8_b")
    parser.add_argument("--tokens", type=int, default=32, help="1: decode, 32: prefill")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    a = parser.parse_args()
    rng = np.random.default_rng(a.seed)
    fmt = a.format
    shapes = {
        "wq_a": (pj.DIM, pj.Q_RANK),
        "wq_b": (pj.Q_RANK, pj.HEADS * pj.HEAD_DIM),
        "wkv": (pj.DIM, pj.HEAD_DIM),
        "wo_a": (pj.O_GROUPS * pj.HEADS * pj.HEAD_DIM // pj.O_GROUPS, pj.O_RANK),
        "wo_b": (pj.O_GROUPS * pj.O_RANK, pj.DIM),
    }
    weights = {}
    for name, (k, n) in shapes.items():
        fan_in = k // pj.O_GROUPS if name == "wo_a" else k
        w = rng.standard_normal((k, n), dtype=np.float32) / np.float32(np.sqrt(fan_in))
        weights[name] = pj.stored(w, fmt)
    gq = bf16(1.0 + 0.1 * rng.standard_normal(pj.Q_RANK, dtype=np.float32))
    gkv = bf16(1.0 + 0.1 * rng.standard_normal(pj.HEAD_DIM, dtype=np.float32))
    x = np.zeros((32, pj.DIM), np.float32)
    x[: a.tokens] = bf16(rng.standard_normal((a.tokens, pj.DIM), dtype=np.float32))
    o = np.zeros((32, pj.HEADS * pj.HEAD_DIM), np.float32)
    o[: a.tokens] = bf16(rng.standard_normal((a.tokens, o.shape[1]), dtype=np.float32))
    reports = []
    # the class's kernels take 103,408 B: opened at that size (a reopen at the load
    # brings a written tensor back at another address)
    with km.Session(log=log, kernel_bytes=KERNEL_BYTES) as s:
        # each W seen as one tile column, as the loader's arenas (first page 0)
        t = {name: ly.column(s.tensor(w, fmt, "tile")) for name, w in weights.items()}
        act = {
            "x": s.tensor(x, "bfloat16", "tile"),
            "o": s.tensor(o, "bfloat16", "tile"),
        }
        for name, cols in (
            ("qa", pj.Q_RANK),
            ("qr", pj.Q_RANK),
            ("q", 32768),
            ("kv0", 512),
            ("kv", 512),
            ("oa", 8192),
            ("y", pj.DIM),
        ):
            act[name] = s.empty((32, cols), "bfloat16", "tile")
        for name, g in (("gq", gq), ("gkv", gkv)):
            pages = pj.norm_pages(g)
            act[name] = s.empty((pages.shape[0], 1024), "float32")
            s.write(act[name], pages.view(np.uint8).reshape(pages.shape[0], -1))
        program = pj.Projections(
            log=log,
            session=s,
            x=act["x"],
            qr=act["qr"],
            o=act["o"],
            oa=act["oa"],
            w_qa=t["wq_a"],
            w_kv=t["wkv"],
            w_qb=t["wq_b"],
            w_oa=t["wo_a"],
            w_ob=t["wo_b"],
            y=act["qa"],
            g=act["gq"],
            out=act["qr"],
        )
        run = s.load(program)

        widths = {
            "qa": pj.Q_RANK,
            "qr": pj.Q_RANK,
            "q": 32768,
            "kv0": 512,
            "kv": 512,
            "oa": 8192,
            "y": pj.DIM,
        }

        def read(name):
            (raw,) = s.read(act[name])
            return np.asarray(
                formats.from_bytes(
                    np.asarray(raw).reshape(-1), (32, widths[name]), "bfloat16", "tile"
                ),
                np.float64,
            )

        steps = [  # (step, handler, ranks, words)
            ("wq_a", "wq_a", pj.cores("wq_a"), (act["x"], t["wq_a"], act["qa"], 0)),
            ("q_norm", "norm_q", 1, (act["qa"], act["gq"], act["qr"])),
            ("wq_b", "wq_b", pj.cores("wq_b"), (act["qr"], t["wq_b"], act["q"], 0)),
            ("wkv", "wkv", pj.cores("wkv"), (act["x"], t["wkv"], act["kv0"], 0)),
            ("kv_norm", "norm_kv", 1, (act["kv0"], act["gkv"], act["kv"])),
            ("wo_a", "wo_a", pj.cores("wo_a"), (act["o"], t["wo_a"], act["oa"], 0)),
            ("wo_b", "wo_b", pj.cores("wo_b"), (act["oa"], t["wo_b"], act["y"], 0)),
        ]
        times = {name: [] for name, *_ in steps}
        for _ in range(a.repeat):
            for name, handler, count, words in steps:
                event = run.wait(
                    run.submit(
                        getattr(pj.Projections, handler), (0, count), args=words
                    ),
                    timeout=60.0,
                )
                if event.kind != "done":
                    raise km.KomorebiError(f"{name}: {event}")
                times[name].append(event.microseconds)
        run.stop()
        got = {name: read(name) for name in ("qa", "qr", "q", "kv0", "kv", "oa", "y")}
    w = {name: v.astype(np.float64) for name, v in weights.items()}
    rows = slice(0, a.tokens)
    oa_want = np.concatenate(
        [
            o[:, g * 4096 : (g + 1) * 4096].astype(np.float64)
            @ w["wo_a"][g * 4096 : (g + 1) * 4096]
            for g in range(pj.O_GROUPS)
        ],
        axis=1,
    )
    wants = {  # each step from the chip's own input
        "wq_a": (got["qa"], x @ w["wq_a"]),
        "q_norm": (got["qr"], rms_norm(got["qa"], gq)),
        "wq_b": (got["q"], got["qr"] @ w["wq_b"]),
        "wkv": (got["kv0"], x @ w["wkv"]),
        "kv_norm": (got["kv"], rms_norm(got["kv0"], gkv)),
        "wo_a": (got["oa"], oa_want),
        "wo_b": (got["y"], got["oa"] @ w["wo_b"]),
    }
    for name, (g, want) in wants.items():
        report = {
            "step": name,
            "format": fmt,
            "tokens": a.tokens,
            "cores": dict((n, c) for n, _, c, _ in steps)[name],
        }
        report.update(errors(g[rows], want[rows]))
        report["us_median"] = float(np.median(times[name]))
        report["us"] = [round(v, 1) for v in times[name]]
        log(json.dumps(report))
        reports.append(report)
    # the whole chain from x and o, float64 throughout (the weights as stored)
    qr = rms_norm(x.astype(np.float64) @ w["wq_a"], gq)
    chain = {
        "q": errors(got["q"][rows], (qr @ w["wq_b"])[rows]),
        "kv": errors(got["kv"][rows], rms_norm(x @ w["wkv"], gkv)[rows]),
        "y": errors(got["y"][rows], (oa_want @ w["wo_b"])[rows]),
    }
    total = sum(r["us_median"] for r in reports)
    log(json.dumps({"chain": chain, "us_total": total}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
