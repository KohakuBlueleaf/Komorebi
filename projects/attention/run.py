"""One attention shape on the card, checked against float64 on sampled query tile rows
(their output tiles are read back, not the whole tensor). Prints one JSON line; exits 1
when the check fails. Usage in README.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from projects.attention import card  # noqa: E402
from projects.attention import plan as pl  # noqa: E402
from projects.attention.reference import sdpa  # noqa: E402

# Bound of the check on standard normal inputs (README.md, "The check's bound"); the
# approximate exponential's relative RMS bound.
RMS_BOUND, MAX_BOUND = 1.5e-2, 6e-2
APPROX_RMS_BOUND = 3e-2
SAMPLED = 16  # query tile rows checked
T0 = time.monotonic()


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shape", choices=pl.SHAPES, default="t1")
    p.add_argument("--calls", type=int, default=1)
    p.add_argument(
        "--broadcast", type=int, default=None, help="broadcast length (default: plan)"
    )
    p.add_argument("--chunks", type=int, default=None, help="causal chunks per head")
    p.add_argument("--approx", action="store_true", help="approximate exponential")
    p.add_argument("--noc", type=int, choices=[0, 1], default=0, help="broadcast's NoC")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--json", default=None)
    return p.parse_args()


def shape_dims(name):
    """(b, h, hk, s, d, causal) of a preset and its Dims (Sq = Sk, D = Dv)."""
    b, h, hk, s, d, causal = pl.SHAPES[name]
    return (b, h, hk, s, d, causal), pl.dims(b, h, hk, s, s, d, d, causal)


def check(session, out, q, k, v, s, causal, approx=False, count=SAMPLED):
    """Sampled query tile rows of O against float64: error statistics and `ok`."""
    group = s.h // s.hk
    got, want = [], []
    for bi, hi, row in sdpa.sample_rows(s.b, s.h, s.sqt, count):
        first = ((bi * s.h + hi) * s.sqt + row) * s.dvt
        tiles = session.read_tiles_bf16(out, first, s.dvt)
        got.append(np.concatenate(list(tiles), axis=1))
        rows = np.arange(row * pl.TILE, (row + 1) * pl.TILE)
        kv = hi // group
        want.append(sdpa.attention(q[bi, hi], k[bi, kv], v[bi, kv], rows, causal))
    res = sdpa.errors(np.concatenate(got), np.concatenate(want))
    res["tile_rows"] = len(got)
    rms = APPROX_RMS_BOUND if approx else RMS_BOUND
    res["ok"] = res["rms_rel"] <= rms and res["max_rel"] <= MAX_BOUND
    return res


def main():
    args = parse()
    (b, h, hk, sq, d, causal), s = shape_dims(args.shape)
    q, k, v = sdpa.inputs(b, h, hk, sq, sq, d, d, args.seed)
    scale = 1.0 / np.sqrt(d)
    out = {"shape": args.shape, "b": b, "h": h, "hk": hk, "s": sq, "d": d}
    out["causal"], out["approx"], out["noc"] = causal, args.approx, args.noc
    with km.Session(log=log) as session:
        nocs = session.nocs()
        p = pl.make(s, nocs.grid, length=args.broadcast, chunks=args.chunks)
        out["plan"] = p.summary()
        t0 = time.perf_counter()
        tensors = {
            "q": card.bf16_tensor(session, q),
            "k": card.bf16_tensor(session, k),
            "v": card.bf16_tensor(session, v),
            "o": card.bf16_tensor(session, np.zeros((b, h, sq, d), np.float32)),
            "tables": card.table_tensor(session, pl.tables(p)),
        }
        out["upload_s"] = round(time.perf_counter() - t0, 3)
        log(f"uploaded, plan {out['plan']}")
        att = card.Attention(session, p, tensors, scale, args.approx, noc=args.noc)
        log("loaded")
        host_s, us = att.calls(args.calls)
        log(f"{args.calls} calls in {host_s:.4f} s")
        out["run"] = att.stop()
        log("stopped")
        out["check"] = check(session, tensors["o"], q, k, v, s, causal, args.approx)
        log("sampled rows read back")
    out["ok"] = out["check"]["ok"] and out["run"]["ok"]
    print(json.dumps(out), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out))
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
