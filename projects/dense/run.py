"""One matrix product shape on the card, checked against float64 on 256 sampled rows
(bit for bit with `--inputs int`, integers in [-2, 2], K at most 64). Prints one JSON
line; exits 1 when the check fails. Usage in README.md.
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
from projects.dense import card  # noqa: E402
from projects.dense import plan as pl  # noqa: E402
from ref import dense as ref  # noqa: E402

SHAPES = pl.SHAPES
# Bound of the check on uniform inputs (README.md, "The check's bound").
RMS_BOUND, MAX_BOUND = 1e-2, 2e-2
SAMPLED_ROWS = 256
EXACT_K = 64  # integer inputs in [-2, 2]: every sum is a bf16 integer up to K = 64
T0 = time.monotonic()


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def check(y, a, w, kind, seed=2):
    """Y on sampled rows against the float64 product: error statistics and `ok`."""
    rows = np.sort(np.random.default_rng(seed).permutation(len(a))[:SAMPLED_ROWS])
    want = ref.matmul(a[rows], w)
    out = ref.errors(y[rows], want) | {"rows": len(rows)}
    if kind == "int":
        out["exact"] = bool(np.array_equal(y[rows], want))
        out["ok"] = out["exact"]
    else:
        out["ok"] = out["rms_rel"] <= RMS_BOUND and out["max_rel"] <= MAX_BOUND
    return out


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shape", choices=SHAPES, default=None)
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--calls", type=int, default=1)
    p.add_argument("--team", type=int, default=0, help="0: one member per row tile")
    p.add_argument("--form", choices=["resident", "stream"], default=None)
    p.add_argument("--inputs", choices=["uniform", "int"], default="uniform")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    args.mkn = SHAPES[args.shape] if args.shape else (args.m, args.k, args.n)
    if args.inputs == "int" and args.mkn[1] > EXACT_K:
        p.error(f"--inputs int needs K <= {EXACT_K}")
    return args


def main():
    args = parse()
    m, k, n = args.mkn
    form = None if args.form is None else pl.K[f"kForm{args.form.title()}"]
    plan = pl.plan(m // 32, k // 32, n // 32, form=form)
    if plan is None:
        raise SystemExit(f"no plan fits L1 for {m} x {k} x {n}")
    out = {"m": m, "k": k, "n": n, "plan": plan.__dict__ | {"l1": plan.l1_bytes()}}
    a, w = ref.inputs(m, k, n, args.seed, args.inputs)
    with km.Session(log=log) as session:
        device = session.device
        t0 = time.perf_counter()
        tensors = {
            "a": card.bf16_tensor(device, a),
            "w": card.bf16_tensor(device, w),
            "y": card.bf16_tensor(device, np.zeros((m, n), np.float32)),
        }
        out["upload_s"] = round(time.perf_counter() - t0, 3)
        count = args.team or pl.team_size(plan, len(session.nocs().words))
        product = card.Product(session, plan, tensors, count)
        log(f"loaded, team {count}")
        out["load_us"] = product.load_us
        host_s, us = product.calls(args.calls)
        log(f"{args.calls} calls in {host_s:.4f} s")
        out["us_per_call_host"] = round(host_s * 1e6 / args.calls, 3)
        out["record_us_median"] = float(np.median(us))
        out["run"] = product.stop()
        t0 = time.perf_counter()
        (y,) = session.read_bf16(tensors["y"])
        out["readback_s"] = round(time.perf_counter() - t0, 3)
        log("Y read back")
    out["check"] = check(y[:m, :n], a, w, args.inputs)
    out["ok"] = out["check"]["ok"] and out["run"]["kernel"] == "none"
    print(json.dumps(out), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out))
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
