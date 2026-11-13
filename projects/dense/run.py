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
FORMS = ["resident", "stream", "grid"]
# Bound of the check on uniform inputs (README.md, "The check's bound").
RMS_BOUND, MAX_BOUND = 1e-2, 2e-2
EXACT_K = 64  # integer inputs in [-2, 2]: every sum is a bf16 integer up to K = 64
T0 = time.monotonic()


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def check(got, rows, a, w, kind):
    """Y's sampled rows `got` (ref.sampled_rows) against the float64 product: error
    statistics and `ok`."""
    want = ref.matmul(a[rows], w)
    out = ref.errors(got, want) | {"rows": len(rows)}
    if kind == "int":
        out["exact"] = bool(np.array_equal(got, want))
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
    p.add_argument("--form", choices=FORMS, default=None)
    p.add_argument(
        "--grid", default=None, help="[layout:]rows,cols,pcm,pcn,bm,bn,kb[,wa,ww]"
    )
    p.add_argument("--inputs", choices=["uniform", "int"], default="uniform")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    args.mkn = SHAPES[args.shape] if args.shape else (args.m, args.k, args.n)
    if args.inputs == "int" and args.mkn[1] > EXACT_K:
        p.error(f"--inputs int needs K <= {EXACT_K}")
    return args


def make_plan(form, mkn, grid=None):
    """The plan of a shape: the given form (or W resident when it fits, else the grid
    form); `grid` a grid plan "[layout:]rows,cols,pcm,pcn,bm,bn,kb[,wa,ww]" or a
    layout name for the rule's plan in it."""
    if any(v % pl.TILE for v in mkn):
        raise SystemExit(f"M, K and N are whole tiles of {pl.TILE}: {mkn}")
    mt, kt, nt = (v // pl.TILE for v in mkn)
    if grid in pl.LAYOUTS:
        return pl.grid_plan(mt, kt, nt, layout=grid)
    if grid:
        lay, _, spec = grid.rpartition(":")
        rows, cols, pcm, pcn, bm, bn, kb, *win = (int(v) for v in spec.split(","))
        sub = pl.subblock(bm, bn)
        g = pl.Grid(mt, kt, nt, rows, cols, pcm, pcn, bm, bn, kb, *sub, *win)
        g = pl.Grid(**(g.__dict__ | {"layout": lay or "2d"}))
        g.check((12, 10))
        return g
    if form == "grid":
        return pl.grid_plan(mt, kt, nt)
    p = pl.plan(mt, kt, nt, form=None if form is None else pl.K[f"kForm{form.title()}"])
    if p is None or (form is None and not p.resident):
        return pl.grid_plan(mt, kt, nt)
    return p


def describe(plan):
    out = plan.__dict__ | {"l1": plan.l1_bytes()}
    if isinstance(plan, pl.Grid):
        out |= {"form": "grid", "nk": plan.nk, "nbh": plan.nbh, "nbw": plan.nbw}
        out |= {"rect": plan.rect((12, 10))}
    return out


def load(session, plan, tensors, team=0, deadline=2.0):
    """The plan's program loaded on the session (each record's `deadline` in s): a
    card.Product or card.GridProduct."""
    if isinstance(plan, pl.Grid):
        return card.GridProduct(session, plan, tensors, deadline=deadline)
    count = team or pl.team_size(plan, len(session.nocs().words))
    return card.Product(session, plan, tensors, count, deadline=deadline)


def main():
    args = parse()
    m, k, n = args.mkn
    plan = make_plan(args.form, args.mkn, args.grid)
    if plan is None:
        raise SystemExit(f"no plan fits L1 for {m} x {k} x {n}")
    out = {"m": m, "k": k, "n": n, "plan": describe(plan)}
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
        product = load(session, plan, tensors, args.team)
        log(f"loaded, team {product.team}")
        out["load_us"] = product.load_us
        try:
            host_s, us = product.calls(args.calls)
        except km.KomorebiError as err:
            out["failed"] = product.failed(err)
            log(f"failed: {json.dumps(out['failed'], default=str)}")
            print(json.dumps(out, default=str), flush=True)
            sys.exit(1)
        log(f"{args.calls} calls in {host_s:.4f} s")
        out["us_per_call_host"] = round(host_s * 1e6 / args.calls, 3)
        out["record_us_median"] = float(np.median(us))
        out["run"] = product.stop()
        t0 = time.perf_counter()
        rows = ref.sampled_rows(m)
        got = card.read_rows(session, tensors["y"], rows, log)
        out["readback_s"] = round(time.perf_counter() - t0, 3)
        log("Y's sampled rows read back")
    out["check"] = check(got[:, :n], rows, a, w, args.inputs)
    out["ok"] = out["check"]["ok"] and out["run"]["kernel"] == "none"
    print(json.dumps(out), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out))
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
