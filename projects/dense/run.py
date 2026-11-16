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
from komorebi.bench import power  # noqa: E402
from projects.dense import card  # noqa: E402
from projects.dense import plan as pl  # noqa: E402
from reference import dense as ref  # noqa: E402

SHAPES = pl.SHAPES
FORMS = ["resident", "stream", "grid", "split"]
# Bound of the check on uniform inputs (README.md, "The check's bound").
RMS_BOUND, MAX_BOUND = 1e-2, 2e-2
EXACT_K = 64  # integer inputs in [-2, 2]: every sum is a bf16 integer up to K = 64
STARTED_AT = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED_AT:7.2f}] {message}", flush=True)


def check(got, rows, a_matrix, w_matrix, kind):
    """Y's sampled rows `got` (ref.sampled_rows) against the float64 product: error
    statistics and `ok`."""
    want = ref.matmul(a_matrix[rows], w_matrix)
    report = ref.errors(got, want) | {"rows": len(rows)}
    if kind == "int":
        report["exact"] = bool(np.array_equal(got, want))
        report["ok"] = report["exact"]
    else:
        report["ok"] = report["rms_rel"] <= RMS_BOUND and report["max_rel"] <= MAX_BOUND
    return report


def clean(run):
    """No kernel error, no core or broadcast given up, and every core ended."""
    ended = all(
        ended_count == run["cores"] for ended_count in run["cores_ended"].values()
    )
    gave_up = run["gave_up"] or run["broadcast_gave_up"]
    return run["kernel"] == "none" and ended and not gave_up


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shape", choices=SHAPES, default=None)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--calls", type=int, default=1)
    parser.add_argument("--group", type=int, default=0, help="0: one core per row tile")
    parser.add_argument("--form", choices=FORMS, default=None)
    parser.add_argument(
        "--grid", default=None, help="[layout:]rows,cols,pcm,pcn,bm,bn,kb[,wa,ww]"
    )
    parser.add_argument("--split", default=None, help="K split: s,oh,ow[,kc]")
    parser.add_argument("--reduce", choices=list(pl.REDUCE_FORMS), default="owner")
    parser.add_argument("--inputs", choices=["uniform", "int"], default="uniform")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    args.dimensions = SHAPES[args.shape] if args.shape else (args.m, args.k, args.n)
    if args.inputs == "int" and args.dimensions[1] > EXACT_K:
        parser.error(f"--inputs int needs K <= {EXACT_K}")
    return args


def make_plan(form, dimensions, grid=None, split=None):
    """The plan of a shape: the given form (or W resident when it fits, else the grid
    form); `grid` a grid plan "[layout:]rows,cols,pcm,pcn,bm,bn,kb[,wa,ww]" or a
    layout name for the rule's plan in it; `split` a K-split plan "s,oh,ow[,kc]"."""
    if any(value % pl.TILE for value in dimensions):
        raise SystemExit(f"M, K and N are whole tiles of {pl.TILE}: {dimensions}")
    m_tiles, k_tiles, n_tiles = (value // pl.TILE for value in dimensions)
    if form == "split" or split:
        return pl.split_plan(
            m_tiles, k_tiles, n_tiles, *(int(value) for value in split.split(","))
        )
    if grid in pl.LAYOUTS:
        return pl.grid_plan(m_tiles, k_tiles, n_tiles, layout=grid)
    if grid:
        layout_name, _, spec = grid.rpartition(":")
        (
            rows,
            cols,
            band_height,
            band_width,
            block_height,
            block_width,
            k_block_tiles,
            *windows,
        ) = (int(value) for value in spec.split(","))
        subblock = pl.subblock(block_height, block_width)
        plan = pl.Grid(
            m_tiles,
            k_tiles,
            n_tiles,
            rows,
            cols,
            band_height,
            band_width,
            block_height,
            block_width,
            k_block_tiles,
            *subblock,
            *windows,
        )
        plan = pl.Grid(**(plan.__dict__ | {"layout": layout_name or "2d"}))
        plan.check((12, 10))
        return plan
    if form == "grid":
        return pl.grid_plan(m_tiles, k_tiles, n_tiles)
    plan = pl.plan(
        m_tiles,
        k_tiles,
        n_tiles,
        form=None if form is None else pl.K[f"kForm{form.title()}"],
    )
    if plan is None or (form is None and not plan.resident):
        return pl.grid_plan(m_tiles, k_tiles, n_tiles)
    return plan


def describe(plan):
    report = plan.__dict__ | {"l1": plan.l1_bytes()}
    if isinstance(plan, pl.Grid):
        report |= {"form": "grid", "nk": plan.nk, "nbh": plan.nbh, "nbw": plan.nbw}
        report |= {"rect": plan.rect((12, 10))}
    if isinstance(plan, pl.Split):
        report |= {"form": "split", "cores": plan.cores()}
    return report


def load(session, plan, tensors, group=0, deadline=2.0, reduce="owner"):
    """The plan's program loaded on the session (each command's `deadline` in s): a
    card.Product, GridProduct or SplitProduct (summing by `reduce`)."""
    if isinstance(plan, pl.Split):
        return card.SplitProduct(session, plan, tensors, reduce, deadline=deadline)
    if isinstance(plan, pl.Grid):
        return card.GridProduct(session, plan, tensors, deadline=deadline)
    count = group or pl.group_size(plan, len(session.nocs().words))
    return card.Product(session, plan, tensors, count, deadline=deadline)


def main():
    args = parse()
    m_rows, k_depth, n_columns = args.dimensions
    plan = make_plan(args.form, args.dimensions, args.grid, args.split)
    if plan is None:
        raise SystemExit(f"no plan fits L1 for {m_rows} x {k_depth} x {n_columns}")
    report = {
        "m": m_rows,
        "k": k_depth,
        "n": n_columns,
        "plan": describe(plan),
        "reduce": args.reduce,
    }
    a_matrix, w_matrix = ref.inputs(m_rows, k_depth, n_columns, args.seed, args.inputs)
    with km.Session(log=log) as session:
        started = time.perf_counter()
        tensors = {
            "a": card.bf16_tensor(session, a_matrix),
            "w": card.bf16_tensor(session, w_matrix),
            "y": card.bf16_tensor(session, np.zeros((m_rows, n_columns), np.float32)),
        }
        report["upload_s"] = round(time.perf_counter() - started, 3)
        product = load(session, plan, tensors, args.group, reduce=args.reduce)
        log(f"loaded, group {product.group}")
        report["load_microseconds"] = product.load_microseconds
        try:
            host_s, microseconds = product.calls(args.calls)
        except km.KomorebiError as error:
            report["failed"] = product.failed(error)
            log(f"failed: {json.dumps(report['failed'], default=str)}")
            print(json.dumps(power.clocked(report), default=str), flush=True)
            sys.exit(1)
        log(f"{args.calls} calls in {host_s:.4f} s")
        report["microseconds_per_call_host"] = round(host_s * 1e6 / args.calls, 3)
        report["command_microseconds_median"] = float(np.median(microseconds))
        report["run"] = product.stop()
        started = time.perf_counter()
        rows = ref.sampled_rows(m_rows)
        got = card.read_rows(session, tensors["y"], rows, log)
        report["readback_s"] = round(time.perf_counter() - started, 3)
        log("Y's sampled rows read back")
    report["check"] = check(got[:, :n_columns], rows, a_matrix, w_matrix, args.inputs)
    report["ok"] = report["check"]["ok"] and clean(report["run"])
    print(json.dumps(power.clocked(report)), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
