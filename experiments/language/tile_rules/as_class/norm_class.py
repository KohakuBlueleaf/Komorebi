"""The row norm of ../rules.py as a class's tile function in the row form (a row of 2
tiles, the weight a field it holds), timed on the math CPU and checked against float64
(build/rownorm-<hash>/ beside this file). See README.md.
Example: python norm_class.py --cores 12 --rows 32
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import rules  # noqa: E402  (the harness: its values, weight tile and reference)

STARTED = time.monotonic()
WIDTH = rules.WIDTH  # tiles a row
BLOCK = 2 * WIDTH  # tiles a block: two rows
SPANS_EMPTY, SPANS_FIRST = 16, 18  # a stamps page: the empty spans, then each call's
TOLERANCE = rules.TOLERANCE["norm"]


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class RowNorm(km.Program):
    """norm: each core's `rows` rows of WIDTH tiles of x from rank x rows, y = x
    rsqrt(mean(x^2) + eps) g into the same pages of out; g's tile filled from w."""

    weight: km.L1[km.fp32, 1, 32, 32]

    @km.tile_function(tolerance=TOLERANCE)
    def norm(row, weight):
        ms = km.reduce_sum(row * row, exact=True) * (1.0 / (32 * WIDTH))
        r = km.rsqrt(ms + rules.EPS)
        return row * r.col() * weight.row()

    @km.handler(stamps="stamps")
    def norm_rows(
        self, x: km.Tensor, w: km.Tensor, out: km.Tensor, stamps: km.Tensor, rows: int
    ):
        first = self.rank * rows * WIDTH
        self.weight.fill(km.Stream(w.pages(0, 1)))
        results = km.Stream(out.pages(first, rows * WIDTH), write=True, block=BLOCK)
        for row in km.Stream(x.pages(first, rows * WIDTH), block=BLOCK, row=WIDTH):
            results.push(self.norm(row, self.weight))


def compared(session, out, wanted):
    """The largest relative error against float64 over values above 1e-30 (as
    ../rules.py judges), and the values that are not finite."""
    relative, bad = 0.0, 0
    for first, got in session.read_chunks(out):
        got = got.view(np.float32).reshape(len(got), -1).astype(np.float64)
        want = wanted[first : first + len(got)]
        judged = np.isfinite(want) & (np.abs(want) > 1e-30)
        error = np.abs(got[judged] - want[judged]) / np.abs(want[judged])
        relative = max(relative, float(error.max()))
        bad += int((~np.isfinite(got[judged])).sum())
        log(f"  output pages from {first} read back")
    return {"max_relative": relative, "not_finite": bad}


def spans(session, stamps, cores, rows):
    """The math CPU's cycles a call: the calls' median less the median empty span."""
    (raw,) = session.read(stamps)
    pages = raw.view(np.uint32).reshape(cores, -1).astype(np.int64)
    empty = float(np.median(pages[:, 2 : 2 + SPANS_EMPTY]))
    calls = pages[:, SPANS_FIRST : SPANS_FIRST + rows]
    return {
        "calls_ok": bool((pages[:, 1] == rows).all()),
        "empty_span": empty,
        "median_less_empty": float(np.median(calls)) - empty,
        "min_call": int(calls.min()),
        "max_call": int(calls.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=12)
    parser.add_argument("--rows", type=int, default=32, help="rows a core takes")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.rows % 2 or not 2 <= args.rows <= 256 - SPANS_FIRST:
        parser.error(f"--rows is even, up to {256 - SPANS_FIRST}")
    report, ok = {"tolerance": TOLERANCE}, True
    with km.Session(log=log) as session:
        cores, count = args.cores, args.cores * args.rows
        pages = count * WIDTH
        log(f"device open; {cores} cores, {args.rows} rows of {WIDTH} tiles each")
        data = rules.inputs_of("norm", count, args.seed)
        x = session.empty((pages, 1024), "float32")
        w = session.empty((1, 1024), "float32")
        session.write(x, data["a"].view(np.uint8).reshape(pages, -1))
        session.write(w, data["p"].view(np.uint8).reshape(1, -1))
        out = session.empty((pages, 1024), "float32")
        stamps = session.empty((cores, 256), "uint32")
        program = RowNorm(x=x, w=w, out=out, stamps=stamps, log=log)
        run = session.load(program)
        report["kernel_text"] = program.sizes.text
        wanted = rules.reference("norm", data)
        try:
            words = (x, w, out, stamps, args.rows)
            submitted = run.submit(RowNorm.norm_rows, (0, cores), args=words)
            event = run.wait(submitted, timeout=30.0)
            log(f"norm: {event}")
            record = {"event": event.kind, "microseconds": event.microseconds}
            record |= compared(session, out, wanted)
            record["spans"] = spans(session, stamps, cores, args.rows)
            report["norm"] = record
            ok &= event.kind == "done" and record["not_finite"] == 0
            ok &= record["max_relative"] <= TOLERANCE
            ok &= record["spans"]["calls_ok"]
        finally:
            run.stop()
            log("run stopped")
    report["ok"] = bool(ok)
    print(json.dumps(report, default=float), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
