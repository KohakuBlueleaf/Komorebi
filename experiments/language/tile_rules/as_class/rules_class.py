"""The residual add and the scale of ../rules.py as tile functions of a class: the
load-macro forms the tile compiler emits, called on a stream's tiles, timed on the
math CPU and checked against float64 (build/rules-<hash>/ here). See README.md.
Example: python rules_class.py --cores 12 --tiles 64
"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km

STARTED = time.monotonic()
SPANS_EMPTY, SPANS_FIRST = 16, 18  # a stamps page: the empty spans, then each call's
SCALE = 0.08838834764831845  # 1 / sqrt(128), rounded to fp32 by the body
TOLERANCE = 2.0**-23  # one fp32 rounding, as ../rules.py states it


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class Rules(km.Program):
    """add: out = a + b; scale: out = a x SCALE; each over the core's `tiles` pages
    from rank x tiles, the math CPU's cycles of each call to page rank of stamps."""

    BLOCK = km.constant(8)  # tiles a block

    @km.tile_function(tolerance=TOLERANCE)
    def residual(a, b):
        return a + b

    @km.tile_function(tolerance=TOLERANCE)
    def scaled(x):
        return x * SCALE

    @km.handler(stamps="stamps")
    def add(
        self, a: km.Tensor, b: km.Tensor, out: km.Tensor, stamps: km.Tensor, tiles: int
    ):
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=self.BLOCK)
        for x, y in zip(
            km.Stream(a.pages(first, tiles), block=self.BLOCK),
            km.Stream(b.pages(first, tiles), block=self.BLOCK),
        ):
            results.push(self.residual(x, y))

    @km.handler(stamps="stamps")
    def scale(self, a: km.Tensor, out: km.Tensor, stamps: km.Tensor, tiles: int):
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=self.BLOCK)
        for x in km.Stream(a.pages(first, tiles), block=self.BLOCK):
            results.push(self.scaled(x))


def inputs(rng, pages):
    """Normal numbers times 8 as float32 (../rules.py's), and one page over
    magnitudes 2^-40 .. 2^40 with random signs."""
    values = (rng.standard_normal((pages, 1024)) * 8).astype(np.float32)
    exponents = rng.uniform(-40, 40, 1024)
    signs = np.sign(rng.standard_normal(1024))
    values[1] = (signs * 2.0**exponents).astype(np.float32)
    return values


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


def spans(session, stamps, cores, tiles):
    """The math CPU's cycles a call: the calls' median less the median empty span."""
    (raw,) = session.read(stamps)
    pages = raw.view(np.uint32).reshape(cores, -1).astype(np.int64)
    empty = float(np.median(pages[:, 2 : 2 + SPANS_EMPTY]))
    calls = pages[:, SPANS_FIRST : SPANS_FIRST + tiles]
    return {
        "calls_ok": bool((pages[:, 1] == tiles).all()),
        "empty_span": empty,
        "median_less_empty": float(np.median(calls)) - empty,
        "min_call": int(calls.min()),
        "max_call": int(calls.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=12)
    parser.add_argument("--tiles", type=int, default=64, help="tiles a core takes")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    block = Rules.BLOCK.default
    if args.tiles % block or not block <= args.tiles <= 256 - SPANS_FIRST:
        parser.error(f"--tiles is a multiple of {block} up to {256 - SPANS_FIRST}")
    report, ok = {"tolerance": TOLERANCE}, True
    with km.Session(log=log) as session:
        cores, pages = args.cores, args.cores * args.tiles
        log(f"device open; {cores} cores, {args.tiles} tiles each")
        rng = np.random.default_rng(args.seed)
        values = {name: inputs(rng, pages) for name in ("a", "b")}
        tensors = {name: session.empty((pages, 1024), "float32") for name in values}
        for name, raw in values.items():
            session.write(tensors[name], raw.view(np.uint8).reshape(pages, -1))
        out = session.empty((pages, 1024), "float32")
        stamps = session.empty((cores, 256), "uint32")
        program = Rules(out=out, stamps=stamps, log=log, **tensors)
        run = session.load(program)
        report["kernel_text"] = program.sizes.text
        a, b = (values[name].astype(np.float64) for name in ("a", "b"))
        wanted = {"add": a + b, "scale": a * float(np.float32(SCALE))}
        try:
            for name in ("add", "scale"):
                handler = getattr(Rules, name)
                words = (tensors["a"], *([tensors["b"]] if name == "add" else []))
                words += (out, stamps, args.tiles)
                submitted = run.submit(handler, (0, cores), args=words)
                event = run.wait(submitted, timeout=30.0)
                log(f"{name}: {event}")
                record = {"event": event.kind, "microseconds": event.microseconds}
                record |= compared(session, out, wanted[name])
                record["spans"] = spans(session, stamps, cores, args.tiles)
                report[name] = record
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
