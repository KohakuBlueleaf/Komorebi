"""silu(a @ w1) * (a @ w3) of ../rules.py's BODY 29 as a class's tile function: two
tile products and the gate in one body, called on three streams' tiles, timed on the
math CPU and checked against float64 (build/mmgate-<hash>/ here). See README.md.
Example: python mm_gate_class.py --cores 12 --tiles 64
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

import rules  # noqa: E402  (BODY 29's values, reference and tolerance)

STARTED = time.monotonic()
BLOCK = 8  # tiles a block
SPANS_EMPTY, SPANS_FIRST = 16, 18  # a stamps page: the empty spans, then each call's
TOLERANCE = rules.TOLERANCE["mm_gate"]  # 2^-7, as ../rules.py states it


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class MmGate(km.Program):
    """gate: each core's `tiles` pages of a, w1 and w3 from rank x tiles, the gate of
    each tile's two products into the same pages of out, the math CPU's cycles of
    each call to page rank of stamps."""

    @km.tile_function(tolerance=TOLERANCE)
    def gate(a, w1, w3):
        return km.silu(a @ w1) * (a @ w3)

    @km.handler(stamps="stamps")
    def run(
        self,
        a: km.Tensor,
        w1: km.Tensor,
        w3: km.Tensor,
        out: km.Tensor,
        stamps: km.Tensor,
        tiles: int,
    ):
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=BLOCK)
        for x, p, q in zip(
            km.Stream(a.pages(first, tiles), block=BLOCK),
            km.Stream(w1.pages(first, tiles), block=BLOCK),
            km.Stream(w3.pages(first, tiles), block=BLOCK),
        ):
            results.push(self.gate(x, p, q))


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
    if args.tiles % BLOCK or not BLOCK <= args.tiles <= 256 - SPANS_FIRST:
        parser.error(f"--tiles is a multiple of {BLOCK} up to {256 - SPANS_FIRST}")
    report = {"tolerance": TOLERANCE}
    with km.Session(log=log) as session:
        cores, pages = args.cores, args.cores * args.tiles
        log(f"device open; {cores} cores, {args.tiles} tiles each")
        data = rules.inputs_of("mm_gate", pages, args.seed)  # fp32 words, face order
        tensors = {}
        for name, key in (("a", "a"), ("w1", "b"), ("w3", "p")):
            tiles = data[key].view(np.float32)
            matrix = np.concatenate([rules.matrix_of(t[None]) for t in tiles])
            tensors[name] = session.tensor(matrix, "bfloat16", "tile")
        out = session.empty((pages, 1024), "float32")
        stamps = session.empty((cores, 256), "uint32")
        program = MmGate(out=out, stamps=stamps, log=log, **tensors)
        run = session.load(program)
        report["kernel_text"] = program.sizes.text
        try:
            words = (tensors["a"], tensors["w1"], tensors["w3"], out, stamps)
            submitted = run.submit(MmGate.run, (0, cores), args=(*words, args.tiles))
            event = run.wait(submitted, timeout=30.0)
            log(f"gate: {event}")
            report["event"], report["microseconds"] = event.kind, event.microseconds
            got = np.concatenate([chunk for _, chunk in session.read_chunks(out)])
            got = got.view(np.uint32).reshape(pages, -1)
            report["error"] = rules.errors(got, rules.reference("mm_gate", data))
            report["spans"] = spans(session, stamps, cores, args.tiles)
        finally:
            run.stop()
            log("run stopped")
    ok = report["event"] == "done" and report["error"]["not_finite"] == 0
    ok = ok and report["error"]["max"] <= TOLERANCE and report["spans"]["calls_ok"]
    report["ok"] = bool(ok)
    print(json.dumps(report, default=float), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
