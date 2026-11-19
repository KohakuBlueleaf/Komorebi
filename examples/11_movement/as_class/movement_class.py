"""Both movers on one stream, written as a class: READ and COPY of ../movement.py as two
methods, compiled into the program's files (build/movement-<hash>/ here); the same
copies checked word by word and the same rates as ../movement.py. See README.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import formats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import movement as hand  # noqa: E402  (the hand program's data and timing)


class Movement(km.Program):
    """Every core reads its `tiles` pages, from page rank x tiles, through one stream
    both movers carry; READ drops them, COPY writes each block to the output."""

    BLOCK = km.constant(16)  # tiles a block; the make gives the run's (--block)

    @km.handler
    def read(
        self, source: km.Tensor, tiles: int, pattern: km.Pattern, repetitions: int
    ):
        pages = source.pages(self.rank * tiles, tiles)
        for _ in km.Stream(
            pages, block=self.BLOCK, pattern=pattern, repeat=repetitions
        ):
            pass

    @km.handler
    def copy(
        self, source: km.Tensor, output: km.Tensor, tiles: int, pattern: km.Pattern
    ):
        first = self.rank * tiles
        out = km.Stream(
            output.pages(first, tiles), write=True, block=self.BLOCK, pattern=pattern
        )
        for tile in km.Stream(
            source.pages(first, tiles), block=self.BLOCK, pattern=pattern
        ):
            out.push(tile)


def check(session, run, tensors, data, cores, tiles, pattern):
    """COPY once and read the output back: the pages that differ from the input."""
    words = Movement.copy.args(tensors["input"], tensors["output"], tiles, pattern)
    hand.submit(run, int(Movement.copy), cores, words)
    wrong = 0
    for first, pages in session.read_chunks(tensors["output"]):
        got = pages.view(np.uint16).reshape(len(pages), -1)
        last = min(first + len(pages), cores * tiles)
        if first < last:
            wrong += int((got[: last - first] != data[first:last]).any(axis=1).sum())
        hand.log(f"  {pattern}: output pages from {first} read back")
    return wrong


def rate(run, tensors, cores, tiles, pattern):
    """GB/s read from R and 4R repetitions, best of three, as the hand program's."""

    def read(repetitions):
        words = Movement.read.args(tensors["input"], tiles, pattern, repetitions)
        return hand.submit(run, int(Movement.read), cores, words)

    once = read(1)
    repetitions = max(2, int(hand.TARGET_MICROSECONDS / max(once, 1.0)))
    best = None
    for _ in range(3):
        span = read(4 * repetitions) - read(repetitions)
        best = span if best is None else min(best, span)
    moved = cores * tiles * hand.PAGE_BYTES * 3 * repetitions
    return round(moved / (best * 1e-6) / 1e9, 2), repetitions


def patterns_of(session, run, tensors, data, args, cores, check_tiles, ours):
    """Each pattern's copy check and rate, by the class's program (`ours`) or the
    hand program's."""
    results = []
    for pattern in args.pattern:
        if ours:
            wrong = check(session, run, tensors, data, cores, check_tiles, pattern)
        else:
            wrong = hand.check(
                session, run, tensors, data, cores, check_tiles, args.block, pattern
            )
        result = {"pattern": str(pattern), "pages_wrong": wrong}
        if not wrong and not args.check_only:
            if ours:
                result["rate_gbs"], result["repetitions"] = rate(
                    run, tensors, cores, args.tiles, pattern
                )
            else:
                result["rate_gbs"], result["repetitions"] = hand.rate(
                    run, tensors, cores, args.tiles, args.block, pattern
                )
        hand.log(f"{'class' if ours else 'hand'} pattern {pattern}: {result}")
        results.append(result)
    return results


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pattern", nargs="+", default=["AB"])
    parser.add_argument("--cores", type=int, default=0, help="0: the session's ranks")
    parser.add_argument("--tiles", type=int, default=256, help="tiles a core reads")
    parser.add_argument(
        "--block", type=int, default=Movement.BLOCK.default, help="tiles per block"
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--beside", action="store_true", help="the hand program first")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    args.pattern = [km.Pattern("A")] + [km.Pattern(p) for p in args.pattern if p != "A"]
    return args


def main():
    args = parse()
    check_tiles = args.block + args.block // 2  # one whole block and a ragged one
    report = {"tiles": args.tiles, "block": args.block, "runs": {}}
    with km.Session() as session:
        grid = session.grid
        cores = args.cores or session.ranks  # the grid less a Tensix dispatcher's
        rng = np.random.default_rng(args.seed)
        hand.log(f"device open, grid {grid}; {cores} cores, {args.tiles} tiles each")
        data = hand.pages_of(rng, cores * args.tiles)
        # The pages as bfloat16 (the same bits): the class's buffers take the format
        # of the tensor a stream reads.
        tensors = {
            "input": session.tensor(formats.bf16_values(data), "bfloat16"),
            "output": session.tensor(
                np.zeros((cores * check_tiles, data.shape[1]), np.float32), "bfloat16"
            ),
        }
        hand.log("tensors written")
        programs = {}
        if args.beside:
            programs["hand"] = hand.program(tensors, args.block)
        programs["class"] = Movement(
            source=tensors["input"],
            output=tensors["output"],
            BLOCK=args.block,
            log=hand.log,
        )
        for name, program in programs.items():
            run = session.load(program)
            try:
                report["runs"][name] = {
                    "kernel_text": program.sizes.text,
                    "patterns": patterns_of(
                        session,
                        run,
                        tensors,
                        data[: cores * check_tiles],
                        args,
                        cores,
                        check_tiles,
                        name == "class",
                    ),
                }
            finally:
                run.stop()
                hand.log(f"{name}: run stopped")
    report["cores"] = cores
    report["exact"] = all(
        result["pages_wrong"] == 0
        for run in report["runs"].values()
        for result in run["patterns"]
    )
    report["wall_s"] = round(time.monotonic() - hand.STARTED, 1)
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report["exact"] else 1)


if __name__ == "__main__":
    main()
