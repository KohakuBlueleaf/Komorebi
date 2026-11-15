"""Overlays written as a class: mix and scale of ../overlays.py as two handlers whose
code is an overlay each (built apart, loaded into a slot when a command needs it,
called), compiled into the program's files (build/overlays-<hash>/ here). The
same turns and checks as ../overlays.py (beside it with --beside). See README.md.
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

import overlays as hand  # noqa: E402  (the hand program, its loop and its values)


class Overlays(km.Program):
    """mix: each core's word mixed `steps` times, written to its page of words. scale:
    each core's `tiles` tiles of source times `scalar` (float32 bits) into output, the
    same pages. Both: each mover's load and call cycles to its page of stamps."""

    @km.handler(overlay=True, stamps="stamps")
    def mix(self, words: km.Tensor, stamps: km.Tensor, seed: int, steps: int):
        word = km.cxx(
            """
            word = seed ^ movement.me();
            for (uint32_t step = 0; step < steps; step++) {
                word = word * 2654435761u + step;
                word ^= word >> 13;
            }
            """,
            takes=[seed, steps],
            gives=int,
            on="mover_b",
        )
        words.pages(self.rank, 1).write(word)

    @km.handler(overlay=True, stamps="stamps")
    def scale(
        self,
        source: km.Tensor,
        output: km.Tensor,
        stamps: km.Tensor,
        tiles: int,
        scalar: int,
    ):
        first = self.rank * tiles
        out = km.Stream(output.pages(first, tiles), write=True)
        for tile in km.Stream(source.pages(first, tiles)):
            scaled = km.cxx(
                """
                #include "api/compute/eltwise_unary/binop_with_scalar.h"
                copy_init(tile_buffer);
                copy_tile(tile_buffer, tile_index, 0);
                binop_with_scalar_tile_init();
                mul_unary_tile(0, scalar);
                """,
                takes=[tile, scalar],
                gives=km.Tile,
            )
            out.push(scaled)


class Turns:
    """../overlays.py's turns with the class's commands, each one command to every
    core, each result checked as overlays.py checks it; the stamps read after the
    turn's time (the hand program's turn reads its lines, stamps in them)."""

    def __init__(self, session, run, cores, args, tensors):
        self.session, self.run, self.cores, self.args = session, run, cores, args
        self.source, self.output, self.words, self.stamps = tensors
        self.times = {}  # overlay -> [us a turn, from the host]
        self.cycles = {}  # (overlay, mover, loading or held) -> [(load, call)]
        self.wrong = 0
        self.commands = 0

    def once(self, name, seed):
        args, run = self.args, self.run
        started = time.perf_counter()
        if name == "mix":
            words = (self.words, self.stamps, seed, args.steps)
            event = run.wait(run.submit(Overlays.mix, (0, self.cores), args=words))
            (raw,) = self.session.read(self.words)
            got = raw.reshape(self.cores, -1).view(np.uint32)[:, 0].astype(np.int64)
            want = [hand.mix(seed ^ core, args.steps) for core in range(self.cores)]
            self.wrong += int((got != want).sum())
        else:
            scalar = float(2 + seed % 3)
            bits = int(np.float32(scalar).view(np.uint32))
            words = (self.source, self.output, self.stamps, args.tiles, bits)
            event = run.wait(run.submit(Overlays.scale, (0, self.cores), args=words))
            (raw,) = self.session.read(self.output)
            got = raw.view(np.uint16).reshape(self.cores * args.tiles, -1)
            self.wrong += int(
                (got != hand.scaled(self.cores, args.tiles, scalar)).sum()
            )
        self.times.setdefault(name, []).append((time.perf_counter() - started) * 1e6)
        self.wrong += int(event.kind != "done")
        self.commands += 1
        (raw,) = self.session.read(self.stamps)
        stamps = raw.reshape(self.cores * 2, -1).view(np.uint32).astype(np.int64)
        number = self.run.program.overlays[name].number
        self.wrong += int((stamps[:, 1] != number).sum())
        for row, (load, call, held) in enumerate(stamps[:, 2:5]):
            key = (name, "AB"[row % 2], "held" if held else "loading")
            self.cycles.setdefault(key, []).append((int(load), int(call)))


def run_class(session, args, cores):
    """One load of the class's program: overlays.py's turns, the stop, the report."""
    pages = cores * args.tiles
    source = session.empty((pages * 32, 32), "bfloat16", "tile")
    output = session.empty((pages * 32, 32), "bfloat16", "tile")
    small = hand.scaled(cores, args.tiles, 1.0)  # value (c + t + j) % 7 + 1, raw
    session.write(source, small.view(np.uint8).reshape(pages, -1))
    words = session.empty((cores, 16), "uint32")
    stamps = session.empty((cores * 2, 16), "uint32")
    program = Overlays(
        source=source, output=output, words=words, stamps=stamps, log=hand.log
    )
    run = session.load(program)
    turns = Turns(session, run, cores, args, (source, output, words, stamps))
    try:
        for seed in range(args.rounds):  # in turn: each load replaces the other
            turns.once("mix", seed)
            turns.once("scale", seed)
        for name in ("mix", "scale"):  # again and again: held after the first
            for seed in range(args.rounds):
                turns.once(name, seed)
        hand.log(f"{turns.commands} commands, {turns.wrong} wrong")
    finally:
        run.stop()
    built = program.overlays_built
    return {
        "commands": turns.commands,
        "wrong": turns.wrong,
        "exact": turns.wrong == 0,
        "kernel_text": program.sizes.text,
        "image_bytes": {
            f"{n}-{p}": len(i) for (n, p), i in sorted(built.images.items())
        },
        "slots_bytes": {part: list(value) for part, value in built.slots.items()},
        "median_turn_us": {
            name: round(float(np.median(us)), 2) for name, us in turns.times.items()
        },
        "median_cycles": {
            " ".join(key): np.median(np.asarray(pairs), axis=0).tolist()
            for key, pairs in sorted(turns.cycles.items())
        },
    }


def run_hand(session, args, cores):
    """overlays.py's program and turns in this session, timed from the host as the
    class's turns are."""
    rounds = hand.Rounds(session, cores, args, {})
    run = session.load(hand.program(args.slots, rounds.lines, rounds.tiles))
    rounds.run = run
    times = {}
    try:
        for seed in range(args.rounds):
            for name in ("mix", "scale"):
                started = time.perf_counter()
                rounds.once(name, seed)
                times.setdefault(name, []).append((time.perf_counter() - started) * 1e6)
        for name in ("mix", "scale"):
            for seed in range(args.rounds):
                started = time.perf_counter()
                rounds.once(name, seed)
                times[name].append((time.perf_counter() - started) * 1e6)
    finally:
        run.stop()
    built = run.program.overlays_built
    return {
        "commands": rounds.commands,
        "wrong": rounds.wrong,
        "exact": rounds.wrong == 0,
        "image_bytes": {
            f"{n}-{p}": len(i) for (n, p), i in sorted(built.images.items())
        },
        "slots_bytes": {part: list(value) for part, value in built.slots.items()},
        "median_turn_us": {
            name: round(float(np.median(us)), 2) for name, us in times.items()
        },
        "median_cycles": {
            f"{name} {how}": np.median(np.asarray(pairs), axis=0).tolist()
            for (name, how), pairs in sorted(rounds.times.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--beside", action="store_true", help="the hand program first")
    parser.add_argument("--rounds", type=int, default=10, help="commands per phase")
    parser.add_argument("--tiles", type=int, default=1, help="scale's tiles a core")
    parser.add_argument("--steps", type=int, default=64, help="mix's loop steps")
    args = parser.parse_args()
    args.slots = 1
    report = {"runs": {}}
    with km.Session(log=hand.log) as session:
        cores = session.grid[0] * session.grid[1]
        hand.log(f"device open; {cores} cores")
        if args.beside:
            report["runs"]["hand"] = run_hand(session, args, cores)
            hand.log(f"hand program: exact {report['runs']['hand']['exact']}")
        report["runs"]["class"] = run_class(session, args, cores)
        hand.log(f"class: exact {report['runs']['class']['exact']}")
    report["ok"] = all(each["exact"] for each in report["runs"].values())
    print(json.dumps(report, default=float), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
