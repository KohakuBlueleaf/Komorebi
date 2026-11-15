"""Overlays written as a class: mix and scale of ../overlays.py as two handlers whose
code is an overlay each (built apart, loaded into a slot when a command needs it,
called), compiled into the program's files (build/overlays/ beside this file). The
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
    """mix: a core's word mixed `steps` times, the command's result. scale: each core's
    `tiles` tiles of source times `scalar` (float32 bits) into output, the same
    pages."""

    @km.handler(overlay=True)
    def mix(self, seed: int, steps: int):
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
        if self.rank == self.leader:
            return word

    @km.handler(overlay=True)
    def scale(self, source: km.Tensor, output: km.Tensor, tiles: int, scalar: int):
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
    """../overlays.py's turns with the class's commands: mix to each core alone (its
    word the command's result: a core's own word to DRAM is not a line of a class),
    scale to every core; each result checked as overlays.py checks it."""

    def __init__(self, session, run, cores, args, source, output):
        self.session, self.run, self.cores, self.args = session, run, cores, args
        self.source, self.output = source, output
        self.times = {}  # overlay -> [us a turn, from the host]
        self.wrong = 0
        self.commands = 0

    def once(self, name, seed):
        args, run = self.args, self.run
        started = time.perf_counter()
        if name == "mix":
            handles = [
                run.submit(Overlays.mix, (core, 1), args=(seed, args.steps))
                for core in range(self.cores)
            ]
            events = run.wait(handles, 30.0)
            want = [hand.mix(seed ^ core, args.steps) for core in range(self.cores)]
            got = [event.result for event in events]
            self.wrong += sum(g != w for g, w in zip(got, want, strict=True))
        else:
            scalar = float(2 + seed % 3)
            bits = int(np.float32(scalar).view(np.uint32))
            words = (self.source, self.output, args.tiles, bits)
            events = [run.wait(run.submit(Overlays.scale, (0, self.cores), args=words))]
            (raw,) = self.session.read(self.output)
            got = raw.view(np.uint16).reshape(self.cores * args.tiles, -1)
            self.wrong += int(
                (got != hand.scaled(self.cores, args.tiles, scalar)).sum()
            )
        self.times.setdefault(name, []).append((time.perf_counter() - started) * 1e6)
        self.wrong += sum(event.kind != "done" for event in events)
        self.commands += len(events)


def run_class(session, args, cores):
    """One load of the class's program: overlays.py's turns, the stop, the report."""
    pages = cores * args.tiles
    source = session.empty((pages * 32, 32), "bfloat16", "tile")
    output = session.empty((pages * 32, 32), "bfloat16", "tile")
    small = hand.scaled(cores, args.tiles, 1.0)  # value (c + t + j) % 7 + 1, raw
    session.write(source, small.view(np.uint8).reshape(pages, -1))
    program = Overlays(source=source, output=output, log=hand.log)
    run = session.load(program)
    turns = Turns(session, run, cores, args, source, output)
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
