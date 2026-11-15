"""Map/collect written as a class: MAP of ../mapcollect.py as one method, compiled into
the program's files (build/mapcollect/ beside this file); the same jobs, checks and
JSON line as ../mapcollect.py. See README.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi.dsl.program import lines_of

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import mapcollect as hand  # noqa: E402  (the hand program's jobs and checks)

TILE = hand.TILE


class MapCollect(km.Program):
    """Each core of a MAP command's group adds its part of `tiles` input tiles from
    `first_tile` into one tile and stores it at output page first_page + rank."""

    @km.handler
    def map(
        self,
        source: km.Tensor,
        tiles: int,
        output: km.Tensor,
        first_page: int,
        first_tile: int,
    ):
        part = self.partition(source.pages(first_tile, tiles))
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(source[part], block=8):
            total += tile
        output.pages(first_page + self.rank, 1).write(total)


def run(session, args, generator):
    """The hand program's run() with the class's program and its command words."""
    grid_width, grid_height = session.grid
    cores = grid_width * grid_height
    groups = hand.make_groups(args, cores, generator)
    job_count, tiles = args.jobs, args.tiles
    input_tiles = max(args.in_tiles, tiles)
    exact_input = generator.integers(-2, 3, (input_tiles * TILE, TILE)).astype(
        np.float32
    )
    random_input = generator.uniform(-1, 1, (input_tiles * TILE, TILE)).astype(
        np.float32
    )
    random_input = formats.bf16_values(formats.bf16_bits(random_input))

    def tile_tensor(array):
        return session.tensor(array, "bfloat16", "tile")

    inputs = [tile_tensor(exact_input), tile_tensor(random_input)]
    pages = args.out_pages
    outputs = [
        tile_tensor(np.zeros((pages * TILE, TILE), np.float32))
        for _ in range(args.out_tensors)
    ]
    first_pages = hand.place_outputs(groups, pages, args.out_tensors)
    first_tiles = hand.input_pages(job_count, tiles, input_tiles)

    program = MapCollect(source=inputs[0], output=outputs[0], depth=args.depth)
    run = session.load(
        program,
        deadline=args.deadline,
        batch=args.batch,
        settle_microseconds=args.settle_microseconds,
    )
    hand.log(f"loaded: {cores} cores, depth {args.depth}")
    words = np.array(
        [
            MapCollect.map.args(
                inputs[job % 2],
                tiles,
                outputs[first_pages[job] // pages],
                first_pages[job] % pages,
                first_tiles[job],
            )
            for job in range(job_count)
        ],
        np.int64,
    )
    handlers = np.full(job_count, int(program.map))
    firsts = np.array([first for first, _ in groups])
    counts = np.array([count for _, count in groups])
    start = time.perf_counter()
    try:
        handles = list(run.submit_many(handlers, firsts, counts, words))
        waited = run.drain(timeout=60.0)
        hand.log(f"{waited} events")
    except km.CommandTimeout as error:
        hand.log(f"timeout: {error}")
        for line in lines_of(program, error):
            hand.log(f"  {line}")
        raise
    host_seconds = time.perf_counter() - start
    run.stop()
    result = run.result()
    report = {
        "cores": cores,
        "depth": args.depth,
        "count": args.count if args.groups in ("fixed", "rotate") else None,
        "tiles": tiles,
        "kernel": {
            key: result[key] for key in ("error_name", "commands", "events", "max_busy")
        },
        "microseconds_per_job_kernel": round(
            result["last"] / 50 / max(result["commands"], 1), 3
        ),
        "microseconds_per_job_host": round(host_seconds * 1e6 / job_count, 3),
        "kernel_text": program.sizes.text if program.sizes else None,
    }
    done = [
        run.event(h) for h in handles if run.event(h) and run.event(h).kind == "done"
    ]
    if done:
        latencies = np.array([event.microseconds for event in done])
        report["latency_microseconds"] = {
            "mean": round(float(latencies.mean()), 3),
            "max": round(float(latencies.max()), 3),
        }
    states = run.state()
    report["cores_magic_ok"] = sum(state["magic_ok"] for state in states)
    report["cores_ended"] = {
        key: sum(state[key] != 0 for state in states)
        for key in ("reader_end", "writer_end", "compute_end")
    }
    report["compute_wrong"] = sum(sum(state["compute_wrong"]) for state in states)
    report["gave_up_cores"] = sum(
        1 for state in states if state["reader"] or state["writer"]
    )
    report.update(hand.core_counts(states, groups, cores))
    report["recovery_refused"] = run.recovery_refused
    report["recoveries"] = []
    if not args.no_check:
        report.update(
            hand.check(
                session,
                args,
                groups,
                first_pages,
                first_tiles,
                outputs,
                [exact_input, random_input],
            )
        )
    report["ok"] = hand.verdict(report, len(states))
    return report


def parse():
    """This script's flags, then the hand program's (../mapcollect.py --help)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--beside", action="store_true", help="the hand program first, same session"
    )
    parser.add_argument(
        "--timing", type=int, default=0, help="then N jobs on the whole grid, unchecked"
    )
    ours, rest = parser.parse_known_args()
    return ours, hand.parse(rest)


def both(session, ours, args):
    """The class's run (and the hand program's before it, with --beside) of `args`,
    each from the same seed."""
    runs = {}
    if ours.beside:
        runs["hand"] = hand.run(session, args, np.random.default_rng(args.seed))
        hand.log(f"hand program: ok {runs['hand']['ok']}")
    runs["class"] = run(session, args, np.random.default_rng(args.seed))
    hand.log(f"class: ok {runs['class']['ok']}")
    return runs


def main():
    ours, args = parse()
    report = {"jobs": args.jobs, "handler": "map", "groups": args.groups}
    with km.Session(log=hand.log) as session:
        hand.log("device open")
        report["runs"] = both(session, ours, args)
        if ours.timing:
            grid = session.grid
            timed = hand.parse(
                ["--jobs", str(ours.timing), "--count", str(grid[0] * grid[1])]
                + ["--no-check"]
            )
            report["timing"] = both(session, ours, timed)
    report["wall_s"] = round(time.monotonic() - hand.STARTED, 1)
    runs = [*report["runs"].values(), *report.get("timing", {}).values()]
    report["ok"] = all(each["ok"] for each in runs)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
