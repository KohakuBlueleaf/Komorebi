"""Broadcast written as a class: BROADCAST of ../broadcast.py as one method, its row
broadcast on mover A and its column broadcast on mover B, compiled into the program's
files (build/broadcasts-<hash>/ beside this file); the same jobs and checks as
../broadcast.py. See README.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.dsl.program import lines_of

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import broadcast as hand  # noqa: E402  (the hand program's data and checks)


class Broadcasts(km.Program):
    """Each core of the rectangle adds every tile of `blocks` blocks of its row's
    `rows` pages and of its column's `columns` pages into one tile, stored at output
    page first_page + rank; block k is data block k mod `data_blocks`."""

    PAGES = km.constant(2)  # tiles a block; the make gives the run's (--pages)
    WINDOW = km.constant(2)  # blocks in flight; the make gives the run's (--window)

    @km.handler
    def broadcast(
        self,
        rows: km.Tensor,
        columns: km.Tensor,
        output: km.Tensor,
        blocks: int,
        first_page: int,
        data_blocks: int,
    ):
        row = km.Broadcast(self, along="row", window=self.WINDOW)
        column = km.Broadcast(self, along="column", window=self.WINDOW)
        total = km.tile.zeros(km.fp32)
        for block in range(blocks):
            first_row = (self.y * data_blocks + block % data_blocks) * self.PAGES
            if row.sender():
                along_row = row.cast(rows.pages(first_row, self.PAGES).read())
            else:
                along_row = row.arrive()
            for tile in along_row:
                total += tile
            first_column = (self.x * data_blocks + block % data_blocks) * self.PAGES
            if column.sender():
                down_column = column.cast(
                    columns.pages(first_column, self.PAGES).read()
                )
            else:
                down_column = column.arrive()
            for tile in down_column:
                total += tile
        output.pages(first_page + self.rank, 1).write(total)


def run(session, args, generator):
    """The hand program's run_host() with the class's program and its command words:
    the same group, data, jobs and check."""
    nocs = session.nocs()
    rect = tuple(int(field) for field in args.rect.split(","))
    group, _, cells = hand.broadcast_plan(
        nocs, len(nocs.words), rect, args.window, args.pages
    )
    sets, inputs, output = hand.make_data(session, generator, group, rect, args)
    program = Broadcasts(
        rows=inputs[0][0],
        columns=inputs[0][1],
        output=output,
        PAGES=args.pages,
        WINDOW=args.window,
        rect=rect,
        log=hand.log,
    )
    run = session.load(program, deadline=args.deadline, wait=args.wait)
    hand.log("loaded")
    start = time.perf_counter()
    try:
        for job in range(args.jobs):
            rows, columns = inputs[job % 2]
            run.submit(
                Broadcasts.broadcast,
                (group.first, group.count),
                args=(
                    rows,
                    columns,
                    output,
                    hand.job_blocks(args, job),
                    job * group.count,
                    args.blocks,
                ),
                exclusive=True,
            )
        hand.log(f"{run.drain(timeout=10.0)} events")
    except km.CommandTimeout as error:
        hand.log(f"timeout: {error}")
        for line in lines_of(program, error):
            hand.log(f"  {line}")
        raise
    host_seconds = time.perf_counter() - start
    run.stop()
    result = run.result()
    states = run.state()
    report = hand.describe(args, rect, group) | {
        "kernel": {key: result[key] for key in ("error_name", "commands", "events")},
        "kernel_text": program.sizes.text,
        "microseconds_per_job_host": round(host_seconds * 1e6 / args.jobs, 2),
        "cores_magic_ok": sum(core["magic_ok"] for core in states),
        "cores_ended": hand.kernel_ends(states),
        **hand.broadcast_summary([core["broadcasts"] for core in states], group, cells),
        "core_gave_up": hand.core_gave_up(states),
    }
    if not args.no_check:
        report.update(hand.check(args, sets, output, group, cells))
    report["ok"] = hand.verdict(
        report, len(states), False, result["error_name"] == "none"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--beside", action="store_true", help="the hand program first")
    ours, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    args = hand.parse()
    if args.read or args.feeder or args.stall_job >= 0:
        raise SystemExit("the class casts (no --read), by the dispatcher, no stall")
    report = {"runs": {}}
    with km.Session(log=hand.log) as session:
        hand.log("device open")
        if ours.beside:
            generator = np.random.default_rng(args.seed)
            report["runs"]["hand"] = hand.run_host(session, args, generator, -1)
            hand.log(f"hand program: ok {report['runs']['hand']['ok']}")
        generator = np.random.default_rng(args.seed)
        report["runs"]["class"] = run(session, args, generator)
        hand.log(f"class: ok {report['runs']['class']['ok']}")
    report["ok"] = all(each["ok"] for each in report["runs"].values())
    report["wall_s"] = round(time.monotonic() - hand.STARTED, 1)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
