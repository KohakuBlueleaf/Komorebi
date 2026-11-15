"""Several casters into one rectangle: the counted VALID's broadcast down one column,
every core of it casting its own part of each long block at once, every word of every
block and the count beside it checked on every core (`--casters 0`: the sender casts
each block whole, as the broadcast does by default). See README.md."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import layout

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
CONSTANTS = km.abi.constants(KERNELS / "casters.h", known=km.abi.headers().constants)
FIELDS = km.abi.struct_fields(KERNELS / "counted.hpp", "BroadcastArgs")  # + casters
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--rect", default="0,0,1,10", help="first_x,first_y,columns,rows"
    )
    parser.add_argument(
        "--casters", type=int, default=None, help="casters a broadcast (default: all)"
    )
    parser.add_argument("--pages", type=int, default=32, help="2 KB pages a block")
    parser.add_argument("--window", type=int, default=2, help="blocks in flight")
    parser.add_argument("--blocks", type=int, default=2000, help="blocks a job")
    parser.add_argument("--jobs", type=int, default=100, help="jobs at most")
    parser.add_argument("--seconds", type=float, default=20.0, help="jobs until then")
    parser.add_argument("--json", default=None, help="the result, written there too")
    return parser.parse_args()


def cast_line(broadcast, noc_xy):
    """The receivers' NoC addresses one step a rank along one axis from the first, as
    a caster finds the others (km::counted::Broadcast::along)."""
    places = [noc_xy[core] for core in broadcast.cores[1:]]
    first, last = places[0], places[-1]
    step = [(b > a) - (b < a) for a, b in zip(first, last, strict=True)]
    for rank, place in enumerate(places):
        if place != (first[0] + rank * step[0], first[1] + rank * step[1]) or (
            step[0] and step[1]
        ):
            raise ValueError(f"{broadcast.cores} are not one line of NoC addresses")


def plan(nocs, rect, casters, window, pages):
    """The group covering `rect`, each core's program arguments and the cores."""
    first_x, first_y, columns, rows = rect
    width = nocs.grid[0]
    first = first_y * width + first_x
    group = layout.Group(
        first, (first_y + rows - 1) * width + first_x + columns - first
    )
    words = layout.broadcast_args(
        nocs.grid,
        nocs.panels,
        group,
        rect,
        "column",
        0,
        nocs.noc0,
        nocs.noc_xy,
        (0, CONSTANTS["kCbBlocks"], window, pages),
    )
    words = {core: word | {"casters": 0} for core, word in words.items()}
    for broadcast in layout.broadcasts(nocs.grid, nocs.panels, rect, "column", 0):
        casting = min(casters, len(broadcast.cores))
        if casting > 1:
            cast_line(broadcast, nocs.noc_xy)
        for x, y in broadcast.cores:
            words[y * width + x]["casters"] = casting
    cells = [
        y * width + x
        for y in range(first_y, first_y + rows)
        for x in range(first_x, first_x + columns)
    ]

    def arguments_of(core):
        if core not in words:
            return [0] * len(FIELDS), [0], [0, window, pages]
        included = int(core in cells)
        return [words[core][field] for field in FIELDS], [0], [included, window, pages]

    return group, arguments_of, cells


def main():
    args = parse()
    rect = tuple(int(field) for field in args.rect.split(","))
    casters = rect[3] if args.casters is None else args.casters
    with km.Session() as session:
        nocs = session.nocs()
        group, arguments_of, cells = plan(nocs, rect, casters, args.window, args.pages)
        log(f"group {group.first}:{group.count}, {len(cells)} cores, {casters} casters")
        # every job's rows, read back once the program has stopped (a read waits for
        # the cores to be idle)
        results = session.tensor(
            np.zeros((args.jobs * group.count, CONSTANTS["kResultWords"]), np.uint32),
            "uint32",
        )
        program = km.Program(
            "casters",
            kernels=KERNELS,
            handlers=KERNELS / "casters.h",
            buffers=[
                km.Buffer(
                    CONSTANTS["kCbBlocks"], "bfloat16", 2048, args.window * args.pages
                ),
                km.Buffer(CONSTANTS["kCbResult"], "uint32", results.page_bytes, 1),
            ],
            args=arguments_of,
            compile_args={"reader": results.tensor_args()},
            defines={"KM_BROADCAST_COUNTED": 1},
        )
        run = session.load(program)
        log("program loaded")
        microseconds, bad, job = [], [], 0
        ended = time.monotonic() + args.seconds

        def check(jobs):
            got = results.read().reshape(args.jobs, group.count, -1)
            for done in jobs:
                for core in cells:
                    row = got[done, core - group.first]
                    wrong, first_block, first_place, blocks, out, high, left = row[:7]
                    right = (
                        wrong == 0
                        and blocks == args.blocks
                        and out == 0
                        and left == 0
                        and high <= casters * args.blocks
                    )
                    if not right:
                        bad.append(
                            {
                                "job": done,
                                "core": core,
                                "words_wrong": int(wrong),
                                "first_block": int(first_block),
                                "first_place": int(first_place),
                                "blocks": int(blocks),
                                "counts_out": int(out),
                                "count_high": int(high),
                                "count_left": int(left),
                            }
                        )

        while job < args.jobs and (job == 0 or time.monotonic() < ended):
            job_args = [
                results.address,
                args.blocks,
                job * args.blocks,
                job * group.count,
            ]
            handle = run.submit(
                "cast", group=(group.first, group.count), args=job_args, exclusive=True
            )
            microseconds.append(run.wait(handle, timeout=10.0).microseconds)
            job += 1
            if job % 20 == 0:
                log(f"job {job} done")
        run.stop()
        check(range(job))
        log(f"{job} jobs checked, {len(bad)} rows wrong")
    block_bytes = args.pages * 2048
    per_job = float(np.median(microseconds))
    out = {
        "rect": rect,
        "cores": len(cells),
        "casters": casters,
        "block_bytes": block_bytes,
        "window": args.window,
        "jobs": job,
        "blocks_per_job": args.blocks,
        "casts": job * args.blocks * max(casters, 1),
        "words_checked": job * args.blocks * block_bytes // 4 * len(cells),
        "rows_bad": len(bad),
        "first_bad": bad[:8],
        "microseconds_per_job_median": round(per_job, 1),
        "microseconds_per_block": round(per_job / args.blocks, 3),
        "ok": not bad,
    }
    print(
        f"{out['casts']} casts, {out['words_checked']} words checked, "
        f"{out['rows_bad']} rows wrong, {out['microseconds_per_block']} us a block"
    )
    print(json.dumps(out))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
