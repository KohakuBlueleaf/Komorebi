"""A cast of a body: the first core of each broadcast of a rectangle reads a 4 KiB body
from DRAM and casts it, and every core of the broadcast folds each word of every body
it receives; each core's fold against numpy, the time a cast. See README.md."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import layout

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels" / "cast"
CONSTANTS = km.abi.constants(KERNELS / "cast.h", known=km.abi.headers().constants)
FIELDS = km.abi.struct_fields(km.abi.headers().device[-1], "BroadcastArgs")
M32 = 0xFFFFFFFF
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--rect", default="0,0,12,1", help="first_x,first_y,columns,rows"
    )
    parser.add_argument("--along", choices=("column", "row"), default="row")
    parser.add_argument("--casts", type=int, default=1000, help="bodies a job")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--bodies", type=int, default=8, help="distinct bodies")
    parser.add_argument("--window", type=int, default=2, help="bodies in flight")
    parser.add_argument(
        "--no-fold", action="store_true", help="compute frees each body unread"
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def plan(nocs, rect, along, window):
    """The group covering `rect`, each core's arguments (reader: the broadcast's words;
    compute: included, window, the area) and the cores of the rectangle."""
    first_x, first_y, columns, rows = rect
    width = nocs.grid[0]
    first = first_y * width + first_x
    group = layout.Group(
        first, (first_y + rows - 1) * width + first_x + columns - first
    )
    common = (nocs.grid, nocs.panels, group, rect, along, 0, nocs.noc0, nocs.noc_xy)
    buffer = (0, CONSTANTS["kCbBody"], window, 1)
    words = layout.broadcast_args(*common, buffer)
    cells = [
        y * width + x
        for y in range(first_y, first_y + rows)
        for x in range(first_x, first_x + columns)
    ]
    broadcasts = layout.broadcasts(nocs.grid, nocs.panels, rect, along, 0)
    return group, words, cells, [len(each.cores) for each in broadcasts]


def folds(bodies, casts):
    """Every receiver's fold of `casts` bodies (body c is bodies[c % len(bodies)])."""
    fold = np.zeros(bodies.shape[1], dtype=np.uint64)
    for cast in range(casts):
        fold = (((fold << 1) & M32) | (fold >> 31)) ^ bodies[cast % len(bodies)]
    return fold


def main():
    args = parse()
    rect = tuple(int(field) for field in args.rect.split(","))
    generator = np.random.default_rng(args.seed)
    words = CONSTANTS["kBodyWords"]
    bodies = generator.integers(0, 1 << 32, (args.bodies, words), dtype=np.uint64)
    with km.Session(log=log) as session:
        grid = session.grid
        nocs = session.nocs()
        group, broadcast_words, cells, sizes = plan(nocs, rect, args.along, args.window)
        log(f"group {group.first}:{group.count}, broadcasts of {sizes} cores")
        bodies_tensor = session.tensor(bodies.astype(np.uint32), "uint32")
        area = session.tensor(
            np.zeros((grid[0] * grid[1], CONSTANTS["kAreaWords"]), np.int32),
            "int32",
            shard_grid=grid,
        )

        def arguments_of(core):
            if core not in broadcast_words:
                return [0] * len(FIELDS), [], [0, args.window, area.address]
            reader = [broadcast_words[core][field] for field in FIELDS]
            return reader, [], [int(core in cells), args.window, area.address]

        # --8<-- [start:program]
        program = km.Program(
            "cast_body",
            kernels=KERNELS,
            handlers={"cast": CONSTANTS["kHandlerCast"]},
            buffers=[
                km.Buffer(
                    CONSTANTS["kCbBody"], "uint32", CONSTANTS["kBodyBytes"], args.window
                )
            ],
            args=arguments_of,
            compile_args={"reader": bodies_tensor.accessor_args()},
        )
        # --8<-- [end:program]
        run = session.load(program)
        log("program loaded")
        fold = 0 if args.no_fold else 1
        command = [bodies_tensor.address, args.bodies, fold]
        microseconds, bad = [], []
        want = folds(bodies, args.casts).astype(np.int64)
        for job in range(args.jobs):  # each job folds afresh: the last is read
            handle = run.submit(
                "cast",
                group=(group.first, group.count),
                args=[args.casts, *command],
                exclusive=True,
            )
            run.drain(timeout=10.0)
            event, sequence = run.event(handle), run.sequence(handle)
            microseconds.append(event.microseconds)
            log(f"job {job + 1} of {args.jobs}: {event.microseconds:.1f} us")
        run.stop()  # the area is read once the grid has returned
        rows = (area.read().astype(np.int64) & M32)[cells]
        log("areas read back")
    for core, row in zip(cells, rows):
        right = (
            row[CONSTANTS["kResCasts"]] == args.casts
            and row[CONSTANTS["kResRun"]] == sequence
            and (args.no_fold or (row[:words] == want).all())
        )
        if not right:
            differ = np.flatnonzero(row[:words] != want)
            bad.append(
                {
                    "core": core,
                    "casts": int(row[CONSTANTS["kResCasts"]]),
                    "run": int(row[CONSTANTS["kResRun"]]),
                    "sequence": int(sequence),
                    "words_wrong": len(differ),
                    "first_wrong": [
                        [int(i), int(row[i]), int(want[i])] for i in differ[:3]
                    ],
                }
            )
    per_job = float(np.median(microseconds))
    out = {
        "rect": rect,
        "along": args.along,
        "broadcasts": sizes,
        "cores": len(cells),
        "casts_per_job": args.casts,
        "fold": not args.no_fold,
        "body_words_checked": 0 if args.no_fold else args.casts * words * len(cells),
        "cores_bad": len(bad),
        "first_bad": bad[:8],
        "microseconds_per_job": [round(each, 1) for each in microseconds],
        "microseconds_per_cast": round(per_job / args.casts, 3),
        "ok": not bad,
    }
    print(
        f"{args.casts} bodies a job to {len(cells)} cores ({sizes}), "
        f"{len(bad)} wrong, {out['microseconds_per_cast']} us a cast"
    )
    print(json.dumps(out))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
