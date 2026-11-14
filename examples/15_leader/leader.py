"""A leader picks what its group does next: the first core of each broadcast reads a
value from DRAM, picks one of four steps from it and casts its choice as a word; every
core of the broadcast takes that step. Every core's every word and its end state are
checked against numpy (`--carry`: casts also carry a tile). See README.md."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import layout

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
CONSTANTS = km.abi.constants(KERNELS / "leader.h", known=km.abi.headers().constants)
FIELDS = km.abi.struct_fields(km.abi.headers().device[-1], "BroadcastArgs")
TILE = 32
M32 = 0xFFFFFFFF
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--rect", default="0,0,1,10", help="first_x,first_y,columns,rows"
    )
    parser.add_argument("--along", choices=("column", "row"), default="column")
    parser.add_argument("--casts", type=int, default=1000, help="casts per job")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--carry",
        choices=("never", "always", "alternate"),
        default="never",
        help="jobs whose casts carry a tile with the word (alternate: odd jobs)",
    )
    parser.add_argument("--window", type=int, default=2, help="tile blocks in flight")
    parser.add_argument("--tiles", type=int, default=8, help="tiles of data cast")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


# --8<-- [start:steps]
def pick(values):
    """The leader's choice from each value: which quarter of the range it falls in."""
    return values >> 30


def steps(choices, first):
    """A core's state after the steps `choices` from value `first` on (leader.h)."""
    state = first
    for offset, choice in enumerate(choices.tolist()):
        index = first + offset
        if choice == 0:
            state = (state * 33 + index) & M32
        elif choice == 1:
            state ^= (state << 7) & M32
        elif choice == 2:
            state = (state + 0x9E3779B9) & M32
        else:
            state = (state >> 3 | state << 29) & M32
    return state


# --8<-- [end:steps]


def plan(nocs, rect, along, window):
    """The group covering `rect`, each core's program arguments (reader: the words of
    broadcast 0, without a buffer, then broadcast 1's; writer and compute: included,
    window) and the cores of the rectangle."""
    first_x, first_y, columns, rows = rect
    width = nocs.grid[0]
    first = first_y * width + first_x
    group = layout.Group(
        first, (first_y + rows - 1) * width + first_x + columns - first
    )
    common = (nocs.grid, nocs.panels, group, rect, along, 0, nocs.noc0, nocs.noc_xy)
    words = layout.broadcast_args(*common, (0, 0, 0, 0))
    tiles = layout.broadcast_args(*common, (1, CONSTANTS["kCbTiles"], window, 1))
    cells = [
        y * width + x
        for y in range(first_y, first_y + rows)
        for x in range(first_x, first_x + columns)
    ]

    def arguments_of(core):
        if core not in words:
            return [0] * 2 * len(FIELDS), [0], [0, window]
        reader = [words[core][field] for field in FIELDS]
        reader += [tiles[core][field] for field in FIELDS]
        included = int(core in cells)
        return reader, [included], [included, window]

    return group, arguments_of, cells


def carries(args, job):
    """True when job `job`'s casts carry a tile with the word (handler CARRY)."""
    return args.carry == "always" or (args.carry == "alternate" and job % 2 == 1)


def main():
    args = parse()
    rect = tuple(int(field) for field in args.rect.split(","))
    generator = np.random.default_rng(args.seed)
    count = args.jobs * args.casts
    value_pages = -(-count // CONSTANTS["kValuesPage"])
    values = generator.integers(
        0, 1 << 32, (value_pages, CONSTANTS["kValuesPage"]), dtype=np.uint64
    ).astype(np.uint32)
    tile_data = generator.integers(-2, 3, (args.tiles, TILE, TILE)).astype(np.float32)
    with km.Session() as session:
        nocs = session.nocs()
        group, arguments_of, cells = plan(nocs, rect, args.along, args.window)
        log(f"group {group.first}:{group.count}, {len(cells)} cores along {args.along}")
        values_tensor = session.tensor(values, "uint32")
        tiles = session.tensor(tile_data.reshape(-1, TILE), "bfloat16", "tile")
        result_rows = args.jobs * group.count
        results = session.tensor(
            np.zeros((result_rows, CONSTANTS["kResultWords"]), np.uint32), "uint32"
        )
        sums = session.tensor(
            np.zeros((result_rows * TILE, TILE), np.float32), "float32", "tile"
        )
        program = km.Program(
            "leader",
            kernels=KERNELS,
            handlers=KERNELS / "leader.h",
            buffers=[
                km.Buffer(CONSTANTS["kCbTiles"], "bfloat16", 2048, args.window),
                km.Buffer(
                    CONSTANTS["kCbValues"], "uint32", values_tensor.page_bytes, 1
                ),
                km.Buffer(CONSTANTS["kCbResult"], "uint32", results.page_bytes, 1),
                km.Buffer(CONSTANTS["kCbSum"], "float32", 4096, 1),
            ],
            args=arguments_of,
            compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
            compile_args={
                "reader": values_tensor.tensor_args()
                + tiles.tensor_args()
                + results.tensor_args(),
                "writer": sums.tensor_args(),
            },
        )
        run = session.load(program)
        log("program loaded")
        microseconds = []
        for job in range(args.jobs):
            # --8<-- [start:submit]
            job_args = [
                values_tensor.address,
                args.casts,
                results.address,
                job * args.casts,
                job * group.count,
                tiles.address,
                args.tiles,
                sums.address,
            ]
            handler = "carry" if carries(args, job) else "lead"
            handle = run.submit(
                handler, group=(group.first, group.count), args=job_args, exclusive=True
            )
            microseconds.append(run.wait(handle, timeout=10.0).microseconds)
            # --8<-- [end:submit]
            if job % 10 == 9 or job == args.jobs - 1:
                log(f"job {job + 1} of {args.jobs} done")
        run.stop()
        rows = results.read().reshape(args.jobs, group.count, -1)
        log("results read back")
        got_sums = (
            sums.read().reshape(args.jobs, group.count, TILE, TILE)
            if args.carry != "never"
            else None
        )
        log("sums read back")
    flat = values.reshape(-1)[:count]
    choices = pick(flat).astype(np.int64)
    bad = []
    for job in range(args.jobs):
        first = job * args.casts
        want = steps(choices[first : first + args.casts], first)
        want_sum = (
            tile_data[np.arange(first, first + args.casts) % args.tiles].sum(0)
            if carries(args, job)
            else None
        )
        for core in cells:
            state, wrong, first_wrong, casts = rows[job, core - group.first, :4]
            right = state == want and wrong == 0 and casts == args.casts
            if want_sum is not None:
                right &= np.array_equal(got_sums[job, core - group.first], want_sum)
            if not right:
                bad.append(
                    {
                        "job": job,
                        "core": core,
                        "state": int(state),
                        "want": want,
                        "words_wrong": int(wrong),
                        "first_wrong": int(first_wrong),
                    }
                )
    per_job = float(np.median(microseconds))
    out = {
        "rect": rect,
        "along": args.along,
        "carry": args.carry,
        "cores": len(cells),
        "jobs": args.jobs,
        "casts_per_job": args.casts,
        "words_checked": args.jobs * args.casts * len(cells),
        "cores_bad": len(bad),
        "first_bad": bad[:8],
        "microseconds_per_job_median": round(per_job, 1),
        "microseconds_per_cast": round(per_job / args.casts, 3),
        "ok": not bad,
    }
    print(
        f"{out['words_checked']} words to {len(cells)} cores, "
        f"{out['cores_bad']} wrong, {out['microseconds_per_cast']} us per cast"
    )
    print(json.dumps(out))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
