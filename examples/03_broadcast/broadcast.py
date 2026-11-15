"""Broadcast on a card: A blocks along the rows of a rectangle group, W blocks
down its columns, every core's checksum against numpy. Commands go through the
dispatcher on X280 cluster 0 (`--feeder`: through the test feeder, no X280). See
README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "tests" / "hardware" / "feeder"), str(ROOT / "src")]

import commands as feeder_commands  # noqa: E402
import fed  # noqa: E402

import komorebi as km  # noqa: E402
from komorebi import layout, power, state  # noqa: E402

RUNTIME_CONSTANTS = feeder_commands.K
KERNELS = HERE / "kernels"
BROADCAST_HPP = feeder_commands.INCLUDE / "komorebi" / "device" / "km_broadcast.hpp"
BROADCAST_FIELDS = feeder_commands.struct_fields(BROADCAST_HPP, "BroadcastArgs")
CONSTANTS = km.abi.constants(KERNELS / "broadcast.h", known=km.abi.headers().constants)
TILE = 32
COLUMN, ROW = 0, 1  # broadcast ids: columns on NoC1 (writer), rows on NoC0 (reader)
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--rect", default="0,0,6,4", help="first_x,first_y,columns,rows"
    )
    parser.add_argument(
        "--blocks", type=int, default=8, help="blocks n per broadcast and job"
    )
    parser.add_argument("--pages", type=int, default=2, help="tiles P per block")
    parser.add_argument("--window", type=int, default=2, help="blocks B in flight")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--read", action="store_true", help="cores read DRAM themselves"
    )
    parser.add_argument("--stall-job", type=int, default=-1)
    parser.add_argument("--stall-rank", type=int, default=0, help="group index")
    parser.add_argument("--stall-block", type=int, default=0)
    parser.add_argument("--again", action="store_true", help="then a clean run")
    parser.add_argument(
        "--empty-job", type=int, default=-1, help="a job multicasting 0 blocks"
    )
    parser.add_argument(
        "--empty-every", type=int, default=0, help="every nth job multicasts 0"
    )
    parser.add_argument(
        "--first-blocks", type=int, default=0, help="job 0 multicasts these, cycling"
    )
    parser.add_argument("--wait", type=float, default=2.0, help="wait limit, seconds")
    parser.add_argument(
        "--deadline", type=float, default=1.0, help="per command, seconds"
    )
    parser.add_argument("--feeder", action="store_true", help="run on the test feeder")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def broadcast_plan(nocs, cores, rect, window, pages):
    """Group, per-core program arguments and the rectangle's cores (row, column), for
    `cores` cores of a grid whose NoC coordinates are `nocs` (a layout.Nocs)."""
    first_x, first_y, columns, rows = rect
    grid_width = nocs.grid[0]
    first = first_y * grid_width + first_x
    group = layout.Group(
        first, (first_y + rows - 1) * grid_width + first_x + columns - first
    )
    if group.first + group.count > cores:
        raise ValueError(f"{rect} reaches core {cores}")
    grid_panels = nocs.panels
    common = (nocs.grid, grid_panels, group, rect)
    noc_coordinates = (nocs.noc0, nocs.noc_xy)
    by_broadcast = {
        COLUMN: layout.broadcast_args(
            *common,
            "column",
            1,
            *noc_coordinates,
            (COLUMN, CONSTANTS["kCbCol"], window, pages),
        ),
        ROW: layout.broadcast_args(
            *common,
            "row",
            0,
            *noc_coordinates,
            (ROW, CONSTANTS["kCbRow"], window, pages),
        ),
    }
    cells = {
        y * grid_width + x: (y - first_y, x - first_x)
        for y in range(first_y, first_y + rows)
        for x in range(first_x, first_x + columns)
    }
    idle = [0] * (len(BROADCAST_FIELDS) + 1)

    def program_args(core):
        if core not in group.indices:
            return idle, idle, [0, window, pages]
        row, column = cells.get(core, (0, 0))
        words = {
            broadcast: [
                by_broadcast[broadcast][core][field] for field in BROADCAST_FIELDS
            ]
            for broadcast in by_broadcast
        }
        included = int(core in cells)
        return words[ROW] + [row], words[COLUMN] + [column], [included, window, pages]

    log(f"group {group.first}:{group.count}, rect {rect}, panels {grid_panels}")
    return group, program_args, cells


def broadcast_summary(broadcasts, group, cells):
    """Blocks every broadcast delivered, the longest broadcast wait (cycles), and each
    broadcast give-up by core: broadcast, where, block, sequence number.
    `broadcasts[core]`: the core's broadcast areas (as run.state() gives them)."""
    areas = [
        broadcasts[core][broadcast] for core in cells for broadcast in (COLUMN, ROW)
    ]
    gave = [
        {"core": core} | broadcasts[core][broadcast]["gave_up"]
        for core in group.indices
        for broadcast in (COLUMN, ROW)
        if broadcasts[core][broadcast]["gave_up"]
    ]
    return {
        "broadcast_blocks": sum(area["blocks"] for area in areas),
        "broadcast_wait_cycles_max": max(
            (area["wait_cycles"] for area in areas), default=0
        ),
        "broadcast_gave_up": gave,
    }


def make_data(session, generator, group, rect, args):
    """Two input sets of integers in [-2, 2] (sums exact in fp32), their tensors and
    the output tensor (one fp32 tile per job and core)."""
    _, _, columns, rows = rect
    blocks, pages = args.blocks, args.pages
    sets = [
        (
            generator.integers(-2, 3, (rows, blocks, pages, TILE, TILE)).astype(
                np.float32
            ),
            generator.integers(-2, 3, (columns, blocks, pages, TILE, TILE)).astype(
                np.float32
            ),
        )
        for _ in range(2)
    ]

    def tensor(array, dtype):
        return session.tensor(array.reshape(-1, TILE), dtype, "tile")

    inputs = [
        (tensor(row_set, "bfloat16"), tensor(column_set, "bfloat16"))
        for row_set, column_set in sets
    ]
    output = tensor(
        np.zeros((args.jobs * group.count, TILE, TILE), np.float32), "float32"
    )
    return sets, inputs, output


def job_blocks(args, job):
    """Blocks job j multicasts per broadcast: 0 for the empty job and every
    `--empty-every`th, `--first-blocks` for job 0 (block k is data block k mod
    `--blocks`), else `--blocks`."""
    every = args.empty_every
    if job == args.empty_job or (every and job % every == every - 1):
        return 0
    return args.first_blocks if job == 0 and args.first_blocks else args.blocks


def job_args(args, inputs, output, group, job, stall_job):
    """CAST / READ arguments of job j."""
    row_tensor, column_tensor = inputs[job % 2]
    no_stall = (CONSTANTS["kNone"], 0)
    stall = (args.stall_rank, args.stall_block) if job == stall_job else no_stall
    return [
        row_tensor.address,
        column_tensor.address,
        output.address,
        job_blocks(args, job),
        *stall,
        job * group.count,
        args.blocks,
    ]


def check(args, sets, output, group, cells, stall_job=-1):
    """Every core's checksum tile of every job but the stalled one against numpy, bit
    for bit."""
    got = output.read().reshape(args.jobs, group.count, TILE, TILE)
    log("outputs read back")
    bad = []
    jobs = [job for job in range(args.jobs) if job != stall_job]
    for job in jobs:
        times, rest = divmod(job_blocks(args, job), args.blocks)
        row_sums, column_sums = (
            times * values.sum((1, 2)) + values[:, :rest].sum((1, 2))
            for values in sets[job % 2]
        )
        for core, (row, column) in cells.items():
            want = row_sums[row] + column_sums[column]
            got_tile = got[job, core - group.first]
            if not np.array_equal(got_tile, want):
                wrong = int((got_tile != want).sum())
                bad.append({"job": job, "core": core, "values_wrong": wrong})
    return {
        "ranks_checked": len(jobs) * len(cells),
        "ranks_bad": len(bad),
        "first_bad": bad[:8],
    }


def describe(args, rect, group):
    return {
        "rect": rect,
        "group": [group.first, group.count],
        "blocks": args.blocks,
        "first_blocks": args.first_blocks,
        "pages": args.pages,
        "window": args.window,
        "jobs": args.jobs,
        "mode": "read" if args.read else "cast",
    }


def kernel_ends(states):
    return {
        key: sum(core[key] != 0 for core in states)
        for key in ("reader_end", "writer_end", "compute_end")
    }


def core_gave_up(states):
    return [
        {
            "core": core["core"],
            **{key: core[key] for key in ("reader", "writer", "compute")},
        }
        for core in states
        if core["reader"] or core["writer"] or core["compute"]
    ][:16]


def verdict(run, cores, stalled, end_ok):
    """Every core ended and the run ended as it should (`end_ok`), every job checked
    right; a clean run with no core and no broadcast given up."""
    ok = set(run["cores_ended"].values()) == {cores} and end_ok
    ok &= run.get("ranks_bad", 0) == 0
    if not stalled:
        ok &= not run["core_gave_up"] and not run["broadcast_gave_up"]
    return bool(ok)


def run_host(session, args, generator, stall_job):
    """One load of the program in the session: the jobs submitted through the
    dispatcher, then the stop (or the timeout), the state read-out and the check."""
    nocs = session.nocs()
    rect = tuple(int(field) for field in args.rect.split(","))
    window, pages = args.window, args.pages
    group, program_args, cells = broadcast_plan(
        nocs, len(nocs.words), rect, window, pages
    )
    sets, inputs, output = make_data(session, generator, group, rect, args)
    program = km.Program(
        "broadcast",
        kernels=KERNELS,
        buffers=[
            km.Buffer(
                CONSTANTS["kCbRow"],
                "bfloat16",
                CONSTANTS["kTileBytes"],
                window * pages,
            ),
            km.Buffer(
                CONSTANTS["kCbCol"],
                "bfloat16",
                CONSTANTS["kTileBytes"],
                window * pages,
            ),
            km.Buffer(CONSTANTS["kCbOut"], "float32", 4096, 1),
        ],
        args=program_args,
        compile_args={
            "reader": inputs[0][0].tensor_args(),
            "writer": inputs[0][1].tensor_args() + output.tensor_args(),
        },
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        handlers=KERNELS / "broadcast.h",
    )
    run = session.load(program, deadline=args.deadline, wait=args.wait)
    log("loaded")
    handler = "read" if args.read else "broadcast"
    group_arg = (group.first, group.count)
    timed_out = None
    start = time.perf_counter()
    try:
        for job in range(args.jobs):
            words = job_args(args, inputs, output, group, job, stall_job)
            run.submit(handler, group_arg, words, exclusive=True)
        log(f"{run.drain(timeout=10.0)} events")
        host_seconds = time.perf_counter() - start
        run.stop()
    except km.CommandTimeout as error:
        host_seconds = time.perf_counter() - start
        timed_out = error
        log(f"timeout: {error}")
    result = run.result()
    states = run.state()
    report = describe(args, rect, group) | {
        "kernel": {key: result[key] for key in ("error_name", "commands", "events")},
        "microseconds_per_job_host": round(host_seconds * 1e6 / args.jobs, 2),
        "cores_magic_ok": sum(core["magic_ok"] for core in states),
        "cores_ended": kernel_ends(states),
        **broadcast_summary([core["broadcasts"] for core in states], group, cells),
        "core_gave_up": core_gave_up(states),
        "recovery_refused": run.recovery_refused,
        "recoveries": [
            {key: recovery[key] for key in ("handle", "cores", "text")}
            for recovery in run.recoveries
        ],
    }
    if timed_out is not None:
        report["timed_out"] = {
            "handle": timed_out.handle,
            "sequence": timed_out.sequence,
            "group": list(timed_out.group),
            "not_reported": timed_out.cores,
        }
    elif not args.no_check:
        report.update(check(args, sets, output, group, cells, stall_job))
    stalled = 0 <= stall_job < args.jobs
    # A stall: the stalled core recovered, or the timeout (no reset lists, or other
    # commands in flight held a core).
    end = ("deadline", True) if stalled and timed_out is not None else ("none", False)
    end_ok = (result["error_name"], timed_out is not None) == end
    if stalled and timed_out is None:
        end_ok &= any(
            group.first + args.stall_rank in recovery["cores"]
            for recovery in run.recoveries
        )
    report["ok"] = verdict(report, len(states), stalled, end_ok)
    return report


def run_fed(session, runner, args, generator, stall_job):
    """The same jobs, submitted by the test feeder on the grid's last core."""
    rect = tuple(int(field) for field in args.rect.split(","))
    window, pages = args.window, args.pages
    nocs = session.nocs()
    group, program_args, cells = broadcast_plan(
        nocs, runner.workers, rect, window, pages
    )
    sets, inputs, output = make_data(session, generator, group, rect, args)
    handler = CONSTANTS["kHandlerRead"] if args.read else CONSTANTS["kHandlerBroadcast"]
    commands = [
        feeder_commands.command(
            handler,
            group.first,
            group.count,
            job_args(args, inputs, output, group, job, stall_job),
            RUNTIME_CONSTANTS["KM_FLAG_EXCLUSIVE"],
        )
        for job in range(args.jobs)
    ]
    kernels = fed.Kernels(
        reader=KERNELS / "reader.cpp",
        compute=KERNELS / "compute.cpp",
        writer=KERNELS / "writer.cpp",
        reader_ct=inputs[0][0].tensor_args(),
        writer_ct=inputs[0][1].tensor_args() + output.tensor_args(),
        compute_options={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        cbs=[
            (
                CONSTANTS["kCbRow"],
                "bfloat16",
                CONSTANTS["kTileBytes"],
                window * pages,
            ),
            (
                CONSTANTS["kCbCol"],
                "bfloat16",
                CONSTANTS["kTileBytes"],
                window * pages,
            ),
            (CONSTANTS["kCbOut"], "float32", 4096, 1),
        ],
    )
    launched = runner.run(
        kernels,
        commands,
        program_args=program_args,
        tensors=[*[tensor for tensor_pair in inputs for tensor in tensor_pair], output],
        depth=1,
        deadline_s=args.deadline,
        wait_s=args.wait,
    )
    feeder_summary = launched.summary
    states = [
        feeder_commands.core_state(launched.state, core)
        for core in range(runner.workers)
    ]
    result = describe(args, rect, group) | {
        "aiclk_mhz": launched.mhz,
        "feeder": {
            key: feeder_summary[key]
            for key in ("status", "slot", "sequence", "index", "done")
        },
        "microseconds_per_job": round(
            feeder_summary["span_cycles"]
            / launched.mhz
            / max(feeder_summary["done"], 1),
            2,
        ),
        "cores_ended": kernel_ends(states),
        **broadcast_summary(
            [state.broadcasts(row) for row in launched.control], group, cells
        ),
        "core_gave_up": core_gave_up(states),
    }
    if feeder_summary["status"] == "done" and not args.no_check:
        result.update(check(args, sets, output, group, cells))
    stalled = 0 <= stall_job < args.jobs
    if stalled:
        end_ok = feeder_summary["status"] == "timeout"
    else:
        end_ok = (
            feeder_summary["status"] == "done" and feeder_summary["done"] == args.jobs
        )
    result["ok"] = verdict(result, len(states), stalled, end_ok)
    return result


def main():
    args = parse()
    generator = np.random.default_rng(args.seed)
    stalls = [args.stall_job] + ([-1] if args.again else [])
    runs = []
    if args.feeder:
        with km.Session(log=log) as session:
            log("device open")
            runner = fed.Fed(session)
            log(f"grid {runner.grid}, aiclk {runner.mhz} MHz")
            for index, stall_job in enumerate(stalls):
                runs.append(run_fed(session, runner, args, generator, stall_job))
                log(f"run {index + 1} done")
    else:
        with km.Session(log=log) as session:
            log("device open")
            for index, stall_job in enumerate(stalls):
                runs.append(run_host(session, args, generator, stall_job))
                log(f"run {index + 1} done")
    ok = all(run["ok"] for run in runs)
    wall = round(time.monotonic() - STARTED, 1)
    print(
        json.dumps(power.clocked({"ok": ok, "runs": runs, "wall_s": wall})), flush=True
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
