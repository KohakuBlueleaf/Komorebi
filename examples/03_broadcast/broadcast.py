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

import commands as rec  # noqa: E402
import fed  # noqa: E402

import komorebi as km  # noqa: E402
from komorebi import layout, state  # noqa: E402

K = rec.K
KERNELS = HERE / "kernels"
BROADCAST_HPP = rec.INCLUDE / "komorebi" / "device" / "km_broadcast.hpp"
BROADCAST_FIELDS = rec.struct_fields(BROADCAST_HPP, "BroadcastArgs")
EX = km.abi.constants(KERNELS / "broadcast.h", known=km.abi.headers().constants)
TILE = 32
COLUMN, ROW = 0, 1  # broadcast ids: columns on NoC1 (writer), rows on NoC0 (reader)
T0 = time.monotonic()


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rect", default="0,0,6,4", help="x0,y0,columns,rows")
    p.add_argument(
        "--blocks", type=int, default=8, help="blocks n per broadcast and job"
    )
    p.add_argument("--pages", type=int, default=2, help="tiles P per block")
    p.add_argument("--window", type=int, default=2, help="blocks B in flight")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--read", action="store_true", help="cores read DRAM themselves")
    p.add_argument("--stall-job", type=int, default=-1)
    p.add_argument("--stall-rank", type=int, default=0, help="group index")
    p.add_argument("--stall-block", type=int, default=0)
    p.add_argument("--again", action="store_true", help="then a clean run")
    p.add_argument(
        "--empty-job", type=int, default=-1, help="a job multicasting 0 blocks"
    )
    p.add_argument(
        "--empty-every", type=int, default=0, help="every nth job multicasts 0"
    )
    p.add_argument(
        "--first-blocks", type=int, default=0, help="job 0 multicasts these, cycling"
    )
    p.add_argument("--wait", type=float, default=2.0, help="wait limit, seconds")
    p.add_argument("--deadline", type=float, default=1.0, help="per command, seconds")
    p.add_argument("--feeder", action="store_true", help="run on the test feeder")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def broadcast_plan(nocs, cores, rect, window, pages):
    """Group, per-core program arguments and the rectangle's cores (row, column), for
    `cores` cores of a grid whose NoC coordinates are `nocs` (a layout.Nocs)."""
    x0, y0, nx, ny = rect
    gx = nocs.grid[0]
    first = y0 * gx + x0
    group = layout.Group(first, (y0 + ny - 1) * gx + x0 + nx - first)
    if group.first + group.count > cores:
        raise ValueError(f"{rect} reaches core {cores}")
    grid_panels = nocs.panels
    common = (nocs.grid, grid_panels, group, rect)
    at = (nocs.noc0, nocs.noc_xy)
    by_broadcast = {
        COLUMN: layout.broadcast_args(
            *common, "column", 1, *at, (COLUMN, EX["kCbCol"], window, pages)
        ),
        ROW: layout.broadcast_args(
            *common, "row", 0, *at, (ROW, EX["kCbRow"], window, pages)
        ),
    }
    place = {
        y * gx + x: (y - y0, x - x0)
        for y in range(y0, y0 + ny)
        for x in range(x0, x0 + nx)
    }
    idle = [0] * (len(BROADCAST_FIELDS) + 1)

    def program_args(core):
        if core not in group.indices:
            return idle, idle, [0, window, pages]
        row, col = place.get(core, (0, 0))
        words = {
            broadcast: [by_broadcast[broadcast][core][f] for f in BROADCAST_FIELDS]
            for broadcast in by_broadcast
        }
        included = int(core in place)
        return words[ROW] + [row], words[COLUMN] + [col], [included, window, pages]

    log(f"group {group.first}:{group.count}, rect {rect}, panels {grid_panels}")
    return group, program_args, place


def broadcast_summary(broadcasts, group, place):
    """Blocks every broadcast delivered, the longest broadcast wait (cycles), and each
    broadcast give-up by core: broadcast, where, block, sequence number.
    `broadcasts[core]`: the core's broadcast areas (as run.state() gives them)."""
    areas = [broadcasts[c][ln] for c in place for ln in (COLUMN, ROW)]
    gave = [
        {"core": core} | broadcasts[core][broadcast]["gave_up"]
        for core in group.indices
        for broadcast in (COLUMN, ROW)
        if broadcasts[core][broadcast]["gave_up"]
    ]
    return {
        "broadcast_blocks": sum(a["blocks"] for a in areas),
        "broadcast_wait_cycles_max": max((a["wait_cycles"] for a in areas), default=0),
        "broadcast_gave_up": gave,
    }


def make_data(session, rng, group, rect, args):
    """Two input sets of integers in [-2, 2] (sums exact in fp32), their tensors and
    the output tensor (one fp32 tile per job and core)."""
    _, _, nx, ny = rect
    n, p = args.blocks, args.pages
    sets = [
        (
            rng.integers(-2, 3, (ny, n, p, TILE, TILE)).astype(np.float32),
            rng.integers(-2, 3, (nx, n, p, TILE, TILE)).astype(np.float32),
        )
        for _ in range(2)
    ]

    def tensor(a, dtype):
        return session.tensor(a.reshape(-1, TILE), dtype, "tile")

    inputs = [(tensor(a, "bfloat16"), tensor(b, "bfloat16")) for a, b in sets]
    out = tensor(np.zeros((args.jobs * group.count, TILE, TILE), np.float32), "float32")
    return sets, inputs, out


def job_blocks(args, j):
    """Blocks job j multicasts per broadcast: 0 for the empty job and every
    `--empty-every`th, `--first-blocks` for job 0 (block k is data block k mod
    `--blocks`), else `--blocks`."""
    n = args.empty_every
    if j == args.empty_job or (n and j % n == n - 1):
        return 0
    return args.first_blocks if j == 0 and args.first_blocks else args.blocks


def job_args(args, inputs, out, group, j, stall_job):
    """CAST / READ arguments of job j."""
    a_t, w_t = inputs[j % 2]
    none = (EX["kNone"], 0)
    stall = (args.stall_rank, args.stall_block) if j == stall_job else none
    return [
        a_t.address,
        w_t.address,
        out.address,
        job_blocks(args, j),
        *stall,
        j * group.count,
        args.blocks,
    ]


def check(args, sets, out, group, place):
    """Every core's checksum tile of every job against numpy, bit for bit."""
    got = out.read().reshape(args.jobs, group.count, TILE, TILE)
    log("outputs read back")
    bad = []
    for j in range(args.jobs):
        times, rest = divmod(job_blocks(args, j), args.blocks)
        a_sum, w_sum = (
            times * x.sum((1, 2)) + x[:, :rest].sum((1, 2)) for x in sets[j % 2]
        )
        for core, (row, col) in place.items():
            want = a_sum[row] + w_sum[col]
            mine = got[j, core - group.first]
            if not np.array_equal(mine, want):
                wrong = int((mine != want).sum())
                bad.append({"job": j, "core": core, "values_wrong": wrong})
    return {
        "ranks_checked": args.jobs * len(place),
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
        k: sum(c[k] != 0 for c in states)
        for k in ("reader_end", "writer_end", "compute_end")
    }


def core_gave_up(states):
    return [
        {"core": c["core"], **{k: c[k] for k in ("reader", "writer", "compute")}}
        for c in states
        if c["reader"] or c["writer"] or c["compute"]
    ][:16]


def verdict(run, cores, stalled, end_ok):
    """Every core ended and the run ended as it should (`end_ok`: the deadline for a
    stalled job, else a clean end); a clean run with no core and no broadcast given up
    and every core's values right."""
    ok = set(run["cores_ended"].values()) == {cores} and end_ok
    if not stalled:
        ok &= not run["core_gave_up"] and not run["broadcast_gave_up"]
        ok &= run.get("ranks_bad", 0) == 0
    return bool(ok)


def run_host(session, args, rng, stall_job):
    """One load of the program in the session: the jobs submitted through the
    dispatcher, then the stop (or the timeout), the state read-out and the check."""
    nocs = session.nocs()
    rect = tuple(int(v) for v in args.rect.split(","))
    w, p = args.window, args.pages
    group, program_args, place = broadcast_plan(nocs, len(nocs.words), rect, w, p)
    sets, inputs, out = make_data(session, rng, group, rect, args)
    program = km.Program(
        "broadcast",
        kernels=KERNELS,
        buffers=[
            km.Buffer(EX["kCbRow"], "bfloat16", EX["kTileBytes"], w * p),
            km.Buffer(EX["kCbCol"], "bfloat16", EX["kTileBytes"], w * p),
            km.Buffer(EX["kCbOut"], "float32", 4096, 1),
        ],
        args=program_args,
        compile_args={
            "reader": inputs[0][0].accessor_args(),
            "writer": inputs[0][1].accessor_args() + out.accessor_args(),
        },
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        handlers=KERNELS / "broadcast.h",
    )
    run = session.load(program, deadline=args.deadline, wait=args.wait)
    log("loaded")
    handler = "read" if args.read else "broadcast"
    group_arg = (group.first, group.count)
    timed_out = None
    t0 = time.perf_counter()
    try:
        for j in range(args.jobs):
            words = job_args(args, inputs, out, group, j, stall_job)
            run.submit(handler, group_arg, words, exclusive=True)
        log(f"{run.drain(timeout=10.0)} events")
        host_s = time.perf_counter() - t0
        run.stop()
    except km.CommandTimeout as err:
        host_s = time.perf_counter() - t0
        timed_out = err
        log(f"timeout: {err}")
    result = run.result()
    states = run.state()
    out_json = describe(args, rect, group) | {
        "kernel": {k: result[k] for k in ("error_name", "commands", "events")},
        "microseconds_per_job_host": round(host_s * 1e6 / args.jobs, 2),
        "cores_magic_ok": sum(c["magic_ok"] for c in states),
        "cores_ended": kernel_ends(states),
        **broadcast_summary([c["broadcasts"] for c in states], group, place),
        "core_gave_up": core_gave_up(states),
    }
    if timed_out is not None:
        out_json["timed_out"] = {
            "handle": timed_out.handle,
            "sequence": timed_out.sequence,
            "group": list(timed_out.group),
            "not_reported": timed_out.cores,
        }
    elif not args.no_check:
        out_json.update(check(args, sets, out, group, place))
    stalled = 0 <= stall_job < args.jobs
    end = ("deadline", True) if stalled else ("none", False)
    end_ok = (result["error_name"], timed_out is not None) == end
    out_json["ok"] = verdict(out_json, len(states), stalled, end_ok)
    return out_json


def run_fed(session, runner, args, rng, stall_job):
    """The same jobs, submitted by the test feeder on the grid's last core."""
    rect = tuple(int(v) for v in args.rect.split(","))
    w, p = args.window, args.pages
    nocs = session.nocs()
    group, program_args, place = broadcast_plan(nocs, runner.workers, rect, w, p)
    sets, inputs, out = make_data(session, rng, group, rect, args)
    handler = EX["kHandlerRead"] if args.read else EX["kHandlerBroadcast"]
    commands = [
        rec.command(
            handler,
            group.first,
            group.count,
            job_args(args, inputs, out, group, j, stall_job),
            K["KM_FLAG_EXCLUSIVE"],
        )
        for j in range(args.jobs)
    ]
    kernels = fed.Kernels(
        reader=KERNELS / "reader.cpp",
        compute=KERNELS / "compute.cpp",
        writer=KERNELS / "writer.cpp",
        reader_ct=inputs[0][0].accessor_args(),
        writer_ct=inputs[0][1].accessor_args() + out.accessor_args(),
        compute_options={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        cbs=[
            (EX["kCbRow"], "bfloat16", EX["kTileBytes"], w * p),
            (EX["kCbCol"], "bfloat16", EX["kTileBytes"], w * p),
            (EX["kCbOut"], "float32", 4096, 1),
        ],
    )
    res = runner.run(
        kernels,
        commands,
        program_args=program_args,
        tensors=[*[t for pair in inputs for t in pair], out],
        depth=1,
        deadline_s=args.deadline,
        wait_s=args.wait,
    )
    s = res.summary
    states = [rec.core_state(res.state, c) for c in range(runner.workers)]
    result = describe(args, rect, group) | {
        "aiclk_mhz": res.mhz,
        "feeder": {k: s[k] for k in ("status", "slot", "sequence", "index", "done")},
        "microseconds_per_job": round(
            s["span_cycles"] / res.mhz / max(s["done"], 1), 2
        ),
        "cores_ended": kernel_ends(states),
        **broadcast_summary(
            [state.broadcasts(row) for row in res.control], group, place
        ),
        "core_gave_up": core_gave_up(states),
    }
    if s["status"] == "done" and not args.no_check:
        result.update(check(args, sets, out, group, place))
    stalled = 0 <= stall_job < args.jobs
    if stalled:
        end_ok = s["status"] == "timeout"
    else:
        end_ok = s["status"] == "done" and s["done"] == args.jobs
    result["ok"] = verdict(result, len(states), stalled, end_ok)
    return result


def main():
    args = parse()
    rng = np.random.default_rng(args.seed)
    stalls = [args.stall_job] + ([-1] if args.again else [])
    runs = []
    if args.feeder:
        with km.Session(log=log) as session:
            log("device open")
            runner = fed.Fed(session)
            log(f"grid {runner.grid}, aiclk {runner.mhz} MHz")
            for i, stall_job in enumerate(stalls):
                runs.append(run_fed(session, runner, args, rng, stall_job))
                log(f"run {i + 1} done")
    else:
        with km.Session(log=log) as session:
            log("device open")
            for i, stall_job in enumerate(stalls):
                runs.append(run_host(session, args, rng, stall_job))
                log(f"run {i + 1} done")
    ok = all(r["ok"] for r in runs)
    wall = round(time.monotonic() - T0, 1)
    print(json.dumps({"ok": ok, "runs": runs, "wall_s": wall}), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
