"""Lane broadcast on a card: A blocks along the rows of a rectangle team, W blocks down
its columns, every member's checksum against numpy. Records go through the dispatch
kernel on X280 cluster 0 (`--feeder`: through the test feeder, no X280). See README.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "tests" / "hardware" / "feeder"), str(ROOT / "src")]

import fed  # noqa: E402
import records as rec  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

import komorebi as km  # noqa: E402
from komorebi import layout, native, state  # noqa: E402

K = rec.K
KERNELS = HERE / "kernels"
LANE_HPP = rec.INCLUDE / "komorebi" / "device" / "km_lane.hpp"
LANE_FIELDS = rec.struct_fields(LANE_HPP, "LaneArgs")
EX = km.abi.constants(KERNELS / "broadcast.h", known=km.abi.headers().constants)
TILE = 32
COLUMN, ROW = 0, 1  # lane ids: columns on NoC1 (writer), rows on NoC0 (reader)
T0 = time.monotonic()


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rect", default="0,0,6,4", help="x0,y0,columns,rows")
    p.add_argument("--blocks", type=int, default=8, help="blocks n per lane and job")
    p.add_argument("--pages", type=int, default=2, help="tiles P per block")
    p.add_argument("--window", type=int, default=2, help="blocks B in flight")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--read", action="store_true", help="members read DRAM themselves")
    p.add_argument("--stall-job", type=int, default=-1)
    p.add_argument("--stall-member", type=int, default=0, help="team index")
    p.add_argument("--stall-block", type=int, default=0)
    p.add_argument("--again", action="store_true", help="then a clean run")
    p.add_argument("--empty-job", type=int, default=-1, help="a job casting 0 blocks")
    p.add_argument("--empty-every", type=int, default=0, help="every nth job casts 0")
    p.add_argument(
        "--first-blocks", type=int, default=0, help="job 0 casts these, cycling"
    )
    p.add_argument("--wait", type=float, default=2.0, help="wait limit, seconds")
    p.add_argument("--deadline", type=float, default=1.0, help="per record, seconds")
    p.add_argument("--feeder", action="store_true", help="run on the test feeder")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def lane_plan(nocs, cores, rect, window, pages):
    """Team, per-core program arguments and the rectangle's cores (row, column), for
    `cores` cores of a grid whose NoC coordinates are `nocs` (a layout.Nocs)."""
    x0, y0, nx, ny = rect
    gx = nocs.grid[0]
    first = y0 * gx + x0
    team = layout.Team(first, (y0 + ny - 1) * gx + x0 + nx - first)
    if team.first + team.count > cores:
        raise ValueError(f"{rect} reaches core {cores}")
    grid_panels = nocs.panels
    common = (nocs.grid, grid_panels, team, rect)
    at = (nocs.noc0, nocs.noc_xy)
    by_lane = {
        COLUMN: layout.lane_args(
            *common, "column", 1, *at, (COLUMN, EX["kCbCol"], window, pages)
        ),
        ROW: layout.lane_args(
            *common, "row", 0, *at, (ROW, EX["kCbRow"], window, pages)
        ),
    }
    place = {
        y * gx + x: (y - y0, x - x0)
        for y in range(y0, y0 + ny)
        for x in range(x0, x0 + nx)
    }
    idle = [0] * (len(LANE_FIELDS) + 1)

    def program_args(core):
        if core not in team.members:
            return idle, idle, [0, window, pages]
        row, col = place.get(core, (0, 0))
        words = {
            lane: [by_lane[lane][core][f] for f in LANE_FIELDS] for lane in by_lane
        }
        member = int(core in place)
        return words[ROW] + [row], words[COLUMN] + [col], [member, window, pages]

    log(f"team {team.first}:{team.count}, rect {rect}, panels {grid_panels}")
    return team, program_args, place


def lane_summary(lanes, team, place):
    """Blocks every lane handed over, the longest lane wait (cycles), and each lane
    give-up by core: lane, where, block, sequence number. `lanes[core]`: the core's
    lane areas (as run.state() gives them)."""
    areas = [lanes[c][ln] for c in place for ln in (COLUMN, ROW)]
    gave = [
        {"core": core} | lanes[core][lane]["gave_up"]
        for core in team.members
        for lane in (COLUMN, ROW)
        if lanes[core][lane]["gave_up"]
    ]
    return {
        "lane_blocks": sum(a["blocks"] for a in areas),
        "lane_wait_cycles_max": max((a["wait_cycles"] for a in areas), default=0),
        "lane_gave_up": gave,
    }


def make_data(device, rng, team, rect, args):
    """Two input sets of integers in [-2, 2] (sums exact in fp32), their tensors and
    the output tensor (one fp32 tile per job and member)."""
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
        return ttnn.from_torch(
            torch.from_numpy(a.reshape(-1, TILE)),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    inputs = [(tensor(a, ttnn.bfloat16), tensor(b, ttnn.bfloat16)) for a, b in sets]
    out = tensor(
        np.zeros((args.jobs * team.count, TILE, TILE), np.float32), ttnn.float32
    )
    return sets, inputs, out


def job_blocks(args, j):
    """Blocks job j casts per lane: 0 for the empty job and every `--empty-every`th,
    `--first-blocks` for job 0 (block k is data block k mod `--blocks`), else
    `--blocks`."""
    n = args.empty_every
    if j == args.empty_job or (n and j % n == n - 1):
        return 0
    return args.first_blocks if j == 0 and args.first_blocks else args.blocks


def job_args(args, inputs, out, team, j, stall_job):
    """CAST / READ arguments of job j."""
    a_t, w_t = inputs[j % 2]
    none = (EX["kNone"], 0)
    stall = (args.stall_member, args.stall_block) if j == stall_job else none
    return [
        a_t.buffer_address(),
        w_t.buffer_address(),
        out.buffer_address(),
        job_blocks(args, j),
        *stall,
        j * team.count,
        args.blocks,
    ]


def check(args, sets, out, team, place):
    """Every member's checksum tile of every job against numpy, bit for bit."""
    got = ttnn.to_torch(out).float().numpy().reshape(args.jobs, team.count, TILE, TILE)
    log("outputs read back")
    bad = []
    for j in range(args.jobs):
        times, rest = divmod(job_blocks(args, j), args.blocks)
        a_sum, w_sum = (
            times * x.sum((1, 2)) + x[:, :rest].sum((1, 2)) for x in sets[j % 2]
        )
        for core, (row, col) in place.items():
            want = a_sum[row] + w_sum[col]
            mine = got[j, core - team.first]
            if not np.array_equal(mine, want):
                wrong = int((mine != want).sum())
                bad.append({"job": j, "core": core, "values_wrong": wrong})
    return {
        "members_checked": args.jobs * len(place),
        "members_bad": len(bad),
        "first_bad": bad[:8],
    }


def describe(args, rect, team):
    return {
        "rect": rect,
        "team": [team.first, team.count],
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
    stalled job, else a clean end); a clean run with no core and no lane given up and
    every member's values right."""
    ok = set(run["cores_ended"].values()) == {cores} and end_ok
    if not stalled:
        ok &= not run["core_gave_up"] and not run["lane_gave_up"]
        ok &= run.get("members_bad", 0) == 0
    return bool(ok)


def run_host(session, args, rng, stall_job):
    """One load of the program in the session: the jobs issued through the dispatch
    kernel, then the stop (or the timeout), the state read-out and the check."""
    device = session.device
    nocs = session.nocs()
    rect = tuple(int(v) for v in args.rect.split(","))
    w, p = args.window, args.pages
    team, program_args, place = lane_plan(nocs, len(nocs.words), rect, w, p)
    sets, inputs, out = make_data(device, rng, team, rect, args)
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
            "reader": ttnn.TensorAccessorArgs(inputs[0][0]).get_compile_time_args(),
            "writer": ttnn.TensorAccessorArgs(inputs[0][1]).get_compile_time_args()
            + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
        },
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        segments=KERNELS / "broadcast.h",
    )
    run = session.load(program, deadline=args.deadline, wait=args.wait)
    log("loaded")
    seg = "read" if args.read else "cast"
    team_arg = (team.first, team.count)
    timed_out = None
    t0 = time.perf_counter()
    try:
        for j in range(args.jobs):
            words = job_args(args, inputs, out, team, j, stall_job)
            run.issue(seg, team_arg, words, exclusive=True)
        log(f"{run.drain(timeout=10.0)} events")
        host_s = time.perf_counter() - t0
        run.stop()
    except km.RecordTimeout as err:
        host_s = time.perf_counter() - t0
        timed_out = err
        log(f"timeout: {err}")
    result = run.result()
    states = run.state()
    out_json = describe(args, rect, team) | {
        "kernel": {k: result[k] for k in ("error_name", "records", "events")},
        "us_per_job_host": round(host_s * 1e6 / args.jobs, 2),
        "cores_magic_ok": sum(c["magic_ok"] for c in states),
        "cores_ended": kernel_ends(states),
        **lane_summary([c["lanes"] for c in states], team, place),
        "core_gave_up": core_gave_up(states),
    }
    if timed_out is not None:
        out_json["timed_out"] = {
            "ticket": timed_out.ticket,
            "seq": timed_out.seq,
            "team": list(timed_out.team),
            "not_reported": timed_out.cores,
        }
    elif not args.no_check:
        out_json.update(check(args, sets, out, team, place))
    stalled = 0 <= stall_job < args.jobs
    end = ("deadline", True) if stalled else ("none", False)
    end_ok = (result["error_name"], timed_out is not None) == end
    out_json["ok"] = verdict(out_json, len(states), stalled, end_ok)
    return out_json


def run_fed(device, runner, args, rng, stall_job):
    """The same jobs, issued by the test feeder on the grid's last core."""
    rect = tuple(int(v) for v in args.rect.split(","))
    w, p = args.window, args.pages
    gx, gy, table, noc0 = native.load_module().core_nocs(device)
    nocs = layout.nocs((gx, gy), table, noc0)
    team, program_args, place = lane_plan(nocs, runner.workers, rect, w, p)
    sets, inputs, out = make_data(device, rng, team, rect, args)
    seg = EX["kSegRead"] if args.read else EX["kSegCast"]
    records = [
        rec.record(
            seg,
            team.first,
            team.count,
            job_args(args, inputs, out, team, j, stall_job),
            K["KM_F_EXCL"],
        )
        for j in range(args.jobs)
    ]
    kernels = fed.Kernels(
        reader=KERNELS / "reader.cpp",
        compute=KERNELS / "compute.cpp",
        writer=KERNELS / "writer.cpp",
        reader_ct=ttnn.TensorAccessorArgs(inputs[0][0]).get_compile_time_args(),
        writer_ct=ttnn.TensorAccessorArgs(inputs[0][1]).get_compile_time_args()
        + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
        compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        cbs=[
            (EX["kCbRow"], ttnn.bfloat16, EX["kTileBytes"], w * p),
            (EX["kCbCol"], ttnn.bfloat16, EX["kTileBytes"], w * p),
            (EX["kCbOut"], ttnn.float32, 4096, 1),
        ],
    )
    res = runner.run(
        kernels,
        records,
        program_args=program_args,
        tensors=[*[t for pair in inputs for t in pair], out],
        depth=1,
        deadline_s=args.deadline,
        wait_s=args.wait,
    )
    s = res.summary
    states = [rec.core_state(res.state, c) for c in range(runner.workers)]
    result = describe(args, rect, team) | {
        "aiclk_mhz": res.mhz,
        "feeder": {k: s[k] for k in ("status", "slot", "seq", "index", "done")},
        "us_per_job": round(s["span_cycles"] / res.mhz / max(s["done"], 1), 2),
        "cores_ended": kernel_ends(states),
        **lane_summary([state.lanes(row) for row in res.ctl], team, place),
        "core_gave_up": core_gave_up(states),
    }
    if s["status"] == "done" and not args.no_check:
        result.update(check(args, sets, out, team, place))
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
        device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        device.disable_and_clear_program_cache()
        log("device open")
        try:
            runner = fed.Fed(device)
            log(f"grid {runner.grid}, aiclk {runner.mhz} MHz")
            for i, stall_job in enumerate(stalls):
                runs.append(run_fed(device, runner, args, rng, stall_job))
                log(f"run {i + 1} done")
        finally:
            ttnn.close_mesh_device(device)
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
