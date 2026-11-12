"""Lane broadcast on a card, fed by the test feeder (no X280): A blocks along the rows
of a rectangle team, W blocks down its columns, every member's checksum against numpy.

Example: python broadcast.py --rect 0,0,6,4 --blocks 8 --pages 2 --window 2"""

import argparse
import json
import re
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

from komorebi import layout  # noqa: E402

K = rec.K
KERNELS = HERE / "kernels"
LANE_HPP = rec.INCLUDE / "komorebi" / "device" / "km_lane.hpp"
LANE_FIELDS = rec.struct_fields(LANE_HPP, "LaneArgs")
EX = rec.constants(KERNELS / "broadcast.h")
TILE = 32
COLUMN, ROW = 0, 1  # lane ids: columns on NoC1 (writer), rows on NoC0 (reader)


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
    p.add_argument("--wait", type=float, default=2.0, help="wait limit, seconds")
    p.add_argument("--deadline", type=float, default=1.0, help="per record, seconds")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def noc0_x():
    """NoC0 x of each logical column, from the card's Tensix harvesting mask."""
    path = ttnn._ttnn.cluster.serialize_cluster_descriptor()
    masks = re.findall(r"\bharvest_mask:\s*(\d+)", Path(path).read_text())
    if len(masks) != 1:
        raise RuntimeError(f"expected one chip's harvest_mask in {path}")
    return layout.blackhole_noc0_x(int(masks[0]))


def lane_plan(runner, rect, window, pages):
    """Team, per-core program arguments and the rectangle's cores (row, column)."""
    x0, y0, nx, ny = rect
    gx = runner.grid[0]
    first = y0 * gx + x0
    team = layout.Team(first, (y0 + ny - 1) * gx + x0 + nx - first)
    if team.first + team.count > runner.workers:
        raise ValueError(f"{rect} reaches the feeder's core {runner.workers}")
    xs = noc0_x()
    grid_panels = layout.panels(xs)
    noc_xy = {c: (v & 0xFFFF, v >> 16) for c, v in zip(runner.logical, runner.noc)}
    noc0 = {c: (xs[c[0]], xy[1]) for c, xy in noc_xy.items()}
    common = (runner.grid, grid_panels, team, rect)
    by_lane = {
        COLUMN: layout.lane_args(
            *common, "column", 1, noc0, noc_xy, (COLUMN, EX["kCbCol"], window, pages)
        ),
        ROW: layout.lane_args(
            *common, "row", 0, noc0, noc_xy, (ROW, EX["kCbRow"], window, pages)
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

    return team, program_args, place, grid_panels


def lane_words(ctl, core, lane):
    """GAVE and STATS words of one lane area of a core's control block."""
    base = (K["KM_CTL_LANE"] + lane * K["KM_LANE_BYTES"]) // 4
    gave = ctl[core, base + K["KM_LANE_GAVE"] // 4 :][:3]
    stats = ctl[core, base + K["KM_LANE_STATS"] // 4 :][:3]
    return [int(v) for v in gave], [int(v) for v in stats]


def gave_ups(res, cores):
    """Lane give-ups by core: lane, where, block, sequence number."""
    wheres = {v: k[6:].lower() for k, v in K.items() if k.startswith("kWhere")}
    out = []
    for core in cores:
        for lane in (COLUMN, ROW):
            g, _ = lane_words(res.ctl, core, lane)
            if g[0]:
                where = wheres.get(g[0] & 0xFF, g[0] & 0xFF)
                lane_id = g[0] >> 8
                out.append(
                    {"core": core, "lane": lane_id, "where": where, "block": g[1]}
                    | {"seq": g[2]}
                )
    return out


def run_once(device, runner, args, rng, stall_job):
    rect = tuple(int(v) for v in args.rect.split(","))
    n, p, w = args.blocks, args.pages, args.window
    team, program_args, place, grid_panels = lane_plan(runner, rect, w, p)
    _, _, nx, ny = rect
    fed.log(f"team {team.first}:{team.count}, rect {rect}, panels {grid_panels}")

    # Two input sets of integers in [-2, 2]: sums are exact in fp32.
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
    seg = EX["kSegRead"] if args.read else EX["kSegCast"]
    records = []
    for j in range(args.jobs):
        a_t, w_t = inputs[j % 2]
        stall = (
            (args.stall_member, args.stall_block)
            if j == stall_job
            else (EX["kNone"], 0)
        )
        words = [
            a_t.buffer_address(),
            w_t.buffer_address(),
            out.buffer_address(),
            n,
            *stall,
            j * team.count,
        ]
        records.append(rec.record(seg, team.first, team.count, words, K["KM_F_EXCL"]))

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
    blocks = sum(lane_words(res.ctl, c, ln)[1][0] for c in place for ln in (0, 1))
    waits = [
        lane_words(res.ctl, c, ln)[1][1] | lane_words(res.ctl, c, ln)[1][2] << 32
        for c in place
        for ln in (0, 1)
    ]
    result = {
        "rect": rect,
        "team": [team.first, team.count],
        "blocks": n,
        "pages": p,
        "window": w,
        "jobs": args.jobs,
        "mode": "read" if args.read else "cast",
        "aiclk_mhz": res.mhz,
        "feeder": {k: s[k] for k in ("status", "slot", "seq", "index", "done")},
        "us_per_job": round(s["span_cycles"] / res.mhz / max(s["done"], 1), 2),
        "lane_blocks": blocks,
        "lane_wait_cycles_max": max(waits) if waits else 0,
        "cores_ended": {
            k: sum(c[k] != 0 for c in states)
            for k in ("reader_end", "writer_end", "compute_end")
        },
        "lane_gave_up": gave_ups(res, team.members),
        "core_gave_up": [
            {"core": c["core"], **{k: c[k] for k in ("reader", "writer", "compute")}}
            for c in states
            if c["reader"] or c["writer"] or c["compute"]
        ][:16],
    }
    if s["status"] == "done" and not args.no_check:
        got = (
            ttnn.to_torch(out)
            .float()
            .numpy()
            .reshape(args.jobs, team.count, TILE, TILE)
        )
        fed.log("outputs read back")
        bad = 0
        for j in range(args.jobs):
            a, b = sets[j % 2]
            a_sum, w_sum = a.sum((1, 2)), b.sum((1, 2))
            for core, (row, col) in place.items():
                ref = a_sum[row] + w_sum[col]
                bad += not np.array_equal(got[j, core - team.first], ref)
        result["members_checked"] = args.jobs * len(place)
        result["members_bad"] = bad
    return result


def main():
    args = parse()
    rng = np.random.default_rng(args.seed)
    t_start = time.time()
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    device.disable_and_clear_program_cache()
    fed.log(f"device open {time.time() - t_start:.1f}s")
    try:
        runner = fed.Fed(device)
        fed.log(f"grid {runner.grid}, aiclk {runner.mhz} MHz")
        runs = [run_once(device, runner, args, rng, args.stall_job)]
        fed.log("run 1 done")
        if args.again:
            runs.append(run_once(device, runner, args, rng, -1))
            fed.log("run 2 done")
    finally:
        ttnn.close_mesh_device(device)
    print(json.dumps({"runs": runs, "wall_s": round(time.time() - t_start, 1)}))


if __name__ == "__main__":
    main()
