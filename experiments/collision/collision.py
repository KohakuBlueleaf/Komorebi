"""Collision experiment: X280 record multicast against lane broadcasts in the grid.

Runs the record and lane kernels on every core and prints one JSON line with both
sides' counts (see README.md). Example: python collision.py --records 100000 --lane 5
"""

import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
# A core that never finishes makes the host's wait throw (naming the cores) in time.
os.environ.setdefault("TT_METAL_OPERATION_TIMEOUT_SECONDS", "20")

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
sys.path.insert(0, BUILD)

import _collision_x280 as x280mod  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

RECORDS_KERNEL = os.path.join(HERE, "kernels", "records.cpp")
LANE_KERNEL = os.path.join(HERE, "kernels", "lane.cpp")
STREAM_KERNEL = os.path.join(HERE, "kernels", "stream.cpp")
IMAGE = os.path.join(BUILD, "record-seq.bin")
N_CORES = 120
WORDS = 32  # counter words per core, both kernels

# CB indices (kernels/lane.h; the template's control block is 8).
CTL_CB, CTL_BYTES = 8, 1024
REC_CNT_CB, SHARE_CB, LANE_CNT_CB, LANE_CB = 9, 11, 12, 13
LANE_HEAD = 512  # lane buffer bytes before the block

# Record counter word indices (kernels/landing.h LC_*), only those this kernel writes.
LC = {
    "magic": 0,
    "accepted": 1,
    "torn": 2,
    "early": 3,
    "idle_lo": 4,
    "idle_hi": 5,
    "torn_seq": 11,
    "torn_word": 12,
    "torn_value": 13,
    "healed": 14,
    "heal_polls": 15,
    "lost": 16,
    "lost_seq": 17,
    "tag": 19,
    "last_seq": 20,
}
LANDING_MAGIC = 0x1A4D0000

# Lane counter word indices (kernels/lane.h LN_*).
LN_NAMES = [
    "magic",
    "role",
    "blocks",
    "bad_blocks",
    "bad_words",
    "bad_k",
    "bad_w",
    "bad_v",
    "where",
    "where_k",
    "missing",
    "peer",
    "peer_phase",
    "peer_k",
    "t_lo",
    "t_hi",
    "stopped",
    "tag",
    "wait_max",
    "busy_lo",
    "busy_hi",
    "check_lo",
    "check_hi",
    "full",
]
LN = {name: i for i, name in enumerate(LN_NAMES)}
LANE_MAGIC = 0x1A9E0000
WHERE = {1: "ready", 2: "stamp", 3: "valid"}
PHASES = {
    1: "start",
    2: "wait_stamps",
    3: "fill",
    4: "barrier_block",
    5: "barrier_valid",
    6: "wait_valid",
    7: "verify",
    8: "barrier_stamp",
    9: "end",
    10: "gave_up",
    0xFFFFFFFF: "unreadable",
}


def log(msg):
    print(msg, flush=True)


def loadavg():
    with open("/proc/loadavg") as f:
        return f.read().split()[0]


def cb(index, size, cores):
    fmt = ttnn.CBFormatDescriptor(
        buffer_index=index, data_format=ttnn.uint32, page_size=size
    )
    return ttnn.CBDescriptor(
        total_size=size, core_ranges=cores, format_descriptors=[fmt]
    )


def lane_roles(plan, args):
    """Per logical core (x, y): role, rank, sender core and multicast rectangle."""
    gx, gy, L = plan.grid_x, plan.grid_y, args.lane
    roles = {}
    if L == 0:
        return roles
    if args.along == "row":
        return row_roles(plan, args)
    assert gy % L == 0, (gy, L)
    cols = (
        range(gx)
        if args.columns == "all"
        else [int(c) for c in args.columns.split(",")]
    )
    for x in cols:
        t = [plan.translated(x, y) for y in range(gy)]
        # Lanes need one translated x and consecutive y down the column.
        assert all(c[0] == t[0][0] and c[1] == t[0][1] + y for y, c in enumerate(t)), t
        for i in range(gy // L):
            first, last = i * L, i * L + L - 1
            if args.lane_noc == 0:  # NoC0 runs +y: sender on top
                snd, step = first, 1
                rect = (t[first + 1][0], t[first + 1][1], t[last][0], t[last][1])
            else:  # NoC1 runs -y: sender at the bottom, rectangle start is the high y
                snd, step = last, -1
                rect = (t[last - 1][0], t[last - 1][1], t[first][0], t[first][1])
            for r in range(L):
                roles[(x, snd + step * r)] = {
                    "role": 1 if r == 0 else 2,
                    "rank": r,
                    "sender": snd * gx + x,
                    "snd": t[snd],
                    "step": 1 if step == 1 else 0,
                    "rect": rect,
                }
    return roles


def row_roles(plan, args):
    """Lanes of --lane cores along every grid row, inside each panel (as lane_roles)."""
    L, roles = args.lane, {}
    for y in range(plan.grid_y):
        for x0, x1 in plan.panels:
            t = [plan.translated(x, y) for x in range(x0, x1 + 1)]
            # A panel's columns are consecutive in NoC0 (its translated x may skip).
            assert all(c[1] == t[0][1] for c in t)
            assert len(t) % L == 0, (len(t), L)
            for i in range(len(t) // L):
                first, last = i * L, i * L + L - 1
                if args.lane_noc == 0:  # NoC0 runs +x: sender at the left
                    snd, step = first, 1
                    rect = (*t[first + 1], *t[last])
                else:  # NoC1 runs -x: sender at the right, rectangle start is high x
                    snd, step = last, -1
                    rect = (*t[last - 1], *t[first])
                for r in range(L):
                    roles[(x0 + snd + step * r, y)] = {
                        "role": 1 if r == 0 else 2,
                        "rank": r,
                        "sender": y * plan.grid_x + x0 + snd,
                        "snd": t[snd],
                        "step": 3 if step == 1 else 2,
                        "rect": rect,
                    }
    return roles


def program(plan, rec_cnt, lane_cnt, args, roles, wait_cycles):
    slots = args.window if args.stream else 1
    gx, gy = plan.grid_x, plan.grid_y
    cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]
    )
    rec_args, lane_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for y in range(gy):
        for x in range(gx):
            me = y * gx + x
            rec_args[x][y] = [rec_cnt, args.records, me] + list(plan.worker_args(x, y))
            r = roles.get((x, y))
            lane = [lane_cnt, me, plan.tag, args.blocks, args.bytes, int(args.until)]
            if r is None:
                lane += [0] * 10
            else:
                lane += [r["role"], r["rank"], args.lane - 1, *r["snd"], r["step"]]
                lane += list(r["rect"])
            lane_args[x][y] = lane + [wait_cycles, args.window, args.check_every]
    # The two kernels of a core use different NoCs; the lane kernel takes --lane-noc.
    if args.lane_noc == 0:
        rec_cfg = (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_1_default)
        lane_cfg = (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_0_default)
    else:
        rec_cfg = (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_0_default)
        lane_cfg = (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_1_default)

    def kernel(source, rt_args, cfg):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[],
            defines=[],
            runtime_args=rt_args,
            config=ttnn.DataMovementConfigDescriptor(processor=cfg[0], noc=cfg[1]),
        )

    return ttnn.ProgramDescriptor(
        kernels=[
            kernel(RECORDS_KERNEL, rec_args, rec_cfg),
            kernel(STREAM_KERNEL if args.stream else LANE_KERNEL, lane_args, lane_cfg),
        ],
        semaphores=[],
        cbs=[
            cb(CTL_CB, CTL_BYTES, cores),
            cb(REC_CNT_CB, WORDS * 4, cores),
            cb(SHARE_CB, 64, cores),
            cb(LANE_CNT_CB, WORDS * 4, cores),
            cb(LANE_CB, LANE_HEAD + slots * max(args.bytes, 64), cores),
        ],
    )


def summarize_records(cnt, n):
    c = {k: cnt[:, i] for k, i in LC.items()}
    idle = c["idle_lo"] + (c["idle_hi"] << 32)
    out = {
        "rec_cores_finished": int(
            sum(int(c["magic"][i]) == (LANDING_MAGIC | i) for i in range(N_CORES))
        ),
        "rec_cores_all_accepted": int((c["accepted"] == n).sum()),
        "accepted": int(c["accepted"].sum()),
        "torn": int(c["torn"].sum()),
        "healed": int(c["healed"].sum()),
        "early": int(c["early"].sum()),
        "idle_polls": int(idle.sum()),
        "lost": int(c["lost"].sum()),
    }
    torn = [
        {
            "core": i,
            "torn": int(c["torn"][i]),
            "seq": int(c["torn_seq"][i]),
            "word": int(c["torn_word"][i]),
            "value": hex(int(c["torn_value"][i])),
        }
        for i in range(N_CORES)
        if c["torn"][i]
    ]
    lost = [
        {"core": i, "seq": int(c["lost_seq"][i])}
        for i in range(N_CORES)
        if c["lost"][i]
    ]
    if torn:
        out["first_torn"] = torn[:8]
    if lost:
        out["lost_cores"] = lost[:8]
    return out


def summarize_lanes(cnt, roles, gx, mhz):
    c = {k: cnt[:, i] for k, i in LN.items()}
    span = c["t_lo"] + (c["t_hi"] << 32)
    senders = [i for i in range(N_CORES) if int(c["role"][i]) == 1]
    out = {
        "lane_cores_finished": int(
            sum(int(c["magic"][i]) == (LANE_MAGIC | i) for i in range(N_CORES))
        ),
        "senders": len(senders),
        "receivers": int((c["role"] == 2).sum()),
        "roles_ok": all(
            int(c["role"][y * gx + x]) == r["role"] for (x, y), r in roles.items()
        ),
        "bad_blocks": int(c["bad_blocks"].sum()),
        "bad_words": int(c["bad_words"].sum()),
        "stamps_missing": int(c["missing"].sum()),
        "gave_up": int((c["where"] != 0).sum()),
        "stopped": int(c["stopped"].sum()),
        "wait_max_us": round(int(c["wait_max"].max()) / mhz, 2),
    }
    if senders:
        blocks = [int(c["blocks"][i]) for i in senders]
        out["blocks_min"], out["blocks_max"] = min(blocks), max(blocks)
        out["blocks_total"] = sum(blocks)
        per = [int(span[i]) / max(int(c["blocks"][i]), 1) / mhz for i in senders]
        out["us_per_block_mean"] = round(sum(per) / len(per), 4)
        out["us_per_block_max"] = round(max(per), 4)
        out["lane_span_ms_max"] = round(
            max(int(span[i]) for i in senders) / mhz / 1e3, 3
        )
        busy = c["busy_lo"] + (c["busy_hi"] << 32)
        duty = [int(busy[i]) / max(int(span[i]), 1) for i in senders]
        out["duty_mean"] = round(sum(duty) / len(duty), 4)
        out["duty_min"] = round(min(duty), 4)
    recv = [i for i in range(N_CORES) if int(c["role"][i]) == 2]
    if recv:
        check = c["check_lo"] + (c["check_hi"] << 32)
        frac = [int(check[i]) / max(int(span[i]), 1) for i in recv]
        out["check_frac_max"] = round(max(frac), 4)
        out["full_checks"] = int(c["full"].sum())
    # Every receiver checked exactly the blocks its sender completed.
    mismatch = []
    for (x, y), r in roles.items():
        if r["role"] != 2:
            continue
        i, s = y * gx + x, r["sender"]
        if int(c["blocks"][i]) != int(c["blocks"][s]):
            mismatch.append({"core": i, "blocks": int(c["blocks"][i]), "sender": s})
    out["block_count_mismatch"] = len(mismatch)
    bad = [
        {
            "core": i,
            "bad_blocks": int(c["bad_blocks"][i]),
            "k": int(c["bad_k"][i]),
            "word": int(c["bad_w"][i]),
            "value": hex(int(c["bad_v"][i])),
        }
        for i in range(N_CORES)
        if c["bad_words"][i]
    ]
    gave = [
        {
            "core": i,
            "where": WHERE.get(int(c["where"][i]), int(c["where"][i])),
            "k": int(c["where_k"][i]),
            "missing": int(c["missing"][i]),
            "peer_rank": int(c["peer"][i]),
            "peer_phase": PHASES.get(int(c["peer_phase"][i]), int(c["peer_phase"][i])),
            "peer_k": int(c["peer_k"][i]),
        }
        for i in range(N_CORES)
        if c["where"][i]
    ]
    if bad:
        out["first_bad"] = bad[:8]
    if gave:
        out["gave_up_cores"] = gave[:12]
    if mismatch:
        out["mismatch"] = mismatch[:8]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=int, required=True, help="0: X280 idle")
    p.add_argument("--depth", type=int, choices=[1, 4], default=4)
    p.add_argument("--lane", type=int, default=0, help="lane length, 0: no lanes")
    p.add_argument("--bytes", type=int, default=64, help="block bytes")
    p.add_argument("--blocks", type=int, default=1000, help="blocks (most, --until)")
    p.add_argument("--until", action="store_true", help="stop when records are done")
    p.add_argument("--lane-noc", type=int, choices=[0, 1], default=0)
    p.add_argument("--columns", default="all", help="all or a list like 3,7")
    p.add_argument("--along", choices=["column", "row"], default="column")
    p.add_argument("--wait-ms", type=float, default=1000.0, help="bounded wait")
    p.add_argument("--stream", action="store_true", help="fixed body, back to back")
    p.add_argument("--window", type=int, default=4, help="stream: blocks ahead")
    p.add_argument("--check-every", type=int, default=1024, help="stream: full check")
    p.add_argument("--pll", type=int, default=1750)
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args()
    assert args.bytes % 64 == 0 and args.bytes > 0
    assert args.lane in ((0, 2, 5, 10) if args.along == "column" else (0, 2, 3, 6))
    if args.stream:
        assert args.bytes >= 128 and 1 <= args.window <= 8 and args.check_every >= 1
        assert args.window * args.bytes <= 256 * 1024
    if args.records == 0:
        args.depth = 1  # the image's final depth-4 wait needs at least one record
    record = {k: v for k, v in vars(args).items() if k not in ("pll", "timeout")}
    record["load_start"] = loadavg()
    t_start = time.time()

    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    device.disable_and_clear_program_cache()
    log(f"device open {time.time() - t_start:.1f}s")
    rc = 0
    try:
        record.update(run(device, args))
    except RuntimeError as e:  # reported in the JSON line; a core timeout is rc 7
        msg = str(e)
        record["error"] = msg[-3000:]
        rc = 7 if "Timeout" in msg else 1
    finally:
        record["load_end"] = loadavg()
        record["wall_s"] = round(time.time() - t_start, 1)
        print(json.dumps(record), flush=True)
        ttnn.close_mesh_device(device)
    sys.exit(rc)


def run(device, args):
    n = args.records
    runtime = x280mod.Runtime(device, pll_mhz=args.pll)
    plan = runtime.plan(IMAGE, args.depth, 0)
    gx, gy = plan.grid_x, plan.grid_y
    assert gx * gy == N_CORES, (gx, gy)
    mhz = plan.aiclk_mhz()
    log(f"plan: grid {gx}x{gy} panels {plan.panels} depth {plan.depth} aiclk {mhz}")

    def zeros(rows):
        return ttnn.from_torch(
            torch.zeros(rows, WORDS, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

    rec_cnt, lane_cnt = zeros(N_CORES), zeros(N_CORES)
    wait_cycles = int(args.wait_ms * 1e3 * mhz)
    assert wait_cycles < 1 << 32

    # The X280 starts first: it picks the tag the kernel args carry.
    session = plan.start(n)
    roles = lane_roles(plan, args)
    prog = program(
        plan,
        rec_cnt.buffer_address(),
        lane_cnt.buffer_address(),
        args,
        roles,
        wait_cycles,
    )
    finished = False
    log(f"launch: {len(roles)} lane cores")
    t0 = time.perf_counter()
    try:
        ttnn.generic_op([rec_cnt, lane_cnt], prog)
        log("program enqueued")
        ticks = plan.finish(session, args.timeout)
        finished = True
    finally:
        if not finished:  # stop the X280 image before anything else
            plan.finish(session, 5.0)
    wall = time.perf_counter() - t0
    log(f"x280 done: {ticks} ticks, host wall {wall:.2f}s")

    # Close the X280 runtime before reading anything back.
    del plan, runtime
    gc.collect()
    log("x280 runtime closed")

    rc = ttnn.to_torch(rec_cnt).long() & 0xFFFFFFFF
    lc = ttnn.to_torch(lane_cnt).long() & 0xFFFFFFFF
    log("counters read back")
    out = {
        "ticks": ticks,
        "us_per_record": (
            round(ticks / x280mod.MTIME_HZ * 1e6 / n, 4) if ticks and n else None
        ),
        "host_wall_s": round(wall, 2),
        "aiclk_mhz": mhz,
    }
    out.update(summarize_records(rc, n))
    out.update(summarize_lanes(lc, roles, gx, mhz))
    rec_ok = (
        (bool(ticks) or n == 0)
        and out["rec_cores_finished"] == N_CORES
        and out["rec_cores_all_accepted"] == N_CORES
        and out["lost"] == 0
        and out["torn"] == 0
    )
    lane_ok = (
        out["lane_cores_finished"] == N_CORES
        and out["roles_ok"]
        and out["bad_words"] == 0
        and out["gave_up"] == 0
        and out["stamps_missing"] == 0
        and out["block_count_mismatch"] == 0
    )
    out["ok"] = bool(rec_ok and lane_ok)
    return out


if __name__ == "__main__":
    main()
