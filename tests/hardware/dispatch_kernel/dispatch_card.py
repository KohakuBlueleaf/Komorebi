"""Dispatch kernel tests on a card: the kernel on X280 cluster 0, fed from the host,
the stand-in worker (kernels/worker.cpp) on every Tensix core. One case per run, one
JSON line at the end: python dispatch_card.py <case> [options] (see README.md).
"""

import argparse
import gc
import itertools
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
sys.path.insert(0, BUILD)

import _km_dispatch_test as dk  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

KERNEL = os.path.join(HERE, "kernels", "worker.cpp")
IMAGE = os.path.join(BUILD, "km-dispatch.bin")
VERIFY_IMAGE = os.path.join(BUILD, "km-dispatch-verify.bin")
MEMCOST = os.path.join(BUILD, "km-memcost.bin")
MEMCOST_TESTS = [
    "loop",
    "stack_load",
    "stack_store",
    "stack_chain",
    "ddr_load",
    "ddr_store",
    "ddr_chain",
    "lim_poll",
    "fence_ir",
    "fence_ow",
    "mtime",
    "store_fence",
    "ddr_fence_ir",
]
IPI_ACK_DDR = 0x470020  # the runtime's IPI ack in the cluster's DRAM bank
DRAM_KEEP = []  # DRAM tensors a running image may use
CTL_CB, OUT_CB = 8, 9
CTL_BYTES = 0xA00  # KM_CTL_BYTES (0x980) rounded up to whole 64-byte lines
AICLK_HZ = 1.35e9  # Tensix clock of this card, for the wait limit only

# km_abi.h and km_dispatch.h
SEG_NOP, SEG_EXIT = 0, 255
SEG_PING, SEG_STOP, SEG_DEADLINE = 254, 253, 252
F_EXCL = 2
EV_DONE, EV_TIMEOUT, EV_PONG, EV_REFUSED = 0, 1, 2, 3
REFUSE_SEG, REFUSE_FLAGS, REFUSE_TEAM, REFUSE_EXITED = 1, 2, 3, 4
STATE_READY, STATE_DONE, STATE_FAILED = 2, 3, 4
ERR_DEADLINE, ERR_STOPPED, ERR_STOP_BUSY = 4, 7, 9
R_NAMES = {
    0: "state",
    1: "error",
    2: "where",
    3: "slot",
    4: "seq",
    5: "req",
    6: "seen",
    7: "commands",
    8: "records",
    9: "events",
    10: "busy",
    11: "max_busy",
    12: "ctl",
    21: "boot_ticks",
    24: "refused",
    25: "held",
    26: "no_room",
    27: "checked",
    28: "bad_lines",
    29: "bad_first",
}
R_WIDE = {"ticks": 13, "idle_passes": 15, "full_passes": 17, "passes": 19, "last": 22}

# kernels/dk_test.h
SEG_CHECK, SEG_DELAY, SEG_STALL = 1, 2, 3
DK_WORDS, DK_MAGIC = 64, 0xD15C0000
C = {
    "magic": 0,
    "tag": 1,
    "run": 2,
    "key_sum": 3,
    "key_mix": 4,
    "ignored": 5,
    "arg_errors": 6,
    "hdr_errors": 7,
    "exit": 8,
    "acks": 9,
    "incs": 10,
    "seg": 11,
    "state": 16,
}
S_GAVE, S_WHERE, S_SEQ = 16, 17, 18
WHERE = {1: "boot", 2: "next", 3: "done", 4: "stall", 5: "delay", 6: "exit"}
GAVE = {0: "none", 1: "stop", 2: "limit"}

M32 = 0xFFFFFFFF
TARGETS = {"lim": 0, "l1": 1}


def log(msg):
    print(msg, flush=True)


def loadavg():
    with open("/proc/loadavg") as f:
        return f.read().split()[0]


def dk_arg(key, param, w):
    """kernels/dk_test.h dk_arg, on Python ints."""
    x = ((key * 0x9E3779B1) ^ (param * 0x85EBCA77) ^ (w * 0xC2B2AE3D)) & M32
    x ^= x >> 15
    x = (x * 0x2C1B3C6D) & M32
    x ^= x >> 12
    return x


def dk_mix(key):
    return dk_arg(key, 0x5EED, 0)


def verify_word(i, w):
    """km_dispatch.h kmd_verify_word, on Python ints."""
    x = ((i * 0x9E3779B1) ^ ((w + 1) * 0x85EBCA77)) & M32
    x ^= x >> 15
    return (x * 0x2C1B3C6D) & M32


def verify_record(i, cores):
    """The record of command i that a KMD_VERIFY_LINES kernel expects (a NOP)."""
    count = 1 + verify_word(i, 10) % cores
    first = verify_word(i, 11) % (cores - count + 1)
    words = [0] * 16
    words[2] = first | (count << 16)
    for j, at in enumerate([3, 5, 6, 7, 9, 10, 11, 13, 14, 15]):
        words[at] = verify_word(i, j)
    return words


def record(seg, first, count, key=0, param=0):
    """The 16 words of a record as the host passes them (seq, landing words and slot
    are filled on the way)."""
    words = [0] * 16
    words[1] = seg
    words[2] = first | (count << 16)
    args = [key & M32, param & M32] + [dk_arg(key, param, w) for w in range(2, 10)]
    for i, value in enumerate(args):
        words[[3, 5, 6, 7, 9, 10, 11, 13, 14, 15][i]] = value
    return words


def command(seg, a0=0):
    """A command to the dispatch kernel itself (KM_SEG_PING, _STOP, _DEADLINE)."""
    words = [0] * 16
    words[1] = seg
    words[3] = a0
    return words


def kind(status):
    return status & 0xFF, status >> 8


# ---------------------------------------------------------------------------------
# Device, programs, the run frame


def all_cores(gx, gy):
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]
    )


def cb(index, size, cores):
    fmt = ttnn.CBFormatDescriptor(
        buffer_index=index, data_format=ttnn.uint32, page_size=size
    )
    return ttnn.CBDescriptor(
        total_size=size, core_ranges=cores, format_descriptors=[fmt]
    )


def program(plan, table, out, limit_s, defines):
    """The stand-in worker on every core, one data-movement kernel."""
    gx, gy = plan.grid_x, plan.grid_y
    cores = all_cores(gx, gy)
    limit = max(1, int(limit_s * AICLK_HZ) >> 20)
    args = ttnn.RuntimeArgs()
    for y in range(gy):
        for x in range(gx):
            args[x][y] = list(plan.worker_args(x, y, table, out, limit))
    kernel = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[],
        defines=defines,
        runtime_args=args,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
        ),
    )
    return ttnn.ProgramDescriptor(
        kernels=[kernel],
        semaphores=[],
        cbs=[cb(CTL_CB, CTL_BYTES, cores), cb(OUT_CB, DK_WORDS * 4, cores)],
    )


def int_tensor(device, array):
    return ttnn.from_torch(
        torch.from_numpy(np.asarray(array, dtype=np.uint32).view(np.int32)),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )


def dram_area(device):
    """The bank address of 4 KiB that is ours in every DRAM bank (bank offset 0):
    page rows of a DRAM tensor of 1024 pages, kept alive with the device."""
    pages = int_tensor(device, np.zeros((1024, 1024), dtype=np.uint32))
    DRAM_KEEP.append(pages)
    ddr = pages.buffer_address()
    if ddr <= IPI_ACK_DDR < ddr + 4096:
        raise SystemExit("the buffer covers the runtime's IPI ack word")
    return ddr


def result_block(words):
    out = {name: words[i] for i, name in R_NAMES.items()}
    for name, lo in R_WIDE.items():
        out[name] = words[lo] | (words[lo + 1] << 32)
    out["spare"] = list(words[30:32])
    return out


def with_kernel(
    device, args, body, depth=1, defines=(), preload=None, stop=False, info=None
):
    """Grid, then kernel (after preload(plan)); body(plan); the kernel's end (with stop,
    the runtime stops it); X280 runtime closed, counters read. (body, finish, cnt).
    `info` gets "started" and, if the kernel had to be stopped, "aborted"."""
    info = {} if info is None else info
    info["started"] = False
    runtime = dk.Runtime(device, pll_mhz=args.pll)
    plan = runtime.plan()
    gx, gy = plan.grid_x, plan.grid_y
    log(f"plan: grid {gx}x{gy} panels {plan.panels}")
    table = int_tensor(device, np.array([plan.core_table()], dtype=np.uint32))
    state = dram_area(device)
    plan.prepare(
        depth, args.host_depth, args.event_depth, 30.0, 1.0, args.idle_s, state
    )
    out = int_tensor(device, np.zeros((gx * gy, DK_WORDS), dtype=np.uint32))
    prog = program(
        plan, table.buffer_address(), out.buffer_address(), args.limit, list(defines)
    )
    body_out, fin = None, None
    if preload is not None:
        preload(plan)
    try:
        if args.image_first:
            plan.start(args.image)
            info["started"] = True
            log("kernel started")
        ttnn.generic_op([table, out], prog)
        log("grid launched")
        if not args.image_first:
            plan.start(args.image)
            info["started"] = True
            log("kernel started")
        state = plan.wait_ready(30.0)
        log(f"kernel state {state}")
        body_out = body(plan) if state == STATE_READY else {"ready_state": state}
        fin = plan.finish(10.0, wait=not stop)
        log(f"kernel finished: result {fin['result']} cleaned {fin['cleaned']}")
    finally:
        if plan.running():
            info["aborted"] = plan.abort()
            log(f"stopping the kernel: cleaned {info['aborted']}")
        del plan, runtime
        gc.collect()
        log("x280 runtime closed")
    cnt = ttnn.to_torch(out).numpy().astype(np.int64) & M32
    log("counters read back")
    fin["block"] = result_block(fin.pop("words"))
    return body_out, fin, cnt


def exit_and_stop(plan):
    """EXIT to the whole grid, then STOP; returns the EXIT event."""
    n = plan.grid_x * plan.grid_y
    i = plan.submit(record(SEG_EXIT, 0, n))
    events = plan.take(i, 10.0)
    plan.submit(command(SEG_STOP))
    return events


def core_report(cnt, n_cores, tag):
    """Cores that finished, with their tag, and the gave-up / where of every core."""
    magic_ok = [
        int(cnt[i, C["magic"]]) == (DK_MAGIC | i) and int(cnt[i, C["tag"]]) == tag
        for i in range(n_cores)
    ]
    state = cnt[:, C["state"] : C["state"] + 19]
    return {
        "cores_finished": int(sum(magic_ok)),
        "where": {
            WHERE.get(w, str(w)): int((state[:, S_WHERE] == w).sum())
            for w in sorted(set(state[:, S_WHERE].tolist()))
        },
        "gave": {
            GAVE.get(g, str(g)): int((state[:, S_GAVE] == g).sum())
            for g in sorted(set(state[:, S_GAVE].tolist()))
        },
        "arg_errors": int(cnt[:, C["arg_errors"]].sum()),
        "hdr_errors": int(cnt[:, C["hdr_errors"]].sum()),
    }


def check_teams(cnt, n_cores, sent):
    """Compare every core's run count, key sum and key mix with the records sent
    ([(first, count, key)]); returns the number of cores that differ and examples."""
    runs = np.zeros(n_cores, dtype=np.int64)
    sums = np.zeros(n_cores, dtype=np.int64)
    mixes = np.zeros(n_cores, dtype=np.int64)
    for first, count, key in sent:
        runs[first : first + count] += 1
        sums[first : first + count] += key
        mixes[first : first + count] ^= dk_mix(key)
    sums &= M32
    bad = [
        {
            "core": i,
            "run": [int(cnt[i, C["run"]]), int(runs[i])],
            "sum_ok": int(cnt[i, C["key_sum"]]) == int(sums[i]),
            "mix_ok": int(cnt[i, C["key_mix"]]) == int(mixes[i]),
        }
        for i in range(n_cores)
        if int(cnt[i, C["run"]]) != runs[i]
        or int(cnt[i, C["key_sum"]]) != sums[i]
        or int(cnt[i, C["key_mix"]]) != mixes[i]
    ]
    return len(bad), bad[:8]


def check_events(events, reqs, first, kinds=(EV_DONE,)):
    """Events numbered first, first + 1, .. in order, each request once, status in
    `kinds`."""
    got = [e[1] for e in events]
    indices_ok = [e[0] for e in events] == list(range(first, first + len(reqs)))
    return {
        "events": len(events),
        "indices_ok": bool(indices_ok),
        "reqs_ok": sorted(got) == sorted(reqs),
        "status_ok": all(kind(e[2])[0] in kinds for e in events),
    }


# ---------------------------------------------------------------------------------
# Cases


def case_limit(device, args):
    runtime = dk.Runtime(device, pll_mhz=args.pll)
    try:
        out = dict(runtime.image_limit())
    finally:
        del runtime
        gc.collect()
    out["image_bytes"] = os.path.getsize(args.image)
    return out


def case_ping(device, args, issue_one=False):
    n_cores = 0

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        out = {"tag": plan.tag}
        out["ping_req"] = plan.submit(command(SEG_PING))
        out["pong"] = plan.take(out["ping_req"], 5.0)
        if issue_one:
            out["issue_req"] = plan.submit(record(SEG_CHECK, 0, n_cores, 77, 5))
            out["issue"] = plan.take(out["issue_req"], 5.0)
        out["exit"] = exit_and_stop(plan)
        return out

    out, fin, cnt = with_kernel(device, args, body)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    out["run_per_core"] = sorted(set(cnt[:n_cores, C["run"]].tolist()))
    out["ok"] = bool(
        fin["result_ready"]
        and fin["block"]["error"] == 0
        and out["pong"]
        and kind(out["pong"][0][2])[0] == EV_PONG
        and out["pong"][0][1] == out["ping_req"]
        and out["cores"]["cores_finished"] == n_cores
        and out["cores"]["arg_errors"] == 0
        and (not issue_one or out["issue"][0][1] == out["issue_req"])
    )
    return out


def case_teams(device, args):
    """args.n records with random teams (CHECK), depth 16."""
    rng = np.random.default_rng(args.seed)
    sent, n_cores = [], 0

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        recs = []
        for _ in range(args.n):
            count = int(rng.integers(1, n_cores + 1))
            first = int(rng.integers(0, n_cores - count + 1))
            key = int(rng.integers(1, 1 << 32))
            param = int(rng.integers(0, 1 << 32))
            recs.append(record(SEG_CHECK, first, count, key, param))
            sent.append((first, count, key))
        reqs = list(range(plan.next, plan.next + args.n))
        first = plan.seen + 1
        events, times = plan.stream(np.array(recs, dtype=np.uint32), 10.0, 64)
        out = {"tag": plan.tag, "times": times}
        out["check"] = check_events(events, reqs, first)
        out["exit"] = exit_and_stop(plan)
        return out

    out, fin, cnt = with_kernel(device, args, body, depth=16)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    out["team_mismatch"], out["first_mismatch"] = check_teams(cnt, n_cores, sent)
    out["ok"] = bool(
        fin["result_ready"]
        and fin["block"]["error"] == 0
        and all(out["check"].values())
        and out["team_mismatch"] == 0
        and out["cores"]["cores_finished"] == n_cores
        and out["cores"]["arg_errors"] == 0
        and out["cores"]["hdr_errors"] == 0
    )
    return out


def case_disjoint(device, args):
    """Team A = first half, one DELAY of args.delay_us; team B = second half, then
    args.n CHECK records. B's events must arrive before A's."""
    n_cores, sent = 0, []
    cycles = int(args.delay_us * 1e-6 * AICLK_HZ)

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        half = n_cores // 2
        a = plan.submit(record(SEG_DELAY, 0, half, 1, cycles))
        sent.append((0, half, 1))
        last = 0
        for k in range(args.n):
            last = plan.submit(record(SEG_CHECK, half, half, 2 + k, k))
            sent.append((half, half, 2 + k))
        events = plan.take(last, 10.0)
        out = {"tag": plan.tag, "events": events, "a_req": a}
        out["exit"] = exit_and_stop(plan)
        return out

    out, fin, cnt = with_kernel(device, args, body, depth=16)
    events = out["events"]
    order = [e[1] for e in events]
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    out["order"] = order
    out["team_mismatch"], out["first_mismatch"] = check_teams(cnt, n_cores, sent)
    a_req = out["a_req"]
    a_ticks = [e[3] for e in events if e[1] == a_req]
    out["ok"] = bool(
        fin["result_ready"]
        and fin["block"]["error"] == 0
        and out["team_mismatch"] == 0
        and out["cores"]["arg_errors"] == 0
        and len(events) == args.n + 1
        and order[-1] == a_req
        and fin["block"]["max_busy"] >= 2
        and a_ticks
        and a_ticks[0] > max(e[3] for e in events if e[1] != a_req)
        and out["cores"]["cores_finished"] == n_cores
    )
    return out


def case_stall(device, args, leader=False):
    """A team of every core whose member --core (with leader, the team's leader, core
    0) waits for the stop word, under KM_SEG_DEADLINE: KM_EV_TIMEOUT, the stop word."""
    n_cores = 0
    stalled = 0 if leader else args.core

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        out = {"tag": plan.tag, "stalled": stalled}
        i = plan.submit(record(SEG_CHECK, 0, n_cores, 3, 4))
        out["before"] = plan.take(i, 5.0)
        plan.submit(command(SEG_DEADLINE, args.deadline_us))
        out["stall_req"] = plan.submit(record(SEG_STALL, 0, n_cores, 9, stalled))
        out["timeout"] = plan.take(out["stall_req"], 10.0)
        out["host_state"] = plan.host_word(0)
        return out

    out, fin, cnt = with_kernel(device, args, body, depth=4)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    state = cnt[:n_cores, C["state"] : C["state"] + 19]
    blk = fin["block"]
    # Cores that did not report the timed-out record: never took it, or stopped other
    # than waiting for the next record or, as its leader, for the members.
    out["not_reported"] = [
        {
            "core": i,
            "where": WHERE.get(int(state[i, S_WHERE])),
            "gave": GAVE.get(int(state[i, S_GAVE])),
            "seq": int(state[i, S_SEQ]),
        }
        for i in range(n_cores)
        if int(state[i, S_SEQ]) != blk["seq"] or int(state[i, S_WHERE]) not in (2, 3)
    ]
    ev = out["timeout"]
    out["ok"] = bool(
        ev
        and kind(ev[-1][2]) == (EV_TIMEOUT, blk["slot"])
        and ev[-1][1] == out["stall_req"]
        and blk["error"] == ERR_DEADLINE
        and blk["req"] == out["stall_req"]
        and ev[-1][3] >= args.deadline_us * 50
        and out["cores"]["cores_finished"] == n_cores
        and out["cores"]["gave"] == {"stop": n_cores}
        and [c["core"] for c in out["not_reported"]] == [stalled]
    )
    return out


def case_bench(device, args):
    """args.n NOP records at depth args.depth to teams of args.team cores (0 = all),
    random first core, then EXIT; microseconds per command. --preload stores every
    command before the kernel starts (kernel only); else the host feeds in batches."""
    rng = np.random.default_rng(args.seed)
    n_cores, sent, recs = 0, [], []

    def commands(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        team = args.team or n_cores
        firsts = rng.integers(0, n_cores - team + 1, size=args.n)
        r = np.zeros((args.n + 1, 16), dtype=np.uint32)
        r[:, 1] = SEG_NOP
        r[:-1, 2] = firsts.astype(np.uint32) | (team << 16)
        r[:, 3] = np.arange(1, args.n + 2, dtype=np.uint32)
        r[-1] = record(SEG_EXIT, 0, n_cores)
        recs.append(r)
        sent.extend(zip(firsts.tolist(), [team] * args.n, r[:-1, 3].tolist()))
        if args.preload:
            for rec in r.tolist():
                plan.submit(rec)

    def body(plan):
        if args.preload:
            events, times = plan.take(args.n + 1, 30.0), {}
        else:
            events, times = plan.stream(recs[0], 30.0, args.batch, args.settle_us)
        plan.submit(command(SEG_STOP))
        out = {"tag": plan.tag, "times": times}
        out["check"] = check_events(events, list(range(1, args.n + 2)), 1)
        ticks = np.array([e[3] for e in events[:-1]], dtype=np.float64)
        out["latency_us"] = {
            "mean": round(float(ticks.mean()) / 50, 3),
            "p50": round(float(np.median(ticks)) / 50, 3),
            "max": round(float(ticks.max()) / 50, 3),
        }
        return out

    if args.preload and args.n + 1 > args.host_depth:
        raise SystemExit("--preload needs n + 1 <= host depth")
    out, fin, cnt = with_kernel(device, args, body, depth=args.depth, preload=commands)
    blk = fin["block"]
    n = args.n
    t = out["times"]
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    out["team_mismatch"], out["first_mismatch"] = check_teams(cnt, n_cores, sent)
    # The kernel's span runs from the first command taken to the EXIT's event, over
    # n + 1 commands; the host's from the first store to the last event read.
    out["us_per_command_kernel"] = round(blk["last"] / dk.MTIME_HZ * 1e6 / (n + 1), 4)
    if t:
        out["us_per_command_host"] = round(t["total_s"] * 1e6 / (n + 1), 4)
        out["us_store_per_command"] = round(t["store_s"] * 1e6 / (n + 1), 4)
    out["idle_pass_share"] = round(blk["idle_passes"] / max(blk["passes"], 1), 4)
    out["full_pass_share"] = round(blk["full_passes"] / max(blk["passes"], 1), 4)
    out["no_room"] = blk["no_room"]
    out["ok"] = bool(
        fin["result_ready"]
        and blk["error"] == 0
        and all(out["check"].values())
        and out["team_mismatch"] == 0
        and out["cores"]["cores_finished"] == n_cores
    )
    return out


def percentiles(us):
    return {
        str(q): round(float(np.percentile(us, q)), 3)
        for q in (0, 1, 10, 25, 50, 75, 90, 99, 100)
    }


def case_store(device, args):
    """Host store cost, no kernel running: --n bursts of --lines 64-byte stores and a
    fence, per idle gap (--gaps-us), keep-busy period (--busy-us, 0 = none), burst
    positions (--spread), a read before each burst (--read); LIM or a Tensix L1."""
    runtime = dk.Runtime(device, pll_mhz=args.pll)
    plan = runtime.plan()
    configs = []
    try:
        grid = itertools.product(args.busy_us, args.spread, args.lines, args.gaps_us)
        for busy, spread, lines, gap in grid:
            target = TARGETS[args.target]
            t = np.array(
                plan.store_probe(args.n, lines, spread, gap, busy, args.read, target)
            ).reshape(-1, 2)
            us = t.sum(axis=1) * 1e6
            log(f"store: busy {busy} lines {lines} gap {gap}: p50 {np.median(us):.2f}")
            configs.append(
                {
                    "busy_us": busy,
                    "lines": lines,
                    "spread": spread,
                    "gap_us": gap,
                    "pct_us": percentiles(us),
                    "store_pct_us": percentiles(t[:, 0] * 1e6),
                    "fence_pct_us": percentiles(t[:, 1] * 1e6),
                    "over_5us": round(float((us > 5).mean()), 4),
                    "raw_store_fence_us": np.round(t * 1e6, 2).tolist(),
                }
            )
    finally:
        del plan, runtime
        gc.collect()
    return {"target": args.target, "read": args.read, "configs": configs, "ok": True}


def case_memcost(device, args):
    """x280/memcost.c: picoseconds per load, store and fence in the stack (LIM),
    cached DRAM (a buffer of ours in every bank) and registers; no grid."""
    ddr = dram_area(device)
    runtime = dk.Runtime(device, pll_mhz=args.pll)
    plan = runtime.plan()
    try:
        fin = plan.run_image(MEMCOST, [ddr], 10.0)
    finally:
        if plan.running():
            plan.abort()
        del plan, runtime
        gc.collect()
    words = fin.pop("words")
    ps = dict(zip(MEMCOST_TESTS, words))
    return {"ddr_offset": ddr, "finish": fin, "ps_per_op": ps, "ok": fin["result"] == 0}


def case_compile_error(device, args):
    """The worker fails to compile: the exception is that build failure, and the X280
    image was never started (or, with --image-first, was stopped)."""
    info = {}
    try:
        with_kernel(
            device,
            args,
            lambda plan: {},
            defines=[("DK_COMPILE_ERROR", "1")],
            info=info,
        )
    except Exception as err:  # noqa: BLE001
        message = str(err)
        compile_error = "build failed" in message and "DK_COMPILE_ERROR" in message
        if args.image_first:
            image_ok = info["started"] and info.get("aborted") is True
        else:
            image_ok = not info["started"]
        return {
            "raised": type(err).__name__,
            "message": message[:300],
            "compile_error": compile_error,
            "image": info,
            "ok": bool(compile_error and image_ok),
        }
    return {"raised": None, "image": info, "ok": False}


def finished_clean(fin, n_cores, cores, error=0):
    """The kernel ended with `error`, every core reached its end, no argument errors."""
    return (
        fin["result_ready"]
        and fin["block"]["error"] == error
        and cores["cores_finished"] == n_cores
        and cores["arg_errors"] == 0
        and cores["hdr_errors"] == 0
    )


def case_fill(device, args):
    """All 16 slots busy: 16 DELAY records of --delay-us on 16 disjoint teams, then --n
    CHECK records with random teams that wait for slots; then EXIT."""
    rng = np.random.default_rng(args.seed)
    n_cores, sent = 0, []
    cycles = int(args.delay_us * 1e-6 * AICLK_HZ)

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        width = n_cores // 16
        s0 = plan.seen
        reqs = []
        for j in range(16):
            reqs.append(plan.submit(record(SEG_DELAY, j * width, width, 1 + j, cycles)))
            sent.append((j * width, width, 1 + j))
        for k in range(args.n):
            count = int(rng.integers(1, n_cores + 1))
            first = int(rng.integers(0, n_cores - count + 1))
            reqs.append(plan.submit(record(SEG_CHECK, first, count, 100 + k, k)))
            sent.append((first, count, 100 + k))
        events = plan.take(s0 + len(reqs), 10.0)
        out = {"tag": plan.tag, "check": check_events(events, reqs, s0 + 1)}
        out["exit"] = exit_and_stop(plan)
        return out

    out, fin, cnt = with_kernel(device, args, body, depth=16)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    out["team_mismatch"], out["first_mismatch"] = check_teams(cnt, n_cores, sent)
    blk = fin["block"]
    out["ok"] = bool(
        finished_clean(fin, n_cores, out["cores"])
        and all(out["check"].values())
        and out["team_mismatch"] == 0
        and blk["max_busy"] == 16
        and blk["full_passes"] > 0
    )
    return out


def case_exit_busy(device, args):
    """EXIT sent while a DELAY of --delay-us on half the grid is in flight: the kernel
    holds EXIT until the DELAY completes, then sends it to every core."""
    n_cores = 0
    cycles = int(args.delay_us * 1e-6 * AICLK_HZ)

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        out = {"tag": plan.tag}
        s0 = plan.seen
        out["delay_req"] = plan.submit(record(SEG_DELAY, 0, n_cores // 2, 1, cycles))
        out["exit_req"] = plan.submit(record(SEG_EXIT, 0, 1))
        out["events"] = plan.take(s0 + 2, 10.0)
        plan.submit(command(SEG_STOP))
        return out

    out, fin, cnt = with_kernel(device, args, body, depth=16)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    out["exit_per_core"] = int((cnt[:n_cores, C["exit"]] != 0).sum())
    order = [e[1] for e in out["events"]]
    out["ok"] = bool(
        finished_clean(fin, n_cores, out["cores"])
        and order == [out["delay_req"], out["exit_req"]]
        and all(kind(e[2])[0] == EV_DONE for e in out["events"])
        and fin["block"]["held"] == 1
        and fin["block"]["max_busy"] == 1
        and out["exit_per_core"] == n_cores
        and out["cores"]["where"] == {"exit": n_cores}
    )
    return out


def case_after_exit(device, args):
    """A CHECK record after a completed EXIT is refused (KMD_REFUSE_EXITED); a PING
    still answers; STOP ends the session without an error."""
    n_cores = 0

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        out = {"tag": plan.tag}
        plan.submit(record(SEG_EXIT, 0, n_cores))
        out["exit"] = plan.take(plan.seen + 1, 10.0)
        s0 = plan.seen
        out["late_req"] = plan.submit(record(SEG_CHECK, 0, n_cores, 5, 5))
        out["ping_req"] = plan.submit(command(SEG_PING))
        out["events"] = plan.take(s0 + 2, 5.0)
        plan.submit(command(SEG_STOP))
        return out

    out, fin, cnt = with_kernel(device, args, body)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    ev = out["events"]
    out["ok"] = bool(
        finished_clean(fin, n_cores, out["cores"])
        and [(e[1], kind(e[2])) for e in ev]
        == [
            (out["late_req"], (EV_REFUSED, REFUSE_EXITED)),
            (out["ping_req"], (EV_PONG, 0)),
        ]
        and fin["block"]["refused"] == 1
        and out["cores"]["gave"] == {"none": n_cores}
        and int(cnt[:n_cores, C["run"]].sum()) == 0
    )
    return out


def case_malformed(device, args):
    """Four bad records (segment 240, flag bit 0, count 0, a team past the last core)
    are refused with their reasons; a good CHECK after them runs; then EXIT."""
    n_cores = 0

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        bad_flag = record(SEG_CHECK, 0, n_cores, 7, 7)
        bad_flag[1] |= 1 << 8
        bad = [
            (record(240, 0, n_cores), REFUSE_SEG),
            (bad_flag, REFUSE_FLAGS),
            (record(SEG_CHECK, 0, 0, 7, 7), REFUSE_TEAM),
            (record(SEG_CHECK, n_cores - 1, 2, 7, 7), REFUSE_TEAM),
        ]
        first_event = plan.seen + 1
        want = [(plan.submit(rec), (EV_REFUSED, why)) for rec, why in bad]
        good = plan.submit(record(SEG_CHECK, 0, n_cores, 11, 3))
        want.append((good, (EV_DONE, 0)))
        events = plan.take(first_event + len(want) - 1, 5.0)
        out = {"tag": plan.tag, "want": want, "events": events}
        out["indices_ok"] = [e[0] for e in events] == list(
            range(first_event, first_event + len(want))
        )
        got = sorted((e[1], kind(e[2])) for e in events)
        out["kinds_ok"] = got == sorted(want)
        out["exit"] = exit_and_stop(plan)
        return out

    out, fin, cnt = with_kernel(device, args, body)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    out["team_mismatch"], _ = check_teams(cnt, n_cores, [(0, n_cores, 11)])
    out["ok"] = bool(
        finished_clean(fin, n_cores, out["cores"])
        and out["indices_ok"]
        and out["kinds_ok"]
        and out["team_mismatch"] == 0
        and fin["block"]["refused"] == 4
    )
    return out


def case_stop_busy(device, args):
    """STOP while a DELAY of --delay-us on every core is in flight: the kernel ends with
    KMD_ERR_STOP_BUSY and the stop word ends every core's wait."""
    n_cores = 0
    cycles = int(args.delay_us * 1e-6 * AICLK_HZ)

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        out = {"tag": plan.tag}
        out["delay_req"] = plan.submit(record(SEG_DELAY, 0, n_cores, 1, cycles))
        out["stop_req"] = plan.submit(command(SEG_STOP))
        return out

    out, fin, cnt = with_kernel(device, args, body, depth=4)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    blk = fin["block"]
    out["ok"] = bool(
        finished_clean(fin, n_cores, out["cores"], ERR_STOP_BUSY)
        and blk["req"] == out["stop_req"]
        and blk["busy"] != 0
        and out["cores"]["gave"] == {"stop": n_cores}
    )
    return out


def case_runtime_stop(device, args):
    """The runtime stops the kernel while a DELAY of 0.5 s on every core is in flight:
    KMD_ERR_STOPPED, and the stop word ends every core's wait."""
    n_cores = 0
    cycles = int(0.5 * AICLK_HZ)

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        out = {"tag": plan.tag}
        out["delay_req"] = plan.submit(record(SEG_DELAY, 0, n_cores, 1, cycles))
        by = time.monotonic() + 5.0
        while plan.host_word(1) < out["delay_req"] and time.monotonic() < by:
            pass
        out["taken"] = plan.host_word(1)
        return out

    out, fin, cnt = with_kernel(device, args, body, depth=4, stop=True)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    blk = fin["block"]
    out["ok"] = bool(
        fin["cleaned"]
        and finished_clean(fin, n_cores, out["cores"], ERR_STOPPED)
        and blk["state"] == STATE_FAILED
        and blk["busy"] != 0
        and out["cores"]["gave"] == {"stop": n_cores}
    )
    return out


def case_excl(device, args):
    """A: DELAY of --delay-us on the first half; C: DELAY twice as long on the second
    half but one core; B: CHECK with KM_F_EXCL on the two cores around the middle,
    overlapping A only. B waits for A, not for C."""
    n_cores = 0

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        half = n_cores // 2
        cycles = int(args.delay_us * 1e-6 * AICLK_HZ)
        b_rec = record(SEG_CHECK, half - 1, 2, 2, 2)
        b_rec[1] |= F_EXCL << 8
        out = {"tag": plan.tag}
        s0 = plan.seen
        out["a"] = plan.submit(record(SEG_DELAY, 0, half, 1, cycles))
        out["c"] = plan.submit(record(SEG_DELAY, half + 1, half - 1, 3, 2 * cycles))
        out["b"] = plan.submit(b_rec)
        out["events"] = plan.take(s0 + 3, 10.0)
        out["exit"] = exit_and_stop(plan)
        return out

    out, fin, cnt = with_kernel(device, args, body, depth=16)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    half = n_cores // 2
    sent = [(0, half, 1), (half - 1, 2, 2), (half + 1, half - 1, 3)]
    out["team_mismatch"], out["first_mismatch"] = check_teams(cnt, n_cores, sent)
    order = [e[1] for e in out["events"]]
    out["order"] = order
    b_ticks = [e[3] for e in out["events"] if e[1] == out["b"]]
    out["ok"] = bool(
        finished_clean(fin, n_cores, out["cores"])
        and order == [out["a"], out["b"], out["c"]]
        and all(kind(e[2])[0] == EV_DONE for e in out["events"])
        and fin["block"]["held"] == 1
        and fin["block"]["max_busy"] >= 2
        and b_ticks
        and b_ticks[0] < args.delay_us * 50 / 2
        and out["team_mismatch"] == 0
    )
    return out


def case_verify(device, args):
    """The KMD_VERIFY_LINES kernel image: --n NOP records whose header, team and ten
    argument words are functions of the command number; the kernel checks each word."""
    if args.image == IMAGE:
        args.image = VERIFY_IMAGE
    n_cores, sent = 0, []

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        first_req, first_event = plan.next, plan.seen + 1
        reqs = list(range(first_req, first_req + args.n))
        recs = np.array([verify_record(i, n_cores) for i in reqs], dtype=np.uint32)
        sent.extend((r[2] & 0xFFFF, r[2] >> 16, r[3]) for r in recs.tolist())
        events, times = plan.stream(recs, 30.0, args.batch, args.settle_us)
        out = {"tag": plan.tag, "times": times}
        out["check"] = check_events(events, reqs, first_event)
        out["exit"] = exit_and_stop(plan)
        return out

    out, fin, cnt = with_kernel(device, args, body, depth=args.depth)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    out["team_mismatch"], out["first_mismatch"] = check_teams(cnt, n_cores, sent)
    blk = fin["block"]
    out["ok"] = bool(
        finished_clean(fin, n_cores, out["cores"])
        and all(out["check"].values())
        and out["team_mismatch"] == 0
        and blk["checked"] == args.n
        and blk["bad_lines"] == 0
    )
    return out


CASES = {
    "limit": case_limit,
    "ping": case_ping,
    "issue": lambda device, args: case_ping(device, args, issue_one=True),
    "teams": case_teams,
    "disjoint": case_disjoint,
    "stall": case_stall,
    "stall-leader": lambda device, args: case_stall(device, args, leader=True),
    "bench": case_bench,
    "compile-error": case_compile_error,
    "fill": case_fill,
    "exit-busy": case_exit_busy,
    "after-exit": case_after_exit,
    "malformed": case_malformed,
    "stop-busy": case_stop_busy,
    "runtime-stop": case_runtime_stop,
    "excl": case_excl,
    "verify": case_verify,
    "store": case_store,
    "memcost": case_memcost,
}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("case", choices=sorted(CASES))
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--team", type=int, default=0, help="team size, 0 = all cores")
    p.add_argument("--host-depth", type=int, default=16)
    p.add_argument("--event-depth", type=int, default=128)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--delay-us", type=float, default=2000.0)
    p.add_argument("--core", type=int, default=37, help="the stalling core")
    p.add_argument("--deadline-us", type=int, default=50000)
    p.add_argument("--limit", type=float, default=20.0, help="worker wait limit, s")
    p.add_argument(
        "--idle-s", type=float, default=0.0, help="kernel idle limit, 0 = none"
    )
    p.add_argument("--pll", type=int, default=1750)
    p.add_argument("--image-first", action="store_true")
    p.add_argument("--batch", type=int, default=1, help="commands per store batch")
    p.add_argument("--settle-us", type=float, default=0.0, help="bench: fence to store")
    p.add_argument(
        "--preload", action="store_true", help="bench: store every command first"
    )
    p.add_argument("--gaps-us", type=float, nargs="+", default=[0.0], help="store")
    p.add_argument("--busy-us", type=float, nargs="+", default=[0.0], help="store")
    p.add_argument("--lines", type=int, nargs="+", default=[2], help="store")
    p.add_argument("--spread", type=int, nargs="+", default=[1], help="store")
    p.add_argument("--read", action="store_true", help="store: read before a burst")
    p.add_argument("--target", choices=sorted(TARGETS), default="lim", help="store")
    p.add_argument("--image", default=IMAGE, type=os.path.abspath, help="kernel image")
    args = p.parse_args()
    out = {"case": args.case, "argv": sys.argv[1:], "load_start": loadavg()}
    t_start = time.time()
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    device.disable_and_clear_program_cache()
    log(f"device open {time.time() - t_start:.1f}s")
    try:
        out.update(CASES[args.case](device, args))
    finally:
        ttnn.close_mesh_device(device)
    out["load_end"] = loadavg()
    out["wall_s"] = round(time.time() - t_start, 1)
    print(json.dumps(out, default=str), flush=True)


if __name__ == "__main__":
    main()
