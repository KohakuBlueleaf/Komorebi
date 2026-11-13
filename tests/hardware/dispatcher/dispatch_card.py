"""Dispatcher tests on a card: the kernel on X280 cluster 0, fed from the host,
the stand-in worker (kernels/worker.cpp) on every Tensix core. One case per run (or
the regression set), one JSON line at the end: python dispatch_card.py <case> [options].
"""

import argparse
import contextlib
import gc
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
sys.path.insert(0, BUILD)

sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))

import _km_dispatch_test as dk  # noqa: E402

import komorebi as km  # noqa: E402
from komorebi import abi  # noqa: E402
from komorebi.engines import stream as ks  # noqa: E402

KERNEL = os.path.join(HERE, "kernels", "worker.cpp")
INCLUDE = os.path.join(HERE, "..", "..", "..", "src", "komorebi_native", "include")
IMAGE = os.path.join(BUILD, "km-dispatch.bin")
VERIFY_IMAGE = os.path.join(BUILD, "km-dispatch-verify.bin")
IPI_ACK_DDR = 0x470020  # the runtime's IPI ack in the cluster's DRAM bank
STATE_BYTES = 0x8000  # KMD_STATE_BYTES: the kernel's state in its DRAM bank
# The longest host wait without a log line; run.sh keeps the watcher's stall above it.
HOST_WAIT_S = 10.0
DRAM_KEEP = []  # DRAM tensors a running image may use
CTL_CB, OUT_CB = 8, 9
CTL_BYTES = -(-abi.headers()["KM_CTL_BYTES"] // 64) * 64  # whole 64-byte lines
AICLK_HZ = 1.35e9  # Tensix clock of this card, for the wait limit only

# km_abi.h and km_dispatch.h
HANDLER_NOP, HANDLER_EXIT = 0, 255
HANDLER_PING, HANDLER_STOP, HANDLER_DEADLINE = 254, 253, 252
HANDLER_RUN, HANDLER_CANCEL = 251, 250
F_EXCL, F_QUEUE = 2, 8
EV_DONE, EV_TIMEOUT, EV_PONG, EV_REJECTED = 0, 1, 2, 3
EV_STREAM, EV_STREAM_END, EV_STREAM_ERROR = 4, 5, 6
REJECT_HANDLER, REJECT_FLAGS, REJECT_GROUP, REJECT_EXITED, REJECT_REGION = 1, 2, 3, 4, 5
REJECT_STREAMS, REJECT_UNKNOWN, REJECT_DEADLINE, REJECT_QUEUE = 6, 7, 8, 9
STREAMS = os.path.join(
    HERE, "..", "..", "..", "experiments", "stream_checks", "streams"
)
HANDLER_CHECK_UPLOAD = 246  # KMD_VERIFY_LINES images only
HANDLER_TRAP = 244  # KMD_VERIFY_LINES images only: core 0 loads from address a0
LIM_UPLOAD = abi.headers()["KMD_LIM_UPLOAD"]  # the upload region, from the layout
HOST_READY, READY_MARK = 4, 0x594452  # the ready block's first word and its mark
TRACE_SUBMIT, TRACE_FREE, TRACE_MARK = 1, 2, 0x54524345  # KMD_TRACE images only
TRACE_QUEUED = 3  # a submit from the job queue
STATE_READY, STATE_DONE, STATE_FAILED = 2, 3, 4
ERR_DEADLINE, ERR_STOPPED, ERR_STOP_BUSY, ERR_TRAP = 4, 7, 9, 11
LOAD_ACCESS_FAULT = 5  # mcause
R_NAMES = {
    0: "state",
    1: "error",
    2: "where",
    3: "slot",
    4: "seq",
    5: "req",
    6: "seen",
    7: "commands",
    8: "commands",
    9: "events",
    10: "busy",
    11: "max_busy",
    12: "ctl",
    21: "boot_ticks",
    24: "rejected",
    25: "held",
    26: "no_room",
    27: "checked",
    28: "bad_lines",
    29: "bad_first",
    30: "results",
    31: "result",
}
R_WIDE = {"ticks": 13, "idle_passes": 15, "full_passes": 17, "passes": 19, "last": 22}
# The core block after the result block's 32 words (KMD_H_*).
H_NAMES = {
    0: "marker0",
    1: "marker1",
    2: "engine",
    3: "engine_passes",
    4: "engine_commands",
    5: "engine_posts",
    6: "engine_events",
    7: "engine_runs",
    8: "lost",
    9: "source",
    13: "abandoned",
    14: "trap_cause",
    15: "trap_pc",
}
ENGINE_STOPPED = 3  # KMD_ENGINE_STOPPED
SOURCE_ENGINE = 1  # KMD_SRC_ENGINE: the stream engine on core 1

# kernels/dk_test.h
HANDLER_CHECK, HANDLER_DELAY, HANDLER_STALL, HANDLER_RESULT = 1, 2, 3, 4
F_RESULT = 4  # KM_F_RESULT
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
    "handler": 11,
    "state": 16,
}
S_GAVE, S_WHERE, S_SEQ = 16, 17, 18
WHERE = {
    1: "boot",
    2: "next",
    3: "done",
    4: "stall",
    5: "delay",
    6: "exit",
    7: "noc_read",
    8: "noc_write",
    9: "noc_atomic",
}
GAVE = {0: "none", 1: "stop", 2: "limit", 3: "noc", 4: "align"}

M32 = 0xFFFFFFFF


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


def dk_result(key, param):
    return dk_arg(key, param, 10)


def dk_mix(key):
    return dk_arg(key, 0x5EED, 0)


def verify_word(i, w):
    """km_dispatch.h kmd_verify_word, on Python ints."""
    x = ((i * 0x9E3779B1) ^ ((w + 1) * 0x85EBCA77)) & M32
    x ^= x >> 15
    return (x * 0x2C1B3C6D) & M32


def upload_words(seed, first, n):
    """kmd_verify_word(seed, w) for w = first .. first + n - 1, as uint32."""
    w = np.arange(first, first + n, dtype=np.uint64)
    x = np.uint64((seed * 0x9E3779B1) & M32) ^ (((w + 1) * 0x85EBCA77) & M32)
    x ^= x >> np.uint64(15)
    return ((x * 0x2C1B3C6D) & M32).astype(np.uint32)


def verify_command(i, cores):
    """The command of command i that a KMD_VERIFY_LINES kernel expects (a NOP)."""
    count = 1 + verify_word(i, 10) % cores
    first = verify_word(i, 11) % (cores - count + 1)
    words = [0] * 16
    words[2] = first | (count << 16)
    for j, at in enumerate([3, 5, 6, 7, 9, 10, 11, 13, 14, 15]):
        words[at] = verify_word(i, j)
    return words


def command(handler, first, count, key=0, param=0):
    """The 16 words of a command as the host passes them (seq, landing words and slot
    are filled on the way)."""
    words = [0] * 16
    words[1] = handler
    words[2] = first | (count << 16)
    args = [key & M32, param & M32] + [dk_arg(key, param, w) for w in range(2, 10)]
    for i, value in enumerate(args):
        words[[3, 5, 6, 7, 9, 10, 11, 13, 14, 15][i]] = value
    return words


def kernel_command(handler, a0=0):
    """A command to the dispatcher itself (KM_HANDLER_PING, _STOP, _DEADLINE)."""
    words = [0] * 16
    words[1] = handler
    words[3] = a0
    return words


def kind(status):
    """(kind, aux) of a status; a DONE or TIMEOUT aux without its seq bits."""
    aux = status >> 8
    return status & 0xFF, aux & 0xFF if status & 0xFF in (EV_DONE, EV_TIMEOUT) else aux


def seq_low(status):
    """The low 16 bits of the command's sequence number in a DONE or TIMEOUT event."""
    return status >> 16 & 0xFFFF


# ---------------------------------------------------------------------------------
# Device, programs, the run frame


def program(session, plan, table, out, limit_s, defines):
    """The stand-in worker on every core, one data-movement kernel."""
    gx, gy = plan.grid_x, plan.grid_y
    cores = [(0, 0, gx - 1, gy - 1)]
    limit = max(1, int(limit_s * AICLK_HZ) >> 20)
    args = [
        (x, y, list(plan.worker_args(x, y, table, out, limit)))
        for y in range(gy)
        for x in range(gx)
    ]
    launch = session.launch()
    include = [os.path.abspath(INCLUDE)]
    launch.kernel(KERNEL, cores, 0, [], list(defines), include, args)
    launch.cb(cores, CTL_CB, "uint32", CTL_BYTES, CTL_BYTES)
    launch.cb(cores, OUT_CB, "uint32", DK_WORDS * 4, DK_WORDS * 4)
    return launch


def int_tensor(session, array):
    return session.tensor(np.asarray(array, dtype=np.uint32), "uint32")


def dram_area(session):
    """A bank address with STATE_BYTES that are ours in every DRAM bank: a DRAM tensor
    of 1024 pages of 4 KiB, interleaved, so each bank holds consecutive pages."""
    pages = int_tensor(session, np.zeros((1024, 1024), dtype=np.uint32))
    DRAM_KEEP.append(pages)
    ddr = pages.address
    if ddr <= IPI_ACK_DDR < ddr + STATE_BYTES:
        raise SystemExit("the buffer covers the runtime's IPI ack word")
    return ddr


def result_block(words):
    out = {name: words[i] for i, name in R_NAMES.items()}
    out.update({name: words[32 + i] for i, name in H_NAMES.items()})
    for name, lo in R_WIDE.items():
        out[name] = words[lo] | (words[lo + 1] << 32)
    return out


def with_kernel(
    session,
    args,
    body,
    depth=1,
    defines=(),
    preload=None,
    stop=False,
    info=None,
    after=None,
):
    """Grid, then kernel (after preload(plan)); body(plan); the kernel's end (with stop,
    the runtime stops it), then after(plan, fin); X280 runtime closed, counters read.
    (body, finish, cnt). `info` gets "started" and "aborted" if the kernel was stopped.
    """
    info = {} if info is None else info
    info["started"] = False
    runtime = dk.Runtime(session.device.mesh_capsule(), pll_mhz=args.pll)
    log(f"X280 clock set: {args.pll} MHz")
    plan = runtime.plan()
    gx, gy = plan.grid_x, plan.grid_y
    log(f"plan: grid {gx}x{gy} panels {plan.panels}")
    table = int_tensor(session, np.array([plan.core_table()], dtype=np.uint32))
    state = dram_area(session)
    plan.prepare(
        depth, args.host_depth, args.event_depth, 30.0, 1.0, args.idle_s, state
    )
    out = int_tensor(session, np.zeros((gx * gy, DK_WORDS), dtype=np.uint32))
    prog = program(session, plan, table.address, out.address, args.limit, defines)
    body_out, fin = None, None
    if preload is not None:
        preload(plan)
    try:
        if args.image_first:
            log("starting the kernel")
            plan.start(args.image)
            info["started"] = True
            log("kernel started")
        log("launching the grid")
        prog.run()
        log("grid launched")
        if not args.image_first:
            log("starting the kernel")
            plan.start(args.image)
            info["started"] = True
            log("kernel started")
        log("waiting for READY")
        state = plan.wait_ready(HOST_WAIT_S)
        log(f"kernel state {state}")
        body_out = body(plan) if state == STATE_READY else {"ready_state": state}
        log("finishing the kernel")
        fin = plan.finish(HOST_WAIT_S, wait=not stop)
        log(f"kernel finished: result {fin['result']} cleaned {fin['cleaned']}")
        if after is not None:
            log("reading after the kernel")
            after(plan, fin)
            log("read after the kernel")
    finally:
        if plan.running():
            log("stopping the kernel")
            info["aborted"] = plan.abort()
            log(f"stopping the kernel: cleaned {info['aborted']}")
        del plan, runtime
        gc.collect()
        log("x280 runtime closed")
    log("reading the counters")
    cnt = out.read().astype(np.int64)
    log("counters read back")
    fin["block"] = result_block(fin.pop("words"))
    return body_out, fin, cnt


def exit_and_stop(plan):
    """EXIT to the whole grid, then STOP; returns the events up to EXIT's."""
    n = plan.grid_x * plan.grid_y
    i = plan.submit(command(HANDLER_EXIT, 0, n))
    events, by = [], time.time() + 10.0
    while not any(e[1] == i for e in events) and time.time() < by:
        with contextlib.suppress(RuntimeError):
            events += plan.take(plan.seen + 1, 1.0)
    plan.submit(kernel_command(HANDLER_STOP))
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


def check_groups(cnt, n_cores, sent):
    """Compare every core's run count, key sum and key mix with the commands sent
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


def case_ping(session, args, submit_one=False):
    n_cores = 0

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        out = {"tag": plan.tag}
        out["ping_req"] = plan.submit(kernel_command(HANDLER_PING))
        out["pong"] = plan.take(out["ping_req"], 5.0)
        if submit_one:
            out["submit_req"] = plan.submit(command(HANDLER_CHECK, 0, n_cores, 77, 5))
            out["submit"] = plan.take(out["submit_req"], 5.0)
        out["exit"] = exit_and_stop(plan)
        return out

    out, fin, cnt = with_kernel(session, args, body)
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
        and (not submit_one or out["submit"][0][1] == out["submit_req"])
    )
    return out


def case_groups(session, args):
    """args.n commands with random groups (CHECK), depth 16."""
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
            recs.append(command(HANDLER_CHECK, first, count, key, param))
            sent.append((first, count, key))
        reqs = list(range(plan.next, plan.next + args.n))
        first = plan.seen + 1
        events, times = plan.stream(np.array(recs, dtype=np.uint32), 10.0, 64)
        out = {"tag": plan.tag, "times": times}
        out["check"] = check_events(events, reqs, first)
        out["exit"] = exit_and_stop(plan)
        return out

    out, fin, cnt = with_kernel(session, args, body, depth=16)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    out["group_mismatch"], out["first_mismatch"] = check_groups(cnt, n_cores, sent)
    out["ok"] = bool(
        fin["result_ready"]
        and fin["block"]["error"] == 0
        and all(out["check"].values())
        and out["group_mismatch"] == 0
        and out["cores"]["cores_finished"] == n_cores
        and out["cores"]["arg_errors"] == 0
        and out["cores"]["hdr_errors"] == 0
    )
    return out


def case_disjoint(session, args):
    """Group A = first half, one DELAY of args.delay_us; group B = second half, then
    args.n CHECK commands. B's events must arrive before A's."""
    n_cores, sent = 0, []
    cycles = int(args.delay_us * 1e-6 * AICLK_HZ)

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        half = n_cores // 2
        a = plan.submit(command(HANDLER_DELAY, 0, half, 1, cycles))
        sent.append((0, half, 1))
        last = 0
        for k in range(args.n):
            last = plan.submit(command(HANDLER_CHECK, half, half, 2 + k, k))
            sent.append((half, half, 2 + k))
        events = plan.take(last, 10.0)
        out = {"tag": plan.tag, "events": events, "a_req": a}
        out["exit"] = exit_and_stop(plan)
        return out

    out, fin, cnt = with_kernel(session, args, body, depth=16)
    events = out["events"]
    order = [e[1] for e in events]
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    out["order"] = order
    out["group_mismatch"], out["first_mismatch"] = check_groups(cnt, n_cores, sent)
    a_req = out["a_req"]
    a_ticks = [e[3] for e in events if e[1] == a_req]
    out["ok"] = bool(
        fin["result_ready"]
        and fin["block"]["error"] == 0
        and out["group_mismatch"] == 0
        and out["cores"]["arg_errors"] == 0
        and len(events) == args.n + 1
        and order[-1] == a_req
        and fin["block"]["max_busy"] >= 2
        and a_ticks
        and a_ticks[0] > max(e[3] for e in events if e[1] != a_req)
        and out["cores"]["cores_finished"] == n_cores
    )
    return out


def case_stall(session, args, leader=False):
    """A group of every core whose member --core (with leader, the group's leader, core
    0) waits for the stop flag, under KM_HANDLER_DEADLINE: KM_EV_TIMEOUT, the stop flag.
    """
    n_cores = 0
    stalled = 0 if leader else args.core

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        out = {"tag": plan.tag, "stalled": stalled}
        i = plan.submit(command(HANDLER_CHECK, 0, n_cores, 3, 4))
        out["before"] = plan.take(i, 5.0)
        plan.submit(kernel_command(HANDLER_DEADLINE, args.deadline_us))
        out["stall_req"] = plan.submit(command(HANDLER_STALL, 0, n_cores, 9, stalled))
        out["timeout"] = plan.take(out["stall_req"], 10.0)
        out["host_state"] = plan.host_word(0)
        return out

    out, fin, cnt = with_kernel(session, args, body, depth=4)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    state = cnt[:n_cores, C["state"] : C["state"] + 19]
    blk = fin["block"]
    # Cores that did not report the timed-out command: never took it, or stopped other
    # than waiting for the next command or, as its leader, for the members.
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
        and seq_low(ev[-1][2]) == blk["seq"] & 0xFFFF
        and ev[-1][1] == out["stall_req"]
        and blk["error"] == ERR_DEADLINE
        and blk["req"] == out["stall_req"]
        and ev[-1][3] >= args.deadline_us * 50
        and out["cores"]["cores_finished"] == n_cores
        and out["cores"]["gave"] == {"stop": n_cores}
        and [c["core"] for c in out["not_reported"]] == [stalled]
    )
    return out


def case_stall_late(session, args):
    """Depth 4: on the last core a command that waits for the stop flag under a deadline
    ten times --deadline-us (slot 0, never freed), then on the other cores a command
    whose member --core waits (slot 1) and two behind it: KM_EV_TIMEOUT names the
    stalled command (its deadline passed first) and its slot 1, not the lower slot 0."""
    n_cores = 0

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        out = {"tag": plan.tag}
        rest = n_cores - 1
        plan.submit(kernel_command(HANDLER_DEADLINE, 10 * args.deadline_us))
        out["early_req"] = plan.submit(command(HANDLER_STALL, rest, 1, 3, rest))
        plan.submit(kernel_command(HANDLER_DEADLINE, args.deadline_us))
        recs = [command(HANDLER_STALL, 0, rest, 9, args.core)]
        recs += [command(HANDLER_CHECK, 0, rest, 5 + k, k) for k in range(2)]
        out["stall_req"] = plan.next
        out["events"], _ = plan.stream(np.array(recs, dtype=np.uint32), HOST_WAIT_S, 8)
        return out

    out, fin, cnt = with_kernel(session, args, body, depth=4)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    blk, ev = fin["block"], out["events"]
    timeouts = [e for e in ev if kind(e[2])[0] == EV_TIMEOUT]
    out["ok"] = bool(
        len(timeouts) == 1
        and timeouts[0][1] == out["stall_req"]
        and kind(timeouts[0][2]) == (EV_TIMEOUT, 1)
        and blk["error"] == ERR_DEADLINE
        and blk["req"] == out["stall_req"]
        and blk["slot"] == 1
        and blk["busy"] == 0b1111
        and out["cores"]["cores_finished"] == n_cores
    )
    return out


def case_compile_error(session, args):
    """The worker fails to compile: the exception is that build failure, and the X280
    image was never started (or, with --image-first, was stopped)."""
    info = {}
    try:
        with_kernel(
            session,
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
    """The kernel ended with `error` and stopped the engine, every core reached its end,
    no argument errors."""
    return (
        fin["result_ready"]
        and fin["block"]["error"] == error
        and fin["block"]["engine"] == ENGINE_STOPPED
        and cores["cores_finished"] == n_cores
        and cores["arg_errors"] == 0
        and cores["hdr_errors"] == 0
    )


def case_fill(session, args):
    """All 16 slots busy: 16 DELAY commands of --delay-us on 16 disjoint groups, then
    --n CHECK commands with random groups that wait for slots; then EXIT."""
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
            reqs.append(
                plan.submit(command(HANDLER_DELAY, j * width, width, 1 + j, cycles))
            )
            sent.append((j * width, width, 1 + j))
        for k in range(args.n):
            count = int(rng.integers(1, n_cores + 1))
            first = int(rng.integers(0, n_cores - count + 1))
            reqs.append(plan.submit(command(HANDLER_CHECK, first, count, 100 + k, k)))
            sent.append((first, count, 100 + k))
        events = plan.take(s0 + len(reqs), 10.0)
        out = {"tag": plan.tag, "check": check_events(events, reqs, s0 + 1)}
        out["exit"] = exit_and_stop(plan)
        return out

    out, fin, cnt = with_kernel(session, args, body, depth=16)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    out["group_mismatch"], out["first_mismatch"] = check_groups(cnt, n_cores, sent)
    blk = fin["block"]
    out["ok"] = bool(
        finished_clean(fin, n_cores, out["cores"])
        and all(out["check"].values())
        and out["group_mismatch"] == 0
        and blk["max_busy"] == 16
        and blk["full_passes"] > 0
    )
    return out


def case_exit_busy(session, args):
    """EXIT sent while a DELAY of --delay-us on half the grid is in flight: the kernel
    holds EXIT until the DELAY completes, then sends it to every core."""
    n_cores = 0
    cycles = int(args.delay_us * 1e-6 * AICLK_HZ)

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        out = {"tag": plan.tag}
        s0 = plan.seen
        out["delay_req"] = plan.submit(
            command(HANDLER_DELAY, 0, n_cores // 2, 1, cycles)
        )
        out["exit_req"] = plan.submit(command(HANDLER_EXIT, 0, 1))
        out["events"] = plan.take(s0 + 2, 10.0)
        plan.submit(kernel_command(HANDLER_STOP))
        return out

    out, fin, cnt = with_kernel(session, args, body, depth=16)
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


def case_after_exit(session, args):
    """A CHECK command after a completed EXIT is rejected (KMD_REJECT_EXITED); a PING
    still answers; STOP ends the session without an error."""
    n_cores = 0

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        out = {"tag": plan.tag}
        plan.submit(command(HANDLER_EXIT, 0, n_cores))
        out["exit"] = plan.take(plan.seen + 1, 10.0)
        s0 = plan.seen
        out["late_req"] = plan.submit(command(HANDLER_CHECK, 0, n_cores, 5, 5))
        out["ping_req"] = plan.submit(kernel_command(HANDLER_PING))
        out["events"] = plan.take(s0 + 2, 5.0)
        plan.submit(kernel_command(HANDLER_STOP))
        return out

    out, fin, cnt = with_kernel(session, args, body)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    ev = out["events"]
    out["ok"] = bool(
        finished_clean(fin, n_cores, out["cores"])
        and [(e[1], kind(e[2])) for e in ev]
        == [
            (out["late_req"], (EV_REJECTED, REJECT_EXITED)),
            (out["ping_req"], (EV_PONG, 0)),
        ]
        and fin["block"]["rejected"] == 1
        and out["cores"]["gave"] == {"none": n_cores}
        and int(cnt[:n_cores, C["run"]].sum()) == 0
    )
    return out


def case_malformed(session, args):
    """Four bad commands (handler 240, flag bit 0, count 0, a group past the last core)
    are rejected with their reasons; a good CHECK after them runs; then EXIT."""
    n_cores = 0

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        bad_flag = command(HANDLER_CHECK, 0, n_cores, 7, 7)
        bad_flag[1] |= 1 << 8
        bad = [
            (command(240, 0, n_cores), REJECT_HANDLER),
            (bad_flag, REJECT_FLAGS),
            (command(HANDLER_CHECK, 0, 0, 7, 7), REJECT_GROUP),
            (command(HANDLER_CHECK, n_cores - 1, 2, 7, 7), REJECT_GROUP),
        ]
        first_event = plan.seen + 1
        want = [(plan.submit(rec), (EV_REJECTED, why)) for rec, why in bad]
        good = plan.submit(command(HANDLER_CHECK, 0, n_cores, 11, 3))
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

    out, fin, cnt = with_kernel(session, args, body)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    out["group_mismatch"], _ = check_groups(cnt, n_cores, [(0, n_cores, 11)])
    out["ok"] = bool(
        finished_clean(fin, n_cores, out["cores"])
        and out["indices_ok"]
        and out["kinds_ok"]
        and out["group_mismatch"] == 0
        and fin["block"]["rejected"] == 4
    )
    return out


def case_stop_busy(session, args):
    """STOP while a DELAY of --delay-us on every core is in flight: the kernel ends with
    KMD_ERR_STOP_BUSY and the stop flag ends every core's wait."""
    n_cores = 0
    cycles = int(args.delay_us * 1e-6 * AICLK_HZ)

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        out = {"tag": plan.tag}
        out["delay_req"] = plan.submit(command(HANDLER_DELAY, 0, n_cores, 1, cycles))
        out["stop_req"] = plan.submit(kernel_command(HANDLER_STOP))
        return out

    out, fin, cnt = with_kernel(session, args, body, depth=4)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    blk = fin["block"]
    out["ok"] = bool(
        finished_clean(fin, n_cores, out["cores"], ERR_STOP_BUSY)
        and blk["req"] == out["stop_req"]
        and blk["busy"] != 0
        and blk["abandoned"] == 1
        and out["cores"]["gave"] == {"stop": n_cores}
    )
    return out


def case_runtime_stop(session, args):
    """The runtime stops the kernel while a DELAY of 0.5 s on every core is in flight:
    KMD_ERR_STOPPED, and the stop flag ends every core's wait."""
    n_cores = 0
    cycles = int(0.5 * AICLK_HZ)

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        out = {"tag": plan.tag}
        out["delay_req"] = plan.submit(command(HANDLER_DELAY, 0, n_cores, 1, cycles))
        by = time.monotonic() + 5.0
        while plan.host_word(1) < out["delay_req"] and time.monotonic() < by:
            pass
        out["taken"] = plan.host_word(1)
        return out

    out, fin, cnt = with_kernel(session, args, body, depth=4, stop=True)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    blk = fin["block"]
    out["ok"] = bool(
        fin["cleaned"]
        and finished_clean(fin, n_cores, out["cores"], ERR_STOPPED)
        and blk["state"] == STATE_FAILED
        and blk["busy"] != 0
        and blk["abandoned"] == 1
        and out["cores"]["gave"] == {"stop": n_cores}
    )
    return out


def case_trap(session, args):
    """The KMD_VERIFY_LINES image: a DELAY of 0.5 s on every core, then KMD_HANDLER_TRAP
    loading from address 8. Core 0 traps; the kernel finishes from LIM with the trap,
    every core gives up on the stop flag, the runtime restores the cluster, and a
    second kernel answers a PING."""
    if args.image == IMAGE:
        args.image = VERIFY_IMAGE
    n_cores = 0
    cycles = int(0.5 * AICLK_HZ)

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        out = {"tag": plan.tag}
        out["delay_req"] = plan.submit(command(HANDLER_DELAY, 0, n_cores, 1, cycles))
        out["trap_req"] = plan.submit(kernel_command(HANDLER_TRAP, 8))
        by = time.monotonic() + 5.0
        while plan.host_word(0) < STATE_DONE and time.monotonic() < by:
            pass
        out["host"] = [plan.host_word(0), plan.host_word(3)]
        return out

    out, fin, cnt = with_kernel(session, args, body, depth=4)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    blk = fin["block"]
    out["after"] = case_ping(session, args)
    out["ok"] = bool(
        fin["cleaned"]
        and finished_clean(fin, n_cores, out["cores"], ERR_TRAP)
        and fin["result"] == ERR_TRAP ^ (1 << 64) - 1
        and out["host"] == [STATE_FAILED, ERR_TRAP]
        and blk["state"] == STATE_FAILED
        and [blk["slot"], blk["req"], blk["seen"]] == [LOAD_ACCESS_FAULT, 0, 8]
        and out["cores"]["gave"] == {"stop": n_cores}
        and out["after"]["ok"]
    )
    return out


def case_excl(session, args):
    """A: DELAY of --delay-us on the first half; C: DELAY twice as long on the second
    half but one core; B: CHECK with KM_F_EXCL on the two cores around the middle,
    overlapping A only. B waits for A, not for C."""
    n_cores = 0

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        half = n_cores // 2
        cycles = int(args.delay_us * 1e-6 * AICLK_HZ)
        b_rec = command(HANDLER_CHECK, half - 1, 2, 2, 2)
        b_rec[1] |= F_EXCL << 8
        out = {"tag": plan.tag}
        s0 = plan.seen
        out["a"] = plan.submit(command(HANDLER_DELAY, 0, half, 1, cycles))
        out["c"] = plan.submit(
            command(HANDLER_DELAY, half + 1, half - 1, 3, 2 * cycles)
        )
        out["b"] = plan.submit(b_rec)
        out["events"] = plan.take(s0 + 3, 10.0)
        out["exit"] = exit_and_stop(plan)
        return out

    out, fin, cnt = with_kernel(session, args, body, depth=16)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    half = n_cores // 2
    sent = [(0, half, 1), (half - 1, 2, 2), (half + 1, half - 1, 3)]
    out["group_mismatch"], out["first_mismatch"] = check_groups(cnt, n_cores, sent)
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
        and out["group_mismatch"] == 0
    )
    return out


def verify_kernel_command(i, cores):
    """Command i of the verify case and the event it must produce, (kind, aux, ticks>0):
    every 7th a PING, every 11th a command of handler 245 (rejected), else a NOP command
    (its DONE names its first core)."""
    if i % 7 == 3:
        return kernel_command(HANDLER_PING), (EV_PONG, 0, False)
    if i % 11 == 5:
        return command(245, 0, cores), (EV_REJECTED, REJECT_HANDLER, False)
    words = verify_command(i, cores)
    return words, (EV_DONE, words[2] & 0xFFFF, True)


def case_verify(session, args):
    """The KMD_VERIFY_LINES image under load: --n commands, mostly NOP commands whose
    words the kernel checks, with PINGs and rejected commands; each event is checked
    against what its command must produce."""
    if args.image == IMAGE:
        args.image = VERIFY_IMAGE
    n_cores, sent, n_commands, n_rejected = 0, [], 0, 0

    def body(plan):
        nonlocal n_cores, n_commands, n_rejected
        n_cores = plan.grid_x * plan.grid_y
        first_req, first_event = plan.next, plan.seen + 1
        reqs = list(range(first_req, first_req + args.n))
        made = {i: verify_kernel_command(i, n_cores) for i in reqs}
        recs = np.array([made[i][0] for i in reqs], dtype=np.uint32)
        for i in reqs:
            if made[i][1][0] == EV_DONE:
                r = made[i][0]
                sent.append((r[2] & 0xFFFF, r[2] >> 16, r[3]))
        n_commands = len(sent)
        n_rejected = sum(1 for i in reqs if made[i][1][0] == EV_REJECTED)
        events, times = plan.stream(recs, HOST_WAIT_S, args.batch, args.settle_us)
        out = {"tag": plan.tag, "times": times, "commands": n_commands}
        out["check"] = check_events(
            events, reqs, first_event, kinds=(EV_DONE, EV_PONG, EV_REJECTED)
        )
        limit = 50 * 1_000_000  # the default command deadline, 1 s in ticks
        wrong, by_index = [], {e[0]: e for e in events}
        for e in events:
            k, aux, timed = made.get(e[1], (None, (None, None, None)))[1]
            ticks_ok = (0 < e[3] <= limit) if timed else e[3] == 0
            if kind(e[2]) != (k, aux) or not ticks_ok:
                # with the event that used the same ring slot before it
                before = by_index.get(e[0] - plan.event_depth)
                wrong.append([list(e), list(before) if before else None])
        out["event_errors"], out["first_event_errors"] = len(wrong), wrong[:8]
        out["exit"] = exit_and_stop(plan)
        return out

    out, fin, cnt = with_kernel(session, args, body, depth=args.depth)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    out["group_mismatch"], out["first_mismatch"] = check_groups(cnt, n_cores, sent)
    blk = fin["block"]
    out["ok"] = bool(
        finished_clean(fin, n_cores, out["cores"])
        and all(out["check"].values())
        and out["event_errors"] == 0
        and out["group_mismatch"] == 0
        and blk["checked"] == n_commands
        and blk["bad_lines"] == 0
        and blk["rejected"] == n_rejected
    )
    return out


def case_result(session, args):
    """KM_F_RESULT under load: --n DK_HANDLER_RESULT commands (groups of --group cores,
    1 if 0), a million at a time; every event must carry dk_result of its command, every
    core must have run exactly its commands."""
    count = args.group or 1
    n_cores = 0

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        out = {"tag": plan.tag, "events": 0, "wrong": 0, "first_wrong": []}
        total_s, done = 0.0, 0
        while done < args.n:
            k = min(args.n - done, 1_000_000)
            times, wrong = plan.result_run(k, count, args.seed, 30.0, args.batch)
            out["events"] += int(times["events"])
            out["wrong"] += int(times["wrong"])
            out["first_wrong"] += [list(e) for e in wrong][
                : 8 - len(out["first_wrong"])
            ]
            total_s += times["total_s"]
            done += k
            log(f"result: {done} of {args.n}, wrong {out['wrong']}")
        out["us_per_command"] = round(total_s / args.n * 1e6, 4)
        runs, sums, mixes = plan.expected()
        out["expect"] = [
            np.array(v[:n_cores], dtype=np.int64) for v in (runs, sums, mixes)
        ]
        out["exit"] = exit_and_stop(plan)
        return out

    out, fin, cnt = with_kernel(session, args, body, depth=args.depth)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    runs, sums, mixes = out.pop("expect")
    got = cnt[:n_cores]
    differ = (
        (got[:, C["run"]] != runs)
        | (got[:, C["key_sum"]] != sums)
        | (got[:, C["key_mix"]] != mixes)
    )
    out["cores_differ"] = int(differ.sum())
    blk = fin["block"]
    out["ok"] = bool(
        finished_clean(fin, n_cores, out["cores"])
        and out["events"] == args.n
        and out["wrong"] == 0
        and out["cores_differ"] == 0
        and blk["results"] == args.n
    )
    return out


def check_upload(offset, size, seed):
    """KMD_HANDLER_CHECK_UPLOAD: a0 = offset, a1 = bytes, a2 = seed."""
    words = kernel_command(HANDLER_CHECK_UPLOAD, offset)
    words[5], words[6] = size, seed
    return words


def case_upload(session, args):
    """The upload region (KMD_VERIFY_LINES image): the ready block names it; a pattern
    of whole lines written through the host window reads back exactly on the kernel
    side, a changed word is found where it is, ranges outside the region are rejected.
    """
    if args.image == IMAGE:
        args.image = VERIFY_IMAGE

    def body(plan):
        n = plan.upload_bytes
        ready = [plan.host_word(HOST_READY + w) for w in range(4)]
        out = {"tag": plan.tag, "ready": ready, "upload_bytes": n}
        out["ready_ok"] = ready == [
            READY_MARK,
            plan.lim_base + LIM_UPLOAD,
            n,
            LIM_UPLOAD,
        ]
        line = 64 * (n // 128)  # a line in the middle

        def run(name, offset, size, seed, want):
            t0 = time.perf_counter()
            i = plan.submit(check_upload(offset, size, seed))
            ev = [e for e in plan.take(i, 10.0) if e[1] == i]
            got = [kind(ev[0][2]), ev[0][3]] if ev else None
            log(f"upload {name}: {got}")
            checks.append(
                {
                    "name": name,
                    "got": got,
                    "ok": got == want,
                    "ms": round((time.perf_counter() - t0) * 1e3, 3),
                }
            )

        checks = []
        uploads = []
        none = M32
        for seed in (args.seed, args.seed + 1):
            data = upload_words(seed, 0, n // 4)
            t0 = time.perf_counter()
            plan.upload(0, data.view(np.uint8))
            uploads.append(round(time.perf_counter() - t0, 6))
            run(f"whole {seed}", 0, n, seed, [(EV_PONG, 0), none])
        seed = args.seed + 1
        run("first line", 0, 64, seed, [(EV_PONG, 0), none])
        run("last line", n - 64, 64, seed, [(EV_PONG, 0), none])
        run("middle 4 KiB", line, 4096, seed, [(EV_PONG, 0), none])
        run("other seed", line, 64, seed + 1, [(EV_PONG, 16), 0])
        bad = upload_words(seed, line // 4, 16)
        bad[5] ^= 1
        plan.upload(line, bad.view(np.uint8))
        run("one word", 0, n, seed, [(EV_PONG, 1), line // 4 + 5])
        plan.upload(line, upload_words(seed, line // 4, 16).view(np.uint8))
        run("restored", 0, n, seed, [(EV_PONG, 0), none])
        run("past the end", n - 64, 128, seed, [(EV_REJECTED, REJECT_REGION), 0])
        run("half line", 32, 64, seed, [(EV_REJECTED, REJECT_REGION), 0])
        out["checks"] = checks
        out["upload_s"] = uploads
        out["upload_mb_s"] = round(n / min(uploads) / 1e6, 1)
        out["exit"] = exit_and_stop(plan)
        return out

    out, fin, cnt = with_kernel(session, args, body, depth=args.depth)
    out["finish"] = fin
    n_cores = cnt.shape[0]
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    out["ok"] = bool(
        finished_clean(fin, n_cores, out["cores"])
        and out["ready_ok"]
        and all(c["ok"] for c in out["checks"])
        and fin["block"]["rejected"] == 2
    )
    return out


def alloc(busy, count, n_cores):
    """The lowest first whose cores [first, first + count) overlap no busy group
    ((first, count) pairs), or None: the kernel's group allocator."""
    for first in range(n_cores - count + 1):
        if all(first + count <= f or f + c <= first for f, c in busy):
            return first
    return None


def read_trace(plan):
    """The KMD_TRACE image's trace from the upload region: (header, entries), each
    entry (kind, slot, req, first, count, low 32 bits of the timer, source)."""
    head = np.frombuffer(plan.upload_read(0, 64), dtype=np.uint32)
    n = int(head[1]) if head[0] == TRACE_MARK else 0
    raw = np.frombuffer(plan.upload_read(64, 16 * n), dtype=np.uint32).reshape(n, 4)
    entries = [
        (
            int(w & 0xFF),
            int(w >> 8 & 0xFF),
            int(r),
            int(t & 0xFFFF),
            int(t >> 16),
            int(p),
            int(w >> 16 & 0xFF),
        )
        for w, r, t, p in raw
    ]
    return [int(v) for v in head[:3]], entries


def replay_trace(entries, sent, n_cores):
    """Replays the slots' submits and frees against the commands sent ({req: (kind,
    first, count)}, kind "queue", "excl" or "plain"); returns the faults found, the
    groups each command ran on and the most queued commands in flight at once."""
    busy, groups, faults, most = {}, {}, [], 0
    for kind_, slot, req, first, count, p, _ in entries:
        if kind_ == TRACE_FREE:
            if busy.get(slot, (None,))[0] != req:
                faults.append(["free of a slot not busy with it", slot, req, p])
            busy.pop(slot, None)
            continue
        if kind_ not in (TRACE_SUBMIT, TRACE_QUEUED) or slot in busy or req in groups:
            faults.append(["bad submit", kind_, slot, req, p])
            continue
        what, want_first, want_count = sent.get(req, ("exit", 0, n_cores))
        others = [(f, c) for _, f, c in busy.values()]
        if what == "queue":
            ok = (
                kind_ == TRACE_QUEUED
                and count == want_count
                and first == alloc(others, count, n_cores)
            )
        elif what == "excl":
            ok = (first, count) == (want_first, want_count) and all(
                first + count <= f or f + c <= first for f, c in others
            )
        else:
            ok = (first, count) == (want_first, want_count)
        if not ok:
            faults.append([what, req, first, count, others, p])
        busy[slot] = (req, first, count)
        groups[req] = (first, count)
        queued = [r for r, _, _ in busy.values() if sent.get(r, ("",))[0] == "queue"]
        most = max(most, len(queued))
    if busy:
        faults.append(["still busy", sorted(busy.items())])
    return faults, groups, most


def case_queue(session, args):
    """KM_F_QUEUE commands from the host among plain and KM_F_EXCL ones (DELAY of up to
    --delay-us, depth 16, KMD_TRACE image): the allocator gives each queued command the
    lowest free group, which its DONE names; EXCL ones overlap no group in flight, the
    grid ran the traced groups; a queued count of 0 or above the grid is rejected.
    """
    if args.image == IMAGE:
        args.image = VERIFY_IMAGE
    rng = np.random.default_rng(args.seed)
    n_cores, sent, trace = 0, {}, {}

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        out = {"tag": plan.tag}
        cycles = int(args.delay_us * 1e-6 * AICLK_HZ)
        recs, rejected = [], []
        req = plan.next
        for i in range(args.n + 2):
            key = int(rng.integers(1, 1 << 32))
            delay = int(rng.integers(0, cycles + 1))
            pick = rng.random()
            if i < 2:
                count = 0 if i == 0 else n_cores + 1
                rec = command(HANDLER_DELAY, 0, 0, key, delay)
                rec[2] = int(rng.integers(0, 1 << 16)) | count << 16
                rec[1] |= F_QUEUE << 8
                rejected.append(req + i)
            elif pick < 0.7:
                count = int(rng.choice([1, 2, 3, 4, 6, 8, 12, 20, 40, 60, n_cores]))
                rec = command(HANDLER_DELAY, 0, count, key, delay)
                rec[2] = int(rng.integers(0, 1 << 16)) | count << 16
                rec[1] |= F_QUEUE << 8
                sent[req + i] = ("queue", None, count)
            else:
                count = int(rng.integers(1, 17))
                first = int(rng.integers(0, n_cores - count + 1))
                rec = command(HANDLER_DELAY, first, count, key, delay)
                what = "excl" if pick < 0.85 else "plain"
                rec[1] |= (F_EXCL << 8) if what == "excl" else 0
                sent[req + i] = (what, first, count)
            recs.append(rec)
        keys = {req + i: r[3] for i, r in enumerate(recs)}
        first_event = plan.seen + 1
        reqs = list(range(req, req + len(recs)))
        events, times = plan.stream(
            np.array(recs, dtype=np.uint32), HOST_WAIT_S, args.batch
        )
        out["times"] = times
        out["check"] = check_events(
            events, reqs, first_event, kinds=(EV_DONE, EV_REJECTED)
        )
        # A queued command may find the queue full: rejected, not run (the host would
        # retry it).
        full = [e[1] for e in events if kind(e[2]) == (EV_REJECTED, REJECT_QUEUE)]
        full_ok = all(sent.get(r, ("",))[0] == "queue" for r in full)
        for r in full:
            sent.pop(r, None)
        out["queue_full"] = len(full)
        out["rejected_ok"] = (
            full_ok
            and sorted(
                e[1] for e in events if kind(e[2]) == (EV_REJECTED, REJECT_GROUP)
            )
            == rejected
            and all(
                kind(e[2])[0] == EV_DONE
                for e in events
                if e[1] not in rejected and e[1] not in full
            )
        )
        out["keys"] = keys
        out["done_first"] = {
            e[1]: kind(e[2])[1] for e in events if kind(e[2])[0] == EV_DONE
        }
        out["exit"] = exit_and_stop(plan)
        return out

    def after(plan, fin):
        trace["head"], trace["entries"] = read_trace(plan)

    out, fin, cnt = with_kernel(session, args, body, depth=args.depth, after=after)
    keys, done_first = out.pop("keys"), out.pop("done_first")
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    head, entries = trace["head"], trace["entries"]
    faults, groups, most = replay_trace(entries, sent, n_cores)
    out["trace"] = {"head": head, "entries": len(entries), "most_in_flight": most}
    out["faults"], out["first_faults"] = len(faults), faults[:8]
    ran = [(groups[r][0], groups[r][1], keys[r]) for r in sent if r in groups]
    out["group_mismatch"], out["first_mismatch"] = check_groups(cnt, n_cores, ran)
    # Every DONE names the first core its command ran on (the allocator's, if queued).
    out["done_first_ok"] = all(
        done_first.get(r) == groups[r][0] for r in sent if r in groups
    )
    out["queued_firsts"] = sorted(
        {groups[r][0] for r, v in sent.items() if v[0] == "queue" and r in groups}
    )[:16]
    out["ok"] = bool(
        finished_clean(fin, n_cores, out["cores"])
        and all(out["check"].values())
        and out["rejected_ok"]
        and head[0] == TRACE_MARK
        and head[2] == 0
        and len(ran) == len(sent)
        and len(entries) == 2 * (len(sent) + 1)
        and out["faults"] == 0
        and out["group_mismatch"] == 0
        and out["done_first_ok"]
        and most > 1
        and fin["block"]["rejected"] == 2 + out["queue_full"]
    )
    return out


def run_command(offset, size):
    """KM_HANDLER_RUN: a0 = offset of the upload in the region, a1 = its bytes."""
    words = kernel_command(HANDLER_RUN, offset)
    words[5] = size
    return words


def stream_base(plan):
    """First 4 KiB boundary above the trace part (KMD_TRACE_PART, the lower half)."""
    return -(-(plan.upload_bytes // 2) // 4096) * 4096


def put_upload(plan, offset, data):
    """Writes `data` at `offset` as whole lines, cut at the region's end; never into the
    trace part, which the KMD_TRACE image overwrites."""
    if offset < stream_base(plan):
        raise ValueError(f"upload at {offset}: inside the trace part")
    room = plan.upload_bytes - offset
    data = bytes(data)[:room]
    data += bytes(-len(data) % 64)
    plan.upload(offset, np.frombuffer(data, dtype=np.uint8))


def stream_events(plan, reqs, timeout_s=20.0):
    """Every event until each stream in `reqs` (one or several) has ended (STREAM_END or
    _ERROR), the kernel finished or timeout_s passed; a line per batch and every 5 s."""
    reqs = {reqs} if isinstance(reqs, int) else set(reqs)
    out, start = [], time.time()
    ended, said = set(), start
    while ended != reqs and time.time() < start + timeout_s:
        batch = []
        with contextlib.suppress(RuntimeError):
            batch = plan.take(plan.seen + 1, 1.0)
        out += batch
        if batch:
            log(f"{len(batch)} events, {len(out)} in all")
        ended = {
            e[1]
            for e in out
            if e[1] in reqs and kind(e[2])[0] in (EV_STREAM_END, EV_STREAM_ERROR)
        }
        if plan.host_word(0) >= STATE_DONE:
            break
        if time.time() - said > 5.0:
            said = time.time()
            log(f"waiting for streams {sorted(reqs - ended)}: {len(out)} events")
    return out


def expected_events(sim):
    """The simulator's events as (kind, req, aux)."""
    return [(e.kind, e.req, e.aux) for e in sim.events]


def got_events(events, req):
    return [(kind(e[2])[0], e[1], kind(e[2])[1]) for e in events if e[1] == req]


def references():
    """The saved reference streams: [(name, upload bytes, saved json)]."""
    out = []
    for name in sorted(os.listdir(STREAMS)):
        if name.endswith(".json"):
            stem = name[:-5]
            with open(os.path.join(STREAMS, stem + ".bin"), "rb") as f:
                data = f.read()
            with open(os.path.join(STREAMS, name)) as f:
                out.append((stem, data, json.load(f)))
    return out


def result_stream(n, n_cores, req, seed):
    """n commands one at a time on random groups (DK_HANDLER_RESULT with KM_F_RESULT,
    key i); after each, its result goes out as a KM_EV_STREAM event."""
    s = ks.Stream(req=req, seed=seed)
    i, count, first, room, res = s.regs(5)
    s.set(i, 0)
    top = s.label()
    s.rand(count)
    s.mod(count, count, s.const(n_cores))
    s.addi(count, count, 1)
    s.sub(room, s.const(n_cores + 1), count)
    s.rand(first)
    s.mod(first, first, room)
    s.submit(HANDLER_RESULT, (first, count), args=[i, 77], result=True)
    s.wait(0)
    s.result(res)
    s.event(res)
    s.addi(i, i, 1)
    s.jlt(i, s.const(n), top)
    s.halt(s.const(0xABCDEF12))
    return s


def loop_stream(req):
    """A stream that never ends by itself: JMP to itself."""
    s = ks.Stream(req=req, max_steps=0xFFFFFFFF)
    top = s.label()
    s.jmp(top)
    return s


def case_streams(session, args):
    """The upload check and the interpreter (test image): RUN and CANCEL rejections and
    errors, every saved reference that ends without commands (bad uploads included)
    against the simulator on this grid, then --n commands from a stream one at a time
    against the simulator; a PING answers after each part."""
    if args.image == IMAGE:
        args.image = VERIFY_IMAGE
    n_cores, trace, sims = 0, {}, {}

    def ping(plan, name, out):
        i = plan.submit(kernel_command(HANDLER_PING))
        ev = [e for e in plan.take(plan.seen + 1, 5.0) if e[1] == i]
        out["pings"][name] = bool(ev and kind(ev[0][2])[0] == EV_PONG)

    def one(plan, name, data, sim, out, offset=0):
        put_upload(plan, offset, data)
        plan.submit(run_command(offset, len(data)))
        req = sim.events[0].req if sim.events else 0
        events = stream_events(plan, req)
        got = got_events(events, req)
        want = expected_events(sim)
        ends = [e[3] for e in events if kind(e[2])[0] == EV_STREAM_END]
        ok = got == want and (sim.halt is None or ends == [sim.halt])
        out["streams"][name] = ok
        if not ok:
            out["wrong"].append({"name": name, "got": got[:6], "want": want[:6]})
        log(f"stream {name}: {'ok' if ok else 'WRONG'}")

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        region = plan.upload_bytes
        base = stream_base(plan)
        out = {"tag": plan.tag, "pings": {}, "streams": {}, "wrong": [], "run": {}}
        ping(plan, "start", out)

        # RUN and CANCEL: rejections and range errors, four loops at once.
        loops = [loop_stream(100 + j).encode() for j in range(5)]
        size = len(loops[0])
        for j in range(5):
            put_upload(plan, base + 4096 * (j + 1), loops[j])
        checks = []

        def expect(name, words, want):
            i = plan.submit(words)
            ev = plan.take(plan.seen + 1, 5.0)
            got = [(kind(e[2])[0], e[1], kind(e[2])[1]) for e in ev]
            checks.append({"name": name, "got": got, "ok": got == [want(i)]})

        put_upload(plan, base + 4096 * 6, bytes(64))
        for j in range(3):
            plan.submit(run_command(base + 4096 * (j + 1), size))
        hdr_req = 100  # the header of loops[0] at base + 4096
        expect(
            "overlap",
            run_command(base + 4096, size),
            lambda i: (EV_STREAM_ERROR, hdr_req, ks.ERR_HEADER),
        )
        expect(
            "unaligned",
            run_command(base + 4096 * 6 + 32, size),
            lambda i: (EV_STREAM_ERROR, 0, ks.ERR_HEADER),
        )
        expect(
            "outside",
            run_command(base + 4096, region),
            lambda i: (EV_STREAM_ERROR, hdr_req, ks.ERR_HEADER),
        )
        put_upload(plan, base + 4096 * 6, loop_stream(101).encode())
        expect(
            "active req",
            run_command(base + 4096 * 6, size),
            lambda i: (EV_STREAM_ERROR, 101, ks.ERR_HEADER),
        )
        plan.submit(run_command(base + 4096 * 4, size))
        expect(
            "fifth RUN",
            run_command(base + 4096 * 5, size),
            lambda i: (EV_REJECTED, i, REJECT_STREAMS),
        )
        expect(
            "cancel unknown",
            kernel_command(HANDLER_CANCEL, 999),
            lambda i: (EV_REJECTED, i, REJECT_UNKNOWN),
        )
        expect(
            "deadline 0",
            kernel_command(HANDLER_DEADLINE, 0),
            lambda i: (EV_REJECTED, i, REJECT_DEADLINE),
        )
        for j in range(4):
            expect(
                f"cancel {100 + j}",
                kernel_command(HANDLER_CANCEL, 100 + j),
                lambda i, j=j: (EV_STREAM_ERROR, 100 + j, ks.ERR_CANCELLED),
            )
        out["run"] = checks
        ping(plan, "after run", out)

        # Saved references that submit no command, against the simulator on this grid.
        for name, data, saved in references():
            sim = ks.simulate(data, n_cores=n_cores, region_size=region - base)
            if saved["expect"]["commands"] or sim.commands:
                continue
            sims[name] = sim
            one(plan, name, data, sim, out, base)
        ping(plan, "after references", out)

        # Commands one at a time from a stream.
        s = result_stream(args.n, n_cores, 7, args.seed)
        data = s.encode()
        sim = s.simulate(
            n_cores=n_cores,
            result=lambda r: dk_result(r.args[0], r.args[1]),
            region_size=region - base,
        )
        sims["commands"] = sim
        one(plan, "commands", data, sim, out, base)
        ping(plan, "after commands", out)
        out["exit"] = exit_and_stop(plan)
        return out

    def after(plan, fin):
        trace["head"], trace["entries"] = read_trace(plan)

    out, fin, cnt = with_kernel(session, args, body, depth=16, after=after)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    submitted = [
        e for e in trace["entries"] if e[0] == TRACE_SUBMIT and e[6] == SOURCE_ENGINE
    ]
    want = [(r.first, r.count) for r in sims["commands"].commands]
    out["commands_in_order"] = [(e[3], e[4]) for e in submitted] == want
    ran = [(r.first, r.count, r.args[0]) for r in sims["commands"].commands]
    out["group_mismatch"], out["first_mismatch"] = check_groups(cnt, n_cores, ran)
    out["ok"] = bool(
        finished_clean(fin, n_cores, out["cores"])
        and all(c["ok"] for c in out["run"])
        and all(out["pings"].values())
        and all(out["streams"].values())
        and out["commands_in_order"]
        and out["group_mismatch"] == 0
        and trace["head"][2] == 0
    )
    return out


def replay_upload(data):
    """A reference upload with every template's handler set to the stand-in worker's
    DK_HANDLER_RESULT, which runs on any group and answers dk_result(a0, a1)."""
    b = bytearray(data)
    if len(b) < 64:
        return bytes(b)
    n_ops, n_commands = (int(v) for v in np.frombuffer(bytes(b[8:16]), dtype=np.uint32))
    templates = 64 + -(-n_ops * 16 // 64) * 64
    for t in range(n_commands):
        at = templates + 80 * t + 4
        if at < len(b) and 1 <= b[at] <= 239:
            b[at] = HANDLER_RESULT
    return bytes(b)


def pass_timed(params):
    """Whether a reference's answer rests on the simulator's passes (a deadline or a
    cancel in passes, a command never acknowledged, a pass limit)."""
    d = params["delay"]
    never = d is None or (isinstance(d, dict) and None in d.get("by_seq", {}).values())
    return bool(
        params["deadline_passes"] is not None
        or params["cancel_pass"] is not None
        or never
        or params["default_deadline_passes"] != 1_000_000
        or params["max_passes"] < 100_000
    )


# Completion models the replay simulates a reference under: the card's own timing is
# none of them, so only what all of them agree on is compared exactly.
REPLAY_DELAYS = [0, 1, 4, 16, lambda r: r.seq % 5, lambda r: 7 * (r.seq % 3)]


def replay_sims(data, n_cores, region):
    """The reference simulated on this grid under every model of REPLAY_DELAYS."""
    return [
        ks.simulate(
            data,
            n_cores=n_cores,
            delay=d,
            result=lambda r: dk_result(*(list(r.args) + [0, 0])[:2]),
            region_size=region,
            ticks_per_pass=tpp,
        )
        for d in REPLAY_DELAYS
        for tpp in (1000, 37)
    ]


def queue_faults(entries, n_cores):
    """Replays the trace's slots: every queued submit must take the lowest group that
    overlaps no busy one. Returns the faults and the most queued commands in flight."""
    busy, faults, most = {}, [], 0
    for kind_, slot, req, first, count, p, src in entries:
        if kind_ == TRACE_FREE:
            busy.pop(slot, None)
            continue
        others = [(f, c) for f, c, _ in busy.values()]
        if kind_ == TRACE_QUEUED and first != alloc(others, count, n_cores):
            faults.append([src, req, first, count, others, p])
        busy[slot] = (first, count, kind_ == TRACE_QUEUED)
        most = max(most, sum(q for _, _, q in busy.values()))
    return faults, most


def case_replay(session, args):
    """Saved references with commands that do not rest on the simulator's passes, as
    DK_HANDLER_RESULT (test image): events and groups in order where every completion
    model agrees, else counts and the last event; queued commands take the lowest free
    group; the grid ran the traced groups."""
    if args.image == IMAGE:
        args.image = VERIFY_IMAGE
    n_cores, trace, plans = 0, {}, []

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        base = stream_base(plan)
        out = {"tag": plan.tag, "timed": 0, "exact": 0, "sets": 0}
        for name, data, saved in references():
            if pass_timed(saved["params"]):
                out["timed"] += 1
                continue
            data = replay_upload(data)
            sims = replay_sims(data, n_cores, plan.upload_bytes - base)
            if not sims[0].commands:
                continue
            commands = {tuple((r.first, r.count) for r in s.commands) for s in sims}
            events = {tuple((e.kind, e.req, e.aux) for e in s.events) for s in sims}
            exact = len(commands) == 1 and len(events) == 1
            out["exact" if exact else "sets"] += 1
            req = sims[0].events[0].req if sims[0].events else 0
            put_upload(plan, base, data)
            plan.submit(run_command(base, len(data)))
            got = got_events(stream_events(plan, req), req)
            # An ended stream's commands may still run: a ping after a short wait.
            time.sleep(0.005)
            i = plan.submit(kernel_command(HANDLER_PING))
            pong = [e for e in plan.take(plan.seen + 1, 5.0) if e[1] == i]
            plans.append((name, exact, sims[0], got, len(events) == 1, bool(pong)))
        out["exit"] = exit_and_stop(plan)
        return out

    def after(plan, fin):
        trace["head"], trace["entries"] = read_trace(plan)

    out, fin, cnt = with_kernel(session, args, body, depth=16, after=after)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    submits = [e for e in trace["entries"] if e[0] in (TRACE_SUBMIT, TRACE_QUEUED)]
    engine = [e for e in submits if e[6] == SOURCE_ENGINE]
    wrong, at = [], 0
    for name, exact, sim, got, events_fixed, pong in plans:
        mine = engine[at : at + len(sim.commands)]
        at += len(sim.commands)
        want_events = [(e.kind, e.req, e.aux) for e in sim.events]
        groups = [(e[3], e[4]) for e in mine]
        if exact:
            ok = groups == [(r.first, r.count) for r in sim.commands]
            ok = ok and got == want_events
        else:
            ok = sorted(c for _, c in groups) == sorted(r.count for r in sim.commands)
            ok = ok and got[-1:] == want_events[-1:]
            ok = ok and (got == want_events or not events_fixed)
        ok = ok and pong
        if not ok:
            wrong.append({"name": name, "got": got[-3:], "want": want_events[-3:]})
    faults, most = queue_faults(trace["entries"], n_cores)
    runs = np.zeros(n_cores, dtype=np.int64)  # EXIT, the host's one command, is no run
    for e in engine:
        runs[e[3] : e[3] + e[4]] += 1
    out["wrong"], out["faults"] = wrong, faults[:8]
    out["most_queued_in_flight"] = most
    out["runs_ok"] = bool((cnt[:n_cores, C["run"]] == runs).all())
    out["ok"] = bool(
        finished_clean(fin, n_cores, out["cores"])
        and not wrong
        and not faults
        and at == len(engine)
        and out["runs_ok"]
        and trace["head"][2] == 0
    )
    return out


def case_stream_deadline(session, args):
    """A stream's command whose member --core waits for the stop flag, header deadline
    --deadline-us: the stream ends with KM_SERR_DEADLINE at that SUBMIT, the kernel with
    the deadline error, and every core gives up on the stop flag."""
    n_cores = 0

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        s = ks.Stream(req=55, deadline_us=args.deadline_us)
        check_args = [3, 4] + [dk_arg(3, 4, w) for w in range(2, 10)]
        s.submit(HANDLER_CHECK, (0, n_cores), args=check_args)
        s.wait(0)
        s.submit(HANDLER_STALL, (0, n_cores), args=[9, args.core])
        s.wait(0)
        s.halt(s.const(0))
        data = s.encode()
        put_upload(plan, stream_base(plan), data)
        plan.submit(run_command(stream_base(plan), len(data)))
        submits = [pc for pc, o in enumerate(s.uops()) if o.op == ks.OP_SUBMIT]
        out = {"tag": plan.tag, "stall_pc": submits[1]}
        log("stream started")
        out["events"] = stream_events(plan, 55)
        log("stream ended")
        return out

    out, fin, cnt = with_kernel(session, args, body, depth=4)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    blk = fin["block"]
    got = [(kind(e[2])[0], e[1], kind(e[2])[1]) for e in out["events"]]
    out["ok"] = bool(
        got == [(EV_STREAM_ERROR, 55, ks.ERR_DEADLINE | out["stall_pc"] << 8)]
        and blk["error"] == ERR_DEADLINE
        and blk["source"] == SOURCE_ENGINE
        and blk["engine"] == ENGINE_STOPPED
        and out["cores"]["cores_finished"] == n_cores
        and out["cores"]["gave"] == {"stop": n_cores}
    )
    return out


def case_engine_busy(session, args):
    """A host DELAY of --delay-us on the grid's first half, then stream 61 with one
    KM_F_EXCL command on that half and stream 62 with --n commands one at a time on the
    other half: 61's command is rejected busy and sent again after the wake, so 62 ends
    before the DELAY's DONE and 61 after it."""
    n_cores = 0

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        half, base = n_cores // 2, stream_base(plan)
        cycles = int(args.delay_us * 1e-6 * AICLK_HZ)
        blocked = ks.Stream(req=61)
        blocked.submit(HANDLER_DELAY, (0, half), args=[2, 0], exclusive=True)
        blocked.wait(0)
        blocked.halt(blocked.const(7))
        flowing = ks.Stream(req=62)
        i = flowing.regs(1)[0]
        flowing.set(i, 0)
        top = flowing.label()
        flowing.submit(HANDLER_DELAY, (half, n_cores - half), args=[i, 0])
        flowing.wait(0)
        flowing.addi(i, i, 1)
        flowing.jlt(i, flowing.const(args.n), top)
        flowing.halt(i)
        out = {"tag": plan.tag}
        out["delay_req"] = plan.submit(command(HANDLER_DELAY, 0, half, 1, cycles))
        for j, st in enumerate((blocked, flowing)):
            data = st.encode()
            put_upload(plan, base + 4096 * j, data)
            plan.submit(run_command(base + 4096 * j, len(data)))
        out["events"] = stream_events(plan, {61, 62})
        out["exit"] = exit_and_stop(plan)
        return out

    out, fin, cnt = with_kernel(session, args, body, depth=4)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    ev, blk = out.pop("events"), fin["block"]
    at = {}
    for j, e in enumerate(ev):
        what = kind(e[2])
        if e[1] == out["delay_req"] and what[0] == EV_DONE:
            at["delay"] = j
        elif e[1] in (61, 62) and what[0] in (EV_STREAM_END, EV_STREAM_ERROR):
            at[e[1]] = j
            out[f"end_{e[1]}"] = [what[0], what[1], e[3]]
    out["order"] = sorted(at, key=at.get)
    out["ok"] = bool(
        out["order"] == [62, "delay", 61]
        and out.get("end_61") == [EV_STREAM_END, 7, 7]
        and out.get("end_62") == [EV_STREAM_END, args.n, args.n]
        and blk["engine_commands"] >= args.n + 2
        and finished_clean(fin, n_cores, out["cores"])
    )
    return out


def case_seq_events(session, args):
    """A host command, a stream of --n commands one at a time, a second host command:
    the host commands' DONE events carry sequence numbers 1 and n + 2 in their low 16
    bits, with the group's first core below them."""
    n_cores = 0

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        s = ks.Stream(req=91)
        i = s.regs(1)[0]
        s.set(i, 0)
        top = s.label()
        s.submit(HANDLER_DELAY, (0, 1), args=[i, 0])
        s.wait(0)
        s.addi(i, i, 1)
        s.jlt(i, s.const(args.n), top)
        s.halt(i)
        out = {"tag": plan.tag}
        a = plan.submit(command(HANDLER_DELAY, 5, 3, 1, 0))
        got = [e for e in plan.take(a, 5.0) if e[1] == a]
        data, base = s.encode(), stream_base(plan)
        put_upload(plan, base, data)
        plan.submit(run_command(base, len(data)))
        ended = stream_events(plan, 91)
        b = plan.submit(command(HANDLER_DELAY, 7, 2, 2, 0))
        got += [e for e in plan.take(b, 5.0) if e[1] == b]
        out["host"] = [[kind(e[2])[0], kind(e[2])[1], seq_low(e[2])] for e in got]
        out["stream_end"] = [list(kind(e[2])) for e in ended if e[1] == 91][-1:]
        out["exit"] = exit_and_stop(plan)
        return out

    out, fin, cnt = with_kernel(session, args, body, depth=4)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    want = [[EV_DONE, 5, 1], [EV_DONE, 7, (args.n + 2) & 0xFFFF]]
    out["ok"] = bool(
        out["host"] == want
        and out["stream_end"] == [[EV_STREAM_END, args.n]]
        and finished_clean(fin, n_cores, out["cores"])
    )
    return out


def case_engine_cost(session, args):
    """Three streams in turn, each --n DELAY commands on core 0 timed by the stream
    itself: SUBMIT then WAIT 0 each; SUBMIT with KM_F_EXCL and no wait, so a command is
    rejected busy while the one before runs and sent again after the wake (both
    --delay-us); ENQUEUE of DELAY 0 jobs."""
    cycles = int(args.delay_us * 1e-6 * AICLK_HZ)

    def timed(req, form):
        s = ks.Stream(req=req)
        i, one, t0, t1 = s.regs(4)
        s.set(i, 0)
        s.set(one, 1)
        s.time(t0)
        top = s.label()
        if form == "enqueue":
            s.enqueue(HANDLER_DELAY, one, args=[i, 0])
        else:
            s.submit(HANDLER_DELAY, (0, 1), args=[i, cycles], exclusive=True)
        if form == "wait":
            s.wait(0)
        s.addi(i, i, 1)
        s.jlt(i, s.const(args.n), top)
        s.wait(0)
        s.time(t1)
        s.sub(t1, t1, t0)
        s.halt(t1)
        return s

    forms = {81: "wait", 82: "rejected", 83: "enqueue"}
    n_cores = 0

    def body(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        out, base = {"tag": plan.tag}, stream_base(plan)
        for j, (req, form) in enumerate(forms.items()):
            before = plan.seen
            data = timed(req, form).encode()
            put_upload(plan, base + 4096 * j, data)
            plan.submit(run_command(base + 4096 * j, len(data)))
            ev = stream_events(plan, req)
            end = [e for e in ev if e[1] == req and kind(e[2])[0] == EV_STREAM_END]
            out[form] = round(end[0][3] / 50.0 / args.n, 3) if end else None
            out[f"{form}_events"] = plan.seen - before
        out["exit"] = exit_and_stop(plan)
        return out

    out, fin, cnt = with_kernel(session, args, body, depth=4)
    out["finish"] = fin
    out["cores"] = core_report(cnt, n_cores, out["tag"])
    out["rejections"] = fin["block"]["rejected"]
    out["us_per_command"] = {f: out.pop(f) for f in forms.values()}
    out["ok"] = bool(
        all(v is not None for v in out["us_per_command"].values())
        and finished_clean(fin, n_cores, out["cores"])
    )
    return out


def case_exit_flush(session, args):
    """A stream submitting commands and a looping one on core 1 when STOP ends the
    kernel; the host then writes a pattern over the state's DRAM buffer, a second
    kernel runs, and the pattern reads back unchanged: no line of the first run was
    left dirty."""
    n_cores = 0

    def first(plan):
        nonlocal n_cores
        n_cores = plan.grid_x * plan.grid_y
        base = stream_base(plan)
        submitting = ks.Stream(req=71)
        i = submitting.regs(1)[0]
        submitting.set(i, 0)
        top = submitting.label()
        submitting.submit(HANDLER_DELAY, (0, n_cores), args=[i, 0])
        submitting.wait(0)
        submitting.addi(i, i, 1)
        submitting.jlt(i, submitting.const(args.n), top)
        submitting.halt(i)
        for j, st in enumerate((submitting, loop_stream(72))):
            data = st.encode()
            put_upload(plan, base + 4096 * j, data)
            plan.submit(run_command(base + 4096 * j, len(data)))
        events = stream_events(plan, 71)
        out = {"tag": plan.tag, "ended": [kind(e[2])[0] for e in events if e[1] == 71]}
        plan.submit(kernel_command(HANDLER_STOP))
        return out

    out, fin, _ = with_kernel(session, args, first, depth=4)
    out["finish"] = {k: fin["block"][k] for k in ("error", "engine", "engine_commands")}
    # The host writes a pattern over the first run's state buffer, in place.
    state = DRAM_KEEP[-1]
    shape = tuple(state.shape)
    pattern = (np.arange(shape[0] * shape[1], dtype=np.uint64) * 2654435761) & M32
    pattern = pattern.astype(np.uint32).reshape(shape)
    state.write(pattern)

    def second(plan):
        i = plan.submit(kernel_command(HANDLER_PING))
        got = [e for e in plan.take(plan.seen + 1, 5.0) if e[1] == i]
        return {"pong": bool(got), "exit": exit_and_stop(plan)}

    out["second"], fin2, _ = with_kernel(session, args, second, depth=4)
    back = state.read()
    out["words_changed"] = int(np.count_nonzero(back != pattern))
    out["ok"] = bool(
        out["ended"] == [EV_STREAM_END]
        and out["finish"]["error"] == 0
        and out["finish"]["engine"] == ENGINE_STOPPED
        and out["second"]["pong"]
        and fin2["block"]["error"] == 0
        and out["words_changed"] == 0
    )
    return out


CASES = {
    "ping": case_ping,
    "submit": lambda session, args: case_ping(session, args, submit_one=True),
    "groups": case_groups,
    "disjoint": case_disjoint,
    "stall": case_stall,
    "stall-leader": lambda session, args: case_stall(session, args, leader=True),
    "stall-late": case_stall_late,
    "compile-error": case_compile_error,
    "fill": case_fill,
    "exit-busy": case_exit_busy,
    "after-exit": case_after_exit,
    "malformed": case_malformed,
    "stop-busy": case_stop_busy,
    "runtime-stop": case_runtime_stop,
    "trap": case_trap,
    "excl": case_excl,
    "queue": case_queue,
    "streams": case_streams,
    "replay": case_replay,
    "stream-deadline": case_stream_deadline,
    "engine-busy": case_engine_busy,
    "engine-cost": case_engine_cost,
    "seq-events": case_seq_events,
    "exit-flush": case_exit_flush,
    "verify": case_verify,
    "upload": case_upload,
    "result": case_result,
}

# The regression set: one behaviour per case on a small input, in one device open.
FED = {"host_depth": 64, "batch": 64}
SET = [
    ("ping", {}),
    ("groups", {"n": 200}),
    ("disjoint", {"n": 8}),
    ("stall", {}),
    ("stall-late", {}),
    ("fill", {"n": 50}),
    ("exit-busy", {}),
    ("after-exit", {}),
    ("malformed", {}),
    ("stop-busy", {}),
    ("trap", {}),
    ("excl", {}),
    ("queue", {"n": 200, "delay_us": 20.0, **FED}),
    ("verify", {"n": 2000, "settle_us": 1.0, **FED}),
    ("result", {"n": 1000, **FED}),
    ("upload", {}),
    ("streams", {"n": 100}),
    ("replay", {}),
    ("stream-deadline", {}),
    ("engine-busy", {"n": 20, "delay_us": 2000.0}),
    ("exit-flush", {"n": 50}),
    ("seq-events", {"n": 300}),
]


def case_set(session, args):
    """Every case of SET in turn; a JSON line per case."""
    oks = {}
    for name, changes in SET:
        log(f"case {name}")
        out = CASES[name](session, argparse.Namespace(**{**vars(args), **changes}))
        print(json.dumps({"case": name, **out}, default=str), flush=True)
        oks[name] = bool(out.get("ok"))
    return {"cases": oks, "ok": all(oks.values())}


def parser(cases):
    """The options of every case in `cases`."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("case", choices=sorted(cases))
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--group", type=int, default=0, help="group size, 0 = all cores")
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
    p.add_argument("--pll", type=int, default=1000, help="X280 clock, MHz")
    p.add_argument("--image-first", action="store_true")
    p.add_argument("--batch", type=int, default=1, help="commands per store batch")
    p.add_argument("--settle-us", type=float, default=0.0, help="fence to next store")
    p.add_argument("--image", default=IMAGE, type=os.path.abspath, help="kernel image")
    return p


def main(cases, args):
    """Opens the session, runs args.case of `cases`, prints its JSON line."""
    out = {"case": args.case, "argv": sys.argv[1:], "load_start": loadavg()}
    t_start = time.time()
    log("opening the device")
    with km.Session() as session:
        log(f"device open {time.time() - t_start:.1f}s")
        try:
            out.update(cases[args.case](session, args))
        finally:
            log("closing the device")
    log("device closed")
    out["load_end"] = loadavg()
    out["wall_s"] = round(time.time() - t_start, 1)
    print(json.dumps(out, default=str), flush=True)


if __name__ == "__main__":
    ALL = {**CASES, "set": case_set}
    main(ALL, parser(ALL).parse_args())
