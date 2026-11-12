"""Measurements of the dispatch kernel on a card, on the harness of its card tests
(tests/hardware/dispatch_kernel): throughput, host store cost, memory cost per
operation, the stream engine's pass. One case per run, one JSON line at the end.
"""

import gc
import itertools
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, "..", "..", "tests", "hardware", "dispatch_kernel")
sys.path.insert(0, TESTS)

import dispatch_card as dc  # noqa: E402

MEMCOST = os.path.join(dc.BUILD, "km-memcost.bin")
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
TARGETS = {"lim": 0, "l1": 1}


def case_limit(device, args):
    runtime = dc.dk.Runtime(device, pll_mhz=args.pll)
    try:
        out = dict(runtime.image_limit())
    finally:
        del runtime
        gc.collect()
    out["image_bytes"] = os.path.getsize(args.image)
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
        r[:, 1] = dc.SEG_NOP
        r[:-1, 2] = firsts.astype(np.uint32) | (team << 16)
        r[:, 3] = np.arange(1, args.n + 2, dtype=np.uint32)
        r[-1] = dc.record(dc.SEG_EXIT, 0, n_cores)
        recs.append(r)
        sent.extend(zip(firsts.tolist(), [team] * args.n, r[:-1, 3].tolist()))
        if args.preload:
            for rec in r.tolist():
                plan.submit(rec)

    def body(plan):
        if args.preload:
            events, times = plan.take(args.n + 1, dc.HOST_WAIT_S), {}
        else:
            events, times = plan.stream(
                recs[0], dc.HOST_WAIT_S, args.batch, args.settle_us
            )
        plan.submit(dc.command(dc.SEG_STOP))
        out = {"tag": plan.tag, "times": times}
        out["check"] = dc.check_events(events, list(range(1, args.n + 2)), 1)
        ticks = np.array([e[3] for e in events[:-1]], dtype=np.float64)
        out["latency_us"] = {
            "mean": round(float(ticks.mean()) / 50, 3),
            "p50": round(float(np.median(ticks)) / 50, 3),
            "max": round(float(ticks.max()) / 50, 3),
        }
        return out

    if args.preload and args.n + 1 > args.host_depth:
        raise SystemExit("--preload needs n + 1 <= host depth")
    out, fin, cnt = dc.with_kernel(
        device, args, body, depth=args.depth, preload=commands
    )
    blk = fin["block"]
    n = args.n
    t = out["times"]
    out["finish"] = fin
    out["cores"] = dc.core_report(cnt, n_cores, out["tag"])
    out["team_mismatch"], out["first_mismatch"] = dc.check_teams(cnt, n_cores, sent)
    # The kernel's span runs from the first command taken to the EXIT's event, over
    # n + 1 commands; the host's from the first store to the last event read.
    out["us_per_command_kernel"] = round(
        blk["last"] / dc.dk.MTIME_HZ * 1e6 / (n + 1), 4
    )
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
    runtime = dc.dk.Runtime(device, pll_mhz=args.pll)
    plan = runtime.plan()
    configs = []
    try:
        grid = itertools.product(args.busy_us, args.spread, args.lines, args.gaps_us)
        for busy, spread, lines, gap in grid:
            target = TARGETS[args.target]
            dc.log(f"store: busy {busy} lines {lines} gap {gap}")
            t = np.array(
                plan.store_probe(args.n, lines, spread, gap, busy, args.read, target)
            ).reshape(-1, 2)
            us = t.sum(axis=1) * 1e6
            dc.log(
                f"store: busy {busy} lines {lines} gap {gap}: p50 {np.median(us):.2f}"
            )
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
    ddr = dc.dram_area(device)
    runtime = dc.dk.Runtime(device, pll_mhz=args.pll)
    plan = runtime.plan()
    try:
        dc.log("running the memory cost image")
        fin = plan.run_image(MEMCOST, [ddr], 10.0)
        dc.log(f"memory cost image finished: result {fin['result']}")
    finally:
        if plan.running():
            plan.abort()
        del plan, runtime
        gc.collect()
    words = fin.pop("words")
    ps = dict(zip(MEMCOST_TESTS, words))
    return {"ddr_offset": ddr, "finish": fin, "ps_per_op": ps, "ok": fin["result"] == 0}


def timed_stream(n, first, team, req):
    """n records one at a time on cores [first, first + team) (DK_SEG_RESULT, key i),
    timed by the stream itself: its end carries the ticks from the first ISSUE to the
    last acknowledgement."""
    s = dc.ks.Stream(req=req)
    i, t0, t1 = s.regs(3)
    s.set(i, 0)
    s.time(t0)
    top = s.label()
    s.issue(dc.SEG_RESULT, (first, team), args=[i, 5], result=True)
    s.wait(0)
    s.addi(i, i, 1)
    s.jlt(i, s.const(n), top)
    s.time(t1)
    s.sub(t1, t1, t0)
    s.halt(t1)
    return s


def compute_stream(n, req):
    """n rounds of ADDI and JLT, no record, timed by the stream (ticks in its end)."""
    s = dc.ks.Stream(max_steps=2 * n + 16, req=req)
    i, t0, t1 = s.regs(3)
    s.set(i, 0)
    s.time(t0)
    top = s.label()
    s.addi(i, i, 1)
    s.jlt(i, s.const(n), top)
    s.time(t1)
    s.sub(t1, t1, t0)
    s.halt(t1)
    return s


def case_stream_bench(device, args):
    """The kernel's pass with --streams busy streams (JMP loops): idle for --seconds
    (--n 0), or --issuers streams each issuing --n records one at a time on its own
    --team cores; then (--compute) a stream of 2 x --compute micro-ops alone. Pass
    period from the result block, per record from the streams, reissue from the trace.
    """
    if args.image == dc.IMAGE:
        args.image = dc.VERIFY_IMAGE
    trace = {}

    def body(plan):
        base = dc.stream_base(plan)
        out = {"tag": plan.tag, "streams": args.streams}
        plan.submit(dc.command(dc.SEG_PING))
        for j in range(args.streams):
            data = dc.loop_stream(200 + j).encode()
            dc.put_upload(plan, base + 4096 * (j + 1), data)
            plan.submit(dc.run_command(base + 4096 * (j + 1), len(data)))
        if args.n:
            team = args.team or 1
            for j in range(args.issuers):
                data = timed_stream(args.n, j * team, team, 9 + j).encode()
                dc.put_upload(plan, base + 65536 * (j + 1), data)
            t = time.perf_counter()
            for j in range(args.issuers):
                plan.submit(dc.run_command(base + 65536 * (j + 1), len(data)))
            dc.log(f"{args.issuers} streams of {args.n} records started")
            ev = dc.stream_events(plan, range(9, 9 + args.issuers))
            dc.log("streams ended")
            out["host_s"] = round(time.perf_counter() - t, 4)
            ends = [e[3] for e in ev if dc.kind(e[2])[0] == dc.EV_STREAM_END]
            out["us_per_record"] = [round(e / 50 / args.n, 4) for e in ends]
            out["stream_events"] = [list(e) for e in ev]
        else:
            dc.log(f"idle {min(args.seconds, dc.HOST_WAIT_S)} s")
            time.sleep(min(args.seconds, dc.HOST_WAIT_S))
            dc.log("idle done")
        for j in range(args.streams):
            plan.submit(dc.command(dc.SEG_CANCEL, 200 + j))
        if args.compute:
            s = compute_stream(args.compute, 11)
            data = s.encode()
            dc.put_upload(plan, base, data)
            plan.submit(dc.run_command(base, len(data)))
            dc.log("compute stream started")
            ev = dc.stream_events(plan, 11)
            dc.log("compute stream ended")
            ends = [e[3] for e in ev if dc.kind(e[2])[0] == dc.EV_STREAM_END]
            ops = 2 * args.compute
            out["uops_per_s"] = round(ops * 50e6 / ends[0]) if ends else None
            out["compute_events"] = [list(e) for e in ev]
        out["exit"] = dc.exit_and_stop(plan)
        return out

    def after(plan, fin):
        trace["head"], trace["entries"] = dc.read_trace(plan)

    out, fin, cnt = dc.with_kernel(device, args, body, depth=16, after=after)
    out["finish"] = fin
    n_cores = cnt.shape[0]
    out["cores"] = dc.core_report(cnt, n_cores, out["tag"])
    blk = fin["block"]
    out["pass_ns"] = round(blk["ticks"] * 20 / max(1, blk["passes"]), 2)
    out["engine"] = {k: blk[k] for k in dc.H_NAMES.values()}
    # Per issuer (records to cores [j * team, (j + 1) * team)): issue to free and free
    # to next issue, in trace ticks (50 per us).
    freed, busy, reissue, latency = {}, {}, [], []
    team = args.team or 1
    for k_, slot, _, first, _, t, src in trace["entries"]:
        j = first // team
        if src != dc.SOURCE_ENGINE or j >= args.issuers:
            continue
        if k_ == dc.TRACE_ISSUE:
            if j in freed:
                reissue.append((t - freed.pop(j)) & dc.M32)
            busy[slot] = t
        elif k_ == dc.TRACE_FREE and slot in busy:
            freed[j] = t
            latency.append((t - busy.pop(slot)) & dc.M32)
    if reissue:
        out["free_to_reissue_us"] = percentiles(np.array(reissue) / 50)
        out["issue_to_free_us"] = percentiles(np.array(latency) / 50)
    out["ok"] = bool(
        dc.finished_clean(fin, n_cores, out["cores"])
        and (not args.n or len(out["us_per_record"]) == args.issuers)
        and (not args.compute or out["uops_per_s"])
    )
    return out


CASES = {
    "limit": case_limit,
    "bench": case_bench,
    "store": case_store,
    "memcost": case_memcost,
    "stream-bench": case_stream_bench,
}


if __name__ == "__main__":
    p = dc.parser(CASES)
    p.add_argument("--preload", action="store_true", help="bench: every command first")
    p.add_argument("--gaps-us", type=float, nargs="+", default=[0.0], help="store")
    p.add_argument("--busy-us", type=float, nargs="+", default=[0.0], help="store")
    p.add_argument("--lines", type=int, nargs="+", default=[2], help="store")
    p.add_argument("--spread", type=int, nargs="+", default=[1], help="store")
    p.add_argument("--read", action="store_true", help="store: read before a burst")
    p.add_argument("--target", choices=sorted(TARGETS), default="lim", help="store")
    p.add_argument("--streams", type=int, default=0, help="stream-bench: loops")
    p.add_argument("--seconds", type=float, default=1.0, help="stream-bench: idle")
    p.add_argument("--compute", type=int, default=0, help="stream-bench: rounds")
    p.add_argument("--issuers", type=int, default=1, help="stream-bench: streams")
    dc.main(CASES, p.parse_args())
