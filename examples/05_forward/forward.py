"""The forward example on a card through the dispatcher on X280 cluster 0: one dynamic
run, messages forward between the group's cores until their forwards are used up, the
kernel ends the run at the quiet end; every core's totals against the numpy replay. See
README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(HERE),
    str(ROOT / "tests" / "hardware" / "feeder"),
    str(ROOT / "src"),
]

import commands as rec  # noqa: E402
import replay  # noqa: E402

import komorebi as km  # noqa: E402
from komorebi import abi  # noqa: E402
from komorebi.engines.stream import Stream  # noqa: E402
from komorebi.layout import counter_tiles  # noqa: E402
from komorebi.program import Dynamic  # noqa: E402

FORWARD = rec.constants(HERE / "kernels" / "forward.h")
RUNTIME = ROOT / "src" / "komorebi_native" / "kernels"
T0 = time.monotonic()


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--count", type=int, default=0, help="group: cores 0 to count - 1")
    p.add_argument("--messages", type=int, default=1)
    p.add_argument("--forwards", type=int, default=1, help="forwards per message")
    p.add_argument(
        "--one-core", type=int, default=-1, help="every forward to this member"
    )
    p.add_argument("--stall", type=int, default=-1, help="member that stalls")
    p.add_argument("--again", action="store_true", help="then one more run, no stall")
    p.add_argument("--repeat", type=int, default=1, help="the first run, this often")
    p.add_argument(
        "--engine", action="store_true", help="then the stream engine runs two halves"
    )
    p.add_argument("--probe", type=int, default=-1, help="member that spends its quota")
    p.add_argument("--max-live", type=int, default=1 << 14)
    p.add_argument("--queue-messages", type=int, default=1 << 14)
    p.add_argument("--bins", type=int, default=16)
    p.add_argument("--idle-cycles", type=int, default=20000)
    p.add_argument("--counters", type=int, default=2, help="counter tiles of the group")
    p.add_argument("--deadline", type=float, default=5.0, help="the run, seconds")
    p.add_argument("--wait", type=float, default=1.0, help="wait limit, seconds")
    return p.parse_args()


def options(a):
    return {
        "max_live": a.max_live,
        "queue_messages": a.queue_messages,
        "bins": a.bins,
        "idle_cycles": a.idle_cycles,
        "counters": a.counters,
    }


def main():
    a = parse()
    try:  # the launcher's rejections, before any device opens
        Dynamic(handler=FORWARD["kHandlerForward"], **options(a)).check(
            a.count or a.counters + 1, 4 << 30, 4
        )
    except ValueError as err:
        print(json.dumps({"ok": False, "rejected": str(err)}), flush=True)
        sys.exit(1)
    stalls = [a.stall] * a.repeat + ([-1] if a.again else [])
    runs = []
    with km.Session(log=log) as session:
        log("device open")
        for i, stall in enumerate(stalls):
            runs.append(run_once(session, a, stall))
            log(f"run {i + 1} done")
        if a.engine:
            runs.append(run_engine(session, a))
            log("engine run done")
    ok = all(r["ok"] for r in runs)
    timed = [r["run_us"] for r in runs if r.get("run_us")]
    summary = {"exact": sum(bool(r.get("exact")) for r in runs), "runs": len(runs)}
    if timed:
        summary["run_us"] = [min(timed), round(float(np.median(timed)), 1), max(timed)]
    out = {"ok": ok, "summary": summary, "runs": runs}
    print(json.dumps(out | {"wall_s": round(time.monotonic() - T0, 1)}))
    sys.exit(0 if ok else 1)


def l1_blocks(session, grid, words):
    """A zeroed L1 area of `words` words on every core, at one address."""
    zeros = np.zeros((grid[0] * grid[1], words), np.int32)
    return session.tensor(zeros, "int32", shard_grid=grid)


def load(session, a):
    """The program loaded: (run, results tensor, program, cores in the grid)."""
    grid = session.grid
    res_t = l1_blocks(session, grid, FORWARD["kResWords"])  # before the grid starts
    program = km.Program(
        "forward",
        kernels={
            "reader": RUNTIME / "dyn_reader.cpp",
            "compute": HERE / "kernels" / "compute.cpp",
            "writer": RUNTIME / "dyn_writer.cpp",
        },
        args=lambda core: ([], [], [res_t.address]),
        handlers={"forward": FORWARD["kHandlerForward"]},
    ).dynamic("forward", **options(a))
    run = session.load(program, deadline=a.deadline, wait=a.wait)
    return run, res_t, program, grid[0] * grid[1]


def command_words(a, stall):
    return [
        a.messages,
        a.forwards,
        FORWARD["kModeOneCore"] if a.one_core >= 0 else FORWARD["kModeRandom"],
        max(a.one_core, 0),
        stall + 1,
        a.probe + 1,
    ]


def run_once(session, a, stall):
    """One load of the program: the RUN command, its end (or the timeout), the check."""
    run, res_t, program, cores = load(session, a)
    count = a.count or cores
    log(f"loaded: group of {count} of {cores} cores")
    words = command_words(a, stall)
    timed_out = None
    t0 = time.perf_counter()
    handle = run.submit("forward", (0, count), words, exclusive=True)
    try:
        run.drain(timeout=a.deadline + 5.0)
        host_s = time.perf_counter() - t0
        run.stop()
    except km.CommandTimeout as err:
        host_s = time.perf_counter() - t0
        timed_out = err
        log(f"timeout: {err}")
    ev, seq = run.event(handle), run.seq(handle)
    out = report(
        a, run, (ev, seq), program.dyn, (count, cores), stall, res_t, timed_out
    )
    out["host_s"] = round(host_s, 4)
    if stall < 0:  # exact, ended by the kernel at the quiet end
        out["ok"] = out["exact"] and out["event"] == "done" and out["gave_core"] == -1
    else:  # the stalled member named, the run ended by its timeout
        out["ok"] = out["event"] == "timeout" and out["gave_core"] == stall
    return out


def run_engine(session, a):
    """One load: a stream on the engine submits the RUN to two halves of the grid one
    after the other; the second waits (rejected busy, then woken) for the first's end.
    """
    run, res_t, program, cores = load(session, a)
    half = cores // 2
    s = Stream(req=21, deadline_us=int(a.deadline * 1e6))
    for first in (0, half):
        s.submit(
            FORWARD["kHandlerForward"],
            (first, half),
            args=command_words(a, -1),
            exclusive=True,
        )
    s.wait(0)
    s.halt(s.const(7))
    t = run.start(run.upload(s))
    ev = run.wait_engine(t, timeout=2 * a.deadline + 5.0)
    run.stop()
    result = run.result()
    res = (res_t.read().astype(np.int64) & 0xFFFFFFFF)[: 2 * half]
    exact = [exact_rows(a, res[first : first + half], half) for first in (0, half)]
    h = abi.headers()
    x280_cores = result.get("x280_cores") or [0] * h["KMD_H_WORDS"]
    return {
        "ok": ev.kind == "end" and all(exact) and result["error_name"] == "none",
        "engine": ev.kind,
        "halt": ev.value,
        "halves_exact": exact,
        "runs": [int(res[0, FORWARD["kResRun"]]), int(res[half, FORWARD["kResRun"]])],
        "kernel": {k: result[k] for k in ("error_name", "commands", "rejected")},
        "checks": int(x280_cores[h["KMD_H_DYN_CHECKS"]]),
    }


def exact_rows(a, res, count):
    """The result rows of a group of `count` cores against the replay over its members;
    the counter tiles (the last a.counters rows) take no message."""
    m = len(counter_tiles(0, count, a.counters).workers)
    one = a.one_core if a.one_core >= 0 else None
    consumed, total, visits = replay.replay(a.messages, a.forwards, m, one)
    if a.probe >= 0:  # the probe's messages visit their own core once
        visits[a.probe] += res[a.probe, FORWARD["kResSent"]]
    totals = res[:m, FORWARD["kResTotalLo"]] | (res[:m, FORWARD["kResTotalHi"]] << 32)
    return (
        bool((res[:m, FORWARD["kResConsumed"]] == consumed).all())
        and bool((totals.astype(np.uint64) == total).all())
        and bool((res[:m, FORWARD["kResVisits"]] == visits).all())
        and not res[m:, FORWARD["kResVisits"]].any()
    )


def report(a, run, event, dyn, group, stall, res_t, timed_out):
    """The checks and numbers of one run as a dict: `event` (the RUN's event, its
    sequence), `group` (count, cores in the grid)."""
    (ev, seq), (count, cores) = event, group
    h = abi.headers()
    result = run.result()
    x280_cores = result.get("x280_cores") or [0] * h["KMD_H_WORDS"]
    res = (res_t.read().astype(np.int64) & 0xFFFFFFFF)[:count]
    ctr = counter_tiles(0, count, a.counters)
    states = run.state()
    gave = {
        c["core"]: c["dynamic"]["gave_up"]
        for c in states[:count]
        if c["dynamic"]["gave_up"]
    }
    out = {
        "count": count,
        "counters": [
            c["core"]
            for c in states[:count]
            if (c["dynamic"]["counter"] or {}).get("run") == seq
        ],
        "counters_expected": list(ctr.counters),
        "messages": a.messages,
        "forwards_per_message": a.forwards,
        "stall": stall,
        "kernel": {k: result[k] for k in ("error_name", "commands", "events")},
        "event": ev.kind if ev else None,
        "run_us": round(ev.us, 1) if ev and ev.kind == "done" else None,
        "exact": bool((res[:, FORWARD["kResRun"]] == seq).all())
        and exact_rows(a, res, count),
        "consumed_by_compute": int(res[:, FORWARD["kResConsumed"]].sum()),
        "expected_consumed": a.messages,
        "forwards": int(res[:, FORWARD["kResVisits"]].sum()) - a.messages,
        "probe_sent": int(res[:, FORWARD["kResSent"]].sum()),
        "rejected": int(res[:, FORWARD["kResRejected"]].sum()),
        "quota_start": dyn.quota(cores),
        "quota_left_min": int(res[: len(ctr.workers), FORWARD["kResQuota"]].min()),
        "checks": int(x280_cores[h["KMD_H_DYN_CHECKS"]]),
        "gave_core": int(x280_cores[h["KMD_H_DYN_GAVE"]]) - 1,
        "dyn_gave": gave,
    }
    if timed_out is not None:
        out["timed_out"] = {
            "handle": timed_out.handle,
            "seq": timed_out.seq,
            "group": list(timed_out.group),
            "not_reported": timed_out.cores[:16],
        }
        out["core_gave_up"] = [
            {"core": c["core"], **{k: c[k] for k in ("reader", "writer", "compute")}}
            for c in states
            if c["reader"] or c["writer"] or c["compute"]
        ][:16]
    return out


if __name__ == "__main__":
    main()
