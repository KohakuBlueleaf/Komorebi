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
    str(ROOT / "src"),
]

import replay  # noqa: E402

import komorebi as km  # noqa: E402
from komorebi import abi  # noqa: E402
from komorebi.bench import power  # noqa: E402
from komorebi.engines.stream import Stream  # noqa: E402
from komorebi.layout import counter_tiles  # noqa: E402
from komorebi.program import Dynamic  # noqa: E402

FORWARD = km.abi.constants(
    HERE / "kernels" / "forward.h", known=km.abi.headers().constants
)
RUNTIME = ROOT / "src" / "komorebi_native" / "kernels"
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--count", type=int, default=0, help="group: cores 0 to count - 1"
    )
    parser.add_argument("--messages", type=int, default=1)
    parser.add_argument("--forwards", type=int, default=1, help="forwards per message")
    parser.add_argument(
        "--one-core", type=int, default=-1, help="every forward to this rank"
    )
    parser.add_argument("--stall", type=int, default=-1, help="rank that stalls")
    parser.add_argument(
        "--again", action="store_true", help="then one more run, no stall"
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="the first run, this often"
    )
    parser.add_argument(
        "--engine", action="store_true", help="then the stream engine runs two halves"
    )
    parser.add_argument(
        "--probe", type=int, default=-1, help="rank that spends its quota"
    )
    parser.add_argument("--max-live", type=int, default=1 << 14)
    parser.add_argument("--queue-messages", type=int, default=1 << 14)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--idle-cycles", type=int, default=20000)
    parser.add_argument(
        "--counters", type=int, default=2, help="counter tiles of the group"
    )
    parser.add_argument("--deadline", type=float, default=5.0, help="the run, seconds")
    parser.add_argument("--wait", type=float, default=1.0, help="wait limit, seconds")
    return parser.parse_args()


def options(args):
    return {
        "max_live": args.max_live,
        "queue_messages": args.queue_messages,
        "bins": args.bins,
        "idle_cycles": args.idle_cycles,
        "counters": args.counters,
    }


def main():
    args = parse()
    try:  # the launcher's rejections, before any device opens
        Dynamic(handler=FORWARD["kHandlerForward"], **options(args)).check(
            args.count or args.counters + 1, 4 << 30, 4
        )
    except ValueError as error:
        print(
            json.dumps(power.clocked({"ok": False, "rejected": str(error)})), flush=True
        )
        sys.exit(1)
    stalls = [args.stall] * args.repeat + ([-1] if args.again else [])
    runs = []
    with km.Session(log=log) as session:
        log("device open")
        for index, stall in enumerate(stalls):
            runs.append(run_once(session, args, stall))
            log(f"run {index + 1} done")
        if args.engine:
            runs.append(run_engine(session, args))
            log("engine run done")
    ok = all(run_report["ok"] for run_report in runs)
    timed = [
        run_report["run_microseconds"]
        for run_report in runs
        if run_report.get("run_microseconds")
    ]
    summary = {
        "exact_runs": sum(bool(run_report.get("exact")) for run_report in runs),
        "runs": len(runs),
    }
    if timed:
        summary["run_microseconds"] = [
            min(timed),
            round(float(np.median(timed)), 1),
            max(timed),
        ]
    result = {"ok": ok, "summary": summary, "runs": runs}
    print(
        json.dumps(
            power.clocked(result | {"wall_s": round(time.monotonic() - STARTED, 1)})
        )
    )
    sys.exit(0 if ok else 1)


def l1_blocks(session, grid, words):
    """A zeroed L1 area of `words` words on every core, at one address."""
    zeros = np.zeros((grid[0] * grid[1], words), np.int32)
    return session.tensor(zeros, "int32", shard_grid=grid)


def load(session, args):
    """The program loaded: (run, results tensor, program, the session's ranks)."""
    grid = session.grid
    # The results area, made before the grid starts.
    results_tensor = l1_blocks(session, grid, FORWARD["kResWords"])
    program = km.Program(
        "forward",
        kernels={
            "reader": RUNTIME / "messages/reader.cpp",
            "compute": HERE / "kernels" / "compute.cpp",
            "writer": RUNTIME / "messages/writer.cpp",
        },
        args=lambda core: ([], [], [results_tensor.address]),
        handlers={"forward": FORWARD["kHandlerForward"]},
    ).dynamic("forward", **options(args))
    run = session.load(program, deadline=args.deadline, wait=args.wait)
    return run, results_tensor, program, session.ranks


def command_words(args, stall):
    return [
        args.messages,
        args.forwards,
        FORWARD["kModeOneCore"] if args.one_core >= 0 else FORWARD["kModeRandom"],
        max(args.one_core, 0),
        stall + 1,
        args.probe + 1,
    ]


def run_once(session, args, stall):
    """One load of the program: the RUN command, its end (or the timeout), the check."""
    run, results_tensor, program, cores = load(session, args)
    count = args.count or cores
    log(f"loaded: group of {count} of {cores} cores")
    words = command_words(args, stall)
    timed_out = None
    start = time.perf_counter()
    handle = run.submit("forward", (0, count), words, exclusive=True)
    try:
        run.drain(timeout=args.deadline + 5.0)
        host_seconds = time.perf_counter() - start
        run.stop()
    except km.CommandTimeout as error:  # a stall: km.MessagesGaveUp naming its core
        host_seconds = time.perf_counter() - start
        timed_out = error
        log(f"timeout: {error}")
    event, sequence = run.event(handle), run.sequence(handle)
    result = report(
        args,
        run,
        (event, sequence),
        program.messages,
        (count, cores),
        stall,
        results_tensor,
        timed_out,
    )
    result["host_s"] = round(host_seconds, 4)
    if stall < 0:  # exact, ended by the kernel at the quiet end
        result["ok"] = (
            result["exact"] and result["event"] == "done" and result["gave_core"] == -1
        )
    else:  # the stalled core named by the run's error
        named = isinstance(timed_out, km.MessagesGaveUp)
        result["named_core"] = timed_out.gave["core"] if named else None
        result["ok"] = result["event"] == "timeout" and result["named_core"] == stall
    return result


def run_engine(session, args):
    """One load: a stream on the engine submits the RUN to two halves of the grid one
    after the other; the second waits (rejected busy, then woken) for the first's end.
    """
    run, results_tensor, program, cores = load(session, args)
    half = cores // 2
    stream = Stream(req=21, deadline_microseconds=int(args.deadline * 1e6))
    for first in (0, half):
        stream.submit(
            FORWARD["kHandlerForward"],
            (first, half),
            args=command_words(args, -1),
            exclusive=True,
        )
    stream.wait(0)
    stream.halt(stream.const(7))
    handle = run.start(run.upload(stream))
    event = run.wait_engine(handle, timeout=2 * args.deadline + 5.0)
    run.stop()
    result = run.result()
    result_rows = (results_tensor.read().astype(np.int64) & 0xFFFFFFFF)[: 2 * half]
    exact = [
        exact_rows(args, result_rows[first : first + half], half) for first in (0, half)
    ]
    headers = abi.headers()
    x280_cores = result.get("x280_cores") or [0] * headers["KMD_H_WORDS"]
    return {
        "ok": event.kind == "end" and all(exact) and result["error_name"] == "none",
        "engine": event.kind,
        "halt": event.value,
        "halves_exact": exact,
        "runs": [
            int(result_rows[0, FORWARD["kResRun"]]),
            int(result_rows[half, FORWARD["kResRun"]]),
        ],
        "kernel": {key: result[key] for key in ("error_name", "commands", "rejected")},
        "checks": int(x280_cores[headers["KMD_H_MESSAGES_CHECKS"]]),
    }


def exact_rows(args, result_rows, count):
    """The result rows of a group of `count` cores against the replay over its cores;
    the counter tiles (the last a.counters rows) take no message."""
    workers = len(counter_tiles(0, count, args.counters).workers)
    one = args.one_core if args.one_core >= 0 else None
    consumed, total, visits = replay.replay(args.messages, args.forwards, workers, one)
    if args.probe >= 0:  # the probe's messages visit their own core once
        visits[args.probe] += result_rows[args.probe, FORWARD["kResSent"]]
    totals = result_rows[:workers, FORWARD["kResTotalLo"]] | (
        result_rows[:workers, FORWARD["kResTotalHi"]] << 32
    )
    return (
        bool((result_rows[:workers, FORWARD["kResConsumed"]] == consumed).all())
        and bool((totals.astype(np.uint64) == total).all())
        and bool((result_rows[:workers, FORWARD["kResVisits"]] == visits).all())
        and not result_rows[workers:, FORWARD["kResVisits"]].any()
    )


def report(args, run, event, messages, group, stall, results_tensor, timed_out):
    """The checks and numbers of one run as a dict: `event` (the RUN's event, its
    sequence), `group` (count, cores in the grid)."""
    (event, sequence), (count, _) = event, group
    headers = abi.headers()
    result = run.result()
    x280_cores = result.get("x280_cores") or [0] * headers["KMD_H_WORDS"]
    result_rows = (results_tensor.read().astype(np.int64) & 0xFFFFFFFF)[:count]
    counter_layout = counter_tiles(0, count, args.counters)
    states = run.state()
    gave = {
        state["core"]: state["dynamic"]["gave_up"]
        for state in states[:count]
        if state["dynamic"]["gave_up"]
    }
    numbers = {
        "count": count,
        "counters": [
            state["core"]
            for state in states[:count]
            if (state["dynamic"]["counter"] or {}).get("run") == sequence
        ],
        "counters_expected": list(counter_layout.counters),
        "messages": args.messages,
        "forwards_per_message": args.forwards,
        "stall": stall,
        "kernel": {key: result[key] for key in ("error_name", "commands", "events")},
        "event": event.kind if event else None,
        "run_microseconds": (
            round(event.microseconds, 1) if event and event.kind == "done" else None
        ),
        "exact": bool((result_rows[:, FORWARD["kResRun"]] == sequence).all())
        and exact_rows(args, result_rows, count),
        "consumed_by_compute": int(result_rows[:, FORWARD["kResConsumed"]].sum()),
        "expected_consumed": args.messages,
        "forwards": int(result_rows[:, FORWARD["kResVisits"]].sum()) - args.messages,
        "probe_sent": int(result_rows[:, FORWARD["kResSent"]].sum()),
        "rejected": int(result_rows[:, FORWARD["kResRejected"]].sum()),
        "quota_start": messages.quota(count),  # max_live // the run's ranks
        "quota_left_min": int(
            result_rows[: len(counter_layout.workers), FORWARD["kResQuota"]].min()
        ),
        "checks": int(x280_cores[headers["KMD_H_MESSAGES_CHECKS"]]),
        "gave_core": int(x280_cores[headers["KMD_H_MESSAGES_GAVE"]]) - 1,
        "messages_gave": gave,
    }
    if timed_out is not None:
        numbers["timed_out"] = {
            "handle": timed_out.handle,
            "sequence": timed_out.sequence,
            "group": list(timed_out.group),
            "not_reported": timed_out.cores[:16],
        }
        numbers["core_gave_up"] = [
            {
                "core": state["core"],
                **{key: state[key] for key in ("reader", "writer", "compute")},
            }
            for state in states
            if state["reader"] or state["writer"] or state["compute"]
        ][:16]
    return numbers


if __name__ == "__main__":
    main()
