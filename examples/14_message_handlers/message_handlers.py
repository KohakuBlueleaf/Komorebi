"""The message handlers example on a card through the dispatcher on X280 cluster 0: one
dynamic run with three dynamic handlers; each thread changes handler as it moves and
reports at its origin; every core's counts against the numpy replay. See README.md."""

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
from komorebi.bench import power  # noqa: E402
from komorebi.layout import counter_tiles  # noqa: E402
from komorebi.program import Dynamic  # noqa: E402

WORDS = km.abi.constants(
    HERE / "kernels" / "handlers.h", known=km.abi.headers().constants
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
    parser.add_argument("--threads", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=10, help="lookups per thread")
    parser.add_argument("--repeat", type=int, default=1, help="runs, each a new load")
    parser.add_argument("--max-live", type=int, default=1 << 14)
    parser.add_argument("--queue-messages", type=int, default=1 << 14)
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
        "counters": args.counters,
    }


def main():
    args = parse()
    try:  # the launcher's rejections, before any device opens
        Dynamic(handler=1, handlers=(1, 2, 3), **options(args)).check(
            args.count or args.counters + 1, 4 << 30, 4
        )
    except ValueError as error:
        print(
            json.dumps(power.clocked({"ok": False, "rejected": str(error)})), flush=True
        )
        sys.exit(1)
    runs = []
    with km.Session(log=log) as session:
        log("device open")
        for index in range(args.repeat):
            runs.append(run_once(session, args))
            log(f"run {index + 1}: {runs[-1]['line']}")
    timed = [one["run_microseconds"] for one in runs if one["run_microseconds"]]
    summary = {"exact": sum(one["exact"] for one in runs), "runs": len(runs)}
    if timed:
        summary["run_microseconds"] = [
            min(timed),
            round(float(np.median(timed)), 1),
            max(timed),
        ]
    ok = all(one["ok"] for one in runs)
    result = {"ok": ok, "summary": summary, "runs": runs}
    print(
        json.dumps(
            power.clocked(result | {"wall_s": round(time.monotonic() - STARTED, 1)})
        )
    )
    sys.exit(0 if ok else 1)


def run_once(session, args):
    """One load of the program, its RUN command (the seeds' handler: lookup), the end
    and the check."""
    grid = session.grid
    cores = grid[0] * grid[1]
    results_tensor = session.tensor(
        np.zeros((cores, WORDS["kResWords"]), np.int32), "int32", shard_grid=grid
    )
    # --8<-- [start:program]
    program = km.Program(
        "message_handlers",
        kernels={
            "reader": RUNTIME / "messages/reader.cpp",
            "compute": HERE / "kernels" / "compute.cpp",
            "writer": RUNTIME / "messages/writer.cpp",
        },
        args=lambda core: ([], [], [results_tensor.address]),
        handlers=HERE / "kernels" / "handlers.h",
    ).dynamic("lookup", "value", "report", **options(args))
    # --8<-- [end:program]
    run = session.load(program, deadline=args.deadline, wait=args.wait)
    count = args.count or session.ranks  # the grid less a Tensix dispatcher's
    handle = run.submit(
        "lookup", (0, count), [args.threads, args.steps], exclusive=True
    )
    timed_out = None
    try:
        run.drain(timeout=args.deadline + 5.0)
        run.stop()
    except km.CommandTimeout as error:
        timed_out = str(error)
        log(f"timeout: {error}")
    event, sequence = run.event(handle), run.sequence(handle)
    rows = (results_tensor.read().astype(np.int64) & 0xFFFFFFFF)[:count]
    states = run.state()[:count]
    return check(args, count, rows, (event, sequence), states, timed_out)


def check(args, count, rows, event, states, timed_out):
    """The run's numbers against the replay, as a dict."""
    event, sequence = event
    workers = len(counter_tiles(0, count, args.counters).workers)
    expected = replay.replay(args.threads, args.steps, workers)
    got = {
        "lookups": rows[:workers, WORDS["kResLookups"]],
        "values": rows[:workers, WORDS["kResValues"]],
        "reports": rows[:workers, WORDS["kResReports"]],
        "own": rows[:workers, WORDS["kResOwn"]],
        "probes": rows[:workers, WORDS["kResProbes"]],
        "sum": (
            rows[:workers, WORDS["kResSumLo"]]
            | rows[:workers, WORDS["kResSumHi"]] << 32
        ).astype(np.uint64),
    }
    differ = [name for name in got if not (got[name] == expected[name]).all()]
    exact = (
        not differ
        and bool((rows[:, WORDS["kResRun"]] == sequence).all())
        and not rows[:, WORDS["kResWrong"]].any()
        and not rows[:, WORDS["kResRejected"]].any()
        and not rows[workers:, : WORDS["kResReports"] + 1].any()
    )
    handled = int(sum(got[name].sum() for name in ("lookups", "values", "reports")))
    microseconds = (
        round(event.microseconds, 1) if event and event.kind == "done" else None
    )
    heads = [state["dynamic"]["head"] for state in states[:workers]]
    gave = {
        state["core"]: state["dynamic"]["gave_up"]
        for state in states
        if state["dynamic"]["gave_up"]
    }
    result = {
        "count": count,
        "threads": args.threads,
        "steps": args.steps,
        "event": event.kind if event else None,
        "run_microseconds": microseconds,
        "exact": bool(exact),
        "differ": differ,
        "messages": handled,
        "by_handler": [
            int(got[name].sum()) for name in ("lookups", "values", "reports")
        ],
        "own_core_forwards": int(got["own"].sum()),
        "messages_per_second": (
            round(handled / microseconds * 1e6) if microseconds else None
        ),
        "queue_messages": args.queue_messages,
        "head_max": max(heads),
        "laps_max": max(heads) // args.queue_messages,
        "messages_gave": gave,
        "timed_out": timed_out,
    }
    result["ok"] = result["exact"] and result["event"] == "done" and not gave
    result["line"] = (
        f"{handled} messages ({result['by_handler'][0]} lookup, "
        f"{result['by_handler'][1]} value, {result['by_handler'][2]} report; "
        f"{result['own_core_forwards']} to their own core), "
        f"{'exact' if exact else 'NOT exact ' + str(differ)}, {microseconds} us, "
        f"{result['messages_per_second']} messages/s"
    )
    return result


if __name__ == "__main__":
    main()
