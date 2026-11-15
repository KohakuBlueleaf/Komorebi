"""The message overlays example on a card: the three handlers of the message handlers
example, each an overlay part that a core loads when a message for it arrives; every
core's counts against the numpy replay. See README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]

import replay  # noqa: E402

import komorebi as km  # noqa: E402
from komorebi import abi, overlay  # noqa: E402
from komorebi.layout import counter_tiles  # noqa: E402
from komorebi.program import Dynamic  # noqa: E402
from komorebi.session import aiclk_mhz  # noqa: E402

KERNELS = HERE / "kernels"
WORDS = abi.constants(KERNELS / "handlers.h", known=abi.headers().constants)
STRAY = 9  # a handler the program does not have (--stray)
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
    parser.add_argument("--slots", type=int, default=3, help="compute slots, 1 to 3")
    parser.add_argument("--repeat", type=int, default=1, help="runs, each a new load")
    parser.add_argument("--max-live", type=int, default=1 << 14)
    parser.add_argument("--queue-messages", type=int, default=1 << 14)
    parser.add_argument(
        "--counters", type=int, default=2, help="counter tiles of the group"
    )
    parser.add_argument("--deadline", type=float, default=5.0, help="the run, seconds")
    parser.add_argument("--wait", type=float, default=1.0, help="wait limit, seconds")
    parser.add_argument(
        "--loads", action="store_true", help="count each core's loads after the run"
    )
    parser.add_argument(
        "--stray",
        action="store_true",
        help=f"rank 0 also seeds a message for handler {STRAY}: the run must stop",
    )
    parser.add_argument(
        "--addon",
        action="store_true",
        help="report is not loaded: run.addon adds it to the live run before RUN",
    )
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
        print(json.dumps({"ok": False, "rejected": str(error)}), flush=True)
        sys.exit(1)
    runs = []
    with km.Session(log=log) as session:
        log("device open")
        mhz = [aiclk_mhz(session.card)]
        for index in range(args.repeat):
            runs.append(run_once(session, args))
            log(f"run {index + 1}: {runs[-1]['line']}")
        mhz.append(aiclk_mhz(session.card))
    timed = [one["run_microseconds"] for one in runs if one["run_microseconds"]]
    summary = {"exact": sum(one["exact"] for one in runs), "runs": len(runs)}
    if timed:
        summary["run_microseconds"] = [
            min(timed),
            round(float(np.median(timed)), 1),
            max(timed),
        ]
    ok = all(one["ok"] for one in runs)
    result = {"ok": ok, "aiclk_mhz": mhz, "summary": summary, "runs": runs}
    print(json.dumps(result | {"wall_s": round(time.monotonic() - STARTED, 1)}))
    sys.exit(0 if ok else 1)


def run_once(session, args):
    """One load of the program, its RUN command (the seeds' handler: lookup), the end
    and the check."""
    grid = session.grid
    cores = grid[0] * grid[1]
    results = session.tensor(
        np.zeros((cores, WORDS["kResWords"]), np.int32), "int32", shard_grid=grid
    )
    loaded = ("lookup", "value") if args.addon else ("lookup", "value", "report")
    # --8<-- [start:program]
    program = km.Program(
        "message_overlays",
        kernels=None,  # the runtime's: each handler's body is a part
        args=lambda core: ([], [], [results.address]),  # the handlers' state in L1
        handlers=KERNELS / "handlers.h",
    ).dynamic(*loaded, parts=KERNELS, **options(args))
    program.slots(count=args.slots, bytes={} if args.addon else None)  # addon: path
    # --8<-- [end:program]
    run = session.load(program, deadline=args.deadline, wait=args.wait)
    if args.addon:  # messages of handler 3 run the part added here, as overlay 3
        report = {"compute": KERNELS / "report.cpp"}
        number = run.addon(
            "report", report, message=True, number=WORDS["kHandlerReport"]
        )
        log(f"report added as dynamic handler {number}: {run.added['report']}")
    count = args.count or cores
    handle = run.submit(
        "lookup",
        (0, count),
        [args.threads, args.steps, STRAY if args.stray else 0],
        exclusive=True,
    )
    timed_out = None
    try:
        run.drain(timeout=args.deadline + 5.0)
        run.stop()
    except km.CommandTimeout as error:
        timed_out = str(error)
        log(f"timeout: {error}")
    event, sequence = run.event(handle), run.sequence(handle)
    rows = (results.read().astype(np.int64) & 0xFFFFFFFF)[:count]
    states = run.state()[:count]
    loads = core_loads(run, count) if args.loads else None
    return check(args, count, rows, (event, sequence), states, timed_out, loads)


def core_loads(run, count):
    """Handlers each core loaded in the run: its tile's epoch steps once for each part
    it reads, three a handler (the overlay area, read back after the stop)."""
    log("reading the overlay areas")
    rows = run.backend.control.read().astype(np.int64) & 0xFFFFFFFF
    epoch = (
        run.backend.control_offset
        + (abi.headers()["KM_CONTROL_BYTES"] + overlay.constants()["kOverlayEpoch"])
        // 4
    )
    return (rows[:count, epoch] & overlay.constants()["kOverlayEpochMask"]) // 3


def check(args, count, rows, event, states, timed_out, loads):
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
        and bool((rows[:workers, WORDS["kResRun"]] == sequence).all())
        and not rows[:, WORDS["kResWrong"]].any()
        and not rows[:, WORDS["kResRejected"]].any()
        and not rows[workers:].any()
    )
    handled = int(sum(got[name].sum() for name in ("lookups", "values", "reports")))
    microseconds = (
        round(event.microseconds, 1) if event and event.kind == "done" else None
    )
    gave = {
        state["core"]: state["dynamic"]["gave_up"]
        for state in states
        if state["dynamic"]["gave_up"]
    }
    result = {
        "count": count,
        "slots": args.slots,
        "addon": args.addon,
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
        "messages_gave": gave,
        "timed_out": timed_out,
    }
    if loads is not None:
        result["loads"] = {
            "total": int(loads.sum()),
            "per_core": [int(loads.min()), float(np.median(loads)), int(loads.max())],
        }
    result["ok"] = result["exact"] and result["event"] == "done" and not gave
    if args.stray:  # the run stopped where the stray message arrived, and said why
        stray = gave.get(0, {})
        result["ok"] = (
            result["event"] != "done"
            and stray.get("where") == "handler"
            and stray.get("seen", 0) >> 16 & 0xFF == STRAY
        )
    result["line"] = (
        f"{handled} messages ({result['by_handler'][0]} lookup, "
        f"{result['by_handler'][1]} value, {result['by_handler'][2]} report; "
        f"{result['own_core_forwards']} to their own core), "
        f"{'exact' if exact else 'NOT exact ' + str(differ)}, {microseconds} us, "
        f"{result['messages_per_second']} messages/s"
        + (f", {result['loads']['total']} loads" if loads is not None else "")
        + (f", gave up: {gave}" if gave else "")
    )
    return result


if __name__ == "__main__":
    main()
