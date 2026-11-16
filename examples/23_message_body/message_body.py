"""The message body example on a card through the dispatcher on X280 cluster 0: one
dynamic run in which each thread is one message with a body, forwarded from rank to
rank and changed at every hop; every body against the numpy replay, the time a hop
beside a message without a body. See README.md."""

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

import commands as feeder_commands  # noqa: E402
import replay  # noqa: E402

import komorebi as km  # noqa: E402
from komorebi import abi, overlay, power  # noqa: E402
from komorebi.layout import counter_tiles  # noqa: E402
from komorebi.program import Dynamic  # noqa: E402

WORDS = feeder_commands.constants(HERE / "kernels" / "chain.h")
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
    parser.add_argument("--threads", type=int, default=1, help="at most the ranks")
    parser.add_argument("--hops", type=int, default=1200, help="arrivals per thread")
    parser.add_argument(
        "--bodies",
        type=int,
        nargs="+",
        default=[0, 4096],
        help="body bytes of each case (0: the 64-byte message alone)",
    )
    parser.add_argument(
        "--keep", action="store_true", help="also each body carried unchanged"
    )
    parser.add_argument(
        "--parts",
        action="store_true",
        help="the handler as an overlay part under the runtime's kernels",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="with --parts: the thread brings chain by name (start is the RUN's)",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="with --parts: each hop by direct() to rank + 1, a forward when busy",
    )
    parser.add_argument("--repeat", type=int, default=1, help="runs of each case")
    parser.add_argument(
        "--flush-age", type=int, default=None, help="the writer's (Dynamic's default)"
    )
    parser.add_argument("--max-live", type=int, default=128)
    parser.add_argument("--queue-messages", type=int, default=128)
    parser.add_argument(
        "--counters", type=int, default=2, help="counter tiles of the group"
    )
    parser.add_argument("--deadline", type=float, default=5.0, help="the run, seconds")
    parser.add_argument("--wait", type=float, default=1.0, help="wait limit, seconds")
    return parser.parse_args()


def options(args, body_bytes):
    return {
        "max_live": args.max_live,
        "queue_messages": args.queue_messages,
        "counters": args.counters,
        "body_bytes": body_bytes,
        "direct": (1,) if args.direct else (),
    } | ({} if args.flush_age is None else {"flush_age": args.flush_age})


def main():
    args = parse()
    if args.direct and (not args.parts or args.send):
        raise SystemExit("--direct is parts/chain.cpp's: give it with --parts alone")
    cases = [(body, True) for body in args.bodies]
    cases += [(body, False) for body in args.bodies if body and args.keep]
    try:  # the launcher's rejections, before any device opens
        for body, _ in cases:
            Dynamic(handler=WORDS["kHandlerChain"], **options(args, body)).check(
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
        for body, change in cases:
            for index in range(args.repeat):
                runs.append(run_once(session, args, body, change))
                log(
                    f"body {body}, change {change}, run {index + 1}: {runs[-1]['line']}"
                )
    ok = all(one["ok"] for one in runs)
    summary = {}
    for body, change in cases:
        timed = [
            one["hop_microseconds"]
            for one in runs
            if (one["body_bytes"], one["change"]) == (body, change)
            and one["hop_microseconds"]
        ]
        if timed:
            summary[f"body {body}{'' if change else ' kept'}"] = {
                "hop_microseconds_median": round(float(np.median(timed)), 3),
                "runs": len(timed),
            }
    result = {"ok": ok, "summary": summary, "runs": runs}
    print(
        json.dumps(
            power.clocked(result | {"wall_s": round(time.monotonic() - STARTED, 1)})
        )
    )
    sys.exit(0 if ok else 1)


def run_once(session, args, body_bytes, change):
    """One load of the program with `body_bytes`, its RUN command, the end and the
    check."""
    grid = session.grid
    cores = grid[0] * grid[1]
    area = session.tensor(
        np.zeros((cores, WORDS["kAreaWords"]), np.int32), "int32", shard_grid=grid
    )
    if area.address % 64:
        raise SystemExit(f"the area at {area.address:#x} is not 64-byte aligned")
    # --8<-- [start:program]
    program = km.Program(
        "message_body",
        kernels={
            "reader": RUNTIME / "messages_reader.cpp",
            "compute": HERE / "kernels" / "compute.cpp",
            "writer": RUNTIME / "messages_writer.cpp",
        },
        args=lambda core: ([], [], [area.address]),
        handlers={"chain": WORDS["kHandlerChain"]},
    ).dynamic("chain", **options(args, body_bytes))
    # --8<-- [end:program]
    parts = HERE / "kernels" / "parts"
    if args.parts and not args.send:  # the same handler as a part: parts/chain.cpp
        program = km.Program(
            "message_body_parts",
            kernels=None,
            args=lambda core: ([], [], [area.address]),
            handlers={"chain": WORDS["kHandlerChain"]},
        ).dynamic("chain", parts=parts, **options(args, body_bytes))
    if args.send:  # start on every core; chain loaded where the thread arrives
        program = km.Program(
            "message_body_send",
            kernels=None,
            args=lambda core: ([], [], [area.address]),
            handlers={
                "start": WORDS["kHandlerStart"],
                "chain": WORDS["kHandlerChain"],
            },
        ).dynamic("start", "chain", parts=parts, **options(args, body_bytes))
    run = session.load(program, deadline=args.deadline, wait=args.wait)
    count = args.count or cores
    words = [args.threads, args.hops, body_bytes, int(change), int(args.direct)]
    handle = run.submit(
        "start" if args.send else "chain", (0, count), words, exclusive=True
    )
    timed_out = None
    try:
        run.drain(timeout=args.deadline + 5.0)
        run.stop()
    except km.CommandTimeout as error:
        timed_out = str(error)
        log(f"timeout: {error}")
    event, sequence = run.event(handle), run.sequence(handle)
    rows = (area.read().astype(np.int64) & 0xFFFFFFFF)[:count]
    states = run.state()[:count]
    result = check(args, (count, body_bytes, change), rows, (event, sequence), states)
    if args.direct:
        result["direct_hops"] = int(rows[:, WORDS["kResDirect"]].sum())
    if args.send:
        result |= laps(args, run, rows, count, result)
    return result | {"timed_out": timed_out}


def laps(args, run, rows, count, result):
    """--send: the handlers each core loaded (its tile's epoch steps three a handler:
    start, then chain where the thread arrived), and rank 0's lap clock in cycles."""
    log("reading the overlay areas")
    blocks = run.backend.control.read().astype(np.int64) & 0xFFFFFFFF
    epoch = (
        run.backend.control_offset
        + (abi.headers()["KM_CONTROL_BYTES"] + overlay.constants()["kOverlayEpoch"])
        // 4
    )
    loads = (blocks[:count, epoch] & overlay.constants()["kOverlayEpochMask"]) // 3
    ranks = result["ranks"]
    stamps = rows[0, WORDS["kResStamps"] : WORDS["kResStamps"] + WORDS["kStamps"]]
    lap_cycles = (np.diff(stamps[: WORDS["kStamps"] - 1]) & 0xFFFFFFFF).tolist()
    last_lap = (args.hops - 1) // ranks
    span = int((stamps[-1] - stamps[0]) & 0xFFFFFFFF)
    mhz = span / (last_lap * ranks * result["hop_microseconds"])
    first_lap, later = lap_cycles[0], float(np.median(lap_cycles[1:]))
    hop = later / ranks
    first_arrival = hop + (first_lap - later) / (ranks - 1)
    return {
        "loads_per_core": loads[:ranks].tolist(),
        "counter_loads": loads[ranks:].tolist(),
        "lap_cycles": lap_cycles,
        "clock_mhz_from_run": round(mhz, 1),
        "later_arrival_cycles": round(hop, 1),
        "first_arrival_cycles": round(first_arrival, 1),
        "later_arrival_us": round(hop / mhz, 3),
        "first_arrival_us": round(first_arrival / mhz, 3),
    }


def check(args, case, rows, event, states):
    """The run's numbers against the replay, as a dict."""
    (count, body_bytes, change), (event, sequence) = case, event
    ranks = len(counter_tiles(0, count, args.counters).workers)
    threads = min(args.threads, ranks)
    body_words = body_bytes // 4
    arrivals, last, state, body = replay.replay(
        threads, args.hops, ranks, body_words, change, 0 if args.send else 1
    )
    ended = np.zeros(ranks, dtype=np.int64)
    ended[last] = np.arange(threads) + 1
    differ = []
    if not (rows[:ranks, WORDS["kResArrivals"]] == arrivals).all():
        differ.append("arrivals")
    if not (rows[:ranks, WORDS["kResThread"]] == ended).all():
        differ.append("thread")
    if not (rows[last, WORDS["kResHop"]] == args.hops).all():
        differ.append("hop")
    if not (rows[last, WORDS["kResState"]] == state.astype(np.int64)).all():
        differ.append("state")
    if body_words and not (rows[last, :body_words] == body.astype(np.int64)).all():
        differ.append("body")
    run_rows = rows[:ranks] if args.parts else rows  # a part never runs on a counter
    if not (run_rows[:, WORDS["kResRun"]] == sequence).all():
        differ.append("run")
    if rows[ranks:, WORDS["kResArrivals"]].any():
        differ.append("counter tiles")
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
        "ranks": ranks,
        "threads": threads,
        "hops": args.hops,
        "body_bytes": body_bytes,
        "change": change,
        "parts": args.parts,
        "event": event.kind if event else None,
        "run_microseconds": microseconds,
        "hop_microseconds": (
            round(microseconds / args.hops, 3) if microseconds else None
        ),
        "exact": not differ,
        "differ": differ,
        "body_words_checked": int(threads * body_words),
        "messages_gave": gave,
    }
    result["ok"] = result["exact"] and result["event"] == "done" and not gave
    result["line"] = (
        f"{threads} thread(s) x {args.hops} hops over {ranks} ranks, "
        f"{'exact' if not differ else 'NOT exact ' + str(differ)}, "
        f"{microseconds} us, {result['hop_microseconds']} us a hop"
    )
    return result


if __name__ == "__main__":
    main()
