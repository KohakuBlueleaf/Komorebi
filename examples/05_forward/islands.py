"""The forward example over two cards joined by a link (`Session(cards=..., links=1)`):
the same program and kernels, one dynamic run over every rank of both islands, so a
forward to a rank of the other island crosses through the islands' link cores; every
message rank's totals against the numpy replay. Prints one JSON line. See README.md."""

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

FORWARD = feeder_commands.constants(HERE / "kernels" / "forward.h")
RUNTIME = ROOT / "src" / "komorebi_native" / "kernels"
LINK_WORDS = 52  # the link region's words before its packet headers (km_link.hpp)
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--messages", type=int, default=16)
    parser.add_argument("--forwards", type=int, default=4, help="forwards per message")
    parser.add_argument(
        "--one-core", type=int, default=-1, help="every forward to this rank"
    )
    parser.add_argument("--max-live", type=int, default=1 << 14)
    parser.add_argument("--queue-messages", type=int, default=1 << 14)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--idle-cycles", type=int, default=20000)
    parser.add_argument("--counters", type=int, default=2)
    parser.add_argument("--deadline", type=float, default=5.0, help="the run, seconds")
    parser.add_argument("--wait", type=float, default=1.0, help="wait limit, seconds")
    return parser.parse_args()


def message_ranks(session, counters):
    """Per island, its ranks that are message ranks: all but its last `counters` (its
    counter tiles), numbered on from the island before; link cores are no ranks."""
    return [session.graph.count(i) - counters for i in range(len(session.sessions))]


def cores(session, island):
    """Island `island`'s cores: its ranks, then its link cores."""
    columns, rows = session.island(island).grid
    return columns * rows


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    report = {"cards": cards, "messages": args.messages, "forwards": args.forwards}
    # The message service on the link core makes the kernels 78.6 KB: past the default.
    with km.Session(cards=cards, links=1, kernel_bytes=80 * 1024, log=log) as session:
        islands = len(session.sessions)
        report["islands"] = [
            [session.graph.first(i), session.graph.count(i)] for i in range(islands)
        ]
        # The results area on every core of every island, at one address.
        results = [
            session.island(i).tensor(
                np.zeros((cores(session, i), FORWARD["kResWords"]), np.int32),
                "int32",
                shard_grid=session.island(i).grid,
            )
            for i in range(islands)
        ]
        if len({tensor.address for tensor in results}) != 1:
            raise km.KomorebiError("the results areas are not at one address")
        address = results[0].address
        program = km.Program(
            "forward",
            kernels={
                "reader": RUNTIME / "messages_reader.cpp",
                "compute": HERE / "kernels" / "compute.cpp",
                "writer": RUNTIME / "messages_writer.cpp",
            },
            args=lambda core: ([], [], [address]),
            handlers={"forward": FORWARD["kHandlerForward"]},
        ).dynamic(
            "forward",
            max_live=args.max_live,
            queue_messages=args.queue_messages,
            bins=args.bins,
            idle_cycles=args.idle_cycles,
            counters=args.counters,
        )
        run = session.load(program, deadline=args.deadline, wait=args.wait)
        report["link_cores"] = [session.link_cores[i][0] for i in range(islands)]
        log(f"loaded on {islands} islands, link cores {report['link_cores']}")
        words = [
            args.messages,
            args.forwards,
            FORWARD["kModeOneCore"] if args.one_core >= 0 else FORWARD["kModeRandom"],
            max(args.one_core, 0),
            0,  # no stall
            0,  # no probe
        ]
        start = time.perf_counter()
        ticket = run.submit("forward", (0, session.ranks), words, exclusive=True)
        try:
            run.drain(timeout=args.deadline + 5.0)
            report["timeout"] = None
        except km.CommandTimeout as error:
            report["timeout"] = str(error)[:400]
            log(f"timeout: {report['timeout']}")
        report["host_s"] = round(time.perf_counter() - start, 4)
        events = run.event(ticket)
        report["events"] = [event.kind if event else None for event in events]
        report["run_microseconds"] = [
            round(event.microseconds, 1) if event and event.kind == "done" else None
            for event in events
        ]
        log(f"events {report['events']}, {report['run_microseconds']} us")
        done = report["events"] == ["done"] * islands
        try:
            results_end = run.stop() if done else None
            if results_end is not None:
                report["kernel_errors"] = [one["error_name"] for one in results_end]
        except km.KomorebiError as error:
            report["stop_error"] = str(error)[:400]
        if not done:
            run.abort()
            report["link"] = [link_words(session, i).tolist() for i in range(islands)]
        rows = [
            (tensor.read().astype(np.int64) & 0xFFFFFFFF).reshape(
                cores(session, i), FORWARD["kResWords"]
            )
            for i, tensor in enumerate(results)
        ]
        report.update(check(args, session, rows))
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    report["ok"] = bool(
        report["exact"]
        and report["events"] == ["done"] * len(cards)
        and report.get("kernel_errors") == ["none"] * len(cards)
    )
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


def link_words(session, island):
    """The link core's region words on `island` (counts, give-ups, stamps)."""
    region = session._regions[island]
    one = session.island(island)
    columns, rows = one.grid
    view = one.empty(
        (columns * rows, LINK_WORDS),
        "uint32",
        shard_grid=one.grid,
        address=region.address,
    )
    got = view.read().astype(np.int64) & 0xFFFFFFFF
    return got.reshape(columns * rows, LINK_WORDS)[session.link_cores[island][0]]


def check(args, session, rows):
    """Every message rank's consumed count, total and visits against the replay over
    all message ranks; the reserved cores visited by none."""
    ranks = message_ranks(session, args.counters)
    workers = sum(ranks)
    one = args.one_core if args.one_core >= 0 else None
    consumed, total, visits = replay.replay(args.messages, args.forwards, workers, one)
    got = np.concatenate(
        [island_rows[:count] for island_rows, count in zip(rows, ranks)]
    )
    reserved = np.concatenate(
        [island_rows[count:] for island_rows, count in zip(rows, ranks)]
    )
    totals = got[:, FORWARD["kResTotalLo"]] | (got[:, FORWARD["kResTotalHi"]] << 32)
    exact = (
        bool((got[:, FORWARD["kResConsumed"]] == consumed).all())
        and bool((totals.astype(np.uint64) == total).all())
        and bool((got[:, FORWARD["kResVisits"]] == visits).all())
        and not reserved[:, FORWARD["kResVisits"]].any()
    )
    split = np.cumsum([0, *ranks])
    across = 0
    if one is None:  # forwards whose next rank is on the other island, replayed
        across = crossings(args.messages, args.forwards, workers, split)
    return {
        "message_ranks": ranks,
        "exact": exact,
        "consumed": int(got[:, FORWARD["kResConsumed"]].sum()),
        "visits": int(got[:, FORWARD["kResVisits"]].sum()),
        "visits_by_island": [
            int(island_rows[:count, FORWARD["kResVisits"]].sum())
            for island_rows, count in zip(rows, ranks)
        ],
        "crossings_replayed": across,
    }


def crossings(messages, forwards, count, split):
    """How many of the replay's forwards go from one island to the other."""
    state = (np.arange(messages, dtype=np.uint64) + 1) * 0x9E3779B1 & 0xFFFFFFFF
    state[state == 0] = 1
    core = replay.first_rank(messages, count)
    total = 0
    for _hop in range(forwards):
        state = replay.xorshift(state)
        after = state % count
        total += int(
            (
                np.searchsorted(split, core, "right")
                != np.searchsorted(split, after, "right")
            ).sum()
        )
        core = after
    return total


if __name__ == "__main__":
    main()
