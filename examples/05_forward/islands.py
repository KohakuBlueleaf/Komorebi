"""The forward example over two cards (`Session(cards=[0, 1])`) or a split card: the
same kernels, one dynamic run over every rank of every island, a forward to another
island crossing through the link cores; every message rank's totals and the messages
that crossed each link against the numpy replay. One JSON line. See README.md."""

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
from komorebi import power  # noqa: E402

FORWARD = feeder_commands.constants(HERE / "kernels" / "forward.h")
RUNTIME = ROOT / "src" / "komorebi_native" / "kernels"
LINK_WORDS = 52  # the link region's words before its packet headers (km_link.hpp)
LINK_MESSAGE_BYTES = 64  # the forward example's messages
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument(
        "--islands", type=int, default=0, help="one card split into this many islands"
    )
    parser.add_argument(
        "--line", action="store_true", help="the islands in a line (neighbours only)"
    )
    parser.add_argument(
        "--links",
        type=int,
        default=1,
        help="links between the cards (1 to 4)",
    )
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
    parser.add_argument(
        "--message-links",
        action="store_true",
        help="a message service on every link (opt-in), else link 0's alone",
    )
    parser.add_argument(
        "--hold-links",
        action="store_true",
        help="two cards: hold_links=True (every rank, links its own); else deliveries",
    )
    parser.add_argument(
        "--kernel-bytes",
        type=int,
        default=None,
        help="kernel buffer bytes (default 163840)",
    )
    parser.add_argument(
        "--group", default=None, help="the run's group, first,count in flat ranks"
    )
    parser.add_argument(
        "--pages", type=int, default=0, help="pages written across during the run"
    )
    parser.add_argument("--deadline", type=float, default=5.0, help="the run, seconds")
    parser.add_argument("--wait", type=float, default=1.0, help="wait limit, seconds")
    return parser.parse_args()


def group_parts(session, group):
    """Per island, (its first rank of the group, the group's ranks there)."""
    parts = []
    for i in range(len(session.sessions)):
        low = max(group[0], session.graph.first(i))
        high = min(group[0] + group[1], session.graph.first(i) + session.graph.count(i))
        parts.append((low, max(high - low, 0)))
    return parts


def message_ranks(session, counters, group):
    """Per island, its ranks of the group that are message ranks: all but its part's
    last `counters` (its counter tiles), numbered on from the island before; link cores
    are no ranks."""
    return [max(n - counters, 0) for _, n in group_parts(session, group)]


def open_session(args, cards):
    """Two cards joined by `--links` links, or one card split into `--islands`
    islands joined by wires (`--line`: neighbours only); the session's default kernel
    buffer."""
    if args.islands:
        (card,) = cards
        return km.Session(card=card, islands=args.islands, line=args.line, log=log)
    return km.Session(
        cards=cards,
        links=args.links,
        kernel_bytes=args.kernel_bytes,
        message_links=args.message_links,
        log=log,
    )


def results_areas(session):
    """The results area on every core of every island, at one address (a split
    card's: one area of the card)."""
    count = 1 if session.split > 1 else len(session.sessions)
    areas = [
        session.island(i).tensor(
            np.zeros((np.prod(session.island(i).grid), FORWARD["kResWords"]), np.int32),
            "int32",
            shard_grid=session.island(i).grid,
        )
        for i in range(count)
    ]
    if len({tensor.address for tensor in areas}) != 1:
        raise km.KomorebiError("the results areas are not at one address")
    return areas


def island_rows(session, areas):
    """Each island's results rows, its cores in order."""
    rows = [
        (tensor.read().astype(np.int64) & 0xFFFFFFFF).reshape(-1, FORWARD["kResWords"])
        for tensor in areas
    ]
    if session.split > 1:  # one card: the island's own cores of it
        return [rows[0][list(island.cores)] for island in session.graph.islands]
    return rows


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    report = {"cards": cards, "messages": args.messages, "forwards": args.forwards}
    with open_session(args, cards) as session:
        islands = len(session.sessions)
        report["islands"] = [
            [session.graph.first(i), session.graph.count(i)] for i in range(islands)
        ]
        results = results_areas(session)
        address = results[0].address
        group = (
            (0, session.ranks)
            if args.group is None
            else tuple(int(v) for v in args.group.split(","))
        )
        report["group"] = list(group)
        source = target = data = None
        if args.pages:  # island 0's pages into island 1's tensor, during the run
            own = [(session.graph.first(i), session.graph.count(i)) for i in (0, 1)]
            data = np.arange(args.pages * 1024, dtype=np.uint32).reshape(args.pages, -1)
            source = session.tensor(data, "uint32", group=own[0])
            target = session.empty(data.shape, "uint32", group=own[1])
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
            hold_links=args.hold_links,
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
        before = run.crossings()
        start = time.perf_counter()
        ticket = run.submit("forward", group, words, exclusive=True)
        if args.pages:
            run.send()  # the run started: the pages share its links
            try:
                crossed = session.cross(source, target, timeout=args.deadline)
                report["pages_crossed"] = crossed["pages"]
            except km.KomorebiError as error:
                report["pages_error"] = str(error)[:400]
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
        if done:  # link 0 serves other commands again once the RUN has ended
            moved = run.crossings() - before
            log(f"crossed:\n{moved}")
            report["messages_crossed"] = [  # each link, both ways
                moved.total(a, b)["messages"]
                for a in range(islands)
                for b in session.graph.neighbours(a)
            ]
            report["messages_by_link"] = {  # link, from island, to island
                f"{link} {source}->{target}": row["messages"]
                for (link, source, target), row in sorted(moved.rows.items())
            }
        try:
            results_end = run.stop() if done else None
            if results_end is not None:
                report["kernel_errors"] = [one["error_name"] for one in results_end]
        except km.KomorebiError as error:
            report["stop_error"] = str(error)[:400]
        if not done:
            run.abort()
            report["link"] = [link_words(session, i) for i in range(islands)]
        report.update(check(args, session, island_rows(session, results), group))
        if args.pages:
            (got,) = session.read(target)
            wrong = (got.view(np.uint32).reshape(data.shape) != data).any(axis=1)
            report["pages_wrong"] = int(wrong.sum())
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    report["crossings_agree"] = (
        sum(report.get("messages_crossed", [-1])) == report["crossings_replayed"]
    )
    report["ok"] = bool(
        report["exact"]
        and report["crossings_agree"]
        and report["events"] == ["done"] * len(report["islands"])
        and report.get("kernel_errors") == ["none"] * len(report["islands"])
        and report.get("pages_wrong", 0) == 0
        and "pages_error" not in report
    )
    print(json.dumps(power.clocked(report)), flush=True)
    sys.exit(0 if report["ok"] else 1)


def link_words(session, island):
    """Each message service's region words on `island` (counts, give-ups, stamps), then
    its state line: one row a service, link core 0's first CPU first."""
    if island >= len(session._regions):
        return []  # no message service (deliveries): no region
    region = session._regions[island]
    one = session.island(island)
    columns, rows = one.grid
    stride = getattr(session, "_stride", 0)
    entry = 16 + LINK_MESSAGE_BYTES
    state = 0x190 + 128 * entry + 5 * 4352 + 0x20 + 2 * 16 * 256 + 16 * 16 + 16
    out = []
    for cpu in range(2 if stride else 1):
        for offset, words in ((0, LINK_WORDS), (state, 4)):
            view = one.empty(
                (columns * rows, words),
                "uint32",
                shard_grid=one.grid,
                address=region.address + cpu * stride + offset,
            )
            got = view.read().astype(np.int64) & 0xFFFFFFFF
            got = got.reshape(columns * rows, words)
            for k, core in enumerate(session.link_cores[island]):
                out.append([k, cpu, offset, *got[core].tolist()])
    return out


def check(args, session, rows, group):
    """Every message rank's consumed count, total and visits against the replay over
    all message ranks; every other core (counter tiles, ranks outside the group)
    visited by none."""
    ranks = message_ranks(session, args.counters, group)
    workers = sum(ranks)
    one = args.one_core if args.one_core >= 0 else None
    consumed, total, visits = replay.replay(args.messages, args.forwards, workers, one)
    starts = [
        low - session.graph.first(i)
        for i, (low, _) in enumerate(group_parts(session, group))
    ]
    got = np.concatenate([one[at : at + n] for one, at, n in zip(rows, starts, ranks)])
    reserved = np.concatenate(
        [
            np.delete(one, np.arange(at, at + n), axis=0)
            for one, at, n in zip(rows, starts, ranks)
        ]
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
    if one is None:  # forwards whose next rank is on another island, replayed
        across = crossings(args.messages, args.forwards, workers, split)
    return {
        "message_ranks": ranks,
        "exact": exact,
        "consumed": int(got[:, FORWARD["kResConsumed"]].sum()),
        "visits": int(got[:, FORWARD["kResVisits"]].sum()),
        "visits_by_island": [  # each island's part of the group
            int(one[at : at + n, FORWARD["kResVisits"]].sum())
            for one, at, n in zip(rows, starts, ranks)
        ],
        "crossings_replayed": across,
    }


def crossings(messages, forwards, count, split):
    """The replay's forwards to another island: one delivery each (counted on the link
    it leaves by, however many links it crosses)."""
    state = (np.arange(messages, dtype=np.uint64) + 1) * 0x9E3779B1 & 0xFFFFFFFF
    state[state == 0] = 1
    core = replay.first_rank(messages, count)
    total = 0
    for _hop in range(forwards):
        state = replay.xorshift(state)
        after = state % count
        apart = np.abs(
            np.searchsorted(split, core, "right").astype(np.int64)
            - np.searchsorted(split, after, "right")
        )
        total += int((apart != 0).sum())
        core = after
    return total


if __name__ == "__main__":
    main()
