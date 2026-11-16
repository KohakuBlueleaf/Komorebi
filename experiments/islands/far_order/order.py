"""Does a far rank ever see an add before the put sent ahead of it by the same CPU?
For each case S:T, in ONE command, island S's first rank puts a page into island T's
DRAM and posts km::add_word to island T's last rank, which reads that page the moment
its counter rises and checks every word. One JSON line; README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import islands
from komorebi.bench import power

KERNELS = Path(__file__).resolve().parent / "kernels"
CLEAR, PAIR, REPORT = 1, 2, 3  # kHandler* (order.h)
PAGE = 4096  # kAreaBytes: the area, a data page and a report page
# order.h's area words
COUNTER, READY, DONE, EARLY, STALE, FIRST, GAVE_UP, SEEN = 0, 1, 2, 3, 4, 5, 7, 8
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--card", type=int, help="one card split into --islands")
    parser.add_argument("--islands", type=int, default=0)
    parser.add_argument("--graph", default="0-1,1-2", help="--islands' pairs")
    parser.add_argument("--cases", default="0:1,1:0", help="S:T island pairs")
    parser.add_argument("--rounds", type=int, default=100000)
    parser.add_argument("--pages", type=int, default=64, help="data pages, in turn")
    parser.add_argument("--most", type=float, default=8.0, help="a side's seconds")
    return parser.parse_args()


def page_of(session, tensor, page):
    """Page `page` of a report tensor as words, read while the run is loaded."""
    (got,) = session.read(tensor, first=page, pages=1)
    return got.view(np.uint32).reshape(-1)


def case(args, run, session, graph, s, t, data, reports):
    """One case: CLEAR both ranks, PAIR over the ranks between them, REPORT both."""
    sender = graph.first(s)
    watcher = graph.first(t) + graph.count(t) - 1
    record = {"case": f"{s}:{t}", "sender": sender, "watcher": watcher}
    record["hops"] = graph.hops(s, t)
    timeout = 2 * args.most + 10

    def one(island, handler, rank, words):
        part = run.island(island)
        event = part.wait(part.submit(handler, (rank, 1), words), timeout=timeout)
        if event.kind != "done":
            raise km.KomorebiError(f"{handler} on rank {rank}: {event}")

    one(t, CLEAR, watcher, [])
    one(s, CLEAR, sender, [])
    low, high = min(sender, watcher), max(sender, watcher)
    most = int(args.most * 1000)
    words = [watcher, args.rounds, data[t].address, args.pages, sender, most]
    event = run.wait(run.submit(PAIR, (low, high - low + 1), words), timeout=timeout)
    record["us"] = round(event.microseconds, 1)
    record["event"] = event.kind
    one(t, REPORT, watcher, [reports[t].address, 0])
    one(s, REPORT, sender, [reports[s].address, 1])
    seen = page_of(session, reports[t], 0)
    sent = page_of(session, reports[s], 1)
    record["sent"] = int(sent[DONE])
    record["sender_gave_up"] = int(sent[GAVE_UP])
    record["counter"] = int(seen[COUNTER])
    record["rises"] = int(seen[SEEN])
    record["checked"], record["early"] = int(seen[DONE]), int(seen[EARLY])
    record["stale"] = int(seen[STALE])
    record["watcher_gave_up"] = int(seen[GAVE_UP])
    if record["early"]:
        record["first_early"] = {
            "round": int(seen[FIRST]),
            "word0": int(seen[FIRST + 1]),
        }
    record["us_a_round"] = round(record["us"] / max(args.rounds, 1), 3)
    record["ok"] = (
        record["event"] == "done"
        and record["sent"] == args.rounds
        and record["counter"] == args.rounds
        and record["checked"] > 0
        and record["early"] == 0
        and record["sender_gave_up"] == 0
        and record["watcher_gave_up"] == 0
    )
    return record


def main():
    args = parse()
    if args.card is not None:
        graph = islands.parse_pairs(args.graph)
        opened = {"card": args.card, "islands": args.islands, "graph": graph}
    else:
        opened = {"cards": [int(card) for card in args.cards.split(",")]}
    cases = [tuple(int(x) for x in c.split(":")) for c in args.cases.split(",")]
    report = {"opened": opened, "rounds": args.rounds, "pages": args.pages}
    report["cases"] = []
    ok = True
    with km.Session(**opened, log=log) as session:
        graph = session.graph
        count = len(graph.islands)
        own = [(graph.first(i), graph.count(i)) for i in range(count)]
        data = [
            session.empty((args.pages, PAGE // 4), "uint32", group=own[i])
            for i in range(count)
        ]
        reports = [
            session.empty((2, PAGE // 4), "uint32", group=own[i]) for i in range(count)
        ]
        program = km.Program(
            "far_order",
            kernels=KERNELS,
            handlers=KERNELS / "order.h",
            buffers=[
                km.Buffer(index=0, format="bfloat16", page_bytes=PAGE, pages=1),
                km.Buffer(index=1, format="bfloat16", page_bytes=PAGE, pages=1),
            ],
            compile_args={"reader": reports[0].accessor_args()},
        )
        run = session.load(program, deadline=2 * args.most + 5)
        try:
            for s, t in cases:
                log(f"case {s}:{t}: {args.rounds} rounds")
                try:
                    record = case(args, run, session, graph, s, t, data, reports)
                except km.KomorebiError as error:
                    record = {"case": f"{s}:{t}", "ok": False}
                    record["error"] = f"{type(error).__name__}: {error}"[:800]
                log(json.dumps(record))
                report["cases"].append(record)
                ok &= record["ok"]
                if "error" in record:
                    break
        finally:
            try:
                report["kernel_errors"] = [r["error_name"] for r in run.stop()]
            except km.KomorebiError as error:
                report["kernel_errors"] = str(error)[:300]
                ok = False
    report["ok"] = bool(ok) and set(report["kernel_errors"]) == {"none"}
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(power.clocked(report)), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
