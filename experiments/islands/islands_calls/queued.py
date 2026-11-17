"""Queued commands (`run.submit(..., queued=True)`) on one card or two: JOBS commands of
COUNT cores each, the kernel choosing the cores; each core adds job + 1 to word 0 of
its page (job x COUNT + its rank in the group). Checked: the pages against one card's,
each ticket's flat group inside one island's ranks. Exits 1 on a failed check (README)."""

import argparse
import json
import sys
import time
from collections import Counter

import numpy as np

import komorebi as km

STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--jobs", type=int, default=64)
    parser.add_argument("--count", type=int, default=8, help="cores a command")
    return parser.parse_args()


class Put(km.Program):
    """put: core `rank` of the group adds job + 1 to word 0 of page job x size + rank
    (a page's word read and written: its first)."""

    @km.handler
    def put(self, out: km.Tensor, job: int):
        page = job * self.size + self.rank
        word = out.pages(page, 1).read()
        word += job + 1
        out.pages(page, 1).write(word)


def placed(session, groups):
    """Each ticket's island (None: its flat group is in no one island's ranks); one
    card (a card's own session): 0 when it is in the card's cores."""
    if not hasattr(session, "graph"):
        cores = session.grid[0] * session.grid[1]
        return [0 if f is not None and f + c <= cores else None for f, c in groups]
    graph, out = session.graph, []
    for first, count in groups:
        held = [
            island
            for island in range(len(session.sessions))
            if first is not None
            and graph.first(island) <= first
            and first + count <= graph.first(island) + graph.count(island)
        ]
        out.append(held[0] if held else None)
    return out


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    report = {"cards": cards, "jobs": args.jobs, "count": args.count}
    opened = {"card": cards[0]} if len(cards) == 1 else {"cards": cards}
    with km.Session(**opened, log=log) as session:  # one card: its own Run
        out = session.tensor(
            np.zeros((args.jobs * args.count, 16), np.uint32), "uint32"
        )
        run = session.load(Put(out=out, log=log), deadline=2.0)
        log(f"loaded; {args.jobs} queued PUTs of {args.count} cores")
        tickets = [
            run.submit(Put.put, args.count, args=(out, job), queued=True)
            for job in range(args.jobs)
        ]
        run.drain(timeout=30.0)  # the events kept: event() and group() read them
        events = [run.event(ticket) for ticket in tickets]
        groups = [run.group(ticket) for ticket in tickets]
        run.forget(tickets)
        groups = [g if isinstance(g, tuple) else (None, 0) for g in groups]  # mock
        report["not_done"] = sum(not event or event.kind != "done" for event in events)
        islands = placed(session, groups)
        report["by_island"] = dict(sorted(Counter(map(str, islands)).items()))
        report["groups_first"] = [first for first, _ in groups[:8]]
        stopped = run.stop()
        blocks = getattr(stopped, "islands", [stopped])  # a block an island
        report["kernel_errors"] = [block["error_name"] for block in blocks]
        log(f"stopped: {report['kernel_errors']}; by island {report['by_island']}")
        got = out.read().astype(np.int64)
    want = np.repeat(np.arange(1, args.jobs + 1), args.count)  # word 0 of each page
    report["pages_wrong"] = int(
        (got[:, 0] != want).sum() + got[:, 1:].any(axis=1).sum()
    )
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    report["ok"] = bool(
        report["not_done"] == 0
        and report["pages_wrong"] == 0
        and "None" not in report["by_island"]
        and all(error == "none" for error in report["kernel_errors"])
    )
    print(json.dumps(report), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
