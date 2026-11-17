"""A tile or a reduce's sum written to two tensors, and written on the branches of an
if, on the card against numpy; then a command whose branch writes nothing, followed by
one whose result must be right. A JSON line a case, the summary last; see README.md.
python experiments/language/write_twice/write_twice.py --cards 0"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi.bench import power

WORDS = 1024  # float32 words a page: one tile's
CORES = 16

STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", file=sys.stderr, flush=True)


class WriteTwice(km.Program):
    """tile, summed: a rank's page (or the group's sum) to t and to y; word, leader:
    to t or y by a command word (under the leader's if for the sum); only: the sum
    to t when the word is 0, nothing else. README.md has each handler's pages."""

    @km.handler
    def tile(self, s: km.Tensor, t: km.Tensor, y: km.Tensor):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        t.pages(self.rank, 1).write(mine)
        y.pages(self.rank, 1).write(mine)

    @km.handler
    def summed(self, s: km.Tensor, t: km.Tensor, y: km.Tensor):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        whole = km.reduce(mine)
        if self.rank == self.leader:
            t.pages(0, 1).write(whole)
            y.pages(0, 1).write(whole)

    @km.handler
    def word(self, s: km.Tensor, t: km.Tensor, y: km.Tensor, which: int):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        if which == 0:
            t.pages(self.rank, 1).write(mine)
        else:
            y.pages(self.rank, 1).write(mine)

    @km.handler
    def leader(self, s: km.Tensor, t: km.Tensor, y: km.Tensor, which: int):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        whole = km.reduce(mine)
        if self.rank == self.leader:
            if which == 0:
                t.pages(0, 1).write(whole)
            else:
                y.pages(0, 1).write(whole)

    @km.handler
    def only(self, s: km.Tensor, t: km.Tensor, y: km.Tensor, which: int):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        whole = km.reduce(mine)
        if self.rank == self.leader:  # noqa: SIM102  (the leader's, then the word's)
            if which == 0:
                t.pages(0, 1).write(whole)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0", help="one card, or two: 0,1")
    parser.add_argument("--copy", action="store_true", help="y a copy on each card")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--kernel-bytes", type=int)
    return parser.parse_args()


def cores_of(session, count, cards):
    """`count` cores from 0, or on two cards the ones meeting at card 1's first."""
    if len(cards) == 1:
        return (0, count)
    edge = session.graph.first(1)
    return (edge - count // 2, count)


def bad(got, want):
    """Words of `got` not equal to `want` (integer sums: exact)."""
    return int(np.count_nonzero(got != want))


def done(run, submitted, what):
    event = run.wait(submitted, timeout=30.0)
    if event.kind != "done":
        raise RuntimeError(f"{what}: {event}")
    return round(float(event.microseconds), 2)


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    count = CORES
    rng = np.random.default_rng(args.seed)
    s = rng.integers(-50, 51, (count, WORDS)).astype(np.float32)
    total = s.sum(0)
    zeros = np.zeros((count, WORDS), np.float32)
    opened = {"cards": cards}
    if args.kernel_bytes:
        opened["kernel_bytes"] = args.kernel_bytes
    cases = []
    with km.Session(log=log, **opened) as session:
        cores = cores_of(session, count, cards)
        log(f"START cards {cards}, cores {cores}, y a copy: {args.copy}")
        st = session.tensor(s, "float32")
        t = session.tensor(zeros, "float32")
        y = session.tensor(zeros, "float32", copy=args.copy)
        program = WriteTwice(log=log, session=session, s=st, t=t, y=y)
        run = session.load(program, wait=0.5, deadline=0.5)

        def case(what, handler, which, want_t, want_y):
            t.write(zeros)
            y.write(zeros)
            words = () if which is None else (which,)
            submitted = run.submit(handler, cores, args=(st, t, y, *words))
            us = done(run, submitted, what)
            checks = {"t": bad(t.read(), want_t), "y": bad(y.read(), want_y)}
            cases.append({"case": what, "us": us, "bad_words": checks})
            log(json.dumps(cases[-1]))

        first = zeros.copy()
        first[0] = total
        case("a rank's tile to t and y", WriteTwice.tile, None, s, s)
        case("the sum to t and y", WriteTwice.summed, None, first, first)
        case("word 0: to t", WriteTwice.word, 0, s, zeros)
        case("word 1: to y", WriteTwice.word, 1, zeros, s)
        case("leader, word 0: the sum to t", WriteTwice.leader, 0, first, zeros)
        case("leader, word 1: the sum to y", WriteTwice.leader, 1, zeros, first)
        case("only, word 1: nothing written", WriteTwice.only, 1, zeros, zeros)
        case("only, word 0, after it: the sum to t", WriteTwice.only, 0, first, zeros)
        case("the sum to t and y, after both", WriteTwice.summed, None, first, first)
        run.stop()
        result = run.result()  # a list over islands, a card's
        results = result if isinstance(result, list) else [result]
    summary = {
        "cards": cards,
        "cores": list(cores),
        "copy": args.copy,
        "cases": cases,
        "kernel": [r.get("error_name", "none") for r in results],
        "bad_words": sum(sum(c["bad_words"].values()) for c in cases),
    }
    print(json.dumps(summary), flush=True)
    print(json.dumps(power.clocked({"cards": cards}, card=cards[0])), flush=True)
    return 0 if summary["bad_words"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
