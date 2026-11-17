"""A gather by a token list, item to item on the movers, against numpy: core r copies
the items its list names (item `token` of the table to item `slot` of out), for items
of 32 B, 64 B and 2 KiB. One command a size; its us and GB/s; see README.md.
python experiments/language/item_gather/item_gather.py --cards 0"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi.bench import power

WORDS = 1024  # uint32 words a page (4,096 B)
TOKENS = 64  # a core's token list
CORES = 8
TABLE = 64  # the table's pages: 256 KiB
SIZES = (32, 64, 2048)  # an item's bytes, a handler each


STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", file=sys.stderr, flush=True)


class Token(km.Entry):
    token: int  # the table's item
    slot: int  # out's item


class Gather(km.Program):
    """Core r reads page r of ids, TOKENS entries of (token, slot), and copies table
    item `token` to out item `slot`: mover A reads it, mover B writes it."""

    @km.handler
    def small(self, ids: km.Tensor, table: km.Tensor, out: km.Tensor):
        for e in ids.pages(self.rank, 1).entries(Token, size=2, fixed=TOKENS):
            results = km.Stream(out.items(e.slot, 1, size=32), write=True)
            for item in km.Stream(table.items(e.token, 1, size=32)):
                results.push(item)

    @km.handler
    def medium(self, ids: km.Tensor, table: km.Tensor, out: km.Tensor):
        for e in ids.pages(self.rank, 1).entries(Token, size=2, fixed=TOKENS):
            results = km.Stream(out.items(e.slot, 1, size=64), write=True)
            for item in km.Stream(table.items(e.token, 1, size=64)):
                results.push(item)

    @km.handler
    def large(self, ids: km.Tensor, table: km.Tensor, out: km.Tensor):
        for e in ids.pages(self.rank, 1).entries(Token, size=2, fixed=TOKENS):
            results = km.Stream(out.items(e.slot, 1, size=2048), write=True)
            for item in km.Stream(table.items(e.token, 1, size=2048)):
                results.push(item)


HANDLERS = dict(zip(SIZES, (Gather.small, Gather.medium, Gather.large), strict=True))


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0", help="one card")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def lists(rng, size):
    """Each core's (token, slot) words: tokens drawn over the table's items, slots
    core r's TOKENS items of out in order."""
    items = TABLE * WORDS * 4 // size
    ids = np.zeros((CORES, WORDS), np.uint32)
    tokens = rng.integers(0, items, (CORES, TOKENS))
    slots = np.arange(CORES * TOKENS).reshape(CORES, TOKENS)
    ids[:, 0 : 2 * TOKENS : 2] = tokens
    ids[:, 1 : 2 * TOKENS : 2] = slots
    return ids, tokens.reshape(-1)


def main():
    args = parse()
    card = int(args.cards)
    rng = np.random.default_rng(args.seed)
    table = rng.integers(1, 1 << 32, (TABLE, WORDS), dtype=np.uint64).astype(np.uint32)
    out_pages = CORES * TOKENS * max(SIZES) // (WORDS * 4)
    cases = []
    with km.Session(log=log, cards=[card]) as session:
        log(f"START card {card}, {CORES} cores x {TOKENS} tokens, sizes {SIZES}")
        lists_of = {size: lists(rng, size) for size in SIZES}
        given = {
            "ids": session.tensor(lists_of[SIZES[0]][0], "uint32"),
            "table": session.tensor(table, "uint32"),
            "out": session.tensor(np.zeros((out_pages, WORDS), np.uint32), "uint32"),
        }
        program = Gather(log=log, session=session, **given)
        run = session.load(program, wait=0.5, deadline=0.5)
        for size in SIZES:
            ids, tokens = lists_of[size]
            session.write(given["ids"], ids.view(np.uint8))  # pages as rows of bytes
            session.write(given["out"], np.zeros((out_pages, WORDS * 4), np.uint8))
            call = (given["ids"], given["table"], given["out"])
            submitted = run.submit(HANDLERS[size], (0, CORES), args=call)
            event = run.wait(submitted, timeout=30.0)
            if event.kind != "done":
                raise RuntimeError(f"{size} B: {event}")
            us = round(float(event.microseconds), 2)
            (raw,) = session.read(given["out"])
            words = size // 4
            got = raw.view(np.uint32).reshape(-1)[: CORES * TOKENS * words]
            want = table.reshape(-1, words)[tokens].reshape(-1)
            copied = CORES * TOKENS * size
            cases.append(
                {
                    "size": size,
                    "us": us,
                    "bytes": copied,
                    "GB/s": round(copied / us / 1e3, 3),
                    "bad_words": int(np.count_nonzero(got != want)),
                }
            )
            log(json.dumps(cases[-1]))
        run.stop()
        result = run.result()
    summary = {
        "card": card,
        "cores": CORES,
        "tokens": TOKENS,
        "cases": cases,
        "kernel": result.get("error_name", "none"),
        "bad_words": sum(c["bad_words"] for c in cases),
    }
    print(json.dumps(summary), flush=True)
    print(json.dumps(power.clocked({"card": card}, card=card)), flush=True)
    return 0 if summary["bad_words"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
