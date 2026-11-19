"""A decode's weight reads alone, on one card and on two cards' islands: each core of a
group streams its K share of its group's W columns and drops the tiles. The same reads,
the same cores a card, through the one-card path and through the islands' tensor table.
Example: python reads.py --cards 0; python reads.py --cards 0,1"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "language" / "bounds"))

import bounds

import komorebi as km

STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class Reads(km.Program):
    """bounds.Split's reads of W without the product, the tree or Y's write."""

    K_TILES = km.constant(168)
    SPLIT = km.constant(8)
    WIDTH = km.constant(28)
    CHUNK = km.constant(2)
    W_BLOCK = CHUNK * WIDTH

    @km.handler
    def reads(self, w: km.Tensor, col0: int):
        share = self.rank % self.SPLIT
        first = col0 + self.rank // self.SPLIT * self.WIDTH
        k0 = share * self.K_TILES // self.SPLIT
        k1 = (share + 1) * self.K_TILES // self.SPLIT
        for _tile in km.Stream(
            w.tiles[k0:k1, first : first + self.WIDTH], block=self.W_BLOCK
        ):
            pass


def one_product(session, name, fmt, repeat):
    """The reads of product `name`'s W in `fmt`: each card's half on two cards; on
    one card a W of half the columns read by the same groups. Its record."""
    shape = bounds.PRODUCTS[name]
    graph = getattr(session, "graph", None)
    kt, nt = shape["k"] // 32, shape["n"] // 32
    half = nt // 2
    split = shape["split"][1]  # the two-card group size, on both paths
    cores = half // shape["width"] * split
    if graph is None:
        w = session.empty((shape["k"], half * 32), fmt, "tile")
        group = (0, cores)
    else:
        hints = bounds.hints_of(session, name)
        w = session.empty((shape["k"], shape["n"]), fmt, "tile", **hints["w"])
        group = (graph.first(1) - cores, 2 * cores)
    program = Reads(
        log=log, K_TILES=kt, SPLIT=split, WIDTH=shape["width"], session=session, w=w
    )
    run = session.load(program)
    times = []
    try:
        for _ in range(repeat):
            event = run.wait(run.submit(program.reads, group, args=(w, 0)), timeout=30)
            if event.kind != "done":
                raise km.KomorebiError(f"{name} {fmt}: {event}")
            times.append(event.microseconds)
    finally:
        run.stop()
    w.free()
    us = float(np.median(times))
    card_bytes = kt * half * bounds.PAGE[fmt]  # a card's half of W
    gbs = card_bytes / us / 1e3
    return {
        "product": name,
        "format": fmt,
        "path": "one card" if graph is None else "islands",
        "group": list(group),
        "us": round(us, 1),
        "times": [round(t, 1) for t in times],
        "card_gbs": round(gbs, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0")
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()
    cards = [int(card) for card in args.cards.split(",")]
    opened = {"cards": cards} if len(cards) > 1 else {"card": cards[0]}
    records = []
    with km.Session(log=log, **opened) as session:
        for fmt in ("bfp8_b", "bfp4_b"):
            for name in ("gate", "down"):
                record = one_product(session, name, fmt, args.repeat)
                log(json.dumps(record))
                records.append(record)
    print(json.dumps({"records": len(records)}), flush=True)


if __name__ == "__main__":
    main()
