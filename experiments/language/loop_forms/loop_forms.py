"""Class lines whose object is made at one loop depth and used at another, on the
card, each checked word for word against numpy: --form total, a running total written
a page a turn; --form held, a weight cast once and read by two products each turn.
Example: python loop_forms.py --form total --cores 8 --turns 4"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi import formats

STARTED = time.monotonic()
K = 2  # tiles a turn adds
WORDS = 1024  # values a tile


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class RunningTotal(km.Program):
    """prefix: each core's sum so far written a page a turn."""

    @km.handler
    def prefix(self, x: km.Tensor, out: km.Tensor, turns: int):
        total = km.tile.zeros(km.fp32)
        for i in range(turns):
            for tile in km.Stream(x.pages((self.rank * turns + i) * K, K)):
                total += tile
            out.pages(self.rank * turns + i, 1).write(total)


KW, NW = 8, 4  # held: the weight's K and N tiles
RECT = (0, 0, 2, 4)  # held: two columns of four cores, a cast down each column


@km.tile_function
def project(x, w):
    return x @ w


class HeldWeight(km.Program):
    """two: the weight cast once down each column, held, and read by two products a
    turn (rows 2i and 2i + 1 of the core's x), each written a turn."""

    @km.handler
    def two(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, turns: int):
        column = km.Broadcast(self, along="column", window=1, mover="A")
        if column.sender():
            ws = column.cast(w.tiles[0:KW, 0:NW].read())
        else:
            ws = column.arrive()
        for i in range(turns):
            r = self.rank * 2 * turns + 2 * i
            acc = km.tile.zeros(km.fp32, NW)
            for xb in km.Stream(x.tiles[r : r + 1, 0:KW], block=2, pattern="B"):
                acc += project(xb, ws)
            y.pages(r * NW, NW).write(acc)
            again = km.tile.zeros(km.fp32, NW)
            for xb in km.Stream(x.tiles[r + 1 : r + 2, 0:KW], block=2, pattern="B"):
                again += project(xb, ws)
            y.pages((r + 1) * NW, NW).write(again)


def run_held(session, turns, seed):
    """HeldWeight.two on RECT: y = x @ w word for word (k / 8 values: every product
    and sum exact in fp32)."""
    rng = np.random.default_rng(seed)
    cores = RECT[2] * RECT[3]
    rows = cores * 2 * turns * 32
    x_values = (rng.integers(-16, 16, (rows, KW * 32)) / 8).astype(np.float32)
    w_values = (rng.integers(-16, 16, (KW * 32, NW * 32)) / 8).astype(np.float32)
    x = session.tensor(x_values, "bfloat16", "tile")
    w = session.tensor(w_values, "bfloat16", "tile")
    y = session.empty((rows, NW * 32), "float32", "tile")
    program = HeldWeight(x=x, w=w, y=y, rect=RECT, session=session, log=log)
    report = {"cores": cores, "turns": turns}
    loaded = session.load(program)
    try:
        group = program.rectangle["group"]
        event = loaded.wait(
            loaded.submit(HeldWeight.two, group, args=(x, w, y, turns)), timeout=5.0
        )
        report["event"], report["us"] = event.kind, event.microseconds
    finally:
        loaded.stop()
    report["kernel_text"] = program.sizes.text
    (got,) = session.read(y)
    got = formats.untilize(got.view(np.float32), rows, NW * 32)
    want = x_values @ w_values
    report["words"] = int(want.size)
    report["differ"] = int((got != want).sum())
    report["ok"] = report["differ"] == 0 and report["event"] == "done"
    return report


def dyadic(rng, pages):
    """Values k / 16 in [-8, 8): exact in bf16 and summed exactly in fp32 in any
    order, so every word is comparable; their sums need more bits than tf32 keeps."""
    return (rng.integers(-128, 128, (pages, WORDS)) / 16).astype(np.float32)


def bf16_bytes(values):
    return (values.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)


def run_total(session, cores, turns, seed):
    rng = np.random.default_rng(seed)
    pages = cores * turns * K
    values = dyadic(rng, pages)
    x = session.empty((pages * 32, 32), "bfloat16", "tile")
    session.write(x, bf16_bytes(values).reshape(pages, -1))
    prefix = session.empty((cores * turns * 32, 32), "float32", "tile")
    program = RunningTotal(x=x, out=prefix, session=session, log=log)
    report = {"cores": cores, "turns": turns}
    loaded = session.load(program)
    try:
        event = loaded.wait(
            loaded.submit(RunningTotal.prefix, (0, cores), args=(x, prefix, turns)),
            timeout=5.0,
        )
        report["event"], report["us"] = event.kind, event.microseconds
    finally:
        loaded.stop()
    report["kernel_text"] = program.sizes.text
    (got,) = session.read(prefix)
    turn_sums = values.reshape(cores, turns, K, WORDS).sum(axis=2)
    want = np.cumsum(turn_sums, axis=1).reshape(cores * turns, WORDS)
    got = got.view(np.float32).reshape(want.shape)
    report["words"] = int(want.size)
    report["differ"] = int((got != want).sum())
    report["ok"] = report["differ"] == 0 and report["event"] == "done"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--form", choices=("total", "held"), default="total")
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    with km.Session(log=log) as session:
        if args.form == "held":
            report = run_held(session, args.turns, args.seed)
        else:
            report = run_total(session, args.cores, args.turns, args.seed)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
