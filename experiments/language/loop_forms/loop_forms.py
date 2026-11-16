"""Class lines whose object is made at one loop depth and used at another, on the
card, each against a proven form and float64: --form total (a running total, written
and reduced a turn), held (a weight cast once, two products a turn), turns (a weight
filled each turn; one reduce's sum written each turn). Example: --form turns"""

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
    """prefix: each core's sum so far written a page a turn; reduced: that sum
    reduced over the cores a turn, the leader writing the turn's page."""

    @km.handler
    def prefix(self, x: km.Tensor, out: km.Tensor, turns: int):
        total = km.tile.zeros(km.fp32)
        for i in range(turns):
            for tile in km.Stream(x.pages((self.rank * turns + i) * K, K)):
                total += tile
            out.pages(self.rank * turns + i, 1).write(total)

    @km.handler
    def reduced(self, x: km.Tensor, out: km.Tensor, turns: int):
        total = km.tile.zeros(km.fp32)
        for i in range(turns):
            for tile in km.Stream(x.pages((self.rank * turns + i) * K, K)):
                total += tile
            whole = km.reduce(total, form="tree")
            if self.rank == self.leader:
                out.pages(i, 1).write(whole)


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
            r = (self.y * 2 + self.x) * 2 * turns + 2 * i  # the core's place in RECT
            acc = km.tile.zeros(km.fp32, NW)
            for xb in km.Stream(x.tiles[r : r + 1, 0:KW], block=2, pattern="B"):
                acc += project(xb, ws)
            y.pages(r * NW, NW).write(acc)
            again = km.tile.zeros(km.fp32, NW)
            for xb in km.Stream(x.tiles[r + 1 : r + 2, 0:KW], block=2, pattern="B"):
                again += project(xb, ws)
            y.pages((r + 1) * NW, NW).write(again)


class CastEachTurn(km.Program):
    """The reference: the same rows, one product a turn of a block cast in that turn
    (the right form before the held block), so the arithmetic is the same."""

    @km.handler
    def one(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, turns: int):
        column = km.Broadcast(self, along="column", window=1, mover="A")
        for i in range(2 * turns):
            if column.sender():
                ws = column.cast(w.tiles[0:KW, 0:NW].read())
            else:
                ws = column.arrive()
            r = (self.y * 2 + self.x) * 2 * turns + i  # the core's place in RECT
            acc = km.tile.zeros(km.fp32, NW)
            for xb in km.Stream(x.tiles[r : r + 1, 0:KW], block=2, pattern="B"):
                acc += project(xb, ws)
            y.pages(r * NW, NW).write(acc)


class CastOnB(km.Program):
    """The reference in broadcast_product's proven arrangement: the cast on mover B,
    x's stream on mover A, one product a turn."""

    @km.handler
    def one(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, turns: int):
        column = km.Broadcast(self, along="column", window=1)
        for i in range(2 * turns):
            if column.sender():
                ws = column.cast(w.tiles[0:KW, 0:NW].read())
            else:
                ws = column.arrive()
            r = (self.y * 2 + self.x) * 2 * turns + i  # the core's place in RECT
            acc = km.tile.zeros(km.fp32, NW)
            for xb in km.Stream(x.tiles[r : r + 1, 0:KW], block=2, pattern="A"):
                acc += project(xb, ws)
            y.pages(r * NW, NW).write(acc)


FK, FN = 2, 2  # turns: each turn's weight is FK x FN tiles
FW = 4  # its pages, FK * FN


class FillEachTurn(km.Program):
    """apply: weight i filled in turn i and read by that turn's product (form 3)."""

    weight: km.L1[km.bf16, FK, FN, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, turns: int):
        for i in range(turns):
            self.weight.fill(km.Stream(w.pages(i * FW, FW)))
            r = self.rank * turns + i
            acc = km.tile.zeros(km.fp32, FN)
            for block in km.Stream(x.pages(r * FK, FK), block=2):
                acc += project(block, self.weight)
            y.pages(r * FN, FN).write(acc)


class FillOnce(km.Program):
    """The reference: the field filled once outside every loop, one command a
    turn (`i` its word)."""

    weight: km.L1[km.bf16, FK, FN, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, turns: int, i: int):
        self.weight.fill(km.Stream(w.pages(i * FW, FW)))
        r = self.rank * turns + i
        acc = km.tile.zeros(km.fp32, FN)
        for block in km.Stream(x.pages(r * FK, FK), block=2):
            acc += project(block, self.weight)
        y.pages(r * FN, FN).write(acc)


class SumEachTurn(km.Program):
    """write: one reduce of the cores' sums, written by the leader to page i each
    turn (form 4)."""

    @km.handler
    def write(self, x: km.Tensor, out: km.Tensor, turns: int):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(x.pages(self.rank * K, K)):
            total += tile
        whole = km.reduce(total, form="tree")
        for i in range(turns):
            if self.rank == self.leader:
                out.pages(i, 1).write(whole)


class SumOnce(km.Program):
    """The reference: the same reduce written once."""

    @km.handler
    def write(self, x: km.Tensor, out: km.Tensor, turns: int):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(x.pages(self.rank * K, K)):
            total += tile
        whole = km.reduce(total, form="tree")
        if self.rank == self.leader:
            out.pages(0, 1).write(whole)


def submitted(session, program, handler, group, calls):
    """`program` loaded, `handler` submitted once per word list of `calls` in turn:
    (each command's event kind, its us)."""
    loaded = session.load(program)
    kinds, times = [], []
    try:
        for args in calls:
            event = loaded.wait(loaded.submit(handler, group, args=args), timeout=5.0)
            kinds.append(event.kind)
            times.append(event.microseconds)
    finally:
        loaded.stop()
    return kinds, times


def differences(got, want, name, report, rows=32):
    """`got` against `want`: the words that differ, where (by `rows` rows: a tile row,
    or a page of the sums), and the largest difference, into `report` under `name`."""
    wrong = got != want
    report[f"{name}_differ"] = int(wrong.sum())
    report[f"{name}_where"] = wrong.reshape(-1, rows * got.shape[1]).sum(1).tolist()
    report[f"{name}_largest"] = float(
        np.abs(got.astype(np.float64) - want.astype(np.float64)).max()
    )


def run_turns(session, cores, turns, seed, dump=None):
    """Form 3 (FillEachTurn against FillOnce, a command a turn) and form 4
    (SumEachTurn's pages against SumOnce's page), word for word, and each against
    float64."""
    rng = np.random.default_rng(seed)
    group = (0, cores)
    report = {"cores": cores, "turns": turns}
    rows = cores * turns * 32
    x_values = (rng.integers(-16, 16, (rows, FK * 32)) / 8).astype(np.float32)
    w_values = (rng.integers(-16, 16, (turns * FK * 32, FN * 32)) / 8).astype(
        np.float32
    )
    x = session.tensor(x_values, "bfloat16", "tile")
    w = session.tensor(w_values, "bfloat16", "tile")
    got = {}
    for name, cls, calls in (
        ("fill_each_turn", FillEachTurn, [None]),
        ("fill_once", FillOnce, list(range(turns))),
    ):
        y = session.empty((rows, FN * 32), "float32", "tile")
        program = cls(x=x, w=w, y=y, session=session, log=log)
        words = [(x, w, y, turns) if i is None else (x, w, y, turns, i) for i in calls]
        kinds, times = submitted(session, program, cls.apply, group, words)
        report[f"{name}_events"], report[f"{name}_us"] = kinds, times
        (raw,) = session.read(y)
        got[name] = formats.untilize(raw.view(np.float32), rows, FN * 32)
    # turn i of core c: x's row tile c * turns + i by weight i
    blocks = x_values.reshape(cores, turns, 32, FK * 32).astype(np.float64)
    weights = w_values.reshape(turns, FK * 32, FN * 32).astype(np.float64)
    want = np.einsum("ctrk,tkn->ctrn", blocks, weights).reshape(rows, FN * 32)
    differences(got["fill_each_turn"], got["fill_once"], "fill_vs_once", report)
    for name in ("fill_each_turn", "fill_once"):
        differences(got[name], want.astype(np.float32), f"{name}_vs_float64", report)

    values = dyadic(rng, cores * K)
    sums = session.empty((cores * K * 32, 32), "bfloat16", "tile")
    session.write(sums, bf16_bytes(values).reshape(cores * K, -1))
    for name, cls in (("sum_each_turn", SumEachTurn), ("sum_once", SumOnce)):
        out = session.empty((turns * 32, 32), "float32", "tile")
        program = cls(x=sums, out=out, session=session, log=log)
        kinds, times = submitted(
            session, program, cls.write, group, [(sums, out, turns)]
        )
        report[f"{name}_events"], report[f"{name}_us"] = kinds, times
        (raw,) = session.read(out)
        got[name] = raw.view(np.float32).reshape(turns, WORDS)
    whole = values.astype(np.float64).sum(axis=0)
    once = np.broadcast_to(got["sum_once"][0], (turns, WORDS))
    differences(got["sum_each_turn"], once, "sum_vs_once", report, rows=1)
    differences(
        got["sum_each_turn"],
        np.broadcast_to(whole, (turns, WORDS)).astype(np.float32),
        "sum_each_turn_vs_float64",
        report,
        rows=1,
    )
    if dump:
        np.savez_compressed(dump, x=x_values, w=w_values, sums=values, **got)
    events = sum(
        (
            report[f"{n}_events"]
            for n in ("fill_each_turn", "fill_once", "sum_each_turn", "sum_once")
        ),
        [],
    )
    report["ok"] = (
        report["fill_vs_once_differ"] == 0
        and report["sum_vs_once_differ"] == 0
        and all(e == "done" for e in events)
    )
    return report


def run_held(session, turns, seed, dump=None):
    """HeldWeight.two and the reference CastEachTurn.one on RECT, the same rows: y
    word for word between the two, each row write's words that differ from numpy's
    x @ w and its largest error, both against float64."""
    rng = np.random.default_rng(seed)
    cores = RECT[2] * RECT[3]
    rows = cores * 2 * turns * 32
    x_values = (rng.integers(-16, 16, (rows, KW * 32)) / 8).astype(np.float32)
    w_values = (rng.integers(-16, 16, (KW * 32, NW * 32)) / 8).astype(np.float32)
    x = session.tensor(x_values, "bfloat16", "tile")
    w = session.tensor(w_values, "bfloat16", "tile")
    report = {"cores": cores, "turns": turns}
    got = {}
    for name, cls, handler in (
        ("held", HeldWeight, HeldWeight.two),
        ("reference", CastEachTurn, CastEachTurn.one),
        ("cast_on_b", CastOnB, CastOnB.one),
    ):
        y = session.empty((rows, NW * 32), "float32", "tile")
        program = cls(x=x, w=w, y=y, rect=RECT, session=session, log=log)
        loaded = session.load(program)
        try:
            group = program.rectangle["group"]
            event = loaded.wait(
                loaded.submit(handler, group, args=(x, w, y, turns)), timeout=5.0
            )
            report[f"{name}_event"] = event.kind
            report[f"{name}_us"] = event.microseconds
        finally:
            loaded.stop()
        report[f"{name}_kernel_text"] = program.sizes.text
        (raw,) = session.read(y)
        got[name] = formats.untilize(raw.view(np.float32), rows, NW * 32)
    want = x_values.astype(np.float64) @ w_values.astype(np.float64)
    report["words"] = int(want.size)
    report["held_vs_reference_differ"] = int((got["held"] != got["reference"]).sum())
    report["reference_vs_cast_on_b_differ"] = int(
        (got["reference"] != got["cast_on_b"]).sum()
    )
    if dump:
        np.savez_compressed(dump, x=x_values, w=w_values, **got)
    for name in got:
        wrong = got[name] != want.astype(np.float32)
        report[f"{name}_vs_numpy_differ"] = int(wrong.sum())
        error = np.abs(got[name] - want)
        report[f"{name}_rms_rel"] = float(np.sqrt((error**2).mean() / (want**2).mean()))
        # each row write (a core's turn and product): its wrong words, largest error
        writes = wrong.reshape(cores, turns, 2, 32 * NW * 32).sum(axis=3)
        largest = error.reshape(cores, turns, 2, -1).max(axis=3)
        report[f"{name}_wrong_by_turn"] = writes.sum(axis=(0, 2)).tolist()
        report[f"{name}_wrong_by_product"] = writes.sum(axis=(0, 1)).tolist()
        report[f"{name}_largest_by_turn"] = largest.max(axis=(0, 2)).tolist()
        # where: the core (its place in RECT), the page of the write's 4, the face
        tiles = wrong.reshape(rows // 32, 2, 16, NW, 2, 16)
        report[f"{name}_wrong_by_core"] = wrong.reshape(cores, -1).sum(axis=1).tolist()
        report[f"{name}_wrong_by_page"] = tiles.sum(axis=(0, 1, 2, 4, 5)).tolist()
        report[f"{name}_wrong_by_face"] = (
            tiles.sum(axis=(0, 2, 3, 5)).reshape(-1).tolist()
        )
    report["ok"] = (
        report["held_vs_reference_differ"] == 0
        and report["held_event"] == report["reference_event"] == "done"
        and report["cast_on_b_event"] == "done"
    )
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
    reduced = session.empty((turns * 32, 32), "float32", "tile")
    program = RunningTotal(x=x, out=prefix, session=session, log=log)
    report = {"cores": cores, "turns": turns}
    loaded = session.load(program)
    try:
        for name, handler, out in (
            ("prefix", RunningTotal.prefix, prefix),
            ("reduced", RunningTotal.reduced, reduced),
        ):
            event = loaded.wait(
                loaded.submit(handler, (0, cores), args=(x, out, turns)), timeout=5.0
            )
            report[f"{name}_event"], report[f"{name}_us"] = (
                event.kind,
                event.microseconds,
            )
    finally:
        loaded.stop()
    report["kernel_text"] = program.sizes.text
    got_prefix, got_reduced = session.read(prefix, reduced)
    turn_sums = values.reshape(cores, turns, K, WORDS).sum(axis=2)
    want_prefix = np.cumsum(turn_sums, axis=1).reshape(cores * turns, WORDS)
    want_reduced = want_prefix.reshape(cores, turns, WORDS).sum(axis=0)
    got_prefix = got_prefix.view(np.float32).reshape(want_prefix.shape)
    got_reduced = got_reduced.view(np.float32).reshape(want_reduced.shape)
    report["prefix_words"] = int(want_prefix.size)
    report["prefix_differ"] = int((got_prefix != want_prefix).sum())
    report["reduced_words"] = int(want_reduced.size)
    report["reduced_differ"] = int((got_reduced != want_reduced).sum())
    report["ok"] = (
        report["prefix_differ"] == report["reduced_differ"] == 0
        and report["prefix_event"] == report["reduced_event"] == "done"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--form", choices=("total", "held", "turns"), default="total")
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--dump", help="held, turns: every output and the inputs to this .npz"
    )
    args = parser.parse_args()
    with km.Session(log=log) as session:
        if args.form == "held":
            report = run_held(session, args.turns, args.seed, args.dump)
        elif args.form == "turns":
            report = run_turns(session, args.cores, args.turns, args.seed, args.dump)
        else:
            report = run_total(session, args.cores, args.turns, args.seed)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
