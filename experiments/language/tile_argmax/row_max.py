"""Row max, min and argmax in a tile function (README.md): rows of 2 tiles and of 64
(read in parts of 16), float32, one core, each row's value against numpy: random rows,
the largest value in several columns (the first column wins the argmax), a NaN in a
row, a row of one value. A JSON line a handler. argv: --cards N (default: the first)."""

import argparse
import json
import time

import numpy as np

import komorebi as km
from komorebi.dsl.tile import row_operand

STARTED = time.monotonic()
ROWS = 1  # rows of tiles a command (32 rows of values)
WIDTHS = (2, 64)  # tiles a row (64: read in 4 parts of 16)
NOWHERE = 1e9  # the argmax's column where the row's largest value is not


def log(message):
    print(f"[{time.monotonic() - STARTED:6.2f}] {message}", flush=True)


@km.tile_function
def row_max(row):
    return km.reduce_max(row, exact=True).col()


@km.tile_function
def row_min(row):
    return km.reduce_min(row).col()


@km.tile_function
def argmax(row, cols):
    m = km.reduce_max(row, exact=True)
    return km.reduce_min(km.where(row == m.col(), cols.row(), NOWHERE)).col()


def make_class(width):
    """The class over rows of `width` tiles: a block a row, the column numbers a
    field of the row's chunks (one per-column tile a chunk of 2 tiles)."""
    chunks = (width + 1) // 2

    class Rows(km.Program):
        BLOCK = km.constant(width)
        CHUNKS = km.constant(chunks)
        PART = km.constant(min(width, 16))  # a block: the row read in parts (L1)
        cols: km.L1[km.fp32, chunks, 32, 32]

        @km.handler
        def run_max(self, x: km.Tensor, c: km.Tensor, out: km.Tensor):
            results = km.Stream(
                out.pages(0, ROWS * self.BLOCK), write=True, block=self.PART
            )
            for row in km.Stream(
                x.pages(0, ROWS * self.BLOCK), block=self.PART, row=self.BLOCK
            ):
                results.push(row_max(row))

        @km.handler
        def run_min(self, x: km.Tensor, c: km.Tensor, out: km.Tensor):
            results = km.Stream(
                out.pages(0, ROWS * self.BLOCK), write=True, block=self.PART
            )
            for row in km.Stream(
                x.pages(0, ROWS * self.BLOCK), block=self.PART, row=self.BLOCK
            ):
                results.push(row_min(row))

        @km.handler
        def run_argmax(self, x: km.Tensor, c: km.Tensor, out: km.Tensor):
            self.cols.fill(km.Stream(c.pages(0, self.CHUNKS)))
            results = km.Stream(
                out.pages(0, ROWS * self.BLOCK), write=True, block=self.PART
            )
            for row in km.Stream(
                x.pages(0, ROWS * self.BLOCK), block=self.PART, row=self.BLOCK
            ):
                results.push(argmax(row, self.cols))

    Rows.__name__ = Rows.__qualname__ = f"Rows{width}"
    return Rows


def inputs(width, seed=3):
    """x: ROWS x 32 rows of 32 x width values; rows 0 .. 7 of each tile row edge
    rows (the largest value in 2, 3 or every column; a NaN; one value; -inf), the
    rest random with ties in about half; c: the column numbers as the field's pages."""
    rng = np.random.default_rng(seed)
    n = 32 * width
    x = rng.integers(-50, 50, (ROWS * 32, n)).astype(np.float32)
    for base in range(0, ROWS * 32, 32):
        x[base + 0, [5, n - 1]] = 99.0  # two places: the first wins
        x[base + 1, [n - 1, 7, 40]] = 99.0
        x[base + 2] = 4.0  # every column
        x[base + 3, 9] = np.nan
        x[base + 4, 0] = 99.0  # the first column
        x[base + 5, n - 1] = 99.0  # the last column
        x[base + 6] = -np.inf
        x[base + 7, 3] = -0.0
        x[base + 7, 4] = 0.0
        x[base + 7, :3] = -5.0
        x[base + 7, 5:] = -5.0
    columns = np.arange(n, dtype=np.float32)
    pages = [row_operand(columns[64 * k : 64 * k + 64], min(2, width - 2 * k))
             for k in range((width + 1) // 2)]  # fmt: skip
    return x, np.asarray(pages, np.float32)


def wants(x):
    """numpy's value a row (argmax: the first of the largest; NaN rows: np.max's)."""
    with np.errstate(invalid="ignore"):
        return {
            "max": x.max(1),
            "min": x.min(1),
            "argmax": np.argmax(x, 1).astype(np.float32),
        }


def compared(name, got, x):
    """Rows whose value (column 0 of the broadcast result) differs from numpy's."""
    want = wants(x)[name]
    value = got[:, 0]
    same = (value == want) | (np.isnan(value) & np.isnan(want))
    rows = [
        [int(r), float(value[r]), float(want[r]), bool((got[r] == got[r, 0]).all())]
        for r in np.nonzero(~same)[0]
    ]
    whole = bool(((got == got[:, :1]) | np.isnan(got)).all())
    return {"differ": int((~same).sum()), "broadcast": whole, "rows": rows[:20]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", default="")
    cards = [int(c) for c in parser.parse_args().cards.split(",") if c]
    opened = {"cards": cards} if len(cards) > 1 else {"card": next(iter(cards), None)}
    with km.Session(log=log, **opened) as session:
        for width in WIDTHS:
            x, c = inputs(width)
            t = {
                "x": session.tensor(x, "float32", "tile"),
                "c": session.empty(c.shape, "float32"),
                "out": session.empty(x.shape, "float32", "tile"),
            }
            session.write(t["c"], c.view(np.uint8).reshape(c.shape[0], -1))
            program = make_class(width)(log=log, session=session, **t)
            run = session.load(program)
            outs = {n: session.empty(x.shape, "float32", "tile") for n in wants(x)}
            kinds = {}
            try:
                for name, out in outs.items():
                    handler = getattr(program, f"run_{name}")
                    sent = run.submit(handler, (0, 1), args=[t["x"], t["c"], out])
                    kinds[name] = run.wait(sent, timeout=30.0).kind
                    log(f"width {width} {name}: {kinds[name]}")
            finally:
                run.stop()
            # read after the stop: a read waits for the cores, which a run holds
            for name, out in outs.items():
                got = np.asarray(out.read(), np.float32)
                row = {"width": width, "handler": name}
                row["kind"] = kinds[name]
                print(json.dumps(row | compared(name, got, x)), flush=True)
            for tensor in [*t.values(), *outs.values()]:
                tensor.free()
    log("done")


if __name__ == "__main__":
    main()
