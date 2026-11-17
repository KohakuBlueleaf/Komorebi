"""km.where under each comparison, and km.floor (README.md): on one core, two tiles of
pairs (every pair of 32 edge values, then random pairs, half of them equal), each lane
against numpy on the values the tensors hold. A JSON line a handler and format.
argv: --cards N (one card; the first visible when not given) or 0,1 (two)."""

import argparse
import json
import time

import numpy as np

import komorebi as km
from komorebi import formats

STARTED = time.monotonic()
TILES = 2
FORMATS = ("float32", "bfloat16")
WHOLE = 2.0**23
EDGES = [
    0.0, -0.0, np.nan, np.inf, -np.inf, 1.0, -1.0, 0.5,
    -0.5, 1.5, -1.5, 2.5, -2.5, 3.0, -3.0, -0.75,
    1e-40, -1e-40, 1.1754944e-38, WHOLE - 0.5,
    -(WHOLE - 0.5), WHOLE, -WHOLE, WHOLE + 1,
    -(WHOLE + 1), 2.0**22 + 0.5, -(2.0**22 + 0.5), 2.0**24 + 2,
    1e30, -1e30, 100.25, -100.75,
]  # fmt: skip
TESTS = {  # handler: numpy's lanes
    "greater": lambda x, y: np.where(x > y, 1.0, 0.0),
    "less": lambda x, y: np.where(x < y, 1.0, 0.0),
    "greater_equal": lambda x, y: np.where(x >= y, 1.0, 0.0),
    "less_equal": lambda x, y: np.where(x <= y, 1.0, 0.0),
    "equal": lambda x, y: np.where(x == y, 1.0, 0.0),
    "not_equal": lambda x, y: np.where(x != y, 1.0, 0.0),
    "smaller": lambda x, y: np.where(x < y, x, y),
    "floor": lambda x, y: np.floor(x),
}


def log(message):
    print(f"[{time.monotonic() - STARTED:6.2f}] {message}", flush=True)


@km.tile_function
def greater(x, y):
    return km.where(x > y, 1.0, 0.0)


@km.tile_function
def less(x, y):
    return km.where(x < y, 1.0, 0.0)


@km.tile_function
def greater_equal(x, y):
    return km.where(x >= y, 1.0, 0.0)


@km.tile_function
def less_equal(x, y):
    return km.where(x <= y, 1.0, 0.0)


@km.tile_function
def equal(x, y):
    return km.where(x == y, 1.0, 0.0)


@km.tile_function
def not_equal(x, y):
    return km.where(x != y, 1.0, 0.0)


@km.tile_function
def smaller(x, y):
    return km.where(x < y, x, y)


class Where(km.Program):
    """Each handler: TILES tiles of x and y in step, one tile function a tile into
    out; floor: x alone."""

    @km.handler
    def run_greater(self, x: km.Tensor, y: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, TILES), write=True)
        for a, b in zip(km.Stream(x.pages(0, TILES)), km.Stream(y.pages(0, TILES))):
            results.push(greater(a, b))

    @km.handler
    def run_less(self, x: km.Tensor, y: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, TILES), write=True)
        for a, b in zip(km.Stream(x.pages(0, TILES)), km.Stream(y.pages(0, TILES))):
            results.push(less(a, b))

    @km.handler
    def run_greater_equal(self, x: km.Tensor, y: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, TILES), write=True)
        for a, b in zip(km.Stream(x.pages(0, TILES)), km.Stream(y.pages(0, TILES))):
            results.push(greater_equal(a, b))

    @km.handler
    def run_less_equal(self, x: km.Tensor, y: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, TILES), write=True)
        for a, b in zip(km.Stream(x.pages(0, TILES)), km.Stream(y.pages(0, TILES))):
            results.push(less_equal(a, b))

    @km.handler
    def run_equal(self, x: km.Tensor, y: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, TILES), write=True)
        for a, b in zip(km.Stream(x.pages(0, TILES)), km.Stream(y.pages(0, TILES))):
            results.push(equal(a, b))

    @km.handler
    def run_not_equal(self, x: km.Tensor, y: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, TILES), write=True)
        for a, b in zip(km.Stream(x.pages(0, TILES)), km.Stream(y.pages(0, TILES))):
            results.push(not_equal(a, b))

    @km.handler
    def run_smaller(self, x: km.Tensor, y: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, TILES), write=True)
        for a, b in zip(km.Stream(x.pages(0, TILES)), km.Stream(y.pages(0, TILES))):
            results.push(smaller(a, b))

    @km.handler
    def run_floor(self, x: km.Tensor, y: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, TILES), write=True)
        for a in km.Stream(x.pages(0, TILES)):
            results.push(km.floor(a))


def inputs(seed=5):
    """x, y: tile 0 every pair of EDGES (x the row's, y the column's), tile 1 random
    values of either sign, y equal to x in half the lanes."""
    edges = np.array(EDGES, np.float32)
    rng = np.random.default_rng(seed)
    x1 = (rng.standard_normal((32, 32)) * 8).astype(np.float32)
    y1 = np.where(rng.random((32, 32)) < 0.5, x1, rng.standard_normal((32, 32)) * 8)
    x = np.concatenate([np.repeat(edges[:, None], 32, 1), x1])
    y = np.concatenate([np.repeat(edges[None, :], 32, 0), y1.astype(np.float32)])
    return x, y


def stored(values, dtype):
    """The values as the tensor holds them (the host's encode, decoded)."""
    raw, _ = formats.to_bytes(values, dtype, "tile")
    return np.asarray(formats.from_bytes(raw, values.shape, dtype, "tile"), np.float32)


def compared(name, got, x, y):
    """The lanes that differ from numpy (a NaN equals a NaN), and those that equal it
    in value but not in sign (a zero's)."""
    with np.errstate(invalid="ignore"):
        want = TESTS[name](x, y).astype(np.float32)
    same = (got == want) | (np.isnan(got) & np.isnan(want))
    sign = same & (np.signbit(got) != np.signbit(want))
    cases = [
        [float(x[i]), float(y[i]), float(got[i]), float(want[i])]
        for i in zip(*np.nonzero(~same | sign))
    ]
    return {"differ": int((~same).sum()), "sign": int(sign.sum()), "cases": cases[:40]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", default="")
    cards = [int(c) for c in parser.parse_args().cards.split(",") if c]
    x, y = inputs()
    opened = {"cards": cards} if len(cards) > 1 else {"card": next(iter(cards), None)}
    with km.Session(log=log, **opened) as session:
        for dtype in FORMATS:
            t = {
                "x": session.tensor(x, dtype, "tile"),
                "y": session.tensor(y, dtype, "tile"),
                "out": session.empty(x.shape, "float32", "tile"),
            }
            program = Where(log=log, session=session, **t)
            run = session.load(program)
            outs = {name: session.empty(x.shape, "float32", "tile") for name in TESTS}
            kinds = {}
            try:
                for name, out in outs.items():
                    handler = getattr(program, f"run_{name}")
                    sent = run.submit(handler, (0, 1), args=[t["x"], t["y"], out])
                    kinds[name] = run.wait(sent, timeout=30.0).kind
                    log(f"{dtype} {name}: {kinds[name]}")
            finally:
                run.stop()
            # read after the stop: a read waits for the cores, which a run holds
            sx, sy = stored(x, dtype), stored(y, dtype)
            for name, out in outs.items():
                got = np.asarray(out.read(), np.float32)
                row = {"format": dtype, "handler": name, "kind": kinds[name]}
                print(json.dumps(row | compared(name, got, sx, sy)), flush=True)
            for tensor in [*t.values(), *outs.values()]:
                tensor.free()
    log("done")


if __name__ == "__main__":
    main()
