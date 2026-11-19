"""Tile functions on Wormhole: each a class of its own, its handler a plain command on
one core over float32 tiles, against numpy; a function the chip refuses is a JSON line
with the refusal. The same script on Blackhole for the figures beside. See README.md.
Example: python wormhole_check.py"""

import argparse
import json
import time

import numpy as np

import komorebi as km
from komorebi.errors import ProgramError

STARTED = time.monotonic()
TILES = 4  # tiles of each tensor, one core's


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class Add(km.Program):
    """run: TILES tiles of a and b through f, one a tile, into out (one core)."""

    @km.tile_function
    def f(a, b):
        return a + b

    @km.handler
    def run(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, TILES), write=True)
        for x, y in zip(km.Stream(a.pages(0, TILES)), km.Stream(b.pages(0, TILES))):
            results.push(self.f(x, y))


class Thrice(Add):  # a MAD result read the next cycle, inside the replayed words
    @km.tile_function
    def f(a, b):
        return a + a + a


class Affine(Add):
    @km.tile_function
    def f(a, b):
        return a * 0.5 + 1.25


class Square(Add):
    @km.tile_function
    def f(a, b):
        return a * b


class Absolute(Add):
    @km.tile_function
    def f(a, b):
        return abs(a)


class Relu(Add):
    @km.tile_function
    def f(a, b):
        return km.relu(a)


class Larger(Add):
    @km.tile_function
    def f(a, b):
        return km.maximum(a, b)


class Difference(Add):  # SFPADD's NEGATE_VC: Blackhole's alone
    @km.tile_function
    def f(a, b):
        return a - b


class Silu(Add):  # SFPGT, SFPARECIP: Blackhole's alone
    @km.tile_function
    def f(a, b):
        return km.silu(a)


def silu(x):
    return x / (1.0 + np.exp(-x))


CLASSES = {  # each function and what numpy computes (float64)
    "add": (Add, lambda x, y: x + y),
    "thrice": (Thrice, lambda x, y: x * 3),
    "affine": (Affine, lambda x, y: x * 0.5 + 1.25),
    "multiply": (Square, lambda x, y: x * y),
    "absolute": (Absolute, lambda x, y: np.abs(x)),
    "relu": (Relu, lambda x, y: np.maximum(x, 0.0)),
    "larger": (Larger, np.maximum),
    "difference": (Difference, lambda x, y: x - y),
    "silu": (Silu, lambda x, y: silu(x)),
}


def inputs(seed=7):
    """x, y: TILES tiles of normals times 4, y equal to x in a quarter of the lanes."""
    rng = np.random.default_rng(seed)
    x, y = (rng.standard_normal((2, 32 * TILES, 32)) * 4).astype(np.float32)
    y = np.where(rng.random(x.shape) < 0.25, x, y).astype(np.float32)
    return x, y


def compared(want, got):
    """The largest error over the result's largest value, the lanes beyond 1e-6 of
    it, the lanes exactly equal, and the relative RMS error."""
    error = np.abs(got.astype(np.float64) - want)
    scale = np.abs(want).max()
    return {
        "max_over_largest": float(error.max() / scale),
        "lanes_over_1e-6": int((error > 1e-6 * scale).sum()),
        "lanes_exact": int((error == 0).sum()),
        "lanes": int(error.size),
        "rms_rel": float(np.sqrt((error**2).mean()) / np.sqrt((want**2).mean())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--card", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=list(CLASSES), choices=CLASSES)
    args = parser.parse_args()
    x, y = inputs()
    with km.Session(log=log, card=args.card) as session:
        arch = session.arch
        a = session.tensor(x, "float32", "tile")
        b = session.tensor(y, "float32", "tile")
        for name in args.only:
            cls, numpy = CLASSES[name]
            out = session.empty(x.shape, "float32", "tile")
            row = {"function": name, "arch": arch}
            try:
                program = cls(log=log, session=session, a=a, b=b, out=out)
                run = session.load(program)
            except ProgramError as refusal:
                print(json.dumps(row | {"refused": str(refusal)}), flush=True)
                out.free()
                continue
            try:
                sent = run.submit(program.run, (0, 1), args=[a, b, out])
                row["kind"] = run.wait(sent, timeout=30.0).kind
            finally:
                run.stop()
            got = np.asarray(out.read(), np.float32)  # after the stop: the cores free
            want = numpy(x.astype(np.float64), y.astype(np.float64))
            print(json.dumps(row | compared(want, got)), flush=True)
            out.free()
    log("done")


if __name__ == "__main__":
    main()
