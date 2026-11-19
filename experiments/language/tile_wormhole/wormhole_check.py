"""Tile functions on Wormhole: each a class of its own, its handler a plain command on
one core over float32 tiles, against numpy; a function the chip refuses is a JSON line
with the refusal. The same script on Blackhole for the figures beside. See README.md.
Example: python wormhole_check.py"""

import argparse
import hashlib
import json
import time

import numpy as np

import komorebi as km
from komorebi import formats
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


class LoFi(km.Program):
    """run: TILES products of a tile of a and a tile of w, one a tile, into out."""

    @km.tile_function(fidelity="LoFi")
    def f(a, w):
        return a @ w

    @km.handler
    def run(self, a: km.Tensor, w: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, TILES), write=True)
        for x, y in zip(km.Stream(a.pages(0, TILES)), km.Stream(w.pages(0, TILES))):
            results.push(self.f(x, y))


class HiFi2(LoFi):
    @km.tile_function(fidelity="HiFi2")
    def f(a, w):
        return a @ w


class HiFi3(LoFi):
    @km.tile_function(fidelity="HiFi3")
    def f(a, w):
        return a @ w


class HiFi4(LoFi):
    @km.tile_function(fidelity="HiFi4")
    def f(a, w):
        return a @ w


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


def products(x, w):
    """Each tile of x times the same tile of w (float64)."""
    tiles = lambda v: v.reshape(TILES, 32, 32)  # noqa: E731
    return np.matmul(tiles(x), tiles(w)).reshape(x.shape)


PRODUCTS = {"LoFi": LoFi, "HiFi2": HiFi2, "HiFi3": HiFi3, "HiFi4": HiFi4}
# the operands' formats a class product takes (B family, then A); out is float32
SOURCES = ("bfloat16", "bfp8_b", "bfp4_b", "bfp2_b", "float16", "bfp8", "bfp4", "bfp2")


def inputs(seed=7):
    """x, y: TILES tiles of normals times 4, y equal to x in a quarter of the lanes."""
    rng = np.random.default_rng(seed)
    x, y = (rng.standard_normal((2, 32 * TILES, 32)) * 4).astype(np.float32)
    y = np.where(rng.random(x.shape) < 0.25, x, y).astype(np.float32)
    return x, y


def stored(values, dtype):
    """The values as a tensor of `dtype` holds them (the host's encode, decoded)."""
    raw, _ = formats.to_bytes(values, dtype, "tile")
    return np.asarray(formats.from_bytes(raw, values.shape, dtype, "tile"), np.float64)


def compared(want, got):
    """The largest error over the result's largest value, the lanes beyond 1e-6 of
    it, the lanes exactly equal, and the relative RMS error."""
    error = np.abs(got.astype(np.float64) - want)
    scale = np.abs(want).max()
    return {
        "words": hashlib.sha1(got.tobytes()).hexdigest()[:12],  # the chips' words alike
        "max_over_largest": float(error.max() / scale),
        "lanes_over_1e-6": int((error > 1e-6 * scale).sum()),
        "lanes_exact": int((error == 0).sum()),
        "lanes": int(error.size),
        "rms_rel": float(np.sqrt((error**2).mean()) / np.sqrt((want**2).mean())),
    }


def ran(session, row, cls, tensors, want, dump):
    """`cls` loaded with `tensors` (a, second operand, out), its run submitted on one
    core, out against `want`: a JSON line (or the load's refusal)."""
    out = tensors[-1]
    names = dict(zip(("a", "w" if issubclass(cls, LoFi) else "b", "out"), tensors))
    try:
        program = cls(log=log, session=session, **names)
        run = session.load(program)
    except ProgramError as refusal:
        print(json.dumps(row | {"refused": str(refusal)}), flush=True)
        return
    try:
        sent = run.submit(program.run, (0, 1), args=list(tensors))
        row["kind"] = run.wait(sent, timeout=30.0).kind
    finally:
        run.stop()
    got = np.asarray(out.read(), np.float32)  # after the stop: the cores free
    if dump is not None:
        dump["/".join(str(v) for v in row.values())] = got
    print(json.dumps(row | compared(want, got)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--card", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=list(CLASSES), choices=CLASSES)
    parser.add_argument("--products", nargs="*", default=[], choices=PRODUCTS)
    parser.add_argument("--sources", nargs="*", default=SOURCES, choices=SOURCES)
    parser.add_argument("--dump", help="an .npz of every output, by its row")
    args = parser.parse_args()
    x, y = inputs()
    dump = {} if args.dump else None
    with km.Session(log=log, card=args.card) as session:
        arch = session.arch
        a = session.tensor(x, "float32", "tile")
        b = session.tensor(y, "float32", "tile")
        for name in args.only:
            cls, numpy = CLASSES[name]
            out = session.empty(x.shape, "float32", "tile")
            want = numpy(x.astype(np.float64), y.astype(np.float64))
            ran(session, {"function": name, "arch": arch}, cls, (a, b, out), want, dump)
            out.free()
        w_values = (y / 4).astype(np.float32)  # normals: sums of 32 products near 6
        for source in args.sources if args.products else ():
            sa, sw = (session.tensor(v, source, "tile") for v in (x / 4, w_values))
            want = products(stored(x / 4, source), stored(w_values, source))
            for fidelity in args.products:
                out = session.empty(x.shape, "float32", "tile")
                row = {"function": "product", "arch": arch, "source": source}
                row["fidelity"] = fidelity
                ran(session, row, PRODUCTS[fidelity], (sa, sw, out), want, dump)
                out.free()
            sa.free()
            sw.free()
    if dump is not None:
        np.savez_compressed(args.dump, **dump)
    log("done")


if __name__ == "__main__":
    main()
