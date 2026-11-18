"""A tile function's result as another's operand (README.md): nested, through a name
bound earlier in the loop, and inside a tile expression, each a composed unit program
on one core, against numpy on the values the tensors hold. A JSON line a handler and
format. argv: --cards N (one card; the first visible when not given) or 0,1 (two)."""

import argparse
import json
import time

import numpy as np

import komorebi as km
from komorebi import formats

STARTED = time.monotonic()
TILES = 4
FORMATS = ("float32", "bfloat16")


def log(message):
    print(f"[{time.monotonic() - STARTED:6.2f}] {message}", flush=True)


@km.tile_function
def larger(x, y):
    return km.where(x > y, x, y * 0.5)


@km.tile_function
def scaled(t, z):
    return t * z + 1.0


@km.tile_function
def biased(t, b):
    return t + b.row()


@km.tile_function
def by_column(t, b):
    return t * b.col()


def numpy_larger(x, y):
    return np.where(x > y, x, y * np.float32(0.5))


WANTS = {  # handler: numpy's lanes
    "nested": lambda x, y, z: numpy_larger(x, y) * z + np.float32(1.0),
    "named": lambda x, y, z: numpy_larger(x, y) * z + np.float32(1.0),
    "expression": lambda x, y, z: (
        numpy_larger(x, y) / (1 + np.exp(-numpy_larger(x, y).astype(np.float64))) * z
    ),
    # z: the field's tile (w), its row 0 or column 0 broadcast over each tile
    "field_row": lambda x, y, z: numpy_larger(x, y) + z[0][None, :],
    "field_col": lambda x, y, z: numpy_larger(x, y) * np.tile(z[:, :1], (TILES, 1)),
}
FIELDS = ("field_row", "field_col")  # handlers given w (float32) in z's place


class Composed(km.Program):
    """Each handler: TILES tiles of x, y and z in step, one composed tile function a
    tile into out; field_*: x and y, and the field filled from w's tile."""

    bias: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def run_nested(self, x: km.Tensor, y: km.Tensor, z: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, TILES), write=True)
        for a, b, c in zip(
            km.Stream(x.pages(0, TILES)),
            km.Stream(y.pages(0, TILES)),
            km.Stream(z.pages(0, TILES)),
        ):
            results.push(scaled(larger(a, b), c))

    @km.handler
    def run_named(self, x: km.Tensor, y: km.Tensor, z: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, TILES), write=True)
        for a, b, c in zip(
            km.Stream(x.pages(0, TILES)),
            km.Stream(y.pages(0, TILES)),
            km.Stream(z.pages(0, TILES)),
        ):
            t = larger(a, b)
            results.push(scaled(t, c))

    @km.handler
    def run_expression(self, x: km.Tensor, y: km.Tensor, z: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, TILES), write=True)
        for a, b, c in zip(
            km.Stream(x.pages(0, TILES)),
            km.Stream(y.pages(0, TILES)),
            km.Stream(z.pages(0, TILES)),
        ):
            results.push(km.silu(larger(a, b)) * c)

    @km.handler
    def run_field_row(self, x: km.Tensor, y: km.Tensor, w: km.Tensor, out: km.Tensor):
        self.bias.fill(km.Stream(w.pages(0, 1)))
        results = km.Stream(out.pages(0, TILES), write=True)
        for a, b in zip(km.Stream(x.pages(0, TILES)), km.Stream(y.pages(0, TILES))):
            results.push(biased(larger(a, b), self.bias))

    @km.handler
    def run_field_col(self, x: km.Tensor, y: km.Tensor, w: km.Tensor, out: km.Tensor):
        self.bias.fill(km.Stream(w.pages(0, 1)))
        results = km.Stream(out.pages(0, TILES), write=True)
        for a, b in zip(km.Stream(x.pages(0, TILES)), km.Stream(y.pages(0, TILES))):
            t = larger(a, b)
            results.push(by_column(t, self.bias))


def inputs(seed=11):
    """x, y, z: TILES tiles of normals times 4, y equal to x in a quarter of lanes."""
    rng = np.random.default_rng(seed)
    x, y, z = (rng.standard_normal((3, 32 * TILES, 32)) * 4).astype(np.float32)
    y = np.where(rng.random(x.shape) < 0.25, x, y).astype(np.float32)
    return x, y, z


def stored(values, dtype):
    """The values as the tensor holds them (the host's encode, decoded)."""
    raw, _ = formats.to_bytes(values, dtype, "tile")
    return np.asarray(formats.from_bytes(raw, values.shape, dtype, "tile"), np.float32)


def compared(name, got, x, y, z):
    """The largest error against numpy over the result's largest value, the lanes
    beyond 1e-6 of it (float32) and its relative error at the median."""
    want = WANTS[name](x, y, z).astype(np.float64)
    error = np.abs(got.astype(np.float64) - want)
    scale = np.abs(want).max()
    return {
        "max_over_largest": float(error.max() / scale),
        "lanes_over_1e-6": int((error > 1e-6 * scale).sum()),
        "rms_rel": float(np.sqrt((error**2).mean()) / np.sqrt((want**2).mean())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", default="")
    cards = [int(c) for c in parser.parse_args().cards.split(",") if c]
    x, y, z = inputs()
    opened = {"cards": cards} if len(cards) > 1 else {"card": next(iter(cards), None)}
    with km.Session(log=log, **opened) as session:
        for dtype in FORMATS:
            t = {n: session.tensor(v, dtype, "tile") for n, v in zip("xyz", (x, y, z))}
            t["out"] = session.empty(x.shape, "float32", "tile")
            t["w"] = session.tensor(z[:32], "float32", "tile")  # the field's tile
            program = Composed(log=log, session=session, **t)
            run = session.load(program)
            outs = {name: session.empty(x.shape, "float32", "tile") for name in WANTS}
            kinds = {}
            try:
                for name, out in outs.items():
                    handler = getattr(program, f"run_{name}")
                    third = t["w"] if name in FIELDS else t["z"]
                    sent = run.submit(
                        handler, (0, 1), args=[t["x"], t["y"], third, out]
                    )
                    kinds[name] = run.wait(sent, timeout=30.0).kind
                    log(f"{dtype} {name}: {kinds[name]}")
            finally:
                run.stop()
            # read after the stop: a read waits for the cores, which a run holds
            sx, sy, sz = (stored(v, dtype) for v in (x, y, z))
            for name, out in outs.items():
                got = np.asarray(out.read(), np.float32)
                row = {"format": dtype, "handler": name, "kind": kinds[name]}
                third = z[:32] if name in FIELDS else sz
                print(json.dumps(row | compared(name, got, sx, sy, third)), flush=True)
            for tensor in [*t.values(), *outs.values()]:
                tensor.free()
    log("done")


if __name__ == "__main__":
    main()
