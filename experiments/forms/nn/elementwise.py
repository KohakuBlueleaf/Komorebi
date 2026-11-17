"""Elementwise layers as tile functions over DRAM streams, each against numpy: SiLU,
GELU in the erf form and in the tanh form (written as x sigmoid(2 z)), a residual add
then SiLU, and a per-channel scale and shift read from a field by .col() (a value a
row) and by .row() (a value a column).
Example: python elementwise.py --cores 8 --tiles 8"""

import argparse
import math
import sys

import numpy as np
from forms import empty, error, finish, log, random, read, tiles, timed

import komorebi as km

TANH_ARGUMENT = 1.5957691216  # 2 x sqrt(2 / pi): 0.5 x (1 + tanh z) = x sigmoid(2 z)


@km.tile_function(tolerance=1e-6)
def silu(a):
    return km.silu(a)


@km.tile_function(tolerance=1e-6)
def gelu_erf(x):
    return 0.5 * x * (1.0 + km.erf(x * 0.7071067811865476))


@km.tile_function(tolerance=1e-5)
def gelu_tanh(x):
    return x * km.sigmoid(TANH_ARGUMENT * (x + 0.044715 * x * x * x))


@km.tile_function(tolerance=1e-6)
def residual_silu(a, b):
    return km.silu(a + b)


@km.tile_function(tolerance=1e-6)
def scale_rows(x, g, b):
    return x * g.col() + b.col()


@km.tile_function(tolerance=1e-6)
def scale_columns(x, g, b):
    return x * g.row() + b.row()


class Layers(km.Program):
    """Each core takes `tiles` pages from rank x tiles; `activate` loops over `chunks`
    of the cores' pages (the form of a tensor larger than the cores' L1)."""

    g: km.L1[km.fp32, 1, 32, 32]
    b: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def activate(self, a: km.Tensor, out: km.Tensor, tiles: int, chunks: int):
        for c in range(chunks):
            first = (c * self.size + self.rank) * tiles
            results = km.Stream(out.pages(first, tiles), write=True, block=4)
            for x in km.Stream(a.pages(first, tiles), block=4):
                results.push(gelu_erf(x))

    @km.handler
    def activate_tanh(self, a: km.Tensor, out: km.Tensor, tiles: int):
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=4)
        for x in km.Stream(a.pages(first, tiles), block=4):
            results.push(gelu_tanh(x))

    @km.handler
    def silu_layer(self, a: km.Tensor, out: km.Tensor, tiles: int):
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=4)
        for x in km.Stream(a.pages(first, tiles), block=4):
            results.push(silu(x))

    @km.handler
    def residual(self, a: km.Tensor, r: km.Tensor, out: km.Tensor, tiles: int):
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=4)
        for x, y in zip(
            km.Stream(a.pages(first, tiles), block=4),
            km.Stream(r.pages(first, tiles), block=4),
        ):
            results.push(residual_silu(x, y))

    @km.handler
    def per_channel_rows(
        self, a: km.Tensor, gv: km.Tensor, bv: km.Tensor, out: km.Tensor, tiles: int
    ):
        first = self.rank * tiles
        self.g.fill(km.Stream(gv.pages(self.rank % 2, 1)))
        self.b.fill(km.Stream(bv.pages(self.rank % 2, 1)))
        results = km.Stream(out.pages(first, tiles), write=True, block=4)
        for x in km.Stream(a.pages(first, tiles), block=4):
            results.push(scale_rows(x, self.g, self.b))

    @km.handler
    def per_channel_columns(
        self, a: km.Tensor, gv: km.Tensor, bv: km.Tensor, out: km.Tensor, tiles: int
    ):
        first = self.rank * tiles
        self.g.fill(km.Stream(gv.pages(0, 1)))
        self.b.fill(km.Stream(bv.pages(0, 1)))
        results = km.Stream(out.pages(first, tiles), write=True, block=4)
        for x in km.Stream(a.pages(first, tiles), block=4):
            results.push(scale_columns(x, self.g, self.b))


def silu_ref(x):
    return x / (1.0 + np.exp(-x))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--tiles", type=int, default=8, help="tiles a core takes")
    parser.add_argument("--chunks", type=int, default=2, help="activate's chunks")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cores, count = args.cores, args.tiles
    pages = cores * count * args.chunks  # activate's pages; the others take the first
    a = random(rng, (pages * 32, 32), 2.0)
    r = random(rng, (pages * 32, 32), 2.0)
    gv = random(rng, (64, 32))
    bv = random(rng, (64, 32))
    erf = np.vectorize(math.erf)
    report = {"cores": cores, "tiles": count, "chunks": args.chunks}
    with km.Session(log=log) as session:
        ta, tr = tiles(session, a), tiles(session, r)
        tg, tb = tiles(session, gv, "float32"), tiles(session, bv, "float32")
        out = empty(session, pages * 32, 32)
        program = Layers(a=ta, r=tr, out=out, gv=tg, bv=tb, session=session, log=log)
        run = session.load(program)
        group, first = (0, cores), slice(0, cores * count * 32)
        results = {}
        try:
            for name, handler, extra, want in (
                ("gelu_erf", Layers.activate, (count, args.chunks),
                 0.5 * a * (1.0 + erf(a / math.sqrt(2.0)))),
                ("gelu_tanh", Layers.activate_tanh, (count,),
                 0.5 * a * (1.0 + np.tanh(TANH_ARGUMENT / 2 * (a + 0.044715 * a**3)))),
                ("silu", Layers.silu_layer, (count,), silu_ref(a)),
            ):  # fmt: skip
                us = timed(run, handler, group, (ta, out, *extra))
                rows = slice(None) if name == "gelu_erf" else first
                results[name] = (us, read(session, out)[rows], want[rows])
            us = timed(run, Layers.residual, group, (ta, tr, out, count))
            results["residual_silu"] = (
                us,
                read(session, out)[first],
                silu_ref(a + r)[first],
            )
            us = timed(run, Layers.per_channel_rows, group, (ta, tg, tb, out, count))
            channel = (
                np.arange(pages * 32) // 32 // count % 2 * 32
                + np.arange(pages * 32) % 32
            )
            want = a * gv[channel, :1] + bv[channel, :1]
            results["scale_rows"] = (us, read(session, out)[first], want[first])
            us = timed(run, Layers.per_channel_columns, group, (ta, tg, tb, out, count))
            want = a * gv[0] + bv[0]
            results["scale_columns"] = (us, read(session, out)[first], want[first])
        finally:
            run.stop()
    for name, (us, got, want) in results.items():
        report[name] = error(got, want) | {"us": us}
        log(f"{name}: {report[name]}")
    finish(report, [(name, "rms", 0.01) for name in results])


if __name__ == "__main__":
    sys.exit(main())
