"""A two-layer MLP over pixel blocks, y = silu(x W1) W2, a tile row of 32 pixels by 64
features, on every core of the grid, against numpy: W1 and the hidden layer held in
L1, and W2 (a) streamed from DRAM or (c) cast along a column of cores and held."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nn"))

import numpy as np
from forms import bf16, error, finish, log, random, read, tiles, timed

import komorebi as km

K, KH, N = 2, 2, 2  # input, hidden and output tiles of a pixel row
W2_BLOCK = KH * N


@km.tile_function(tolerance=1e-6)
def act(x):
    return km.silu(x)


class Streamed(km.Program):
    """(a): W1 a field, W2 read from DRAM for each row."""

    w1: km.L1[km.bf16, K, KH, 32, 32]
    h: km.L1[km.bf16, KH, 32, 32]

    @km.tile_function(fidelity="HiFi4")
    def project(a, b):
        return a @ b

    @km.handler
    def load(self, w1: km.Tensor):
        self.w1.fill(km.Stream(w1.tiles[0:K, 0:KH]))

    @km.handler
    def mlp(self, x: km.Tensor, w2: km.Tensor, y: km.Tensor, rows: int):
        for r in range(rows):
            row = self.rank * rows + r
            acc = km.tile.zeros(km.fp32, KH)
            for xb in km.Stream(x.tiles[row : row + 1, 0:K], block=K):
                acc += self.project(xb, self.w1)
            self.h.write(act(acc))
            out = km.tile.zeros(km.fp32, N)
            for wb in km.Stream(w2.tiles[0:KH, 0:N], block=W2_BLOCK):
                out += self.project(self.h, wb)
            y.pages(row * N, N).write(out)


class Cast(km.Program):
    """(c): W1 a field, W2 cast once along the column and held."""

    w1: km.L1[km.bf16, K, KH, 32, 32]
    h: km.L1[km.bf16, KH, 32, 32]

    @km.tile_function(fidelity="HiFi4")
    def project(a, b):
        return a @ b

    @km.handler
    def load(self, w1: km.Tensor):
        self.w1.fill(km.Stream(w1.tiles[0:K, 0:KH]))

    @km.handler
    def mlp(self, x: km.Tensor, w2: km.Tensor, y: km.Tensor, rows: int):
        column = km.Broadcast(self, along="column", window=1, mover="A")
        if column.sender():
            ws = column.cast(w2.tiles[0:KH, 0:N].read())
        else:
            ws = column.arrive()
        for r in range(rows):
            row = self.rank * rows + r
            acc = km.tile.zeros(km.fp32, KH)
            for xb in km.Stream(x.tiles[row : row + 1, 0:K], block=K):
                acc += self.project(xb, self.w1)
            self.h.write(act(acc))
            out = km.tile.zeros(km.fp32, N)
            out += self.project(self.h, ws[0:KH, :])
            y.pages(row * N, N).write(out)


def silu(x):
    return x / (1.0 + np.exp(-x))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rows", type=int, default=2, help="pixel rows a core")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = args.rows
    with km.Session(log=log) as session:
        cores = session.grid[0] * session.grid[1]  # the cast's rectangle
        x = random(rng, (cores * rows * 32, K * 32))
        w1 = random(rng, (K * 32, KH * 32), 1 / 8)
        w2 = random(rng, (KH * 32, N * 32), 1 / 8)
        want = bf16(silu(x @ w1)) @ w2
        report = {"cores": cores, "rows": rows}
        report["bf16_h_vs_float64"] = error(want, silu(x @ w1) @ w2)
        group = (0, cores)
        tx, t1, t2 = tiles(session, x), tiles(session, w1), tiles(session, w2)
        for name, cls in (("streamed", Streamed), ("cast", Cast)):
            y = tiles(session, np.zeros((x.shape[0], N * 32)), "float32")
            program = cls(x=tx, w1=t1, w2=t2, y=y, session=session, log=log)
            run = session.load(program)
            try:
                timed(run, cls.load, group, (t1,))
                us = timed(run, cls.mlp, group, (tx, t2, y, rows))
                report[f"{name}_us"] = us
            finally:
                run.stop()
            got = read(session, y)
            report[name] = error(got, want)  # against h rounded to bf16 by numpy
            report[f"{name}_vs_float64"] = error(got, silu(x @ w1) @ w2)
            log(f"{name}: {report[name]}, {us} us")
    checks = [
        ("streamed_vs_float64", "rms", 2.5e-3),
        ("cast_vs_float64", "rms", 2.5e-3),
    ]
    finish(report, checks)


if __name__ == "__main__":
    sys.exit(main())
