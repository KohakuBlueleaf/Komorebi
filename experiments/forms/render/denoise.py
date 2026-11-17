"""A separable 3x3 filter over a frame of tiles with a one-tile halo, as products,
against numpy: a horizontal pass (3 tiles of a row by the taps held in L1) into a
padded frame, then a vertical pass with the roles swapped (the taps stream, the 3
tiles of a column are held). A one-hot tap (a shift) and a blur."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nn"))

import numpy as np
from forms import bf16, error, finish, log, random, read, tiles, timed

import komorebi as km

R, C = 4, 4  # output tile rows and columns; core r C + c makes tile (r, c)


class Denoise(km.Program):
    """Products at HiFi4: a product by a 0/1 matrix is then a copy."""

    taps: km.L1[km.bf16, 3, 1, 32, 32]
    column: km.L1[km.bf16, 3, 1, 32, 32]

    @km.tile_function(fidelity="HiFi4")
    def product(a, b):
        return a @ b

    @km.handler
    def load_taps(self, h: km.Tensor):
        self.taps.fill(km.Stream(h.pages(0, 3)))

    @km.handler
    def horizontal(self, img: km.Tensor, mid: km.Tensor):
        """Padded tile (r + 1, c + 1) of mid: [X(c) X(c + 1) X(c + 2)] @ taps."""
        r = self.rank // C + 1
        c = self.rank % C
        acc = km.tile.zeros(km.fp32, 1)
        for xb in km.Stream(img.tiles[r : r + 1, c : c + 3], block=3):
            acc += self.product(xb, self.taps)
        mid.pages(r * (C + 2) + c + 1, 1).write(acc)

    @km.handler
    def vertical(self, img: km.Tensor, v: km.Tensor, out: km.Tensor):
        """Tile (r, c) of out: V @ [X(r); X(r + 1); X(r + 2)], the column held."""
        r = self.rank // C
        c = self.rank % C + 1
        self.column.fill(km.Stream(img.tiles[r : r + 3, c : c + 1]))
        acc = km.tile.zeros(km.fp32, 1)
        for vb in km.Stream(v.tiles[0:1, 0:3], block=3):
            acc += self.product(vb, self.column)
        out.pages(self.rank, 1).write(acc)


def tap_matrices(taps):
    """The horizontal taps as a 96 x 32 matrix (block 3 tiles of a row on the left)
    and the vertical ones as 32 x 96 (on the left of a column's 3 tiles)."""
    h = np.zeros((96, 32))
    for j in range(32):
        for k, w in enumerate(taps):
            h[32 + j - 1 + k, j] = w
    return h, h.T.copy()


def filtered(image, taps, rounded=bf16):
    """numpy: the 3-tap filter along x, `rounded`, then along y; zero border."""
    p = np.pad(image, 1)
    rows, cols = image.shape
    h = sum(w * p[1:-1, k : k + cols] for k, w in enumerate(taps))
    p = np.pad(rounded(h), 1)
    return sum(w * p[k : k + rows, 1:-1] for k, w in enumerate(taps))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    image = random(rng, (R * 32, C * 32))
    padded = np.pad(image, 32)
    cases = {"shift": (1.0, 0.0, 0.0), "blur": (0.25, 0.5, 0.25)}
    report = {"rows": R, "columns": C}
    group = (0, R * C)
    got = {}
    with km.Session(log=log) as session:
        img = tiles(session, padded)
        taps = {}
        for name, weights in cases.items():
            h, v = tap_matrices(weights)
            taps[name] = (tiles(session, h), tiles(session, v))
        mid = tiles(session, np.zeros_like(padded))
        out = tiles(session, np.zeros_like(image), "float32")
        program = Denoise(
            img=img, h=taps["shift"][0], v=taps["shift"][1], mid=mid, out=out,
            session=session, log=log,
        )  # fmt: skip
        run = session.load(program)
        try:
            for name, (h, v) in taps.items():
                timed(run, Denoise.load_taps, group, (h,))
                us = timed(run, Denoise.horizontal, group, (img, mid))
                report[f"{name}_horizontal_us"] = us
                us = timed(run, Denoise.vertical, group, (mid, v, out))
                report[f"{name}_vertical_us"] = us
                got[name] = read(session, out)
        finally:
            run.stop()
    for name, weights in cases.items():
        report[name] = error(got[name], filtered(image, weights))
        exact = filtered(image, weights, np.asarray)
        report[f"{name}_vs_float64"] = error(got[name], exact)
        report[f"{name}_bf16_mid_vs_float64"] = error(filtered(image, weights), exact)
        log(f"{name}: {report[name]}, against float64 {report[f'{name}_vs_float64']}")
    finish(report, [("shift", "max", 0.0), ("blur_vs_float64", "rms", 2.5e-3)])


if __name__ == "__main__":
    sys.exit(main())
