"""Convolution 3x3, stride 1, padding 1, as products only, against numpy. The feature
map is a matrix of rows (y, channel) and columns x with a zero tile row and column on
each side; a shift along x is a product by a 0/1 matrix on the right, a shift along y
is a tile row, and the channels mix by the weights on the left (README.md)."""

import argparse
import sys

import numpy as np
from forms import error, finish, log, random, read, tiles, timed

import komorebi as km

CT, H, J = 2, 16, 2  # channel tiles (64 channels in and out), rows, tile columns
ROWS = (H + 2) * CT  # tile rows with the zero rows: (y, channel tile)
COLS = J + 2  # tile columns with the zero columns
MODES = ("constant", "edge")  # the border tiles: zeros, or the edge's copy


class Shift(km.Program):
    """The shifts: products by a 0/1 matrix at HiFi4, the fidelity that keeps all 8
    bits of x, so each is a copy (a class's products share one fidelity)."""

    @km.tile_function(fidelity="HiFi4")
    def copy_product(a, b):
        return a @ b

    @km.handler
    def shift(self, x: km.Tensor, s: km.Tensor, zl: km.Tensor, zr: km.Tensor):
        """zl and zr of tile row `rank`: each pixel's left and right neighbour."""
        r = self.rank
        for j in range(1, J + 1):
            k = j - 1
            acc = km.tile.zeros(km.fp32, 1)
            for xb, sb in zip(
                km.Stream(x.tiles[r : r + 1, k : k + 2], block=2),
                km.Stream(s.tiles[0:2, 0:1], block=2),
            ):
                acc += self.copy_product(xb, sb)
            zl.tiles[r : r + 1, j : j + 1].write(acc)
            acc2 = km.tile.zeros(km.fp32, 1)
            for xb, sb in zip(
                km.Stream(x.tiles[r : r + 1, j : j + 2], block=2),
                km.Stream(s.tiles[2:4, 0:1], block=2),
            ):
                acc2 += self.copy_product(xb, sb)
            zr.tiles[r : r + 1, j : j + 1].write(acc2)


class Conv(km.Program):
    """The 9 taps' products over the shifted copies, at HiFi2."""

    @km.tile_function(fidelity="HiFi2")
    def product(a, b):
        return a @ b

    @km.handler
    def conv(
        self, w: km.Tensor, z0: km.Tensor, zl: km.Tensor, zr: km.Tensor, y: km.Tensor
    ):
        """Output tile (row, channel tile ro, tile column j): the 9 taps' products."""
        row = self.rank // CT
        ro = self.rank % CT
        for j in range(1, J + 1):
            acc = km.tile.zeros(km.fp32, 1)
            for dy in range(3):
                t0 = dy * 3 * CT
                t1 = t0 + CT
                t2 = t1 + CT
                r0 = (row + dy) * CT
                for wb, zb in zip(
                    km.Stream(w.tiles[ro : ro + 1, t0 : t0 + CT], block=CT),
                    km.Stream(zl.tiles[r0 : r0 + CT, j : j + 1], block=CT),
                ):
                    acc += self.product(wb, zb)
                for wb, zb in zip(
                    km.Stream(w.tiles[ro : ro + 1, t1 : t1 + CT], block=CT),
                    km.Stream(z0.tiles[r0 : r0 + CT, j : j + 1], block=CT),
                ):
                    acc += self.product(wb, zb)
                for wb, zb in zip(
                    km.Stream(w.tiles[ro : ro + 1, t2 : t2 + CT], block=CT),
                    km.Stream(zr.tiles[r0 : r0 + CT, j : j + 1], block=CT),
                ):
                    acc += self.product(wb, zb)
            out = (row + 1) * CT + ro
            y.tiles[out : out + 1, j : j + 1].write(acc)


def shift_matrices():
    """[4 x 32, 32]: tiles 0, 1 take each pixel's left neighbour from the tile before
    and the tile itself; tiles 2, 3 the right one from the tile itself and the next."""
    s = np.zeros((4, 32, 32))
    s[0, 31, 0] = 1.0
    s[1, np.arange(31), np.arange(1, 32)] = 1.0
    s[2, np.arange(1, 32), np.arange(31)] = 1.0
    s[3, 0, 31] = 1.0
    return s.reshape(128, 32)


def reference(image, weights, mode="constant"):
    """out[co, y, x]: the sum over dy, dx and ci of W[co, ci, dy, dx] times
    image[ci, y + dy - 1, x + dx - 1]; outside the image zero, or with mode="edge"
    the nearest edge value."""
    channels, height, width = image.shape
    padded = np.pad(image, ((0, 0), (1, 1), (1, 1)), mode=mode)
    out = np.zeros((weights.shape[0], height, width))
    for dy in range(3):
        for dx in range(3):
            window = padded[:, dy : dy + height, dx : dx + width]
            out += np.einsum("oc,cyx->oyx", weights[:, :, dy, dx], window)
    return out


def to_rows(image, mode="constant"):
    """[channels, H, W] as the layout's matrix: rows (y + 1, channel), columns
    x + 32; the border rows and tile columns zero, or with mode="edge" the edge's copy.
    """
    channels, height, width = image.shape
    matrix = np.pad(image, ((0, 0), (1, 1), (32, 32)), mode=mode)
    return matrix.transpose(1, 0, 2).reshape((height + 2) * channels, -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    channels, width = CT * 32, J * 32
    image = random(rng, (channels, H, width))
    weights = random(rng, (channels, channels, 3, 3), 1.0 / 24)  # 1 / sqrt(9 x 64)
    w = weights.transpose(0, 2, 3, 1).reshape(channels, 9 * channels)  # (dy, dx, ci)
    zeros = np.zeros((ROWS * 32, COLS * 32))
    report = {"channels": channels, "height": H, "width": width}
    with km.Session(log=log) as session:
        s, tw = tiles(session, shift_matrices()), tiles(session, w)
        inputs = {mode: tiles(session, to_rows(image, mode)) for mode in MODES}
        x = inputs["constant"]
        zeros_t = tiles(session, zeros)
        made = {}  # border: the input, its shifted copies and the output
        run = session.load(
            Shift(x=x, s=s, zl=zeros_t, zr=zeros_t, session=session, log=log)
        )
        try:
            for mode in MODES:  # the border zero, then the edge's copy
                zl, zr = tiles(session, zeros), tiles(session, zeros)
                call = (inputs[mode], s, zl, zr)
                report[f"{mode}_shift_us"] = timed(run, Shift.shift, (0, ROWS), call)
                made[mode] = (mode, inputs[mode], zl, zr, tiles(session, zeros))
        finally:
            run.stop()
        _, xm, zl, zr, y = made["constant"]
        run = session.load(
            Conv(w=tw, z0=xm, zl=zl, zr=zr, y=y, session=session, log=log)
        )
        got = {}
        try:
            for name, (mode, xm, zl, zr, y) in made.items():
                call = (tw, xm, zl, zr, y)
                report[f"{name}_conv_us"] = timed(run, Conv.conv, (0, H * CT), call)
                got[name] = mode, read(session, zl), read(session, y)
        finally:
            run.stop()
    checks = []
    for name, (mode, got_left, got_y) in got.items():
        rows = to_rows(image, mode)
        want_left = np.zeros_like(rows)
        want_left[:, 32 : 32 + width] = rows[:, 31 : 31 + width]
        report[f"{name}_shift"] = error(got_left, want_left)
        got_y = got_y.reshape(H + 2, channels, -1)[1:-1, :, 32 : 32 + width]
        want = reference(image, weights, mode)
        report[f"{name}_conv"] = error(got_y.transpose(1, 0, 2), want)
        log(f"{name}: shift {report[f'{name}_shift']}, conv {report[f'{name}_conv']}")
        checks += [(f"{name}_shift", "max", 0.0)]  # a copy at HiFi4: exact
        checks += [(f"{name}_conv", "rms", 0.01)]
    finish(report, checks)


if __name__ == "__main__":
    sys.exit(main())
