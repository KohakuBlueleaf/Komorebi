"""Resampling a feature map by 2 in conv_shift.py's layout, against numpy: up nearest,
up bilinear (half-pixel centres, edges clamped), down nearest and down by the mean of
2 x 2. Along x a product by a constant matrix on the right, along y a choice of tile
rows and a weighted sum of two; the border rows and columns hold the edge's copy.
Example: python resample.py"""

import argparse
import sys

import numpy as np
from forms import error, finish, log, random, read, tiles, timed

import komorebi as km

CT, H, J = 2, 8, 2  # channel tiles, rows and tile columns of the smaller map


@km.tile_function(tolerance=1e-6)
def lerp(a, b):
    return 0.75 * a + 0.25 * b


@km.tile_function(tolerance=1e-6)
def mean2(a, b):
    return 0.5 * (a + b)


class Up(km.Program):
    """Up by 2: the product along x into z, then the rows along y into y. The products
    at HiFi4: x keeps its 8 bits, so nearest is a copy (conv_shift.py)."""

    @km.tile_function(fidelity="HiFi4")
    def product(a, b):
        return a @ b

    @km.handler
    def up_x_nearest(self, x: km.Tensor, d: km.Tensor, z: km.Tensor):
        """Tile column j of x gives output tile columns 2j - 1 and 2j."""
        r = self.rank
        for j in range(1, J + 1):
            for half in range(2):
                acc = km.tile.zeros(km.fp32, 1)
                for xb, db in zip(
                    km.Stream(x.tiles[r : r + 1, j : j + 1], block=1),
                    km.Stream(d.tiles[half : half + 1, 0:1], block=1),
                ):
                    acc += self.product(xb, db)
                o = 2 * j - 1 + half
                z.tiles[r : r + 1, o : o + 1].write(acc)

    @km.handler
    def up_x_bilinear(self, x: km.Tensor, d: km.Tensor, z: km.Tensor):
        """Output tile 2j - 1 + half = [X(j - 1 + half), X(j + half)] @ D(half)."""
        r = self.rank
        for j in range(1, J + 1):
            for half in range(2):
                acc = km.tile.zeros(km.fp32, 1)
                k = j - 1 + half
                t = half * 2
                for xb, db in zip(
                    km.Stream(x.tiles[r : r + 1, k : k + 2], block=2),
                    km.Stream(d.tiles[t : t + 2, 0:1], block=2),
                ):
                    acc += self.product(xb, db)
                o = 2 * j - 1 + half
                z.tiles[r : r + 1, o : o + 1].write(acc)

    @km.handler
    def up_y_nearest(self, z: km.Tensor, y: km.Tensor):
        """Output rows 2y and 2y + 1 are input row y."""
        row = self.rank // CT
        c = self.rank % CT
        src = (row + 1) * CT + c
        for k in range(2):
            dst = (2 * row + k + 1) * CT + c
            results = km.Stream(
                y.tiles[dst : dst + 1, 1 : 1 + 2 * J], write=True, block=4
            )
            for tile in km.Stream(z.tiles[src : src + 1, 1 : 1 + 2 * J], block=4):
                results.push(tile)

    @km.handler
    def up_y_bilinear(self, z: km.Tensor, y: km.Tensor):
        """Output row 2y = 0.75 in[y] + 0.25 in[y - 1], 2y + 1 = 0.75 in[y] + 0.25
        in[y + 1]."""
        row = self.rank // CT
        c = self.rank % CT
        src = (row + 1) * CT + c
        up = src - CT
        down = src + CT
        dst0 = (2 * row + 1) * CT + c
        dst1 = (2 * row + 2) * CT + c
        out0 = km.Stream(y.tiles[dst0 : dst0 + 1, 1 : 1 + 2 * J], write=True, block=4)
        for a, b in zip(
            km.Stream(z.tiles[src : src + 1, 1 : 1 + 2 * J], block=4),
            km.Stream(z.tiles[up : up + 1, 1 : 1 + 2 * J], block=4),
        ):
            out0.push(lerp(a, b))
        out1 = km.Stream(y.tiles[dst1 : dst1 + 1, 1 : 1 + 2 * J], write=True, block=4)
        for a, b in zip(
            km.Stream(z.tiles[src : src + 1, 1 : 1 + 2 * J], block=4),
            km.Stream(z.tiles[down : down + 1, 1 : 1 + 2 * J], block=4),
        ):
            out1.push(lerp(a, b))


class Down(km.Program):
    """Down by 2 (a class of its own: its tensors' widths are not Up's)."""

    @km.tile_function(fidelity="HiFi4")
    def product(a, b):
        return a @ b

    @km.handler
    def down_x(self, x: km.Tensor, s: km.Tensor, z: km.Tensor):
        """Output tile j = [X(2j - 1), X(2j)] @ S: a selection or a pair's mean."""
        r = self.rank
        for j in range(1, J + 1):
            acc = km.tile.zeros(km.fp32, 1)
            k = 2 * j - 1
            for xb, sb in zip(
                km.Stream(x.tiles[r : r + 1, k : k + 2], block=2),
                km.Stream(s.tiles[0:2, 0:1], block=2),
            ):
                acc += self.product(xb, sb)
            z.tiles[r : r + 1, j : j + 1].write(acc)

    @km.handler
    def down_y_nearest(self, z: km.Tensor, y: km.Tensor):
        """Output row y is input row 2y."""
        row = self.rank // CT
        c = self.rank % CT
        src = (2 * row + 1) * CT + c
        dst = (row + 1) * CT + c
        results = km.Stream(y.tiles[dst : dst + 1, 1 : 1 + J], write=True, block=J)
        for tile in km.Stream(z.tiles[src : src + 1, 1 : 1 + J], block=J):
            results.push(tile)

    @km.handler
    def down_y_mean(self, z: km.Tensor, y: km.Tensor):
        """Output row y is the mean of input rows 2y and 2y + 1."""
        row = self.rank // CT
        c = self.rank % CT
        a0 = (2 * row + 1) * CT + c
        a1 = a0 + CT
        dst = (row + 1) * CT + c
        results = km.Stream(y.tiles[dst : dst + 1, 1 : 1 + J], write=True, block=J)
        for a, b in zip(
            km.Stream(z.tiles[a0 : a0 + 1, 1 : 1 + J], block=J),
            km.Stream(z.tiles[a1 : a1 + 1, 1 : 1 + J], block=J),
        ):
            results.push(mean2(a, b))


def matrices():
    """(nearest up [2 x 32, 32], bilinear up [4 x 32, 32], down selection and down mean
    [2 x 32, 32]): the products' right operands."""
    c = np.arange(32)
    near = np.zeros((2, 32, 32))
    for half in range(2):
        near[half, half * 16 + c // 2, c] = 1.0
    bil = np.zeros((2, 64, 32))
    for half in range(2):
        base = (32 if half == 0 else 16) + c // 2
        bil[half, base, c] = 0.75
        bil[half, base + np.where(c % 2, 1, -1), c] += 0.25
    pick, mean = np.zeros((64, 32)), np.zeros((64, 32))
    pick[2 * c, c] = 1.0
    mean[2 * c, c] = mean[2 * c + 1, c] = 0.5
    return near.reshape(64, 32), bil.reshape(128, 32), pick, mean


def layout(image):
    """[channels, H, W] as rows (y + 1, channel), columns x + 32, the border rows and
    tile columns holding the edge's copy."""
    channels, height, width = image.shape
    padded = np.pad(image, ((0, 0), (1, 1), (32, 32)), mode="edge")
    return padded.transpose(1, 0, 2).reshape((height + 2) * channels, width + 64)


def interior(matrix, height, channels, width):
    m = matrix.reshape(height + 2, channels, -1)[1:-1, :, 32 : 32 + width]
    return m.transpose(1, 0, 2)


def up_bilinear(image):
    """x2 along both axes with half-pixel centres and the edges clamped."""
    p = np.pad(image, ((0, 0), (1, 1), (1, 1)), mode="edge")
    rows = np.empty((image.shape[0], 2 * image.shape[1], p.shape[2]))
    rows[:, 0::2] = 0.75 * p[:, 1:-1] + 0.25 * p[:, :-2]
    rows[:, 1::2] = 0.75 * p[:, 1:-1] + 0.25 * p[:, 2:]
    out = np.empty((image.shape[0], rows.shape[1], 2 * image.shape[2]))
    out[:, :, 0::2] = 0.75 * rows[:, :, 1:-1] + 0.25 * rows[:, :, :-2]
    out[:, :, 1::2] = 0.75 * rows[:, :, 1:-1] + 0.25 * rows[:, :, 2:]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    channels, width = CT * 32, J * 32
    small = random(rng, (channels, H, width))
    large = random(rng, (channels, 2 * H, 2 * width))
    near, bil, pick, mean = matrices()
    report = {"channels": channels, "small": [H, width], "large": [2 * H, 2 * width]}
    results = {}
    with km.Session(log=log) as session:
        x_up, x_dn = tiles(session, layout(small)), tiles(session, layout(large))
        z_up = tiles(session, np.zeros(((H + 2) * channels, (2 * J + 2) * 32)))
        y_up = tiles(session, np.zeros(((2 * H + 2) * channels, (2 * J + 2) * 32)))
        z_dn = tiles(session, np.zeros(((2 * H + 2) * channels, (J + 2) * 32)))
        y_dn = tiles(session, np.zeros(((H + 2) * channels, (J + 2) * 32)))
        tn, tb, tp, tm = (tiles(session, m) for m in (near, bil, pick, mean))
        small_rows, large_rows = (0, (H + 2) * CT), (0, (2 * H + 2) * CT)
        out_rows = (0, H * CT)
        program = Up(x=x_up, d=tn, z=z_up, y=y_up, session=session, log=log)
        run = session.load(program)
        try:
            for name, x_handler, matrix, y_handler in (
                ("up_nearest", Up.up_x_nearest, tn, Up.up_y_nearest),
                ("up_bilinear", Up.up_x_bilinear, tb, Up.up_y_bilinear),
            ):
                us = timed(run, x_handler, small_rows, (x_up, matrix, z_up))
                us2 = timed(run, y_handler, out_rows, (z_up, y_up))
                got = interior(read(session, y_up), 2 * H, channels, 2 * width)
                results[name] = (name, got, [us, us2])
        finally:
            run.stop()
        program = Down(x=x_dn, s=tp, z=z_dn, y=y_dn, session=session, log=log)
        run = session.load(program)
        try:
            for name, matrix, y_handler in (
                ("down_nearest", tp, Down.down_y_nearest),
                ("down_mean", tm, Down.down_y_mean),
            ):
                us = timed(run, Down.down_x, large_rows, (x_dn, matrix, z_dn))
                us2 = timed(run, y_handler, out_rows, (z_dn, y_dn))
                got = interior(read(session, y_dn), H, channels, width)
                results[name] = (name, got, [us, us2])
        finally:
            run.stop()
    wants = {
        "up_nearest": small.repeat(2, axis=1).repeat(2, axis=2),
        "up_bilinear": up_bilinear(small),
        "down_nearest": large[:, 0::2, 0::2],
        "down_mean": large.reshape(channels, H, 2, width, 2).mean(axis=(2, 4)),
    }
    for key, (name, got, us) in results.items():
        report[key] = error(got, wants[name]) | {"us": us}
        log(f"{key}: {report[key]}")
    exact = ("up_nearest", "down_nearest")  # copies at HiFi4
    finish(report, [(k, "max" if k in exact else "rms", 0.0 if k in exact else 0.01)
                    for k in results])  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
