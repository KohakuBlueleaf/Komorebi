"""More convolutions in conv_shift.py's layout, each against numpy: 1x1 with SiLU in
the product's write, a bias as its own pass (a field read by .col()), 3x3 stride 2
(selection matrices in place of shift matrices, input rows 2y + dy) and depthwise 3x3
(a diagonal weight tile a tap and channel tile).
Example: python conv_more.py"""

import argparse
import sys

import numpy as np
from conv_shift import Shift, shift_matrices, to_rows
from forms import error, finish, log, random, read, tiles, timed

import komorebi as km

CT, H, J = 2, 16, 2  # channel tiles, rows, tile columns of the input
HO = H // 2  # rows of the stride-2 output


@km.tile_function(tolerance=1e-6)
def silu(a):
    return km.silu(a)


@km.tile_function(tolerance=1e-6)
def bias_add(a, b):
    return a + b.col()


class Pointwise(km.Program):
    """1x1 + SiLU: Y(row y, ro, j) = silu(sum over ci of W[ro, ci] X(row y, ci, j))."""

    @km.tile_function(fidelity="HiFi2")
    def product(a, b):
        return a @ b

    @km.handler
    def conv1x1(self, w: km.Tensor, x: km.Tensor, y: km.Tensor):
        row = self.rank // CT
        ro = self.rank % CT
        for j in range(J):
            acc = km.tile.zeros(km.fp32, 1)
            for wb, xb in zip(
                km.Stream(w.tiles[ro : ro + 1, 0:CT], block=CT),
                km.Stream(x.tiles[row * CT : row * CT + CT, j : j + 1], block=CT),
            ):
                acc += self.product(wb, xb)
            y.tiles[row * CT + ro : row * CT + ro + 1, j : j + 1].write(silu(acc))


class Bias(km.Program):
    """The bias as its own pass: out = in + bias.col(), a core a tile row."""

    bias: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def add(self, a: km.Tensor, bvec: km.Tensor, out: km.Tensor, tiles: int):
        first = self.rank * tiles
        self.bias.fill(km.Stream(bvec.pages(self.rank % CT, 1)))
        results = km.Stream(out.pages(first, tiles), write=True, block=2)
        for x in km.Stream(a.pages(first, tiles), block=2):
            results.push(bias_add(x, self.bias))


class Select(km.Program):
    """3x3, stride 2, padding 1, its taps: z holds tap dx of output tile column j at
    tile column dx x (J + 2) + j, the K = 3 product of input tiles 2j - 2 .. 2j by
    selection tiles. A copy, so HiFi4 (conv_shift.py)."""

    @km.tile_function(fidelity="HiFi4")
    def copy_product(a, b):
        return a @ b

    @km.handler
    def select(self, x: km.Tensor, s: km.Tensor, z: km.Tensor):
        r = self.rank
        for j in range(1, J // 2 + 1):
            k = 2 * j - 2
            for dx in range(3):
                acc = km.tile.zeros(km.fp32, 1)
                t = dx * 3
                for xb, sb in zip(
                    km.Stream(x.tiles[r : r + 1, k : k + 3], block=3),
                    km.Stream(s.tiles[t : t + 3, 0:1], block=3),
                ):
                    acc += self.copy_product(xb, sb)
                c = dx * (J + 2) + j
                z.tiles[r : r + 1, c : c + 1].write(acc)


class Stride2(km.Program):
    """The stride-2 conv over the taps, input rows 2y + dy, at HiFi2."""

    @km.tile_function(fidelity="HiFi2")
    def product(a, b):
        return a @ b

    @km.handler
    def conv(self, w: km.Tensor, z: km.Tensor, y: km.Tensor):
        row = self.rank // CT
        ro = self.rank % CT
        for j in range(1, J // 2 + 1):
            acc = km.tile.zeros(km.fp32, 1)
            for dy in range(3):
                r0 = (2 * row + dy) * CT
                for dx in range(3):
                    t0 = (dy * 3 + dx) * CT
                    c = dx * (J + 2) + j
                    for wb, zb in zip(
                        km.Stream(w.tiles[ro : ro + 1, t0 : t0 + CT], block=CT),
                        km.Stream(z.tiles[r0 : r0 + CT, c : c + 1], block=CT),
                    ):
                        acc += self.product(wb, zb)
            out = (row + 1) * CT + ro
            y.tiles[out : out + 1, j : j + 1].write(acc)


class Depthwise(km.Program):
    """Depthwise 3x3: channel c meets only channel c, so a tap's weight is the diagonal
    tile diag(w[channels, tap]) and each product has K = 1; the taps along x are
    conv_shift.py's shifted copies zl, x, zr."""

    @km.tile_function(fidelity="HiFi2")
    def product(a, b):
        return a @ b

    @km.handler
    def conv(
        self, d: km.Tensor, zl: km.Tensor, z0: km.Tensor, zr: km.Tensor, y: km.Tensor
    ):
        row = self.rank // CT
        ro = self.rank % CT
        for j in range(1, J + 1):
            acc = km.tile.zeros(km.fp32, 1)
            for dy in range(3):
                r0 = (row + dy) * CT + ro
                t0 = dy * 3 * CT + ro
                t1 = t0 + CT
                t2 = t1 + CT
                for wb, zb in zip(
                    km.Stream(d.tiles[ro : ro + 1, t0 : t0 + 1], block=1),
                    km.Stream(zl.tiles[r0 : r0 + 1, j : j + 1], block=1),
                ):
                    acc += self.product(wb, zb)
                for wb, zb in zip(
                    km.Stream(d.tiles[ro : ro + 1, t1 : t1 + 1], block=1),
                    km.Stream(z0.tiles[r0 : r0 + 1, j : j + 1], block=1),
                ):
                    acc += self.product(wb, zb)
                for wb, zb in zip(
                    km.Stream(d.tiles[ro : ro + 1, t2 : t2 + 1], block=1),
                    km.Stream(zr.tiles[r0 : r0 + 1, j : j + 1], block=1),
                ):
                    acc += self.product(wb, zb)
            out = (row + 1) * CT + ro
            y.tiles[out : out + 1, j : j + 1].write(acc)


def selection():
    """[9 x 32, 32]: tap dx's three tiles; output column c takes block pixel
    32 + 2c + dx - 1 (the block starts one tile left of the window)."""
    s = np.zeros((3, 96, 32))
    for dx in range(3):
        s[dx, 32 + 2 * np.arange(32) + dx - 1, np.arange(32)] = 1.0
    return s.reshape(288, 32)


def selected(rows):
    """Tap dx of output tile column 1: column c holds input column 2c + dx - 1
    (matrix column 2c + dx + 31), [3, rows, 32]."""
    c = np.arange(32)
    return np.stack([rows[:, 2 * c + dx + 31] for dx in range(3)])


def taps(z):
    """Stride2.select's three taps of output tile column 1 from z, [3, rows, 32]."""
    return np.stack([z[:, (dx * (J + 2) + 1) * 32 :][:, :32] for dx in range(3)])


def padded(image):
    channels, height, width = image.shape
    out = np.zeros((channels, height + 2, width + 2))
    out[:, 1:-1, 1:-1] = image
    return out


def stride2_ref(image, weights):
    p = padded(image)
    channels, height, width = image.shape
    out = np.zeros((weights.shape[0], height // 2, width // 2))
    for dy in range(3):
        for dx in range(3):
            window = p[:, dy : dy + height : 2, dx : dx + width : 2]
            out += np.einsum("oc,cyx->oyx", weights[:, :, dy, dx], window)
    return out


def depthwise_ref(image, weights):
    p = padded(image)
    channels, height, width = image.shape
    out = np.zeros_like(image)
    for dy in range(3):
        for dx in range(3):
            out += (
                weights[:, dy, dx, None, None] * p[:, dy : dy + height, dx : dx + width]
            )
    return out


def plain_rows(image):
    """[channels, H, W] as rows (y, channel), columns x, no zero rows or columns."""
    channels, height, width = image.shape
    return image.transpose(1, 0, 2).reshape(height * channels, width)


def from_padded(matrix, height, channels, width):
    m = matrix.reshape(height + 2, channels, -1)[1:-1, :, 32 : 32 + width]
    return m.transpose(1, 0, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    channels, width = CT * 32, J * 32
    image = random(rng, (channels, H, width))
    w1 = random(rng, (channels, channels), 1.0 / 8)
    w9 = random(rng, (channels, channels, 3, 3), 1.0 / 24)
    wd = random(rng, (channels, 3, 3), 1.0 / 3)
    bias = random(rng, (channels,))
    diag = np.zeros((channels, 9 * channels))  # (dy, dx, channel) blocks
    for tap in range(9):
        diag[np.arange(channels), tap * channels + np.arange(channels)] = wd.reshape(
            channels, 9
        )[:, tap]
    bias_tiles = np.zeros((CT * 32, 32))
    bias_tiles[:, 0] = bias
    rows = (H + 2) * CT
    report = {"channels": channels, "height": H, "width": width}
    results = {}
    with km.Session(log=log) as session:
        x = tiles(session, to_rows(image))
        flat = tiles(session, plain_rows(image))

        def zero(columns):
            return tiles(session, np.zeros((rows * 32, columns * 32)))

        zl, zr, y3, yd = (zero(J + 2) for _ in range(4))
        z = zero(3 * (J + 2))
        y1, yb = (tiles(session, np.zeros((H * channels, width))) for _ in range(2))
        tw1 = tiles(session, w1)
        tw9 = tiles(session, w9.transpose(0, 2, 3, 1).reshape(channels, 9 * channels))
        tdiag, tsel = tiles(session, diag), tiles(session, selection())
        tshift = tiles(session, shift_matrices())
        tbias = tiles(session, bias_tiles, "float32")
        programs = [
            Pointwise(w=tw1, x=flat, y=y1, session=session, log=log),
            Bias(a=flat, bvec=tbias, out=yb, session=session, log=log),
            Select(x=x, s=tsel, z=z, session=session, log=log),
            Stride2(w=tw9, z=z, y=y3, session=session, log=log),
            Shift(x=x, s=tshift, zl=zl, zr=zr, session=session, log=log),
            Depthwise(d=tdiag, zl=zl, z0=x, zr=zr, y=yd, session=session, log=log),
        ]
        jobs = [
            [(Pointwise.conv1x1, (0, H * CT), (tw1, flat, y1))],
            [(Bias.add, (0, H * CT), (flat, tbias, yb, J))],
            [(Select.select, (0, rows), (x, tsel, z))],
            [(Stride2.conv, (0, HO * CT), (tw9, z, y3))],
            [(Shift.shift, (0, rows), (x, tshift, zl, zr))],
            [(Depthwise.conv, (0, H * CT), (tdiag, zl, x, zr, yd))],
        ]  # fmt: skip
        for program, commands in zip(programs, jobs, strict=True):
            run = session.load(program)
            try:
                for handler, group, call in commands:
                    key = f"{type(program).__name__}.{handler.name}_us"  # a list
                    report.setdefault(key, []).append(timed(run, handler, group, call))
            finally:
                run.stop()
            log(f"{type(program).__name__}: {program.sizes.text}")
        got1, gotb = read(session, y1), read(session, yb)
        got3, gotd = read(session, y3), read(session, yd)
        gotz = read(session, z)
    want1 = np.einsum("oc,cyx->oyx", w1, image)
    results["conv1x1_silu"] = (got1, plain_rows(want1 / (1 + np.exp(-want1))))
    results["bias_pass"] = (gotb, plain_rows(image + bias[:, None, None]))
    results["stride2"] = (
        from_padded(got3, H, channels, width)[:, :HO, : width // 2],
        stride2_ref(image, w9),
    )
    picked = selected(to_rows(image))
    results["selection"] = (taps(gotz), picked)
    results["depthwise"] = (
        from_padded(gotd, H, channels, width),
        depthwise_ref(image, wd),
    )
    for name, (got, want) in results.items():
        report[name] = error(got, want)
        log(f"{name}: {report[name]}")
    checks = [(name, "rms", 0.01) for name in results if name != "selection"]
    finish(report, checks + [("selection", "max", 0.0)])  # a copy at HiFi4: exact


if __name__ == "__main__":
    sys.exit(main())
