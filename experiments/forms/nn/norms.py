"""Layer norm over a row of 64 tiles, and group norm over a rectangle that is not one
row (32 channels by y by x), each against numpy. Group norm is four passes: the
squares, the sums by a product of a tile of ones on the left with the sum along the
row by the row form, the mean and rstd on the host, and the apply (README.md)."""

import argparse
import sys

import numpy as np
from conv_shift import to_rows
from forms import (
    card_options,
    error,
    finish,
    group,
    log,
    open_session,
    random,
    read,
    tiles,
    timed,
)

import komorebi as km
from komorebi.dsl.tile import row_operand

WIDE = 64  # a layer norm row: 2,048 values, 64 tiles, read in blocks of 8
CHUNKS = WIDE // 2  # the chunk form's chunks of 2 tiles: a field tile each
LN_ROWS = 4  # tile rows of the layer norm's input, a core each
H, CT, J = 16, 2, 2  # the group norm's input: conv_shift.py's layout
EPS = 1e-5


@km.tile_function(tolerance=1e-5)
def layer_norm(row, weight, bias):
    mean = km.reduce_sum(row, exact=True) * (1.0 / (32 * WIDE))
    d = row - mean.col()
    var = km.reduce_sum(d * d, exact=True) * (1.0 / (32 * WIDE))
    return d * km.rsqrt(var + EPS).col() * weight.row() + bias.row()


class LayerNorm(km.Program):
    """Each core normalises the 32 rows of its tile row of WIDE tiles."""

    weight: km.L1[km.fp32, CHUNKS, 32, 32]
    bias: km.L1[km.fp32, CHUNKS, 32, 32]

    @km.handler
    def rows(self, x: km.Tensor, w: km.Tensor, b: km.Tensor, out: km.Tensor):
        first = self.rank * WIDE
        self.weight.fill(km.Stream(w.pages(0, CHUNKS)))
        self.bias.fill(km.Stream(b.pages(0, CHUNKS)))
        results = km.Stream(out.pages(first, WIDE), write=True, block=8)
        for row in km.Stream(x.pages(first, WIDE), block=8, row=WIDE):
            results.push(layer_norm(row, self.weight, self.bias))


@km.tile_function(tolerance=1e-6)
def square(a):
    return a * a


@km.tile_function(tolerance=1e-5)
def lane_sum(row):
    """The sum along a row of 2 tiles in every column of both."""
    return row * 0.0 + km.reduce_sum(row, exact=True).col()


@km.tile_function(tolerance=1e-5)
def apply_norm(x, scale, shift):
    return x * scale.col() + shift.col()


class GroupNormStats(km.Program):
    """The squares of each tile row; then the sums over (y, x) of group g (tile row g
    of every y): a product of a tile of ones by each tile, summed in DST over y, then
    the row form's sum along the J tiles of the result."""

    @km.tile_function(fidelity="HiFi4")
    def product(a, b):
        return a @ b

    @km.handler
    def squares(self, x: km.Tensor, sq: km.Tensor):
        r = self.rank
        results = km.Stream(sq.tiles[r : r + 1, 0 : J + 2], write=True, block=2)
        for xb in km.Stream(x.tiles[r : r + 1, 0 : J + 2], block=2):
            results.push(square(xb))

    @km.handler
    def column_sums(self, ones: km.Tensor, x: km.Tensor, t: km.Tensor):
        g = self.rank
        first = CT + g  # y = 0 is tile row CT + g: tile rows 0 to CT - 1 are zeros
        acc = km.tile.zeros(km.fp32, J)
        for ob, xb in zip(
            km.Stream(ones.tiles[0:1, 0:H], block=1),
            km.Stream(x.tiles[first : first + H * CT : CT, 1 : 1 + J], block=J),
        ):
            acc += self.product(ob, xb)
        t.tiles[g : g + 1, 0:J].write(acc)

    @km.handler
    def lanes(self, t: km.Tensor, out: km.Tensor):
        r = self.rank
        results = km.Stream(out.tiles[r : r + 1, 0:J], write=True, block=J)
        for row in km.Stream(t.tiles[r : r + 1, 0:J], block=J, row=J):
            results.push(lane_sum(row))


class GroupNormApply(km.Program):
    """(x - mean) rstd gamma + beta as x * scale.col() + shift.col(), the fields
    holding group g's channels' scale and shift, over the group's tiles of every y."""

    scale: km.L1[km.fp32, 1, 32, 32]
    shift: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, sc: km.Tensor, sh: km.Tensor, y: km.Tensor):
        g = self.rank
        first = CT + g
        self.scale.fill(km.Stream(sc.pages(g, 1)))
        self.shift.fill(km.Stream(sh.pages(g, 1)))
        results = km.Stream(
            y.tiles[first : first + H * CT : CT, 1 : 1 + J], write=True, block=J
        )
        for xb in km.Stream(x.tiles[first : first + H * CT : CT, 1 : 1 + J], block=J):
            results.push(apply_norm(xb, self.scale, self.shift))


def field_pages(values):
    """A .row() field of the chunk form: chunk c's per-column tile (row_operand of
    its 64 values) in page c, [CHUNKS, 1024] float32."""
    pages = [row_operand(values[64 * c : 64 * c + 64], 2) for c in range(CHUNKS)]
    return np.asarray(pages, np.float32)


def column(values):
    """A [len, 32] float32 tile array with `values` in column 0 (read by .col())."""
    out = np.zeros((len(values), 32))
    out[:, 0] = values
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    card_options(parser)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    channels, width = CT * 32, J * 32
    xw = random(rng, (LN_ROWS * 32, WIDE * 32), 2.0) + 0.5
    weight, bias = random(rng, WIDE * 32), random(rng, WIDE * 32)
    image = random(rng, (channels, H, width)) + 0.25
    gamma, beta = random(rng, channels), random(rng, channels)
    rows = (H + 2) * CT
    report = {"wide": WIDE, "channels": channels, "height": H, "width": width}
    with open_session(args) as session:

        def at(count):
            return group(session, args, count)

        x, txw = tiles(session, to_rows(image)), tiles(session, xw, "float32")
        tw = session.tensor(field_pages(weight), "float32")
        tb = session.tensor(field_pages(bias), "float32")
        out = tiles(session, np.zeros_like(xw), "float32")
        sq = tiles(session, np.zeros((rows * 32, (J + 2) * 32)))
        y = tiles(session, np.zeros((rows * 32, (J + 2) * 32)))
        ones = tiles(session, np.ones((32, H * 32)))
        t = tiles(session, np.zeros((CT * 32, J * 32)), "float32")
        s1, s2 = (tiles(session, np.zeros((CT * 32, J * 32)), "float32") for _ in "12")
        run = session.load(
            LayerNorm(x=txw, w=tw, b=tb, out=out, session=session, log=log)
        )
        try:
            report["layer_norm_us"] = timed(
                run, LayerNorm.rows, at(LN_ROWS), (txw, tw, tb, out)
            )
        finally:
            run.stop()
        stats = GroupNormStats(
            ones=ones, x=x, t=t, out=s1, sq=sq, session=session, log=log
        )
        run = session.load(stats)
        try:
            report["squares_us"] = timed(run, GroupNormStats.squares, at(rows), (x, sq))
            for name, source, total in (("sum", x, s1), ("sum_squares", sq, s2)):
                us = timed(run, GroupNormStats.column_sums, at(CT), (ones, source, t))
                report[f"{name}_us"] = us
                report[f"{name}_lanes_us"] = timed(
                    run, GroupNormStats.lanes, at(CT), (t, total)
                )
        finally:
            run.stop()
        got1, got2 = read(session, s1), read(session, s2)
        count = 32 * H * width  # values of a group
        mean = got1[::32, 0] / count  # the first value of each group's tile row
        var = got2[::32, 0] / count - mean * mean
        rstd = 1.0 / np.sqrt(var + EPS)
        scale = np.repeat(rstd, 32) * gamma
        shift = beta - np.repeat(mean, 32) * scale
        sc, sh = tiles(session, column(scale), "float32"), tiles(
            session, column(shift), "float32"
        )
        run = session.load(
            GroupNormApply(x=x, sc=sc, sh=sh, y=y, session=session, log=log)
        )
        try:
            report["apply_us"] = timed(
                run, GroupNormApply.apply, at(CT), (x, sc, sh, y)
            )
        finally:
            run.stop()
        got_ln, got_y = read(session, out), read(session, y)
    m = xw.mean(axis=1, keepdims=True)
    v = ((xw - m) ** 2).mean(axis=1, keepdims=True)
    report["layer_norm"] = error(got_ln, (xw - m) / np.sqrt(v + EPS) * weight + bias)
    groups = image.reshape(CT, 32, H, width)
    want1 = groups.sum(axis=(1, 2, 3))
    want2 = (groups * groups).sum(axis=(1, 2, 3))
    report["group_sum"] = error(
        got1.reshape(CT, 32, -1)[:, :, :].mean(axis=(1, 2)), want1
    )
    report["group_sum_spread"] = float(np.ptp(got1.reshape(CT, -1), axis=1).max())
    report["group_sum_squares"] = error(got2[::32, 0], want2)
    gm = groups.mean(axis=(1, 2, 3), keepdims=True)
    gv = groups.var(axis=(1, 2, 3), keepdims=True)
    want = ((groups - gm) / np.sqrt(gv + EPS)).reshape(channels, H, width)
    want = want * gamma[:, None, None] + beta[:, None, None]
    got = got_y.reshape(H + 2, channels, -1)[1:-1, :, 32 : 32 + width].transpose(
        1, 0, 2
    )
    report["group_norm"] = error(got, want)
    for name in ("layer_norm", "group_sum", "group_sum_squares", "group_norm"):
        log(f"{name}: {report[name]}")
    finish(
        report,
        [("layer_norm", "rms", 0.01), ("group_sum", "max", 1e-2),
         ("group_sum_squares", "max", 1e-2), ("group_norm", "rms", 0.01)],
    )  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
