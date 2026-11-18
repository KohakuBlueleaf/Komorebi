"""Group norm written as one class from the existing forms, every step on the card,
against numpy: the squares, the sums (a product of a tile of ones, then the row form's
lane sum), the derive (mean and rstd into scale and shift tiles) and the apply, the
image group-major (tile row g H + y); as commands, and with --chain as one chain."""

import argparse
import sys

import numpy as np
from forms import (
    across_list,
    card_options,
    copied,
    error,
    finish,
    group,
    log,
    open_session,
    placed,
    random,
    read,
    tiles,
    timed,
)
from norms import CT, EPS, H, J, apply_norm, column, lane_sum, square

import komorebi as km

COUNT = 32 * H * J * 32  # values of a group: 32 channels by H rows by J x 32 columns


@km.tile_function(tolerance=1e-5)
def scale_of(s1, s2, gamma):
    """gamma rstd, from the group's sum and sum of squares (every lane holds them)."""
    mean = s1 * (1.0 / COUNT)
    var = s2 * (1.0 / COUNT) - mean * mean
    return gamma * km.rsqrt(var + EPS)


@km.tile_function(tolerance=1e-5)
def shift_of(s1, scale, beta):
    return beta - s1 * (1.0 / COUNT) * scale


class GroupNorm(km.Program):
    """The steps' handlers; group g is tile rows g H to g H + H - 1 (rank g)."""

    scale: km.L1[km.fp32, 1, 32, 32]
    shift: km.L1[km.fp32, 1, 32, 32]

    @km.tile_function(fidelity="HiFi4")
    def product(a, b):
        return a @ b

    @km.handler
    def squares(self, x: km.Tensor, sq: km.Tensor):
        r = self.rank
        results = km.Stream(sq.tiles[r : r + 1, 0:J], write=True, block=J)
        for xb in km.Stream(x.tiles[r : r + 1, 0:J], block=J):
            results.push(square(xb))

    @km.handler
    def sums(self, ones: km.Tensor, x: km.Tensor, sq: km.Tensor, t: km.Tensor):
        """Tile row g of t: x's column sums in tiles 0 to J - 1, sq's after them."""
        g = self.rank
        acc = km.tile.zeros(km.fp32, J)
        for ob, xb in zip(
            km.Stream(ones.tiles[0:1, 0:H], block=1),
            km.Stream(x.tiles[g * H : g * H + H, 0:J], block=J),
        ):
            acc += self.product(ob, xb)
        t.tiles[g : g + 1, 0:J].write(acc)
        acc2 = km.tile.zeros(km.fp32, J)
        for ob, qb in zip(
            km.Stream(ones.tiles[0:1, 0:H], block=1),
            km.Stream(sq.tiles[g * H : g * H + H, 0:J], block=J),
        ):
            acc2 += self.product(ob, qb)
        t.tiles[g : g + 1, J : 2 * J].write(acc2)

    @km.handler
    def lanes(self, t: km.Tensor, s: km.Tensor):
        """Tile row g of s: the lane sums of t's two rows of J tiles."""
        g = self.rank
        results = km.Stream(s.tiles[g : g + 1, 0 : 2 * J], write=True, block=J)
        for row in km.Stream(t.tiles[g : g + 1, 0 : 2 * J], block=J, row=J):
            results.push(lane_sum(row))

    @km.handler
    def derive_scale(self, s: km.Tensor, gm: km.Tensor, sc: km.Tensor):
        g = self.rank
        results = km.Stream(sc.tiles[g : g + 1, 0:1], write=True)
        for a, b, c in zip(
            km.Stream(s.tiles[g : g + 1, 0:1]),
            km.Stream(s.tiles[g : g + 1, J : J + 1]),
            km.Stream(gm.tiles[g : g + 1, 0:1]),
        ):
            results.push(scale_of(a, b, c))

    @km.handler
    def derive_shift(self, s: km.Tensor, sc: km.Tensor, bt: km.Tensor, sh: km.Tensor):
        g = self.rank
        results = km.Stream(sh.tiles[g : g + 1, 0:1], write=True)
        for a, b, c in zip(
            km.Stream(s.tiles[g : g + 1, 0:1]),
            km.Stream(sc.tiles[g : g + 1, 0:1]),
            km.Stream(bt.tiles[g : g + 1, 0:1]),
        ):
            results.push(shift_of(a, b, c))

    @km.handler
    def apply(self, x: km.Tensor, sc: km.Tensor, sh: km.Tensor, y: km.Tensor):
        g = self.rank
        self.scale.fill(km.Stream(sc.pages(g, 1)))
        self.shift.fill(km.Stream(sh.pages(g, 1)))
        results = km.Stream(y.tiles[g * H : g * H + H, 0:J], write=True, block=J)
        for xb in km.Stream(x.tiles[g * H : g * H + H, 0:J], block=J):
            results.push(apply_norm(xb, self.scale, self.shift))


def group_major(image):
    """(channels, H, width) as tile rows g H + y of group g's 32 channels at y."""
    return image.reshape(CT, 32, H, -1).transpose(0, 2, 1, 3).reshape(CT * H * 32, -1)


def reference(image, gamma, beta):
    """numpy's group norm of image (channels, H, width), 32 channels a group."""
    groups = image.reshape(CT, 32, H, -1)
    mean = groups.mean(axis=(1, 2, 3), keepdims=True)
    var = groups.var(axis=(1, 2, 3), keepdims=True)
    want = ((groups - mean) / np.sqrt(var + EPS)).reshape(image.shape)
    return want * gamma[:, None, None] + beta[:, None, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--chain", action="store_true", help="also as one chain")
    card_options(parser)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    channels, width = CT * 32, J * 32
    image = random(rng, (channels, H, width)) + 0.25
    gamma, beta = random(rng, channels), random(rng, channels)
    rows = H * CT
    report = {"channels": channels, "height": H, "width": width, "place": args.place}
    got = {}
    with open_session(args) as session:
        across = across_list(args)

        def at(count):
            return group(session, args, count)

        own, cts = placed(session, args, rows), placed(session, args, CT)
        ones = tiles(session, np.ones((32, H * 32)), **copied(args))
        gm = tiles(session, column(gamma), "float32", **cts)
        bt = tiles(session, column(beta), "float32", **cts)
        sums = np.zeros((CT * 32, 2 * J * 32))

        def tensors():
            x = tiles(session, group_major(image), **own)
            sq, y = (tiles(session, np.zeros((rows * 32, width)), **own) for _ in "ab")
            t, s = (tiles(session, sums, "float32", **cts) for _ in "ab")
            sc, sh = (
                tiles(session, np.zeros((CT * 32, 32)), "float32", **cts) for _ in "ab"
            )
            return {"x": x, "sq": sq, "y": y, "t": t, "s": s, "sc": sc, "sh": sh}

        def calls(n, back=lambda t: t):
            """The six calls; `back` marks a read of a call further back."""
            return [
                ("squares", rows, GroupNorm.squares, (n["x"], n["sq"])),
                ("sums", CT, GroupNorm.sums, (ones, n["x"], n["sq"], n["t"])),
                ("lanes", CT, GroupNorm.lanes, (n["t"], n["s"])),
                ("scale", CT, GroupNorm.derive_scale, (n["s"], gm, n["sc"])),
                (
                    "shift",
                    CT,
                    GroupNorm.derive_shift,
                    (back(n["s"]), n["sc"], bt, n["sh"]),
                ),
                (
                    "apply",
                    CT,
                    GroupNorm.apply,
                    (n["x"], back(n["sc"]), n["sh"], n["y"]),
                ),
            ]

        sets = {"commands": tensors()}
        if args.chain:
            sets["chain"] = tensors()
        made = dict(sets["commands"], ones=ones, gm=gm, bt=bt)
        run = session.load(GroupNorm(**made, session=session, log=log))
        try:
            for name, count, handler, call in calls(sets["commands"]):
                us = timed(run, handler, at(count), call, across=across)
                report[f"{name}_us"] = us
        finally:
            run.stop()
        got["commands"] = read(session, sets["commands"]["y"])
        if args.chain:
            try:
                made = dict(sets["chain"], ones=ones, gm=gm, bt=bt, chain=True)
                run = session.load(GroupNorm(**made, session=session, log=log))
                try:
                    steps = [
                        handler(*call, cores=at(count))
                        for _, count, handler, call in calls(sets["chain"], km.past)
                    ]
                    chain = run.chain(steps, cores=at(rows))
                    event = run.wait(run.submit(chain), timeout=30.0)
                    report["chain"] = event.kind
                    report["chain_us"] = round(event.microseconds, 1)
                finally:
                    run.stop()
                got["chain"] = read(session, sets["chain"]["y"])
            except Exception as refused:  # noqa: BLE001 - the refusal is the result
                report["chain"] = f"{type(refused).__name__}: {refused}"[:700]
            log(f"one chain: {report['chain']}")
    if across is not None:
        report["across"] = across
    want = reference(image, gamma, beta)
    checks = []
    for name, got_y in got.items():
        result = got_y.reshape(CT, H, 32, width).transpose(0, 2, 1, 3)
        key = "group_norm" if name == "commands" else name
        report[key] = error(result.reshape(channels, H, width), want)
        checks.append((key, "rms", 0.01))
    log(f"group norm: {report['group_norm']}")
    finish(report, checks)


if __name__ == "__main__":
    sys.exit(main())
