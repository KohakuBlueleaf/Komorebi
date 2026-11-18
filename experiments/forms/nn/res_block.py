"""One residual block of a UNet-style model written from the existing forms, against
numpy: conv 3x3, group norm and SiLU, conv 3x3, the skip added, then the nearest 2x
upsample; as commands, and with --chain each of its three runs as one chain (the
block's buffers do not fit one class; README.md)."""

import argparse
import sys

import numpy as np
from conv_shift import reference as conv_reference
from conv_shift import shift_matrices, to_rows
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
from group_norm import reference as norm_reference
from norms import EPS, column, lane_sum, square
from resample import matrices

import komorebi as km

CT, H, J = 2, 16, 2  # channel tiles (64 channels, 2 groups of 32), rows, tile columns
ROWS, COLS = (H + 2) * CT, J + 2  # tile rows and columns with the border
UP_J = 2 * J  # the upsampled map's tile columns
UP_ROWS, UP_COLS = (2 * H + 2) * CT, UP_J + 2  # its tile rows and columns, bordered
TAPS = 9 * CT  # a conv's K in tiles: 9 taps of CT channel tiles
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


@km.tile_function(tolerance=1e-5)
def norm_silu(x, scale, shift):
    return km.silu(x * scale.col() + shift.col())


@km.tile_function(tolerance=1e-6)
def added(a, b):
    return a + b


class Convs(km.Program):
    """The convs' copies and taps, the skip and the upsample. x, the norm's output and
    the block's are in conv_shift's layout; z and a conv's output are not bordered."""

    @km.tile_function(fidelity="HiFi4")
    def product(a, b):
        return a @ b

    @km.handler
    def copies(self, x: km.Tensor, sm: km.Tensor, z: km.Tensor):
        """Tile row r of x into z's rows 3 r + dx (J tiles, no border): its left
        neighbours (dx 0), itself (1) and its right neighbours (2)."""
        r = self.rank
        base = r * 3
        mid = base + 1
        right = base + 2
        results = km.Stream(z.tiles[mid : mid + 1, 0:J], write=True, block=J)
        for xb in km.Stream(x.tiles[r : r + 1, 1 : 1 + J], block=J):
            results.push(xb)
        for j in range(J):
            acc = km.tile.zeros(km.fp32, 1)
            for xb, sb in zip(
                km.Stream(x.tiles[r : r + 1, j : j + 2], block=2),
                km.Stream(sm.tiles[0:2, 0:1], block=2),
            ):
                acc += self.product(xb, sb)
            z.tiles[base : base + 1, j : j + 1].write(acc)
            k = j + 1
            acc2 = km.tile.zeros(km.fp32, 1)
            for xb, sb in zip(
                km.Stream(x.tiles[r : r + 1, k : k + 2], block=2),
                km.Stream(sm.tiles[2:4, 0:1], block=2),
            ):
                acc2 += self.product(xb, sb)
            z.tiles[right : right + 1, j : j + 1].write(acc2)

    @km.handler
    def conv(self, w: km.Tensor, z: km.Tensor, y: km.Tensor):
        """Tile row r = ro H + row of y (group-major): the 9 taps' products, z's rows
        row 3 CT to row 3 CT + 9 CT - 1 in the weights' (dy, ci, dx) order."""
        r = self.rank
        ro = r // H
        first = (r % H) * 3 * CT
        for j in range(J):
            acc = km.tile.zeros(km.fp32, 1)
            for wb, zb in zip(
                km.Stream(w.tiles[ro : ro + 1, 0:TAPS], block=CT),
                km.Stream(z.tiles[first : first + TAPS, j : j + 1], block=CT),
            ):
                acc += self.product(wb, zb)
            y.tiles[r : r + 1, j : j + 1].write(acc)

    @km.handler
    def skip(self, x: km.Tensor, y: km.Tensor, out: km.Tensor):
        """x's tile row r + CT (row, c) plus y's row c H + row."""
        c = self.rank % CT
        r = self.rank + CT
        q = c * H + self.rank // CT
        results = km.Stream(out.tiles[r : r + 1, 1 : 1 + J], write=True, block=J)
        for a, b in zip(
            km.Stream(x.tiles[r : r + 1, 1 : 1 + J], block=J),
            km.Stream(y.tiles[q : q + 1, 0:J], block=J),
        ):
            results.push(added(a, b))

    @km.handler
    def up_x(self, x: km.Tensor, d: km.Tensor, v: km.Tensor):
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
                v.tiles[r : r + 1, o : o + 1].write(acc)

    @km.handler
    def up_y(self, v: km.Tensor, u: km.Tensor):
        """Output row y (rank y CT + c) is input row y // 2."""
        c = self.rank % CT
        dst = self.rank + CT
        src = (self.rank // CT // 2 + 1) * CT + c
        results = km.Stream(
            u.tiles[dst : dst + 1, 1 : 1 + UP_J], write=True, block=UP_J
        )
        for tile in km.Stream(v.tiles[src : src + 1, 1 : 1 + UP_J], block=UP_J):
            results.push(tile)


class Norm(km.Program):
    """Group norm and SiLU of a conv's group-major output (tile row g H + row); a
    group is a channel tile (rank g of CT); the output in conv_shift's layout."""

    scale: km.L1[km.fp32, 1, 32, 32]
    shift: km.L1[km.fp32, 1, 32, 32]

    @km.tile_function(fidelity="HiFi4")
    def product(a, b):
        return a @ b

    @km.handler
    def squares(self, y: km.Tensor, sq: km.Tensor):
        r = self.rank
        results = km.Stream(sq.tiles[r : r + 1, 0:J], write=True, block=J)
        for xb in km.Stream(y.tiles[r : r + 1, 0:J], block=J):
            results.push(square(xb))

    @km.handler
    def sums(self, ones: km.Tensor, y: km.Tensor, sq: km.Tensor, t: km.Tensor):
        """Tile row g of t: y's column sums in tiles 0 to J - 1, sq's after them."""
        g = self.rank
        acc = km.tile.zeros(km.fp32, J)
        for ob, xb in zip(
            km.Stream(ones.tiles[0:1, 0:H], block=1),
            km.Stream(y.tiles[g * H : g * H + H, 0:J], block=J),
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
    def apply(self, y: km.Tensor, sc: km.Tensor, sh: km.Tensor, n: km.Tensor):
        """SiLU of the group norm of y's row g H + row, into n's row r + CT."""
        g = self.rank % CT
        q = g * H + self.rank // CT
        r = self.rank + CT
        self.scale.fill(km.Stream(sc.pages(g, 1)))
        self.shift.fill(km.Stream(sh.pages(g, 1)))
        results = km.Stream(n.tiles[r : r + 1, 1 : 1 + J], write=True, block=J)
        for xb in km.Stream(y.tiles[q : q + 1, 0:J], block=J):
            results.push(norm_silu(xb, self.scale, self.shift))


def reference(image, w1, w2, gamma, beta):
    """numpy's block (float64): the output before the upsample, and after it."""
    a = conv_reference(image, w1)
    a = norm_reference(a, gamma, beta)
    a = a / (1.0 + np.exp(-a))
    out = image + conv_reference(a, w2)
    return out, out.repeat(2, axis=1).repeat(2, axis=2)


def interior(values, rows):
    """The layout's matrix [(rows + 2) channels, width + 64] as [channels, rows, W]."""
    m = values.reshape(rows + 2, CT * 32, -1)[1:-1, :, 32:-32]
    return m.transpose(1, 0, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--chain", action="store_true", help="also as chains")
    card_options(parser)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    channels, width = CT * 32, J * 32
    image = random(rng, (channels, H, width))
    weights = [random(rng, (channels, channels, 3, 3), 1.0 / 24) for _ in "12"]
    gamma, beta = random(rng, channels), random(rng, channels)
    report = {"channels": channels, "height": H, "width": width, "place": args.place}
    got = {}
    with open_session(args) as session:
        across = across_list(args)

        def at(count):
            return group(session, args, count)

        own = placed(session, args, ROWS)
        sm = tiles(session, shift_matrices(), **copied(args))
        d = tiles(session, matrices()[0], **copied(args))
        tw = [
            tiles(
                session,
                w.reshape(channels, CT, 32, 3, 3)
                .transpose(0, 3, 1, 4, 2)
                .reshape(channels, 9 * channels),
                **copied(args),
            )
            for w in weights
        ]
        ones = tiles(session, np.ones((32, H * 32)), **copied(args))
        gm = tiles(session, column(gamma), "float32", **copied(args))
        bt = tiles(session, column(beta), "float32", **copied(args))
        x = tiles(session, to_rows(image), **own)
        zeros = np.zeros((ROWS * 32, COLS * 32))
        taps = np.zeros((3 * ROWS * 32, J * 32))  # copies: rows 3 r + dx
        plain = np.zeros((H * CT * 32, J * 32))  # a conv's output, group-major
        flat = placed(session, args, H * CT)

        def tensors():
            n = {k: tiles(session, zeros, **own) for k in ("g", "out")}
            n |= {k: tiles(session, plain, **flat) for k in ("y1", "sq", "y2")}
            n |= {k: tiles(session, taps, **own) for k in ("z1", "z2")}
            n["t"], n["s"] = (
                tiles(session, np.zeros((CT * 32, 2 * J * 32)), "float32") for _ in "ab"
            )
            n["sc"], n["sh"] = (
                tiles(session, np.zeros((CT * 32, 32)), "float32") for _ in "ab"
            )
            n["v"] = tiles(session, np.zeros((ROWS * 32, UP_COLS * 32)), **own)
            n["u"] = tiles(session, np.zeros((UP_ROWS * 32, UP_COLS * 32)), **own)
            return n

        def segments(n, back=lambda t: t):
            """The block's calls in three runs (one class can not hold the block's
            buffers): the first conv, the norm, the rest; `back`: a read further back.
            """
            inner = H * CT
            return [
                [
                    ("copies1", ROWS, Convs.copies, (x, sm, n["z1"])),
                    ("conv1", inner, Convs.conv, (tw[0], n["z1"], n["y1"])),
                ],
                [
                    ("squares", inner, Norm.squares, (n["y1"], n["sq"])),
                    ("sums", CT, Norm.sums, (ones, n["y1"], n["sq"], n["t"])),
                    ("lanes", CT, Norm.lanes, (n["t"], n["s"])),
                    ("scale", CT, Norm.derive_scale, (n["s"], gm, n["sc"])),
                    (
                        "shift",
                        CT,
                        Norm.derive_shift,
                        (back(n["s"]), n["sc"], bt, n["sh"]),
                    ),
                    (
                        "norm_silu",
                        inner,
                        Norm.apply,
                        (n["y1"], back(n["sc"]), n["sh"], n["g"]),
                    ),
                ],
                [
                    ("copies2", ROWS, Convs.copies, (n["g"], sm, n["z2"])),
                    ("conv2", inner, Convs.conv, (tw[1], n["z2"], n["y2"])),
                    ("skip", inner, Convs.skip, (x, n["y2"], n["out"])),
                    ("up_x", ROWS, Convs.up_x, (n["out"], d, n["v"])),
                    ("up_y", 2 * inner, Convs.up_y, (n["v"], n["u"])),
                ],
            ]

        def program(index, n, chain):
            """The class of segment `index`, made on set n's tensors."""
            more = {"chain": True} if chain else {}
            if index == 1:
                made = {"y": n["y1"], "ones": ones, "gm": gm, "bt": bt, "n": n["g"]}
                made |= {k: n[k] for k in ("sq", "t", "s", "sc", "sh")}
                return Norm(**made, **more, session=session, log=log)
            made = {"x": x, "sm": sm, "z": n["z1"], "w": tw[0], "y": n["y1"], "d": d}
            made |= {k: n[k] for k in ("out", "v", "u")}
            return Convs(**made, **more, session=session, log=log)

        def chained(index, part, n):
            """Run `index`'s calls as one chain: its us, or the refusal and the calls'
            us as commands (the next run reads their outputs)."""
            try:
                run = session.load(program(index, n, True))
            except Exception as refused:  # noqa: BLE001 - the refusal is the result
                return commands(index, part, n, refused)
            try:
                steps = [handler(*c, cores=at(k)) for _, k, handler, c in part]
                chain = run.chain(steps, cores=at(2 * H * CT))
            except Exception as refused:  # noqa: BLE001 - the refusal is the result
                run.stop()
                return commands(index, part, n, refused)
            try:
                event = run.wait(run.submit(chain), timeout=30.0)
            finally:
                run.stop()
            return {
                "run": index,
                "event": event.kind,
                "us": round(event.microseconds, 1),
            }

        def commands(index, part, n, refused):
            text = f"{type(refused).__name__}: {refused}"[:500]
            log(f"run {index} as one chain: {text}")
            part = segments(n)[index]  # the calls unmarked (a command takes no km.past)
            run = session.load(program(index, n, False))
            try:
                us = [timed(run, h, at(k), c, across=across) for _, k, h, c in part]
            finally:
                run.stop()
            return {"run": index, "refused": text, "commands_us": round(sum(us), 1)}

        sets = {"commands": tensors()}
        for index, part in enumerate(segments(sets["commands"])):
            run = session.load(program(index, sets["commands"], False))
            try:
                for name, count, handler, call in part:
                    us = timed(run, handler, at(count), call, across=across)
                    report[f"{name}_us"] = us
            finally:
                run.stop()
        names = [name for part in segments(sets["commands"]) for name, *_ in part]
        report["block_us"] = round(sum(report[f"{k}_us"] for k in names[:11]), 1)
        report["up_us"] = round(report["up_x_us"] + report["up_y_us"], 1)
        got["commands"] = [read(session, sets["commands"][k]) for k in ("out", "u")]
        if args.chain:  # each run's calls as one chain; a refused one as commands
            for tensor in sets["commands"].values():
                tensor.free()  # one set at a time in the tensor table
            sets["chain"] = tensors()
            report["chains"] = []
            for index, part in enumerate(segments(sets["chain"], km.past)):
                report["chains"].append(chained(index, part, sets["chain"]))
            got["chain"] = [read(session, sets["chain"][k]) for k in ("out", "u")]
    if across is not None:
        report["across"] = across
    want, want_up = reference(image, *weights, gamma, beta)
    checks = []
    for name, (out, up) in got.items():
        report[f"{name}_block"] = error(interior(out, H), want)
        report[f"{name}_up"] = error(interior(up, 2 * H), want_up)
        checks += [(f"{name}_block", "rms", 0.01), (f"{name}_up", "rms", 0.01)]
    for name, _, _ in checks:
        log(f"{name}: {report[name]}")
    finish(report, checks)


if __name__ == "__main__":
    sys.exit(main())
