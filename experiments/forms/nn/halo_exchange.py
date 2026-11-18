"""The 3x3 conv of conv_shift.py on cards 0 and 1, its halo brought over in one step,
against numpy: the tensors hold 4 more tile rows (island 0 rows 0 to 19, island 1 rows
16 to 35); a rank of each island reads the other's two halo rows in one read, then the
shifts and the conv read only their own island. --chain: also as one chain."""

import argparse
import sys

import numpy as np
from conv_shift import (
    COLS,
    CT,
    ROWS,
    Conv,
    H,
    J,
    Shift,
    reference,
    shift_matrices,
    to_rows,
)
from forms import error, finish, log, random, read, tiles, timed

import komorebi as km

PAD = 2 * CT  # tile rows of halo: one image row each way
PROWS = ROWS + PAD  # tile rows of a padded tensor: half on each island
LOW = PROWS // 2  # island 0 holds indices 0 to LOW - 1, rows 0 to LOW - 1
HALO = PAD * COLS // 2  # pages of one island's halo
SENTINEL = 7.0  # the halo rows before the step: a wrong conv if it does nothing
STEP = """
km::read_pages(movement, command, x, source, pages, get_write_ptr(stage));
noc_async_read_barrier();
km::write_pages(movement, command, x, target, pages, get_write_ptr(stage));
noc_async_write_barrier();
"""


class Halo(km.Program):
    """Rank 0 (island 0) takes rows LOW - 2.. from island 1's copy at LOW + 2..; rank 1
    (island 1) takes rows LOW.. from island 0's rows LOW - 4.. (16 and 17)."""

    stage: km.L1[km.bf16, HALO, 32, 32] = km.own(push=None)

    @km.handler
    def halo(self, x: km.Tensor):
        source = (LOW + PAD // 2 - self.rank * (PAD + PAD // 2)) * COLS
        target = (LOW - PAD // 2 + self.rank * (PAD // 2)) * COLS
        pages = HALO
        km.cxx(
            STEP,
            takes=[x, source, target, pages, self.stage],
            reads=[x],
            writes=[(x, target, pages)],
            on="mover_b",
        )


class HaloConv(Conv):
    """Conv reading island 1's rows PAD further on (rows 8 and on add PAD); y has no
    halo rows, so a core's row of y is on its own island at tile row rank + 2."""

    @km.handler
    def padded(
        self, w: km.Tensor, z0: km.Tensor, zl: km.Tensor, zr: km.Tensor, y: km.Tensor
    ):
        row = self.rank // CT
        ro = self.rank % CT
        off = (row // (H // 2)) * PAD
        for j in range(1, J + 1):
            acc = km.tile.zeros(km.fp32, 1)
            for dy in range(3):
                t0 = dy * CT  # W's columns (dx, dy, ci): a stream's taps side by side
                t1 = t0 + 3 * CT
                t2 = t1 + 3 * CT
                r0 = (row + dy) * CT + off
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


class HaloConv4(HaloConv):
    """The padded conv at HiFi4: a class's products share one fidelity, the shift's."""

    @km.tile_function(fidelity="HiFi4")
    def product(a, b):
        return a @ b


class Model(Halo, Shift, HaloConv4):
    """The three classes' handlers in one program, for one chain (its conv at HiFi4)."""


def padded(rows):
    """A [ROWS x 32, ...] matrix as PROWS tile rows: island 1's half from row LOW - PAD
    on; the rows the halo step writes (LOW - 2, LOW - 1, LOW, LOW + 1) the sentinel."""
    out = np.zeros((PROWS * 32, rows.shape[1]))
    out[: LOW * 32] = rows[: LOW * 32]
    out[LOW * 32 :] = rows[(LOW - PAD) * 32 :]
    out[(LOW - PAD // 2) * 32 : (LOW + PAD // 2) * 32] = SENTINEL
    return out


def unpadded(matrix):
    """The ROWS tile rows of a padded matrix, each from the island that computed it."""
    split = LOW - PAD // 2  # rows below from island 0's half, the rest from island 1's
    return np.concatenate([matrix[: split * 32], matrix[(split + PAD) * 32 :]])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cards", default="0,1")
    parser.add_argument(
        "--chain",
        action="store_true",
        help="also as one chain: 837.2 us, the conv's covering range across the edge",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    channels, width = CT * 32, J * 32
    image = random(rng, (channels, H, width))
    weights = random(rng, (channels, channels, 3, 3), 1.0 / 24)
    w = weights.transpose(0, 3, 2, 1).reshape(channels, 9 * channels)  # (dx, dy, ci)
    rows = to_rows(image)
    zeros = np.zeros((PROWS * 32, COLS * 32))
    report = {"channels": channels, "height": H, "width": width}
    cards = [int(card) for card in args.cards.split(",")]
    with km.Session(cards=cards, log=log) as session:
        edge = session.graph.first(1)
        place = {"split": 0, "group": (edge - LOW, PROWS)}
        own = {"split": 0, "group": (edge - ROWS // 2, ROWS)}  # y: no halo
        s = tiles(session, shift_matrices(), copy=True)
        tw = tiles(session, w, copy=True)
        sets = {}
        for name in ("commands", "chain") if args.chain else ("commands",):
            x = tiles(session, padded(rows), **place)
            zl, zr = (tiles(session, zeros, **place) for _ in range(2))
            y = tiles(session, zeros[: ROWS * 32], **own)
            sets[name] = (x, zl, zr, y)
        x, zl, zr, y = sets["commands"]
        groups = {
            "halo": (edge - 1, 2),
            "shift": (edge - LOW, PROWS),
            "conv": (edge - H * CT // 2, H * CT),
        }
        across = []
        programs = [
            Halo(x=x, session=session, log=log),
            Shift(x=x, s=s, zl=zl, zr=zr, session=session, log=log),
            HaloConv(w=tw, z0=x, zl=zl, zr=zr, y=y, session=session, log=log),
        ]
        steps = [
            (Halo.halo, "halo", (x,)),
            (Shift.shift, "shift", (x, s, zl, zr)),
            (HaloConv.padded, "conv", (tw, x, zl, zr, y)),
        ]
        for program, (handler, name, call) in zip(programs, steps):
            run = session.load(program)
            try:
                us = timed(run, handler, groups[name], call, across=across)
                report[f"{name}_us"] = us
            finally:
                run.stop()
        report["across"] = across
        got = {"commands": (read(session, zl), read(session, y))}
        if args.chain:  # 837.2 us against 65.1: the conv waits on a covering range
            try:
                x, zl, zr, y = sets["chain"]
                model = Model(
                    x=x, s=s, zl=zl, zr=zr, w=tw, z0=x, y=y, chain=True,
                    session=session, log=log,
                )  # fmt: skip
                run = session.load(model)
                try:
                    calls = [
                        Model.halo(x, cores=groups["halo"]),
                        Model.shift(x, s, zl, zr, cores=groups["shift"]),
                        Model.padded(tw, km.past(x), zl, zr, y, cores=groups["conv"]),
                    ]
                    chain = run.chain(calls, cores=groups["shift"])
                    event = run.wait(run.submit(chain), timeout=30.0)
                    report["chain"] = event.kind
                    report["chain_us"] = round(event.microseconds, 1)
                finally:
                    run.stop()
                got["chain"] = (read(session, zl), read(session, y))
            except Exception as refused:  # noqa: BLE001 - the refusal is the result
                report["chain"] = f"{type(refused).__name__}: {refused}"[:500]
            log(f"one chain: {report['chain']}")
    return finish_checks(report, got, rows, image, weights)


def finish_checks(report, got, rows, image, weights):
    """Each form's shifts and conv against numpy; prints the report and exits."""
    channels, width = CT * 32, J * 32
    want_left = np.zeros_like(rows)
    want_left[:, 32 : 32 + width] = rows[:, 31 : 31 + width]
    want = reference(image, weights)
    checks = []
    for name, (got_left, got_y) in got.items():
        report[f"{name}_shift"] = error(unpadded(got_left), want_left)
        conv = got_y.reshape(H + 2, channels, -1)[1:-1, :, 32 : 32 + width]
        report[f"{name}_conv"] = error(conv.transpose(1, 0, 2), want)
        checks += [(f"{name}_shift", "max", 0.0), (f"{name}_conv", "rms", 0.01)]
    finish(report, checks)


if __name__ == "__main__":
    sys.exit(main())
