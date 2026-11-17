"""A held weight whose rows change by command, y = x W with W of K tile rows held in a
field, K 4 then 8, against numpy: (a) a build a size (the class constant ROWS given at
the make: two programs), (b) one build, the field sized to the most and K a word of
the command (W stored with zero rows to the most, x's width by km.UPTO(x=MOST))."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nn"))

import numpy as np
from forms import error, finish, log, random, read, tiles, timed

import komorebi as km

N, MOST = 2, 8  # output tiles of a row; the most rows of W
SIZES = (4, 8)


class Sized(km.Program):
    """(a): the field's rows are the class constant ROWS."""

    ROWS = km.constant(8)
    w: km.L1[km.bf16, ROWS, N, 32, 32]

    @km.tile_function(fidelity="HiFi4")
    def project(a, b):
        return a @ b

    @km.handler
    def load(self, w: km.Tensor):
        self.w.fill(km.Stream(w.tiles[0 : self.ROWS, 0:N]))

    @km.handler
    def apply(self, x: km.Tensor, y: km.Tensor, rows: int):
        for i in range(rows):
            row = self.rank * rows + i
            acc = km.tile.zeros(km.fp32, N)
            for xb in km.Stream(x.tiles[row : row + 1, 0 : self.ROWS], block=1):
                acc += self.project(xb, self.w)
            y.pages(row * N, N).write(acc)


class Worded(km.Program):
    """(b): the field holds MOST rows; K is a word of the command."""

    w: km.L1[km.bf16, MOST, N, 32, 32]

    @km.tile_function(fidelity="HiFi4")
    def project(a, b):
        return a @ b

    @km.handler
    def load(self, w: km.Tensor):
        self.w.fill(km.Stream(w.tiles[0:MOST, 0:N]))

    @km.handler
    def apply(self, x: km.Tensor, y: km.Tensor, rows: int, k: int):
        for i in range(rows):
            row = self.rank * rows + i
            acc = km.tile.zeros(km.fp32, N)
            for xb in km.Stream(x.tiles[row : row + 1, 0:k], block=1):
                acc += self.project(xb, self.w)
            y.pages(row * N, N).write(acc)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--rows", type=int, default=2, help="tile rows a core")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cores, rows = args.cores, args.rows
    group = (0, cores)
    report = {"cores": cores, "rows": rows, "sizes": SIZES}
    checks = []
    with km.Session(log=log) as session:
        made = {}
        for k in SIZES:
            x = random(rng, (cores * rows * 32, k * 32))
            w = random(rng, (k * 32, N * 32), 1 / 8)
            padded = np.zeros((MOST * 32, N * 32))
            padded[: k * 32] = w
            made[k] = (x, w, tiles(session, x), tiles(session, w))
            made[k] += (tiles(session, padded),)
        y = tiles(session, np.zeros((cores * rows * 32, N * 32)), "float32")
        for k in SIZES:  # (a): a program a size, each its own build and load
            x, w, tx, tw, _ = made[k]
            program = Sized(x=tx, w=tw, y=y, ROWS=k, session=session, log=log)
            run = session.load(program)
            try:
                timed(run, Sized.load, group, (tw,))
                report[f"sized_{k}_us"] = timed(run, Sized.apply, group, (tx, y, rows))
            finally:
                run.stop()
            report[f"sized_{k}"] = error(read(session, y), x @ w)
            checks.append((f"sized_{k}", "rms", 2.5e-3))
        x, _, tx, _, tp = made[MOST]
        widths = {"apply": [km.UPTO(x=MOST)]}  # x's tile columns a word, up to MOST
        program = Worded(x=tx, w=tp, y=y, handlers=widths, session=session, log=log)
        run = session.load(program)
        try:  # (b): one build; a load a weight, K at each apply
            for k in SIZES:
                x, w, tx, _, tp = made[k]
                report[f"worded_{k}_load_us"] = timed(run, Worded.load, group, (tp,))
                call = (tx, y, rows, k)
                report[f"worded_{k}_us"] = timed(run, Worded.apply, group, call)
                report[f"worded_{k}"] = error(read(session, y), x @ w)
                checks.append((f"worded_{k}", "rms", 2.5e-3))
        finally:
            run.stop()
    finish(report, checks)


if __name__ == "__main__":
    sys.exit(main())
