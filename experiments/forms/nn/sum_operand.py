"""A sum as a product's operand, (x + r) @ W, against numpy: the direct form (the sum
in the product's line, refused when it is built: its text is reported), (a) two
products into one sum, x @ W + r @ W, and (b) x + r written by a pass, then s @ W."""

import argparse
import sys

import numpy as np
from forms import bf16, error, finish, log, random, read, tiles, timed

import komorebi as km

K, N = 4, 2  # input and output tiles of a row; W is a field of K x N tiles


@km.tile_function(tolerance=1e-6)
def add(a, b):
    return a + b


class Sums(km.Program):
    """W held as a field; each core takes `rows` tile rows of x and r from its rank."""

    w: km.L1[km.bf16, K, N, 32, 32]

    @km.tile_function(fidelity="HiFi4")
    def project(a, b):
        return a @ b

    @km.handler
    def load(self, w: km.Tensor):
        self.w.fill(km.Stream(w.tiles[0:K, 0:N]))

    @km.handler
    def two(self, x: km.Tensor, r: km.Tensor, y: km.Tensor, rows: int):
        """(a): x @ W and r @ W summed in one accumulator."""
        for i in range(rows):
            row = self.rank * rows + i
            acc = km.tile.zeros(km.fp32, N)
            for xb in km.Stream(x.tiles[row : row + 1, 0:K], block=K):
                acc += self.project(xb, self.w)
            for rb in km.Stream(r.tiles[row : row + 1, 0:K], block=K):
                acc += self.project(rb, self.w)
            y.pages(row * N, N).write(acc)

    @km.handler
    def summed(self, x: km.Tensor, r: km.Tensor, s: km.Tensor, rows: int):
        """(b), pass 1: s = x + r, bf16."""
        first = self.rank * rows * K
        results = km.Stream(s.pages(first, rows * K), write=True, block=K)
        for xb, rb in zip(
            km.Stream(x.pages(first, rows * K), block=K),
            km.Stream(r.pages(first, rows * K), block=K),
        ):
            results.push(add(xb, rb))

    @km.handler
    def one(self, s: km.Tensor, y: km.Tensor, rows: int):
        """(b), pass 2: s @ W."""
        for i in range(rows):
            row = self.rank * rows + i
            acc = km.tile.zeros(km.fp32, N)
            for sb in km.Stream(s.tiles[row : row + 1, 0:K], block=K):
                acc += self.project(sb, self.w)
            y.pages(row * N, N).write(acc)


class Direct(km.Program):
    """The direct form: the sum in the product's line."""

    w: km.L1[km.bf16, K, N, 32, 32]

    @km.tile_function(fidelity="HiFi4")
    def project(a, b):
        return a @ b

    @km.handler
    def direct(self, x: km.Tensor, r: km.Tensor, y: km.Tensor, rows: int):
        for i in range(rows):
            row = self.rank * rows + i
            acc = km.tile.zeros(km.fp32, N)
            for xb, rb in zip(
                km.Stream(x.tiles[row : row + 1, 0:K], block=K),
                km.Stream(r.tiles[row : row + 1, 0:K], block=K),
            ):
                acc += self.project(xb + rb, self.w)
            y.pages(row * N, N).write(acc)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rows", type=int, default=2, help="tile rows a core")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = args.rows
    with km.Session(log=log) as session:
        cores = session.grid[0] * session.grid[1]
        group = (0, cores)
        x = random(rng, (cores * rows * 32, K * 32))
        r = random(rng, x.shape)
        w = random(rng, (K * 32, N * 32), 1 / 8)
        report = {"cores": cores, "rows": rows}
        tx, tr, tw = tiles(session, x), tiles(session, r), tiles(session, w)
        ty = tiles(session, np.zeros((x.shape[0], N * 32)), "float32")
        try:
            session.load(
                Direct(x=tx, r=tr, w=tw, y=ty, session=session, log=log)
            ).stop()
            report["direct_refused"] = None
        except Exception as refusal:  # noqa: BLE001 - the refusal is the result
            report["direct_refused"] = f"{type(refusal).__name__}: {refusal}"
        log(f"direct: {report['direct_refused']}")
        ts = tiles(session, np.zeros_like(x))
        run = session.load(Sums(x=tx, r=tr, w=tw, s=ts, y=ty, session=session, log=log))
        try:
            timed(run, Sums.load, group, (tw,))
            report["two_us"] = timed(run, Sums.two, group, (tx, tr, ty, rows))
            got_two = read(session, ty)
            report["sum_us"] = timed(run, Sums.summed, group, (tx, tr, ts, rows))
            report["one_us"] = timed(run, Sums.one, group, (ts, ty, rows))
            got_one = read(session, ty)
        finally:
            run.stop()
    report["two"] = error(got_two, x @ w + r @ w)
    report["sum_first"] = error(got_one, bf16(x + r) @ w)  # s is stored as bf16
    report["sum_first_vs_float64"] = error(got_one, (x + r) @ w)
    checks = [("two", "rms", 2.5e-3), ("sum_first", "rms", 2.5e-3)]
    finish(report, checks)


if __name__ == "__main__":
    sys.exit(main())
