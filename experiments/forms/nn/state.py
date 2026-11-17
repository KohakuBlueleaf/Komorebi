"""A recurrent state update, h = a h + u, one token a command (decode), the state
kept in DRAM and updated in place page by page, against numpy over a few tokens.
Example: python state.py --cores 8 --tiles 4 --tokens 6"""

import argparse
import sys

import numpy as np
from forms import error, finish, log, random, read, tiles, timed

import komorebi as km


@km.tile_function(tolerance=1e-6)
def gated(a, h, u):
    return a * h + u


class StateUpdate(km.Program):
    """Core r reads and writes the state's pages r tiles to r tiles + tiles - 1."""

    @km.handler
    def step(self, a: km.Tensor, u: km.Tensor, state: km.Tensor, tiles: int):
        first = self.rank * tiles
        results = km.Stream(state.pages(first, tiles), write=True, block=2)
        for g, h, x in zip(
            km.Stream(a.pages(first, tiles), block=2),
            km.Stream(state.pages(first, tiles), block=2),
            km.Stream(u.pages(first, tiles), block=2),
        ):
            results.push(gated(g, h, x))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--tiles", type=int, default=4, help="state tiles a core")
    parser.add_argument("--tokens", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = args.cores * args.tiles * 32
    gates = [rng.uniform(0.5, 1.0, (rows, 32)) for _ in range(args.tokens)]
    inputs = [random(rng, (rows, 32)) for _ in range(args.tokens)]
    report = {"cores": args.cores, "tiles": args.tiles, "tokens": args.tokens}
    us = []
    with km.Session(log=log) as session:
        state = tiles(session, np.zeros((rows, 32)), "float32")
        a = [tiles(session, g, "float32") for g in gates]
        u = [tiles(session, x, "float32") for x in inputs]
        program = StateUpdate(a=a[0], u=u[0], state=state, session=session, log=log)
        run = session.load(program)
        try:
            for t in range(args.tokens):
                call = (a[t], u[t], state, args.tiles)
                us.append(timed(run, StateUpdate.step, (0, args.cores), call))
        finally:
            run.stop()
        got = read(session, state)
    h = np.zeros((rows, 32))
    for g, x in zip(gates, inputs):
        h = g.astype(np.float32).astype(np.float64) * h + x
    report["state"] = error(got, h)
    report["step_us"] = us
    log(f"state {report['state']}, us a token {us}")
    finish(report, [("state", "rms", 1e-5)])


if __name__ == "__main__":
    sys.exit(main())
