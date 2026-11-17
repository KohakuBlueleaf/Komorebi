"""Two reduces in one handler on the card: the second a turn after the first, its
senders waiting for their receiver's credit. Each handler's sums against numpy: over
the command's group, in groups of G, and as a chain's call (a share of K a rank); on
two cards (--cards 0,1) the cores meet at the second card's first rank, so a group
crosses the edge. A JSON line a case, the summary last. See README.md.
python experiments/language/two_reduces/two_reduces.py --cards 0"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi.bench import power

WORDS = 1024  # float32 words a page: one tile's
G = 4  # a group's ranks in `grouped`
K = 2  # ranks a group in `chained`
CORES = 16  # a command's cores: `spread`'s mean divides by them
INVERSE = 1.0 / CORES


@km.tile_function
def spread(x, m):
    return (x - m * INVERSE) * (x - m * INVERSE)


STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", file=sys.stderr, flush=True)


class TwoReduces(km.Program):
    """grid: rank r's page r of s, then of z, summed over the command's group, the
    leader writing pages 0 and 1 of t; grouped: the same in groups of G, root g
    writing page g of t and of y; chained: groups of K sharing s's and z's tiles,
    root g writing page g of t and of y (a chain's call). mean, apart and total: a
    second pass that needs the first's result, as a chain of three calls: the sum
    of s's pages written to m; each rank's (s_r - m / CORES)^2 pushed to d; d's sum to
    u (the layer norm's two passes)."""

    @km.handler
    def grid(self, s: km.Tensor, z: km.Tensor, t: km.Tensor):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        whole = km.reduce(mine)
        if self.rank == self.leader:
            t.pages(0, 1).write(whole)
        other = km.tile.zeros(km.fp32)
        for tile in km.Stream(z.pages(self.rank, 1)):
            other += tile
        second = km.reduce(other)
        if self.rank == self.leader:
            t.pages(1, 1).write(second)

    @km.handler
    def grouped(self, s: km.Tensor, z: km.Tensor, t: km.Tensor, y: km.Tensor):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        whole = km.reduce(mine, form="tree", group=G)
        if self.rank == self.leader:
            t.pages(self.rank // G, 1).write(whole)
        other = km.tile.zeros(km.fp32)
        for tile in km.Stream(z.pages(self.rank, 1)):
            other += tile
        second = km.reduce(other, form="tree", group=G)
        if self.rank == self.leader:
            y.pages(self.rank // G, 1).write(second)

    @km.handler
    def chained(
        self, s: km.Tensor, z: km.Tensor, t: km.Tensor, y: km.Tensor, tiles: int
    ):
        share = self.rank % K
        k0 = share * (tiles // K) + min(share, tiles % K)
        k1 = (share + 1) * (tiles // K) + min(share + 1, tiles % K)
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(k0, k1 - k0)):
            mine += tile
        whole = km.reduce(mine, form="tree", group=K)
        if self.rank == self.leader:
            t.pages(self.rank // K, 1).write(whole)
        other = km.tile.zeros(km.fp32)
        for tile in km.Stream(z.pages(k0, k1 - k0)):
            other += tile
        second = km.reduce(other, form="tree", group=K)
        if self.rank == self.leader:
            y.pages(self.rank // K, 1).write(second)

    @km.handler
    def mean(self, s: km.Tensor, m: km.Tensor):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        whole = km.reduce(mine)
        if self.rank == self.leader:
            m.pages(0, 1).write(whole)

    @km.handler
    def apart(self, s: km.Tensor, m: km.Tensor, d: km.Tensor):
        out = km.Stream(d.pages(self.rank, 1), write=True)
        for xt, mt in zip(km.Stream(s.pages(self.rank, 1)), km.Stream(m.pages(0, 1))):
            out.push(spread(xt, mt))

    @km.handler
    def total(self, d: km.Tensor, u: km.Tensor):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(d.pages(self.rank, 1)):
            mine += tile
        whole = km.reduce(mine)
        if self.rank == self.leader:
            u.pages(0, 1).write(whole)

    @km.chain
    def two_pass(self, s: km.Tensor, m: km.Tensor, d: km.Tensor, u: km.Tensor):
        self.mean(s, m)
        self.apart(s, m, d)
        self.total(d, u)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0", help="one card, or two: 0,1")
    parser.add_argument("--tiles", type=int, default=6, help="chained's K tiles")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--kernel-bytes", type=int)
    return parser.parse_args()


def cores_of(session, count, cards):
    """`count` cores from 0, or on two cards the ones meeting at card 1's first."""
    if len(cards) == 1:
        return (0, count)
    edge = session.graph.first(1)
    return (edge - count // 2, count)


def bad(got, want):
    """Words of `got` not equal to `want` (integer sums: exact)."""
    return int(np.count_nonzero(got != want))


def done(run, submitted, what):
    event = run.wait(submitted, timeout=30.0)
    if event.kind != "done":
        raise RuntimeError(f"{what}: {event}")
    return round(float(event.microseconds), 2)


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    count = CORES
    rng = np.random.default_rng(args.seed)
    s = rng.integers(-50, 51, (count, WORDS)).astype(np.float32)
    z = rng.integers(-50, 51, (count, WORDS)).astype(np.float32)
    opened = {"cards": cards}
    if args.kernel_bytes:
        opened["kernel_bytes"] = args.kernel_bytes
    cases = []
    with km.Session(log=log, **opened) as session:
        cores = cores_of(session, count, cards)
        log(f"START cards {cards}, cores {cores}: 2 commands and 2 chains, one load")
        given = {n: session.tensor(v, "float32") for n, v in (("s", s), ("z", z))}
        for name in "tyd":
            given[name] = session.tensor(
                np.zeros((count, WORDS), np.float32), "float32"
            )
        for name in "mu":
            given[name] = session.tensor(np.zeros((1, WORDS), np.float32), "float32")
        program = TwoReduces(log=log, session=session, chain=True, **given)
        run = session.load(program, wait=0.5, deadline=0.5)
        st, zt, t, y = (given[n] for n in "szty")

        def case(what, us, checks):
            cases.append({"case": what, "us": us, "bad_words": checks})
            log(json.dumps(cases[-1]))

        us = done(run, run.submit(TwoReduces.grid, cores, args=(st, zt, t)), "grid")
        got = t.read()
        case("grid", us, {"s": bad(got[0], s.sum(0)), "z": bad(got[1], z.sum(0))})
        us = done(
            run, run.submit(TwoReduces.grouped, cores, args=(st, zt, t, y)), "grouped"
        )
        got_t, got_y = t.read(), y.read()
        groups = count // G
        case(
            f"groups of {G}",
            us,
            {
                "s": bad(got_t[:groups], s.reshape(groups, G, WORDS).sum(1)),
                "z": bad(got_y[:groups], z.reshape(groups, G, WORDS).sum(1)),
            },
        )
        tiles = args.tiles
        chain = run.chain([program.chained(st, zt, t, y, tiles)], cores=cores)
        us = done(run, run.submit(chain), "chain")
        got_t, got_y = t.read(), y.read()
        groups = count // K
        want_s = np.broadcast_to(s[:tiles].sum(0), (groups, WORDS))
        want_z = np.broadcast_to(z[:tiles].sum(0), (groups, WORDS))
        case(
            f"chain, groups of {K}",
            us,
            {"s": bad(got_t[:groups], want_s), "z": bad(got_y[:groups], want_z)},
        )
        m, d, u = (given[n] for n in "mdu")
        passes = run.chain(program.two_pass(st, m, d, u), cores=cores)
        us = done(run, run.submit(passes), "two passes")
        mean = s.astype(np.float64).sum(0) * INVERSE
        want = ((s - mean) ** 2).sum(0)
        got = u.read()[0].astype(np.float64)
        scale = np.sqrt((want**2).mean())
        rms = float(np.sqrt(((got - want) ** 2).mean()) / scale)
        cases.append({"case": "two passes, a chain of three", "us": us, "rms_rel": rms})
        log(json.dumps(cases[-1]))
        result = run.result()  # a list over islands, a card's
        results = result if isinstance(result, list) else [result]
        run.stop()
        chain.free()
        passes.free()
    summary = {
        "cards": cards,
        "cores": list(cores),
        "cases": cases,
        "kernel": [r.get("error_name", "none") for r in results],
        "bad_words": sum(sum(c.get("bad_words", {}).values()) for c in cases),
    }
    print(json.dumps(summary), flush=True)
    print(json.dumps(power.clocked({"cards": cards}, card=cards[0])), flush=True)
    rms = cases[-1]["rms_rel"]
    return 0 if summary["bad_words"] == 0 and rms < 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main())
