"""Entry 97 proof on two cards: a class whose reduce is local=True, in groups of 7 on
each island's ranks, exact against numpy; its arrival buffer against the same class
without local=; a command whose groups cross the islands' edge refused on the host.
Prints one JSON line."""

import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi import formats

STARTED = time.monotonic()
GROUPS = 16  # groups of 7 a command: 112 ranks


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class Sevens(km.Program):
    """A tile a rank summed over groups of 7, each group's sum at page rank // 7."""

    @km.handler
    def sevens(self, s: km.Tensor, t: km.Tensor, base: int):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(base + self.rank, 1)):
            total += tile
        summed = km.reduce(total, form="tree", group=7, local=True)
        if self.rank == self.leader:
            t.pages((base + self.rank) // 7, 1).write(summed)


class Crossing(km.Program):
    """Sevens without local=."""

    @km.handler
    def sevens(self, s: km.Tensor, t: km.Tensor, base: int):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(base + self.rank, 1)):
            total += tile
        summed = km.reduce(total, form="tree", group=7)
        if self.rank == self.leader:
            t.pages((base + self.rank) // 7, 1).write(summed)


def arrival(program):
    name = type(program).__name__.lower()
    header = (program.built.folder / f"{name}.h").read_text()
    line = next(line for line in header.splitlines() if "Arrival kArrival0 " in line)
    return line.split("{")[1].split("}")[0]


def main():
    rng = np.random.default_rng(1)
    out = {}
    with km.Session(cards=[0, 1], log=log) as session:
        edge = session.graph.first(1)
        ranks = session.ranks
        out["edge"], out["ranks"] = edge, ranks
        data = rng.integers(-8, 9, (ranks * 32, 32)).astype(np.float32)
        s = session.tensor(data, "float32", "tile")
        t = session.tensor(np.zeros((ranks * 32, 32), np.float32), "float32", "tile")
        local = Sevens(log=log, session=session, s=s, t=t)
        crossing = Crossing(log=log, session=session, s=s, t=t)
        out["arrival_local"], out["arrival_crossing"] = arrival(local), arrival(
            crossing
        )
        run = session.load(local, wait=0.5)
        try:
            for first in (0, edge):
                handle = run.submit(Sevens.sevens, (first, 7 * GROUPS), (s, t, first))
                event = run.wait(handle)
                out[f"from_{first}"] = getattr(event, "kind", str(event)[:80])
            try:
                run.submit(Sevens.sevens, (edge - 10, 7 * 3), (s, t, 0))
                out["crossing"] = "not refused"
            except ValueError as error:
                out["crossing"] = str(error)
        finally:
            out["errors"] = [r["error_name"] for r in run.stop().islands]
        (raw,) = session.read(t)
        got = formats.from_bytes(raw, t.shape, t.dtype, t.layout).reshape(-1, 32, 32)
        tiles = data.reshape(ranks, 32, 32)
        exact = True
        for first in (0, edge):
            for g in range(GROUPS):
                rank = first + 7 * g
                want = tiles[rank : rank + 7].sum(axis=0)
                exact &= bool(np.array_equal(got[rank // 7], want))
        out["exact"] = exact
        t2 = session.tensor(np.zeros((ranks * 32, 32), np.float32), "float32", "tile")
        across = Crossing(log=log, session=session, s=s, t=t2)
        run = session.load(across, wait=0.5)
        try:
            base = edge - 10  # groups of 7 from 108: one holds ranks 115 to 121
            event = run.wait(run.submit(Crossing.sevens, (base, 21), (s, t2, base)))
            out["crossing_kind"] = getattr(event, "kind", str(event)[:80])
        finally:
            out["crossing_errors"] = [r["error_name"] for r in run.stop().islands]
        (raw,) = session.read(t2)
        got = formats.from_bytes(raw, t2.shape, t2.dtype, t2.layout).reshape(-1, 32, 32)
        out["crossing_exact"] = all(
            np.array_equal(
                got[(base + 7 * g) // 7],
                tiles[base + 7 * g : base + 7 * g + 7].sum(axis=0),
            )
            for g in range(3)
        )
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
