"""An embedding lookup, a data-dependent read by a token word of a page, against numpy.
E1: the table in tile layout, a vocabulary entry a tile row (32 x the memory), the
token's E tiles one rectangle. E2 (--e2): the table in row layout, a page an entry,
the token's row as items of 32 B pushed to items of the output, copied by the movers."""

import argparse
import sys

import numpy as np
from forms import error, finish, log, random, read, tiles, timed, words

import komorebi as km

E = 4  # hidden tiles: 128 values an entry
ITEMS = E * 32 * 2 // 32  # an entry's 32 B items in row layout (bf16)


class Entry(km.Entry):
    token: int


class Embed(km.Program):
    """E1: core r copies the token's tile row into tile row r of out."""

    @km.handler
    def lookup(self, ids: km.Tensor, table: km.Tensor, out: km.Tensor):
        r = self.rank
        for e in ids.pages(r, 1).entries(Entry, size=1, fixed=1):
            results = km.Stream(out.tiles[r : r + 1, 0:E], write=True, block=E)
            for tile in km.Stream(table.tiles[e.token : e.token + 1, 0:E], block=E):
                results.push(tile)


class EmbedItems(km.Program):
    """E2: core r pushes the token's row (ITEMS items of 32 B) to row r of out: an
    item copy on the movers, mover A reading, mover B writing (README.md)."""

    @km.handler
    def lookup(self, ids: km.Tensor, table: km.Tensor, out: km.Tensor):
        r = self.rank
        for e in ids.pages(r, 1).entries(Entry, size=1, fixed=1):
            first = e.token * ITEMS
            results = km.Stream(out.items(r * ITEMS, ITEMS, size=32), write=True)
            for item in km.Stream(table.items(first, ITEMS, size=32)):
                results.push(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--vocabulary", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--e2", action="store_true", help="also the E2 form")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cores, vocabulary = args.cores, args.vocabulary
    entries = random(rng, (vocabulary, E * 32))
    tokens = rng.integers(0, vocabulary, cores)
    tokens[-1] = vocabulary - 1  # the last entry
    tokens[0] = tokens[1]  # two cores read one entry
    ids = np.zeros((cores, 32), np.uint32)
    ids[:, 0] = tokens
    table_tiles = np.repeat(entries, 32, axis=0)  # an entry in each row of its tile row
    report = {"cores": cores, "vocabulary": vocabulary, "tokens": tokens.tolist()}
    with km.Session(log=log) as session:
        tids = session.tensor(ids, "uint32", "row")
        t1 = tiles(session, table_tiles)
        out1 = tiles(session, np.zeros((cores * 32, E * 32)))
        t2 = session.tensor(entries.astype(np.float32), "bfloat16", "row")
        out2 = session.tensor(np.zeros((cores, E * 32), np.float32), "bfloat16", "row")
        run = session.load(
            Embed(ids=tids, table=t1, out=out1, session=session, log=log)
        )
        try:
            report["e1_us"] = timed(run, Embed.lookup, (0, cores), (tids, t1, out1))
        finally:
            run.stop()
        got1 = read(session, out1)
        if args.e2:
            program = EmbedItems(ids=tids, table=t2, out=out2, session=session, log=log)
            run = session.load(program)
            try:
                call = (tids, t2, out2)
                report["e2_us"] = timed(run, EmbedItems.lookup, (0, cores), call)
            finally:
                run.stop()
            got2 = words(session, out2).view(np.uint16).reshape(cores, E * 32)
    want = entries[tokens]
    report["e1"] = error(got1, np.repeat(want, 32, axis=0))
    checks = [("e1", "max", 0.0)]
    if args.e2:
        bits = np.asarray(got2, np.uint32) << 16
        report["e2"] = error(bits.view(np.float32).astype(np.float64), want)
        checks.append(("e2", "max", 0.0))
    log(f"e1 {report['e1']}, e2 {report.get('e2', 'not run')}")
    finish(report, checks)


if __name__ == "__main__":
    sys.exit(main())
