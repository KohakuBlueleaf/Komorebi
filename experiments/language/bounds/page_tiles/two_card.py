"""two_card.py's layer on two cards with W in DRAM pages of k tiles (page_tiles) and W's
stream on the pattern given: its chain with the gathers and without, words against
float64. README.md. python experiments/language/bounds/page_tiles/two_card.py
--format bfp8_b --pages 1 2/7 --patterns AAAB AB"""

import argparse
import json
import math
import sys

import numpy as np

import komorebi as km
from experiments.language.bounds import bounds as bd
from experiments.language.bounds import decode_cost as dc
from experiments.language.bounds import two_card as tc
from komorebi.bench import power

log = bd.log


class PagedTwoCard(tc.TwoCard):
    """TwoCard with W's stream on the pattern `wp`: its W in pages of the example's
    tiles."""

    @km.handler
    def qkv(
        self,
        x_qkv: km.Tensor,
        w_qkv: km.Tensor,
        y_qkv: km.Tensor,
        k: int,
        col0: int,
        wp: km.Pattern,
    ):
        share = self.rank % self.S_QKV
        first = col0 + self.rank // self.S_QKV * self.W8
        k0 = share * (k // self.S_QKV) + min(share, k % self.S_QKV)
        k1 = (share + 1) * (k // self.S_QKV) + min(share + 1, k % self.S_QKV)
        for xb, wb in zip(
            km.Stream(x_qkv.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(
                w_qkv.tiles[k0:k1, first : first + self.W8], block=self.B8, pattern=wp
            ),
        ):
            self.sums8 += self.project(xb, wb)
        whole = km.reduce(self.sums8, form="tree", group=self.S_QKV)
        if self.rank == self.leader:
            y_qkv.tiles[0:1, first : first + self.W8].write(whole)

    @km.handler
    def o(
        self,
        x_o: km.Tensor,
        w_o: km.Tensor,
        y_o: km.Tensor,
        k: int,
        col0: int,
        wp: km.Pattern,
    ):
        share = self.rank % self.S_O
        first = col0 + self.rank // self.S_O * self.W7
        k0 = share * (k // self.S_O) + min(share, k % self.S_O)
        k1 = (share + 1) * (k // self.S_O) + min(share + 1, k % self.S_O)
        for xb, wb in zip(
            km.Stream(x_o.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(
                w_o.tiles[k0:k1, first : first + self.W7], block=self.B7, pattern=wp
            ),
        ):
            self.sums7 += self.project(xb, wb)
        whole = km.reduce(self.sums7, form="tree", group=self.S_O)
        if self.rank == self.leader:
            y_o.tiles[0:1, first : first + self.W7].write(whole)

    @km.handler
    def gate_up(
        self,
        x_gu: km.Tensor,
        w_gu: km.Tensor,
        y_gu: km.Tensor,
        k: int,
        col0: int,
        wp: km.Pattern,
    ):
        share = self.rank % self.S_G
        first = col0 + self.rank // self.S_G * self.W7
        k0 = share * (k // self.S_G) + min(share, k % self.S_G)
        k1 = (share + 1) * (k // self.S_G) + min(share + 1, k % self.S_G)
        for xb, wb in zip(
            km.Stream(x_gu.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(
                w_gu.tiles[k0:k1, first : first + self.W7], block=self.B7, pattern=wp
            ),
        ):
            self.sums7 += self.project(xb, wb)
        whole = km.reduce(self.sums7, form="tree", group=self.S_G)
        if self.rank == self.leader:
            y_gu.tiles[0:1, first : first + self.W7].write(whole)

    @km.handler
    def down(
        self,
        x_down: km.Tensor,
        w_down: km.Tensor,
        y_down: km.Tensor,
        k: int,
        col0: int,
        wp: km.Pattern,
    ):
        share = self.rank % self.S_O
        first = col0 + self.rank // self.S_O * self.W7
        k0 = share * (k // self.S_O) + min(share, k % self.S_O)
        k1 = (share + 1) * (k // self.S_O) + min(share + 1, k % self.S_O)
        for xb, wb in zip(
            km.Stream(x_down.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(
                w_down.tiles[k0:k1, first : first + self.W7], block=self.B7, pattern=wp
            ),
        ):
            self.sums7 += self.project(xb, wb)
        whole = km.reduce(self.sums7, form="tree", group=self.S_O)
        if self.rank == self.leader:
            y_down.tiles[0:1, first : first + self.W7].write(whole)

    @km.chain
    def layer(
        self, x, wq, q, a, wo, h, wg, g, wu, u, m, wd, y, cq, co, cg, kq: int, wp
    ):
        """two_card.py's chain, W's stream on `wp`."""
        self.qkv(x, wq, q, kq, 0, wp, cores=cq)
        self.o(a, wo, h, 128, 0, wp, cores=co)
        km.all_gather(h, 0, 168)
        self.gate_up(h, wg, g, kq, 0, wp, cores=cg)
        self.gate_up(km.past(h), wu, u, kq, 0, wp, cores=cg)
        self.down(m, wd, y, 672, 0, wp, cores=co)
        km.all_gather(y, 0, 168)

    @km.chain
    def bare(self, x, wq, q, a, wo, h, wg, g, wu, u, m, wd, y, cq, co, cg, kq: int, wp):
        """The same calls without the gathers."""
        self.qkv(x, wq, q, kq, 0, wp, cores=cq)
        self.o(a, wo, h, 128, 0, wp, cores=co)
        self.gate_up(h, wg, g, kq, 0, wp, cores=cg)
        self.gate_up(km.past(h), wu, u, kq, 0, wp, cores=cg)
        self.down(m, wd, y, 672, 0, wp, cores=co)

    @km.chain
    def whole(self, x, wq, q, wo, h, wg, g, wu, u, wd, y, cq, co, cg, kq: int, wp):
        """All four hand-overs gathered: o reads q's first 128 columns (attention's
        place), down reads g (the MLP product's place; up first); no host copy."""
        self.qkv(x, wq, q, kq, 0, wp, cores=cq)
        km.all_gather(q, 0, 256)
        self.o(q, wo, h, 128, 0, wp, cores=co)
        km.all_gather(h, 0, 168)
        self.gate_up(h, wu, u, kq, 0, wp, cores=cg)
        self.gate_up(km.past(h), wg, g, kq, 0, wp, cores=cg)
        km.all_gather(g, 0, 672)
        self.down(g, wd, y, 672, 0, wp, cores=co)
        km.all_gather(y, 0, 168)


WHOLE_SOURCES = {"qkv": "x", "o": "q", "gate": "h", "up": "h", "down": "g"}
# The rows that are checks: `words` is read before q, g and u (copies) are gathered, so
# the other island's half of those is stale DRAM; only o and down are checks there.
CHECK_ROWS = {
    "words": ("o", "down"),
    "whole_words": ("qkv", "o", "gate", "down"),
}


def finite(report):
    """ok says the check rows are finite; it is not an accuracy bound."""
    return all(
        math.isfinite(report[key][name])
        for key, names in CHECK_ROWS.items()
        if key in report
        for name in names
    )


def words_of(session, t, checked, sources):
    """two_card.words with each product's input named by `sources`."""
    out = {}
    for name, source in sources.items():
        target = tc.TARGETS[name]
        n, k = t[target].shape[1], bd.PRODUCTS[name]["k"]
        got = bd.row_tile(session, t[target], 0, n)
        given = bd.row_tile(session, t[source], 0, t[source].shape[1])[:, :k]
        cols, wcols = checked[name]
        out[name] = bd.rms(got[:, cols], given @ wcols.astype(np.float64))
    return out


def tiles_of(spec):
    """{product: tiles a page} of `spec`: "1" every W, "2/7" qkv's 2 and the others' 7
    (a page is not split: qkv is 8 columns a group, the others 7)."""
    parts = [int(part) for part in spec.split("/")]
    return {n: parts[0] if n == "qkv" else parts[-1] for n in tc.SOURCES}


def split_of(session, name, k):
    """bounds.hints_of's W split counted in pages of k tiles, so each island holds
    exactly its half of the pages (a group sized in tiles would leave them on one)."""
    pages, edge = bd.PRODUCTS[name]["n"] // 32 // k, bd.islands_of(session).first(1)
    each = next(
        e
        for e in range(1, pages + 1)
        if pages % e == 0 and pages // e % 2 == 0 and pages // e <= 2 * edge
    )
    return {"split": 1, "group": (edge - pages // each // 2, pages // each)}


def tensors_of(session, fmt, seed, check_cols, tiles, whole=False):
    """two_card.tensors_of with each W in pages of `tiles[name]` tiles; `whole`: q, g
    and u copies, to be gathered, and a as wide as q (o's handler is built for one)."""
    t, checked = tc.tensors_of(session, fmt, seed, check_cols)
    if whole:
        t["a"].free()
        values = bd.activations(1, t["q"].shape[1], seed + 1)
        t["a"] = session.tensor(values, "bfloat16", "tile", copy=True)
    for name in ("q", "g", "u") if whole else ():
        shape = t[name].shape
        t[name].free()
        t[name] = session.empty(shape, "bfloat16", "tile", copy=True)
    for name, k in tiles.items():
        if k == 1:
            continue
        pages = bd.weights(name, fmt, seed, check_cols)[0]
        shape = bd.PRODUCTS[name]
        t["w_" + name].free()
        t["w_" + name] = session.empty(
            (shape["k"], shape["n"]),
            fmt,
            "tile",
            page_tiles=k,
            **split_of(session, name, k),
        )
        session.write(t["w_" + name], pages.reshape(-1, k * pages.shape[1]))
    return t, checked


def made(session, t, chunk):
    """PagedTwoCard, its examples the layer's tensors."""
    examples = {
        "x_qkv": t["x"],
        "w_qkv": t["w_qkv"],
        "y_qkv": t["q"],
        "x_o": t["a"],
        "w_o": t["w_o"],
        "y_o": t["h"],
        "x_gu": t["h"],
        "w_gu": t["w_gate"],
        "y_gu": t["g"],
        "x_down": t["m"],
        "w_down": t["w_down"],
        "y_down": t["y"],
    }
    return PagedTwoCard(log=log, session=session, CHUNK=chunk, **examples)


def rated(out, fmt, names=("chain", "bare")):
    """tok/s (62 layers, the five products and the gathers) and W's GB/s a row."""
    tiles = sum(bd.PRODUCTS[n]["k"] * bd.PRODUCTS[n]["n"] // 1024 for n in tc.SOURCES)
    weight = tiles * bd.PAGE[fmt]
    for name in names:
        out[name + "_tok_s"] = round(1e6 / (out[name + "_us"] * bd.LAYERS), 2)
        out[name + "_gb_s"] = round(weight / out[name + "_us"] / 1e3, 1)
    out["ceiling_gb_s"] = 991.2
    return out


def measure_whole(session, fmt, spec, args):
    """Each pattern: two_card.py's chain (h and y gathered) and `whole` (all four),
    `repeat` times each, in one load."""
    tiles = tiles_of(spec)
    t, checked = tensors_of(session, fmt, args.seed, args.check_cols, tiles, True)
    program = made(session, t, args.chunk)
    run = session.load(program)
    names = ("x", "w_qkv", "q", "a", "w_o", "h", "w_gate", "g", "w_up", "u", "m")
    given = [t[n] for n in names] + [t["w_down"], t["y"]]
    names = ("x", "w_qkv", "q", "w_o", "h", "w_gate", "g", "w_up", "u", "w_down")
    taken = [t[n] for n in names] + [t["y"]]
    groups = [tc.group_of(session, n) for n in ("qkv", "o", "gate")]
    rows = []
    try:
        for pattern in args.patterns:
            wp = km.Pattern(pattern)
            out = {"class": "PagedTwoCard", "format": fmt, "page_tiles": tiles}
            out["pattern"] = pattern
            chain = run.chain(program.layer(*given, *groups, bd.HIDDEN // 32, wp))
            run.wait(run.submit(chain), timeout=10.0)
            out["words"] = words_of(session, t, checked, tc.SOURCES)
            out["chain_us"], out["chain_parts_us"] = tc.run_chain(
                run, chain, args.repeat
            )
            chain.free()
            whole = run.chain(program.whole(*taken, *groups, bd.HIDDEN // 32, wp))
            run.wait(run.submit(whole), timeout=10.0)
            out["whole_words"] = words_of(session, t, checked, WHOLE_SOURCES)
            out["whole_us"], out["whole_parts_us"] = tc.run_chain(
                run, whole, args.repeat
            )
            whole.free()
            rows.append(rated(out, fmt, ("chain", "whole")))
            log(json.dumps(rows[-1], default=float))
    finally:
        run.stop()
    for tensor in t.values():
        tensor.free()
    return rows


def measure(session, fmt, spec, args):
    """Each pattern: the chain with its gathers and without, `repeat` times each."""
    tiles = tiles_of(spec)
    t, checked = tensors_of(session, fmt, args.seed, args.check_cols, tiles)
    program = made(session, t, args.chunk)
    run = session.load(program)
    names = ("x", "w_qkv", "q", "a", "w_o", "h", "w_gate", "g", "w_up", "u", "m")
    given = [t[n] for n in names] + [t["w_down"], t["y"]]
    groups = [tc.group_of(session, n) for n in ("qkv", "o", "gate")]
    rows = []
    try:
        for pattern in args.patterns:
            wp = km.Pattern(pattern)
            words = (*given, *groups, bd.HIDDEN // 32, wp)
            out = {"class": "PagedTwoCard", "format": fmt, "page_tiles": tiles}
            out["pattern"] = pattern
            chain = run.chain(program.layer(*words))
            out["first_us"] = dc.parts_us(run.wait(run.submit(chain), timeout=10.0))
            out["words"] = tc.words(session, t, checked)
            out["chain_us"], out["chain_parts_us"] = tc.run_chain(
                run, chain, args.repeat
            )
            chain.free()
            bare = run.chain(program.bare(*words))
            dc.parts_us(run.wait(run.submit(bare), timeout=10.0))
            out["bare_us"], out["bare_parts_us"] = tc.run_chain(run, bare, args.repeat)
            out["bare_words"] = tc.words(session, t, checked)
            bare.free()
            rows.append(rated(out, fmt))
            log(json.dumps(rows[-1], default=float))
    finally:
        run.stop()
    for tensor in t.values():
        tensor.free()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--format", nargs="+", default=["bfp8_b"])
    parser.add_argument("--pages", nargs="+", default=["1", "2/7", "4/7"])
    parser.add_argument("--patterns", nargs="+", default=["AAAB", "AB"])
    parser.add_argument("--chunk", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--check-cols", type=int, default=256)
    parser.add_argument("--kernel-bytes", type=int)
    parser.add_argument("--whole", action="store_true", help="all four gathered")
    args = parser.parse_args()
    cards = [int(card) for card in args.cards.split(",")]
    opened = {"cards": cards}
    if args.kernel_bytes:
        opened["kernel_bytes"] = args.kernel_bytes
    reports = []
    with km.Session(log=log, **opened) as session:
        for fmt in args.format:
            for spec in args.pages:
                reports += (measure_whole if args.whole else measure)(
                    session, fmt, spec, args
                )
    for report in reports:
        print(json.dumps(report, default=float), flush=True)
    ok = all(finite(report) for report in reports)
    print(
        json.dumps(power.clocked({"cards": cards, "ok": ok}, card=cards[0])), flush=True
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
