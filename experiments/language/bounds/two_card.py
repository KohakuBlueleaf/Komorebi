"""A 27B-class decode layer on two cards as one chain, each card on one card's groups,
o's and down's outputs gathered by km.all_gather (q's and g/u's cannot be: a gather
is 255 pages at most), and the chain without its gathers. README.md.
Example: python two_card.py --format bfp8_b bfp4_b"""

import argparse
import json
import sys

import numpy as np

import komorebi as km
from experiments.language.bounds import bounds as bd
from experiments.language.bounds import decode_cost as dc
from komorebi.bench import power

# product: (width, group size); each card's half of N in groups of that width.
PLAN = {"qkv": (8, 7), "o": (7, 9), "gate": (7, 2), "up": (7, 2), "down": (7, 9)}
A_TILES, M_TILES = bd.HEADS * bd.HEAD_DIM // 32, bd.MLP // 32  # o's and down's K
log = bd.log


class TwoCard(km.Program):
    """The five products by the K split, K a word: qkv in groups of S_QKV at width
    W8, o and down of S_O, gate and up of S_G at width W7; the gathers in `layer`."""

    S_QKV = km.constant(7)
    S_O = km.constant(9)
    S_G = km.constant(2)
    W8 = km.constant(8)
    W7 = km.constant(7)
    CHUNK = km.constant(2)
    B8 = CHUNK * W8  # W's blocks
    B7 = CHUNK * W7
    sums8: km.L1[km.fp32, 1, W8, 32, 32]
    sums7: km.L1[km.fp32, 1, W7, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def qkv(
        self, x_qkv: km.Tensor, w_qkv: km.Tensor, y_qkv: km.Tensor, k: int, col0: int
    ):
        share = self.rank % self.S_QKV  # km::partition's share of K
        first = col0 + self.rank // self.S_QKV * self.W8
        k0 = share * (k // self.S_QKV) + min(share, k % self.S_QKV)
        k1 = (share + 1) * (k // self.S_QKV) + min(share + 1, k % self.S_QKV)
        for xb, wb in zip(
            km.Stream(x_qkv.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(w_qkv.tiles[k0:k1, first : first + self.W8], block=self.B8),
        ):
            self.sums8 += self.project(xb, wb)
        whole = km.reduce(self.sums8, form="tree", group=self.S_QKV)
        if self.rank == self.leader:
            y_qkv.tiles[0:1, first : first + self.W8].write(whole)

    @km.handler
    def o(self, x_o: km.Tensor, w_o: km.Tensor, y_o: km.Tensor, k: int, col0: int):
        share = self.rank % self.S_O
        first = col0 + self.rank // self.S_O * self.W7
        k0 = share * (k // self.S_O) + min(share, k % self.S_O)
        k1 = (share + 1) * (k // self.S_O) + min(share + 1, k % self.S_O)
        for xb, wb in zip(
            km.Stream(x_o.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(w_o.tiles[k0:k1, first : first + self.W7], block=self.B7),
        ):
            self.sums7 += self.project(xb, wb)
        whole = km.reduce(self.sums7, form="tree", group=self.S_O)
        if self.rank == self.leader:
            y_o.tiles[0:1, first : first + self.W7].write(whole)

    @km.handler
    def gate_up(
        self, x_gu: km.Tensor, w_gu: km.Tensor, y_gu: km.Tensor, k: int, col0: int
    ):
        share = self.rank % self.S_G
        first = col0 + self.rank // self.S_G * self.W7
        k0 = share * (k // self.S_G) + min(share, k % self.S_G)
        k1 = (share + 1) * (k // self.S_G) + min(share + 1, k % self.S_G)
        for xb, wb in zip(
            km.Stream(x_gu.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(w_gu.tiles[k0:k1, first : first + self.W7], block=self.B7),
        ):
            self.sums7 += self.project(xb, wb)
        whole = km.reduce(self.sums7, form="tree", group=self.S_G)
        if self.rank == self.leader:
            y_gu.tiles[0:1, first : first + self.W7].write(whole)

    @km.handler
    def down(
        self, x_down: km.Tensor, w_down: km.Tensor, y_down: km.Tensor, k: int, col0: int
    ):
        share = self.rank % self.S_O
        first = col0 + self.rank // self.S_O * self.W7
        k0 = share * (k // self.S_O) + min(share, k % self.S_O)
        k1 = (share + 1) * (k // self.S_O) + min(share + 1, k % self.S_O)
        for xb, wb in zip(
            km.Stream(x_down.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(w_down.tiles[k0:k1, first : first + self.W7], block=self.B7),
        ):
            self.sums7 += self.project(xb, wb)
        whole = km.reduce(self.sums7, form="tree", group=self.S_O)
        if self.rank == self.leader:
            y_down.tiles[0:1, first : first + self.W7].write(whole)

    @km.chain
    def layer(self, x, wq, q, a, wo, h, wg, g, wu, u, m, wd, y, cq, co, cg, kq: int):
        """qkv, o, h gathered, gate, up, down, y gathered (o reads a, down reads m)."""
        self.qkv(x, wq, q, kq, 0, cores=cq)
        self.o(a, wo, h, 128, 0, cores=co)
        km.all_gather(h, 0, 168)
        self.gate_up(h, wg, g, kq, 0, cores=cg)
        self.gate_up(km.past(h), wu, u, kq, 0, cores=cg)
        self.down(m, wd, y, 672, 0, cores=co)
        km.all_gather(y, 0, 168)

    @km.chain
    def bare(self, x, wq, q, a, wo, h, wg, g, wu, u, m, wd, y, cq, co, cg, kq: int):
        """The same calls without the gathers: each card reads its own half."""
        self.qkv(x, wq, q, kq, 0, cores=cq)
        self.o(a, wo, h, 128, 0, cores=co)
        self.gate_up(h, wg, g, kq, 0, cores=cg)
        self.gate_up(km.past(h), wu, u, kq, 0, cores=cg)
        self.down(m, wd, y, 672, 0, cores=co)


def group_of(session, name):
    """A product's call over both cards: each card's groups of its half of N, the
    groups meeting at the second card's first rank."""
    width, size = PLAN[name]
    cores = bd.PRODUCTS[name]["n"] // 32 // 2 // width * size
    edge = session.graph.first(1)
    return (edge - cores, 2 * cores)


def tensors_of(session, fmt, seed, check_cols):
    """W (b1's pages), split by columns; X, o's input a and down's input m copies of
    host values; h and y copies (gathered); q, g and u split as W."""
    t, checked = {}, {}
    for name in ("qkv", "o", "gate", "up", "down"):
        hints = bd.hints_of(session, name)
        pages, cols, wcols = bd.weights(name, fmt, seed, check_cols)
        shape = bd.PRODUCTS[name]
        t["w_" + name] = session.empty(
            (shape["k"], shape["n"]), fmt, "tile", **hints["w"]
        )
        session.write(t["w_" + name], pages)
        checked[name] = (cols, wcols)
    for name, k in (("x", bd.HIDDEN), ("a", A_TILES * 32), ("m", M_TILES * 32)):
        values = bd.activations(1, k, seed + len(t))
        t[name] = session.tensor(values, "bfloat16", "tile", copy=True)
    for name, product in (("q", "qkv"), ("g", "gate"), ("u", "up")):
        split = bd.hints_of(session, product)["y"]
        n = bd.PRODUCTS[product]["n"]
        t[name] = session.empty((32, n), "bfloat16", "tile", **split)
    for name in ("h", "y"):
        t[name] = session.empty((32, bd.HIDDEN), "bfloat16", "tile", copy=True)
    return t, checked


SOURCES = {"qkv": "x", "o": "a", "gate": "h", "up": "h", "down": "m"}
TARGETS = {"qkv": "q", "o": "h", "gate": "g", "up": "u", "down": "y"}


def words(session, t, checked):
    """Each product's relative RMS against float64 of its input as the card held it."""
    out = {}
    for name, source in SOURCES.items():
        target = TARGETS[name]
        n, k = t[target].shape[1], bd.PRODUCTS[name]["k"]
        got = bd.row_tile(session, t[target], 0, n)
        given = bd.row_tile(session, t[source], 0, t[source].shape[1])[:, :k]
        cols, wcols = checked[name]
        out[name] = bd.rms(got[:, cols], given @ wcols.astype(np.float64))
    return out


def made(session, t, fmt, chunk):
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
    return TwoCard(log=log, session=session, CHUNK=chunk, **examples)


def run_chain(run, chain, repeat):
    """`repeat` chains: the median of the slower card's microseconds and each card's."""
    times = [
        dc.parts_us(run.wait(run.submit(chain), timeout=10.0)) for _ in range(repeat)
    ]
    return float(np.median([max(p) for p in times])), np.median(times, axis=0).tolist()


def measure(session, fmt, args):
    t, checked = tensors_of(session, fmt, args.seed, args.check_cols)
    program = made(session, t, fmt, args.chunk)
    run = session.load(program)
    names = ("x", "w_qkv", "q", "a", "w_o", "h", "w_gate", "g", "w_up", "u", "m")
    given = [t[n] for n in names] + [t["w_down"], t["y"]]
    groups = [group_of(session, n) for n in ("qkv", "o", "gate")]
    out = {"format": fmt, "groups": groups, "chunk": args.chunk}
    try:
        chain = run.chain(program.layer(*given, *groups, bd.HIDDEN // 32))
        before = dc.crossings_now(run)
        out["first_us"] = dc.parts_us(run.wait(run.submit(chain), timeout=10.0))
        out["crossed"] = dc.crossed(run, before)
        out["words"] = words(session, t, checked)
        out["chain_us"], out["chain_parts_us"] = run_chain(run, chain, args.repeat)
        chain.free()
        bare = run.chain(program.bare(*given, *groups, bd.HIDDEN // 32))
        before = dc.crossings_now(run)
        dc.parts_us(run.wait(run.submit(bare), timeout=10.0))
        out["bare_crossed"] = dc.crossed(run, before)
        out["bare_us"], out["bare_parts_us"] = run_chain(run, bare, args.repeat)
        bare.free()
    finally:
        run.stop()
    for tensor in t.values():
        tensor.free()
    out["bound_us"] = sum(bd.bounds_of(n, fmt, 1, 2)["memory_us"] for n in SOURCES)
    out["ratio"] = out["chain_us"] / out["bound_us"]
    out["bare_ratio"] = out["bare_us"] / out["bound_us"]
    out["gathers_us"] = out["chain_us"] - out["bare_us"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--format", nargs="+", default=["bfp8_b"])
    parser.add_argument("--chunk", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--check-cols", type=int, default=256)
    parser.add_argument("--kernel-bytes", type=int)
    args = parser.parse_args()
    cards = [int(card) for card in args.cards.split(",")]
    opened = {"cards": cards}
    if args.kernel_bytes:
        opened["kernel_bytes"] = args.kernel_bytes
    reports = []
    with km.Session(log=log, **opened) as session:
        for fmt in args.format:
            reports.append(measure(session, fmt, args))
            log(json.dumps(reports[-1], default=float))
    for report in reports:
        print(json.dumps(report, default=float), flush=True)
    print(json.dumps(power.clocked({"cards": cards}, card=cards[0])), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
