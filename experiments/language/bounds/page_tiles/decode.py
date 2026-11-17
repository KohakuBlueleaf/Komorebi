"""The decode layer's five products as one chain with W in DRAM pages of k tiles
(page_tiles), W's stream on the default pattern or on AB, beside decode_cost.py's
class unchanged at a page a tile; words against float64. README.md.
python experiments/language/bounds/page_tiles/decode.py --cards 0 --pages 1 2 4/7"""

import argparse
import json
import sys

import numpy as np

import komorebi as km
from experiments.language.bounds import bounds as bd
from experiments.language.bounds import decode_cost as dc
from komorebi.bench import power

log = bd.log
PATTERNS = ("AAAB", "AB")  # W's stream: the default, and both movers in turn


class PagedLayer(km.Program):
    """decode_cost.Layer with W's stream on the pattern given (`wp`): its tensors in
    pages of the example's tiles."""

    S_QKV = km.constant(7)
    S_O = km.constant(10)
    S_G = km.constant(2)
    W16 = km.constant(16)
    W14 = km.constant(14)
    CHUNK = km.constant(2)
    B16 = CHUNK * W16
    B14 = CHUNK * W14
    sums16: km.L1[km.fp32, 1, W16, 32, 32]
    sums14: km.L1[km.fp32, 1, W14, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

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
        first = col0 + self.rank // self.S_QKV * self.W16
        k0 = share * (k // self.S_QKV) + min(share, k % self.S_QKV)
        k1 = (share + 1) * (k // self.S_QKV) + min(share + 1, k % self.S_QKV)
        for xb, wb in zip(
            km.Stream(x_qkv.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(
                w_qkv.tiles[k0:k1, first : first + self.W16],
                block=self.B16,
                pattern=wp,
            ),
        ):
            self.sums16 += self.project(xb, wb)
        whole = km.reduce(self.sums16, form="tree", group=self.S_QKV)
        if self.rank == self.leader:
            y_qkv.tiles[0:1, first : first + self.W16].write(whole)

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
        first = col0 + self.rank // self.S_O * self.W14
        k0 = share * (k // self.S_O) + min(share, k % self.S_O)
        k1 = (share + 1) * (k // self.S_O) + min(share + 1, k % self.S_O)
        for xb, wb in zip(
            km.Stream(x_o.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(
                w_o.tiles[k0:k1, first : first + self.W14], block=self.B14, pattern=wp
            ),
        ):
            self.sums14 += self.project(xb, wb)
        whole = km.reduce(self.sums14, form="tree", group=self.S_O)
        if self.rank == self.leader:
            y_o.tiles[0:1, first : first + self.W14].write(whole)

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
        first = col0 + self.rank // self.S_O * self.W14
        k0 = share * (k // self.S_O) + min(share, k % self.S_O)
        k1 = (share + 1) * (k // self.S_O) + min(share + 1, k % self.S_O)
        for xb, wb in zip(
            km.Stream(x_down.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(
                w_down.tiles[k0:k1, first : first + self.W14],
                block=self.B14,
                pattern=wp,
            ),
        ):
            self.sums14 += self.project(xb, wb)
        whole = km.reduce(self.sums14, form="tree", group=self.S_O)
        if self.rank == self.leader:
            y_down.tiles[0:1, first : first + self.W14].write(whole)

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
        first = col0 + self.rank // self.S_G * self.W14
        k0 = share * (k // self.S_G) + min(share, k % self.S_G)
        k1 = (share + 1) * (k // self.S_G) + min(share + 1, k % self.S_G)
        for xb, wb in zip(
            km.Stream(x_gu.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(
                w_gu.tiles[k0:k1, first : first + self.W14], block=self.B14, pattern=wp
            ),
        ):
            self.sums14 += self.project(xb, wb)
        whole = km.reduce(self.sums14, form="tree", group=self.S_G)
        if self.rank == self.leader:
            y_gu.tiles[0:1, first : first + self.W14].write(whole)


def tiles_of(spec):
    """{product: tiles a page} of `spec`: "2" every W, "4/7" qkv's 4 and the others' 7
    (qkv is 16 columns a group, the others 14)."""
    parts = [int(part) for part in spec.split("/")]
    qkv, rest = parts[0], parts[-1]
    return {name: qkv if name == "qkv" else rest for name, *_ in dc.CALLS}


def tensors_of(session, fmt, seed, check_cols, tiles):
    """decode_cost.tensors_of with each W in pages of `tiles[name]` tiles."""
    t, checked = {}, {}
    for name, _, source, out, _ in dc.CALLS:
        hints = bd.hints_of(session, name)
        pages, cols, wcols = bd.weights(name, fmt, seed, check_cols)
        shape = bd.PRODUCTS[name]
        k = tiles[name]
        t["w_" + name] = session.empty(
            (shape["k"], shape["n"]), fmt, "tile", page_tiles=k, **hints["w"]
        )
        session.write(t["w_" + name], pages.reshape(-1, k * pages.shape[1]))
        checked[name] = (cols, wcols)
        t[out] = session.empty((32, dc.SIZES[out]), "bfloat16", "tile", **hints["y"])
        if source == "x":
            x = bd.activations(1, bd.HIDDEN, seed)
            t["x"] = session.tensor(x, "bfloat16", "tile", **hints["x"])
    return t, checked


def made(session, t, chunk):
    """PagedLayer at the session's group sizes, its examples the layer's tensors."""
    at = 1 if bd.islands_of(session) is not None else 0
    examples = {}
    for name, handler, source, target, _ in dc.CALLS:
        x, w, y = dc.PARAMS[handler]
        examples |= {x: t[source], w: t["w_" + name], y: t[target]}
    return PagedLayer(
        log=log,
        session=session,
        chain=True,
        S_QKV=dc.PLAN["qkv"][1][at],
        S_O=dc.PLAN["o"][1][at],
        S_G=dc.PLAN["gate"][1][at],
        CHUNK=chunk,
        **examples,
    )


def calls_of(session, t, pattern):
    """The layer's five calls, W's stream on `pattern`."""
    out = []
    wp = km.Pattern(pattern)
    for name, handler, source, target, k in dc.CALLS:
        _, group, col0 = dc.groups_of(session, name)[0]
        given = km.past(t[source]) if name == "up" else t[source]
        method = getattr(PagedLayer, handler)
        out.append(method(given, t["w_" + name], t[target], k, col0, wp, cores=group))
    return out


def timed(run, chain, repeat):
    """The chain's first run and the median of `repeat` more (the slower part)."""
    first = dc.parts_us(run.wait(run.submit(chain), timeout=10.0))
    times = [
        max(dc.parts_us(run.wait(run.submit(chain), timeout=10.0)))
        for _ in range(repeat)
    ]
    return max(first), float(np.median(times))


def rated(out, fmt, cards):
    """A row's tok/s (62 layers, the five products) and W's GB/s (a layer's weight
    bytes over its time) beside the bound's."""
    tiles = sum(bd.PRODUCTS[n]["k"] * bd.PRODUCTS[n]["n"] // 1024 for n, *_ in dc.CALLS)
    weight = tiles * bd.PAGE[fmt]
    out["tok_s"] = round(1e6 / (out["chain_us"] * bd.LAYERS), 1)
    out["gb_s"] = round(weight / out["chain_us"] / 1e3, 1)
    out["ceiling_gb_s"] = round(495.6 * cards, 1)
    return out


def unchanged(session, fmt, args):
    """decode_cost.Layer, as built before page_tiles: its chain's time and words."""
    t, checked = dc.tensors_of(session, fmt, args.seed, args.check_cols)
    run = session.load(dc.made(session, t, args.chunk))
    try:
        chain = run.chain(dc.calls_of(session, t))
        first, median = timed(run, chain, args.repeat)
        words = dc.words(session, t, checked)
        chain.free()
    finally:
        run.stop()
    for tensor in t.values():
        tensor.free()
    cards = 2 if bd.islands_of(session) else 1
    out = {"class": "decode_cost.Layer", "format": fmt, "chain_us": median}
    out |= {"first_us": first, "words": words}
    return rated(out, fmt, cards)


def paged(session, fmt, spec, args):
    """PagedLayer with W in pages of `spec`'s tiles, a row a pattern of W's stream."""
    tiles = tiles_of(spec)
    t, checked = tensors_of(session, fmt, args.seed, args.check_cols, tiles)
    cards = 2 if bd.islands_of(session) else 1
    rows = []
    run = session.load(made(session, t, args.chunk))
    try:
        for pattern in args.patterns:
            for target in ("q", "h", "g", "u", "y"):
                session.write(t[target], np.zeros_like(session.read(t[target])[0]))
            chain = run.chain(calls_of(session, t, pattern))
            first, median = timed(run, chain, args.repeat)
            words = dc.words(session, t, checked)
            chain.free()
            out = {"class": "PagedLayer", "format": fmt, "page_tiles": tiles}
            out |= {"pattern": pattern, "chain_us": median, "first_us": first}
            rows.append(rated(out | {"words": words}, fmt, cards))
            log(json.dumps(rows[-1], default=float))
    finally:
        run.stop()
    for tensor in t.values():
        tensor.free()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0", help="0, or 0,1")
    parser.add_argument("--format", nargs="+", default=["bfp8_b", "bfp4_b"])
    parser.add_argument("--pages", nargs="*", default=["1", "2", "4/7"])
    parser.add_argument("--patterns", nargs="+", default=list(PATTERNS))
    parser.add_argument("--unchanged", action="store_true", help="decode_cost's too")
    parser.add_argument("--chunk", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--check-cols", type=int, default=256)
    parser.add_argument("--kernel-bytes", type=int)
    args = parser.parse_args()
    cards = [int(card) for card in args.cards.split(",")]
    opened = {"cards": cards} if len(cards) > 1 else {"card": cards[0]}
    if args.kernel_bytes:
        opened["kernel_bytes"] = args.kernel_bytes
    reports = []
    with km.Session(log=log, **opened) as session:
        for fmt in args.format:
            if args.unchanged:
                reports.append(unchanged(session, fmt, args))
                log(json.dumps(reports[-1], default=float))
            for spec in args.pages:
                reports += paged(session, fmt, spec, args)
    for report in reports:
        print(json.dumps(report, default=float), flush=True)
    print(json.dumps(power.clocked({"cards": cards}, card=cards[0])), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
