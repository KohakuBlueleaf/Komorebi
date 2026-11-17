"""A decode layer on two cards (qkv and gate/up by output columns, o and down by their
input, the partials crossing by km.all_gather) as a chain and two add commands, as four
commands in order, and as ONE chain whose adds write each card's copy (a write by
island). README.md. python experiments/language/bounds/page_tiles/one_chain.py"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from experiments.language.bounds import bounds as bd
from experiments.language.bounds import decode_cost as dc
from komorebi import formats
from komorebi.bench import power

PAGE = bd.PAGE

A_TILES, M_TILES, H_TILES = 128, bd.MLP // 32, bd.HIDDEN // 32  # o's, down's K; h's
GU_TILES = 2 * M_TILES  # gate and up as one product: 1,344 columns
# product: (width, group size, ranks a card); each card's groups of its own part.
PLAN = {"qkv": (8, 7, 112), "part": (14, 8, 96), "gate_up": (14, 2, 96)}
ADDERS = 84  # add's ranks a card, 2 tiles each
PATTERNS = ("AAAB", "AB")
# the chain's sizes: h's tiles, a card's K of o and of down, part's ranks a card, the
# partials' pages (both cards')
SIZES = (H_TILES, A_TILES // 2, M_TILES // 2, PLAN["part"][2], 2 * H_TILES)
log = bd.log


class Alternate(km.Program):
    """qkv by output columns (groups of S_QKV at width W8, as two_card.py); o and down
    by input (each card all of N on its half of K in groups of S_P at width W14, its
    partial its row tile); add sums the two; gate and up as one product."""

    S_QKV = km.constant(7)
    S_P = km.constant(8)
    S_G = km.constant(2)
    W8 = km.constant(8)
    W14 = km.constant(14)
    H = km.constant(H_TILES)
    TILES = km.constant(H_TILES // ADDERS)  # add's tiles a rank
    CHUNK = km.constant(2)
    B8 = CHUNK * W8  # W's blocks
    B14 = CHUNK * W14
    sums8: km.L1[km.fp32, 1, W8, 32, 32]
    sums14: km.L1[km.fp32, 1, W14, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def qkv(
        self, x_q: km.Tensor, w_q: km.Tensor, y_q: km.Tensor, k: int, p: km.Pattern
    ):
        share = self.rank % self.S_QKV  # km::partition's share of K
        first = self.rank // self.S_QKV * self.W8
        k0 = share * (k // self.S_QKV) + min(share, k % self.S_QKV)
        k1 = (share + 1) * (k // self.S_QKV) + min(share + 1, k % self.S_QKV)
        for xb, wb in zip(
            km.Stream(x_q.tiles[0:1, k0:k1], block=self.CHUNK, pattern=p),
            km.Stream(
                w_q.tiles[k0:k1, first : first + self.W8], block=self.B8, pattern=p
            ),
        ):
            self.sums8 += self.project(xb, wb)
        whole = km.reduce(self.sums8, form="tree", group=self.S_QKV)
        if self.rank == self.leader:
            y_q.tiles[0:1, first : first + self.W8].write(whole)

    @km.handler
    def part_o(
        self,
        x_o: km.Tensor,
        w_o: km.Tensor,
        h_o: km.Tensor,
        k: int,
        per: int,
        p: km.Pattern,
    ):
        """A card's partial of all of N over its half of K (k tiles: x_o's and w_o's
        copy on each card holds that card's half), written as the card's row tile of
        h_o; part_down is the same on down's."""
        card = self.rank // per
        local = self.rank % per
        share = local % self.S_P
        first = local // self.S_P * self.W14
        k0 = share * (k // self.S_P)
        k1 = k0 + k // self.S_P
        for xb, wb in zip(
            km.Stream(x_o.tiles[0:1, k0:k1], block=self.CHUNK, pattern=p),
            km.Stream(
                w_o.tiles[k0:k1, first : first + self.W14], block=self.B14, pattern=p
            ),
        ):
            self.sums14 += self.project(xb, wb)
        whole = km.reduce(self.sums14, form="tree", group=self.S_P)
        if self.rank == self.leader:
            h_o.tiles[card : card + 1, first : first + self.W14].write(whole)

    @km.handler
    def part_down(
        self,
        x_d: km.Tensor,
        w_d: km.Tensor,
        y_d: km.Tensor,
        k: int,
        per: int,
        p: km.Pattern,
    ):
        card = self.rank // per
        local = self.rank % per
        share = local % self.S_P
        first = local // self.S_P * self.W14
        k0 = share * (k // self.S_P)
        k1 = k0 + k // self.S_P
        for xb, wb in zip(
            km.Stream(x_d.tiles[0:1, k0:k1], block=self.CHUNK, pattern=p),
            km.Stream(
                w_d.tiles[k0:k1, first : first + self.W14], block=self.B14, pattern=p
            ),
        ):
            self.sums14 += self.project(xb, wb)
        whole = km.reduce(self.sums14, form="tree", group=self.S_P)
        if self.rank == self.leader:
            y_d.tiles[card : card + 1, first : first + self.W14].write(whole)

    @km.handler
    def gate_up(self, h_g: km.Tensor, w_g: km.Tensor, y_g: km.Tensor, p: km.Pattern):
        share = self.rank % self.S_G
        first = self.rank // self.S_G * self.W14
        k0 = share * (self.H // self.S_G)
        k1 = k0 + self.H // self.S_G
        for xb, wb in zip(
            km.Stream(h_g.tiles[0:1, k0:k1], block=self.CHUNK, pattern=p),
            km.Stream(
                w_g.tiles[k0:k1, first : first + self.W14], block=self.B14, pattern=p
            ),
        ):
            self.sums14 += self.project(xb, wb)
        whole = km.reduce(self.sums14, form="tree", group=self.S_G)
        if self.rank == self.leader:
            y_g.tiles[0:1, first : first + self.W14].write(whole)

    @km.handler
    def add(self, parts: km.Tensor, total: km.Tensor, per: int):
        """total = the two cards' partials (parts' row tiles 0 and 1) summed, in bf16:
        each card's per ranks write their card's copy, TILES a rank."""
        first = self.rank % per * self.TILES
        results = km.Stream(total.tiles[0:1, first : first + self.TILES], write=True)
        for x0, x1 in zip(
            km.Stream(parts.tiles[0:1, first : first + self.TILES]),
            km.Stream(parts.tiles[1:2, first : first + self.TILES]),
        ):
            results.push(x0 + x1)

    @km.chain
    def layer(
        self,
        x,
        wq,
        q,
        a,
        wo,
        h,
        hs,
        wgu,
        gu,
        m,
        wd,
        y,
        cq,
        cp,
        cg,
        p: km.Pattern,
        hidden: int,
        ko: int,
        kd: int,
        per: int,
        pages: int,
    ):
        """qkv; o's partials and their gather; gate_up on hs, the add command's sum
        (run apart: so gate_up reads last turn's hs); down's partials and their
        gather. o reads a, down m."""
        self.qkv(x, wq, q, hidden, p, cores=cq)
        self.part_o(a, wo, h, ko, per, p, cores=cp)
        km.all_gather(h, 0, pages)
        self.gate_up(hs, wgu, gu, p, cores=cg)
        self.part_down(m, wd, y, kd, per, p, cores=cp)
        km.all_gather(y, 0, pages)

    @km.chain
    def first_half(self, x, wq, q, a, wo, h, cq, cp, p: km.Pattern, hidden: int,
                   ko: int, per: int, pages: int):  # fmt: skip
        """qkv, o's partials and their gather: what h's add command waits for."""
        self.qkv(x, wq, q, hidden, p, cores=cq)
        self.part_o(a, wo, h, ko, per, p, cores=cp)
        km.all_gather(h, 0, pages)

    @km.chain
    def second_half(self, hs, wgu, gu, m, wd, y, cg, cp, p: km.Pattern, kd: int,
                    per: int, pages: int):  # fmt: skip
        """gate_up on hs (the add command's), down's partials and their gather."""
        self.gate_up(hs, wgu, gu, p, cores=cg)
        self.part_down(m, wd, y, kd, per, p, cores=cp)
        km.all_gather(y, 0, pages)

    @km.chain
    def layer_one(
        self,
        x,
        wq,
        q,
        a,
        wo,
        h,
        hs,
        wgu,
        gu,
        m,
        wd,
        y,
        ys,
        cq,
        cp,
        cg,
        ca,
        p: km.Pattern,
        hidden: int,
        ko: int,
        kd: int,
        per: int,
        pages: int,
        adders: int,
    ):
        """The layer as one chain: each add after its gather, on each card its own
        copy of the sum (hs, ys: a write by island)."""
        self.qkv(x, wq, q, hidden, p, cores=cq)
        self.part_o(a, wo, h, ko, per, p, cores=cp)
        km.all_gather(h, 0, pages)
        self.add(h, hs, adders, cores=ca)
        self.gate_up(hs, wgu, gu, p, cores=cg)
        self.part_down(m, wd, y, kd, per, p, cores=cp)
        km.all_gather(y, 0, pages)
        self.add(y, ys, adders, cores=ca)


def halves(graph, steps, dim):
    """A split of `steps` along `dim` over a group centred on island 1's first rank
    whose ranks each take the same count: each island exactly its half."""
    edge = graph.first(1)
    each = next(
        e
        for e in range(1, steps + 1)
        if steps % e == 0 and steps // e % 2 == 0 and steps // e <= 2 * edge
    )
    return {"split": dim, "group": (edge - steps // each // 2, steps // each)}


def cores_of(session, name):
    """A call's ranks: each card's PLAN ranks (add's ADDERS), meeting at island 1's
    first rank."""
    each = ADDERS if name == "add" else PLAN[name][2]
    return (session.graph.first(1) - each, 2 * each)


def gate_up_pages(fmt, seed, check_cols):
    """gate's and up's W (bounds.py's) as one W of 1,344 columns, each card's half
    its gate half then its up half; the checked columns where they land there."""
    gate, gcols, gw = bd.weights("gate", fmt, seed, check_cols)
    up, ucols, uw = bd.weights("up", fmt, seed, check_cols)
    kt, half = bd.HIDDEN // 32, M_TILES // 2
    g, u = (pages.reshape(kt, M_TILES, -1) for pages in (gate, up))
    order = [g[:, :half], u[:, :half], g[:, half:], u[:, half:]]
    pages = np.concatenate(order, axis=1).reshape(-1, gate.shape[1])
    edge = half * 32  # a card's half of gate's columns

    def moved(cols, shift):
        return cols + np.where(cols < edge, shift, shift + edge)

    cols = np.concatenate([moved(gcols, 0), moved(ucols, edge)])
    return pages, cols, np.concatenate([gw, uw], axis=1)


def tensors_of(session, fmt, seed, check_cols):
    """W: qkv's and gate_up's split by columns; o's and down's, and their inputs a and
    m, copies whose card's copy holds that card's half of K; x and the partials
    copies."""
    graph, t, checked = session.graph, {}, {}
    pages, cols, wcols = bd.weights("qkv", fmt, seed, check_cols)
    split = halves(graph, bd.QKV // 32, 1)
    t["w_qkv"] = session.empty((bd.HIDDEN, bd.QKV), fmt, "tile", **split)
    t["q"] = session.empty((32, bd.QKV), "bfloat16", "tile", **split)
    session.write(t["w_qkv"], pages)
    checked["qkv"] = (cols, wcols)
    pages, cols, wcols = gate_up_pages(fmt, seed, check_cols)
    split = halves(graph, GU_TILES, 1)
    t["w_gu"] = session.empty((bd.HIDDEN, GU_TILES * 32), fmt, "tile", **split)
    t["gu"] = session.empty((32, GU_TILES * 32), "bfloat16", "tile", **split)
    session.write(t["w_gu"], pages)
    checked["gate_up"] = (cols, wcols)
    x = bd.activations(1, bd.HIDDEN, seed)
    t["x"] = session.tensor(x, "bfloat16", "tile", copy=True)
    for name, source, k in (("o", "a", A_TILES), ("down", "m", M_TILES)):
        pages, cols, wcols = bd.weights(name, fmt, seed, check_cols)
        values = bd.activations(1, k * 32, seed + len(t))
        half = k // 2
        t["w_" + name] = session.empty((half * 32, bd.HIDDEN), fmt, "tile", copy=True)
        t[source] = session.empty((32, half * 32), "bfloat16", "tile", copy=True)
        rows = pages.reshape(k, bd.HIDDEN // 32, -1)
        for card in (0, 1):  # each card's copy holds its half of K
            mine = slice(card * half, (card + 1) * half)
            island = session.island(card)
            island.write(t["w_" + name].parts[card], rows[mine].reshape(-1, PAGE[fmt]))
            raw, _ = formats.to_bytes(
                values[:, mine.start * 32 : mine.stop * 32], "bfloat16", "tile"
            )
            island.write(t[source].parts[card], raw.reshape(-1, 2048))
        checked[name] = (cols, wcols)
        checked[source] = values
    for name in ("h", "y"):  # the two cards' partials: card c's row tile c (336 pages)
        t[name] = session.empty((64, bd.HIDDEN), "float32", "tile", copy=True)
    # h's sum (the add's command; y's is the same on the next layer's x)
    t["hs"] = session.empty((32, bd.HIDDEN), "bfloat16", "tile", copy=True)
    t["ys"] = session.empty((32, bd.HIDDEN), "bfloat16", "tile", copy=True)
    return t, checked


def copy_of(session, tensor, island, dtype):
    """Island `island`'s copy of a tile tensor as float64 values of its shape."""
    shape = tuple(tensor.shape)
    pages = shape[0] * shape[1] // 1024
    (raw,) = session.island(island).read(tensor.parts[island], first=0, pages=pages)
    values = formats.from_bytes(np.asarray(raw).reshape(-1), shape, dtype, "tile")
    return np.asarray(values, np.float64).reshape(shape)


def partials(session, t, name, island):
    """The two cards' partials [2, 32, HIDDEN] as island `island`'s copy holds them."""
    held = copy_of(session, t[name], island, "float32")
    return np.stack([held[:32], held[32:]])


def words(session, t, checked, ys=False):
    """Relative RMS against float64 of each input as the card held it: q's and
    gate_up's checked columns; each card's partial of h and of y on both copies (the
    gathers' pages) on o's and down's; the adds' sums on each copy."""
    out = {}
    x = bd.row_tile(session, t["x"], 0, bd.HIDDEN)
    cols, wcols = checked["qkv"]
    q = bd.row_tile(session, t["q"], 0, bd.QKV)
    out["qkv"] = bd.rms(q[:, cols], x @ wcols.astype(np.float64))
    for name, source, sums, k in (
        ("o", "a", "h", A_TILES),
        ("down", "m", "y", M_TILES),
    ):
        given = checked[source].astype(np.float64)  # the host's values, both halves
        cols, wcols = checked[name]
        half = k * 32 // 2
        for island in (0, 1):
            got = partials(session, t, sums, island)
            for card in (0, 1):
                part = slice(card * half, (card + 1) * half)
                want = given[:, part] @ wcols[part].astype(np.float64)
                out[f"{name}_card{card}_on{island}"] = bd.rms(got[card][:, cols], want)
    for island in (0, 1) if ys else ():  # y's sum on each copy (the one chain's)
        want = partials(session, t, "y", island).sum(axis=0)
        out[f"add_y_on{island}"] = bd.rms(
            copy_of(session, t["ys"], island, "bfloat16"), want
        )
    cols, wcols = checked["gate_up"]
    gu = bd.row_tile(session, t["gu"], 0, GU_TILES * 32)
    for island in (0, 1):  # each card's sums of the partials, and its columns of gu
        want = partials(session, t, "h", island).sum(axis=0)
        got = copy_of(session, t["hs"], island, "bfloat16")
        out[f"add_on{island}"] = bd.rms(got, want)
        h = copy_of(session, t["hs"], island, "bfloat16")  # the add's sum
        mine = (cols >= island * M_TILES * 32) & (cols < (island + 1) * M_TILES * 32)
        want = h @ wcols[:, mine].astype(np.float64)
        out[f"gate_up_card{island}"] = bd.rms(gu[:, cols[mine]], want)
    return out


def made(session, t, chunk):
    examples = {"x_q": t["x"], "w_q": t["w_qkv"], "y_q": t["q"]}
    examples |= {"x_o": t["a"], "w_o": t["w_o"], "h_o": t["h"]}
    examples |= {"x_d": t["m"], "w_d": t["w_down"], "y_d": t["y"]}
    examples |= {"h_g": t["hs"], "w_g": t["w_gu"], "y_g": t["gu"]}
    examples |= {"parts": t["h"], "total": t["hs"]}
    return Alternate(log=log, session=session, CHUNK=chunk, **examples)


def timed(run, submitted, repeat):
    """`repeat` runs of a chain or command: the median of the slower card's
    microseconds, and each card's."""
    times = [dc.parts_us(run.wait(submitted(), timeout=10.0)) for _ in range(repeat)]
    return float(np.median([max(p) for p in times])), np.median(times, axis=0).tolist()


def measure(session, fmt, args):
    t, checked = tensors_of(session, fmt, args.seed, args.check_cols)
    program = made(session, t, args.chunk)
    run = session.load(program)
    names = ("x", "w_qkv", "q", "a", "w_o", "h", "hs", "w_gu", "gu", "m", "w_down")
    names += ("y",)
    given = [t[n] for n in names]
    groups = [cores_of(session, n) for n in ("qkv", "part", "gate_up")]
    out = {"format": fmt, "groups": groups, "chunk": args.chunk}
    w_bytes = sum(bd.bounds_of(n, fmt, 1, 1)["w_bytes_a_card"] for n in bd.PRODUCTS)
    add = (program.add, cores_of(session, "add"))
    add_words = [t["h"], t["hs"], ADDERS]
    try:
        for pattern in PATTERNS:
            log(f"{fmt} {pattern}: the chain, the add, then {args.repeat} of each")
            chain = run.chain(program.layer(*given, *groups, pattern, *SIZES))
            dc.parts_us(run.wait(run.submit(chain), timeout=10.0))  # h's partials
            dc.parts_us(run.wait(run.submit(*add, args=add_words), timeout=10.0))
            before = dc.crossings_now(run)
            first = dc.parts_us(run.wait(run.submit(chain), timeout=10.0))  # on hs
            crossed = dc.crossed(run, before)
            checked_words = words(session, t, checked)
            us, parts = timed(run, lambda chain=chain: run.submit(chain), args.repeat)
            chain.free()
            add_us, _ = timed(
                run, lambda: run.submit(*add, args=add_words), args.repeat
            )
            layer = us + 2 * add_us  # h's sum and y's (the next layer's x)
            commands = {}  # each product alone as a command: where the chain goes
            for name, handler, cores, words_of in (
                ("qkv", program.qkv, groups[0], ("x", "w_qkv", "q", 0)),
                ("o", program.part_o, groups[1], ("a", "w_o", "h", 1)),
                ("gate_up", program.gate_up, groups[2], ("hs", "w_gu", "gu", None)),
                ("down", program.part_down, groups[1], ("m", "w_down", "y", 2)),
            ):
                *tensors, size = words_of
                extra = [] if size is None else [SIZES[size]]
                if size in (1, 2):
                    extra.append(SIZES[3])
                command = [t[n] for n in tensors] + extra + [pattern]
                commands[name], _ = timed(
                    run,
                    lambda h=handler, c=cores, w=command: run.submit(h, c, args=w),
                    args.repeat,
                )
            # the layer in order as four commands: a chain, h's add, a chain, y's add
            first_half = run.chain(
                program.first_half(t["x"], t["w_qkv"], t["q"], t["a"], t["w_o"], t["h"],
                                   groups[0], groups[1], pattern, *SIZES[0:2],
                                   SIZES[3], SIZES[4])
            )  # fmt: skip
            second_half = run.chain(
                program.second_half(t["hs"], t["w_gu"], t["gu"], t["m"], t["w_down"],
                                    t["y"], groups[2], groups[1], pattern, SIZES[2],
                                    SIZES[3], SIZES[4])
            )  # fmt: skip
            y_words = [t["y"], t["ys"], ADDERS]

            def four(order):  # wall microseconds of the four, each waited
                started = time.perf_counter()
                for submitted in order:
                    run.wait(submitted(), timeout=10.0)
                return (time.perf_counter() - started) * 1e6

            order = (
                lambda one=first_half: run.submit(one),
                lambda: run.submit(*add, args=add_words),
                lambda two=second_half: run.submit(two),
                lambda words=y_words: run.submit(*add, args=words),
            )

            halves_us = [
                [
                    float(np.max(dc.parts_us(run.wait(run.submit(c), timeout=10.0))))
                    for c in (first_half, second_half)
                ]
                for _ in range(args.repeat)
            ]
            four_wall = [four(order) for _ in range(args.repeat)]
            first_half.free()
            second_half.free()
            zero = np.zeros((H_TILES, 2048), np.uint8)  # the one chain's sums afresh
            for name in ("hs", "ys"):
                for island in (0, 1):
                    session.island(island).write(t[name].parts[island], zero)
            session.write(t["gu"], np.zeros((GU_TILES, 2048), np.uint8))
            one = run.chain(
                program.layer_one(
                    *given, t["ys"], *groups, cores_of(session, "add"), pattern,
                    *SIZES, ADDERS,
                )
            )  # fmt: skip
            records = len(getattr(one, "chains", [one])[0].calls)
            log(f"{fmt} {pattern}: one chain, {records} records")
            one_first = dc.parts_us(run.wait(run.submit(one), timeout=10.0))
            one_words = words(session, t, checked, ys=True)
            one_us, one_parts = timed(run, lambda one=one: run.submit(one), args.repeat)
            one.free()
            out[pattern + "_one"] = {
                "records": records,
                "commands_a_layer": 1,
                "first_us": one_first,
                "words": one_words,
                "layer_us": one_us,
                "parts_us": one_parts,
                "tok_s": 1e6 / (bd.LAYERS * one_us),
                "gb_s": w_bytes / one_us / 1e3,
            }
            halves = np.median(halves_us, axis=0).tolist()
            out[pattern + "_four"] = {
                "commands_a_layer": 4,
                "halves_us": halves,
                "layer_us": sum(halves) + 2 * add_us,  # the card's: two chains, adds
                "wall_us": float(np.median(four_wall)),
            }
            out[pattern] = {
                "commands_a_layer": 3,
                "first_us": first,
                "crossed": crossed,
                "words": checked_words,
                "chain_us": us,
                "commands_us": commands,
                "parts_us": parts,
                "add_us": add_us,
                "layer_us": layer,
                "tok_s": 1e6 / (bd.LAYERS * layer),
                "gb_s": w_bytes / layer / 1e3,
                "chain_tok_s": 1e6 / (bd.LAYERS * us),
                "chain_gb_s": w_bytes / us / 1e3,
            }
    finally:
        run.stop()
    for tensor in t.values():
        tensor.free()
    out["bound_us"] = sum(bd.bounds_of(n, fmt, 1, 2)["memory_us"] for n in bd.PRODUCTS)
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
    parser.add_argument("--weights", type=Path, default=bd.RESULTS, help="W's pages")
    args = parser.parse_args()
    bd.RESULTS = args.weights  # bounds.weights reads (or makes) them there
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
