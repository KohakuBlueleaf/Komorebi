"""A wide layer's products as a class, against the card's two ceilings: Y = X W
with W in bfp8_b or bfp4_b, one row tile of X by the K split (decode) and many by the
grid plan (prefill), on one card or the output dimension sharded over two. README.md.
Example: python bounds.py --cards 0 --format bfp8_b --products o --rows 1 8"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi.bench import power

STARTED = time.monotonic()
RESULTS = Path(__file__).resolve().parent / "results"  # W's pages, made once
# The assumed shapes (the brief's set): hidden, MLP, heads, KV heads, head dimension.
HIDDEN, MLP, HEADS, KV_HEADS, HEAD_DIM, LAYERS = 5376, 21504, 32, 16, 128, 62
QKV = (HEADS + 2 * KV_HEADS) * HEAD_DIM
# Each product: K, N, the K split's group width and group size on one card and on two
# (an island's groups on its half of N), the grid's tile columns a command.
PRODUCTS = {
    "qkv": {"k": HIDDEN, "n": QKV, "width": 16, "split": (7, 14), "block": 128},
    "o": {
        "k": HEADS * HEAD_DIM,
        "n": HIDDEN,
        "width": 14,
        "split": (10, 19),
        "block": 84,
    },
    "gate": {"k": HIDDEN, "n": MLP, "width": 28, "split": (5, 8), "block": 112},
    "up": {"k": HIDDEN, "n": MLP, "width": 28, "split": (5, 8), "block": 112},
    "down": {"k": MLP, "n": HIDDEN, "width": 14, "split": (10, 19), "block": 84},
}
COLUMNS, ROWS = 12, 8  # the grid's rectangle of cores (an island's on two cards)
DRAM_GBS = 495.6  # the card's DRAM read ceiling (projects/dense/README.md)
TILE_PRODUCTS = 4.70e9  # a second, 120 ranks at HiFi2 (projects/dense/README.md)
PAGE = {"bfloat16": 2048, "bfp8_b": 1088, "bfp4_b": 576}


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class Split(km.Program):
    """Y = X W for one row tile of X (decode) in groups of SPLIT cores, a group WIDTH
    columns of Y from col0, a core 1/SPLIT of K; the group's sums reduced by a tree."""

    K_TILES = km.constant(168)
    SPLIT = km.constant(10)
    WIDTH = km.constant(14)
    CHUNK = km.constant(2)
    W_BLOCK = CHUNK * WIDTH
    sums: km.L1[km.fp32, 1, WIDTH, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def split(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, col0: int):
        share = self.rank % self.SPLIT
        first = col0 + self.rank // self.SPLIT * self.WIDTH
        k0 = share * self.K_TILES // self.SPLIT
        k1 = (share + 1) * self.K_TILES // self.SPLIT
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(w.tiles[k0:k1, first : first + self.WIDTH], block=self.W_BLOCK),
        ):
            self.sums += self.project(xb, wb)
        whole = km.reduce(self.sums, form="tree", group=self.SPLIT)
        if self.rank == self.leader:
            y.tiles[0:1, first : first + self.WIDTH].write(whole)


class Grid(km.Program):
    """Y's columns col0 .. col0 + BLOCK for many row tiles (prefill) on a COLUMNS x ROWS
    rectangle, OH row tiles and OW columns a core; X's chunks cast along each row of
    cores, W's down each column (the dense project's grid plan)."""

    K_TILES = km.constant(168)
    BLOCK = km.constant(84)  # the grid's columns a command
    OW = km.constant(7)  # BLOCK / COLUMNS, rounded up
    KC = km.constant(2)
    OH = km.symbol()  # the plan's
    part: km.L1[km.fp32, OH, OW, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def grid(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, col0: int):
        along = km.Broadcast(self, along="row", window=2)
        down = km.Broadcast(self, along="column", window=2)
        top = self.y * self.OH
        left = col0 + self.x * self.OW
        for chunk in range(self.K_TILES // self.KC):
            k0 = chunk * self.KC
            if along.sender():
                xs = along.cast(x.tiles[top : top + self.OH, k0 : k0 + self.KC].read())
            else:
                xs = along.arrive()
            if down.sender():
                ws = down.cast(
                    w.tiles[
                        k0 : k0 + self.KC, left : min(left + self.OW, col0 + self.BLOCK)
                    ].read()
                )
            else:
                ws = down.arrive()
            self.part += self.project(xs, ws)
        y.tiles[
            top : top + self.OH, left : min(left + self.OW, col0 + self.BLOCK)
        ].write(self.part)


def bounds_of(name, fmt, rows, cards):
    """The two bounds of a product, microseconds: W's bytes at the DRAM ceiling and its
    tile products at the matrix unit's rate, each card its half on two."""
    shape = PRODUCTS[name]
    kt, nt = shape["k"] // 32, shape["n"] // 32
    w_bytes = kt * nt * PAGE[fmt] / cards
    memory = w_bytes / (DRAM_GBS * 1e3)
    compute = rows * kt * nt / cards / TILE_PRODUCTS * 1e6
    return {"memory_us": memory, "compute_us": compute, "w_bytes_a_card": w_bytes}


def islands_of(session):
    """The session's graph, or None on one card."""
    return getattr(session, "graph", None)


def made(session, name, rows, tensors, chunk=2):
    """The class for `rows` row tiles at product `name`'s constants: Split for one
    (`chunk` X tiles a block), Grid (its OH) for more."""
    shape = PRODUCTS[name]
    two = islands_of(session) is not None
    if rows == 1:
        return Split(
            log=log,
            K_TILES=shape["k"] // 32,
            SPLIT=shape["split"][1 if two else 0],
            WIDTH=shape["width"],
            CHUNK=chunk,
            session=session,
            **tensors,
        )
    return Grid(
        log=log,
        plan={"OH": rows // ROWS},
        K_TILES=shape["k"] // 32,
        BLOCK=shape["block"],
        OW=-(-shape["block"] // COLUMNS),
        rect=(0, 0, COLUMNS, ROWS),
        session=session,
        **tensors,
    )


def rms(got, want):
    return float(np.sqrt(((got - want) ** 2).mean()) / np.sqrt((want**2).mean()))


def weights(name, fmt, seed, check_cols):
    """W, float32 normal / sqrt(K), as its pages in `fmt` and `check_cols` of its
    columns (float32) with their indices: made once into results/ (a card job then
    only writes the pages)."""
    path = RESULTS / f"{name}-{fmt}-s{seed}-c{check_cols}.npz"
    if path.exists():
        held = np.load(path)
        return held["pages"], held["cols"], held["wcols"]
    shape = PRODUCTS[name]
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((shape["k"], shape["n"]), dtype=np.float32)
    w /= np.float32(np.sqrt(shape["k"]))
    raw, _ = formats.to_bytes(w, fmt, "tile")
    pages = raw.reshape(-1, PAGE[fmt])
    cols = np.sort(
        rng.choice(shape["n"], size=min(check_cols, shape["n"]), replace=False)
    )
    wcols = np.ascontiguousarray(w[:, cols])
    RESULTS.mkdir(parents=True, exist_ok=True)
    # under a name of this process's own, then moved: a line beside this one (the
    # host checks run them together) reads it whole or makes its own
    temporary = path.with_name(f".{path.stem}-{os.getpid()}{path.suffix}")
    try:
        np.savez(temporary, pages=pages, cols=cols, wcols=wcols)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return pages, cols, wcols


def row_tile(session, tensor, row, n):
    """Row tile `row` of a [M, n] bfloat16 tile tensor as [32, n] float64 values."""
    nt = n // 32
    (raw,) = session.read(tensor, first=row * nt, pages=nt)
    values = formats.from_bytes(
        np.asarray(raw).reshape(-1), (32, n), "bfloat16", "tile"
    )
    return np.asarray(values, np.float64).reshape(32, n)


def activations(rows, k, seed):
    """X, bfloat16 values (normal) as float32."""
    rng = np.random.default_rng(seed + 1)
    bits = rng.standard_normal((rows * 32, k), dtype=np.float32).view(np.uint32)
    return ((bits + 0x8000) & 0xFFFF0000).view(np.float32)


def hints_of(session, name):
    """Two cards: W and Y split by tile columns over a group centred on island 1's
    first rank whose ranks each take the same count of columns, so each island holds
    exactly its half (km::partition gives a remainder to the first ranks); X a copy.
    One card: none."""
    graph = islands_of(session)
    if graph is None:
        return {"x": {}, "w": {}, "y": {}}
    nt, edge = PRODUCTS[name]["n"] // 32, graph.first(1)
    each = next(
        e
        for e in range(1, nt + 1)
        if nt % e == 0 and nt // e % 2 == 0 and nt // e <= 2 * edge
    )
    split = {"split": 1, "group": (edge - nt // each // 2, nt // each)}
    return {"x": {"copy": True}, "w": split, "y": split}


def commands(session, name, rows):
    """The commands of a product at `rows` row tiles: (handler name, group, col0) each
    and the island each runs on."""
    shape = PRODUCTS[name]
    nt = shape["n"] // 32
    graph = islands_of(session)
    islands = len(graph.islands) if graph else 1
    half = nt // islands
    out = []
    for island in range(islands):
        first = graph.first(island) if graph else 0
        if rows == 1:
            width, split = shape["width"], shape["split"][1 if graph else 0]
            cores = half // width * split
            if graph:  # both islands' groups in one command, centred on the edge
                if island == 0:
                    edge = graph.first(1)
                    out.append(("split", (edge - cores, 2 * cores), 0, None))
                continue
            out.append(("split", (0, cores), 0, None))
        else:
            out.extend(
                ("grid", (first, COLUMNS * ROWS), col0, island)
                for col0 in range(island * half, (island + 1) * half, shape["block"])
            )
    return out


def measure(session, name, fmt, row_list, repeat, seed, check_cols, chunk=2):
    """Each shape of `row_list` (row tiles) of product `name` with W in `fmt`: its
    commands' time against the bounds, Y against float64 on `check_cols` columns."""
    shape = PRODUCTS[name]
    hints = hints_of(session, name)
    pages, cols, wcols = weights(name, fmt, seed, check_cols)
    tw = session.empty((shape["k"], shape["n"]), fmt, "tile", **hints["w"])
    session.write(tw, pages)
    if (
        islands_of(session) is not None
    ):  # each island's pages of W: its half, or crossings
        held = [tw.placement.part_pages(i) for i in range(len(tw.placement.parts))]
        log(f"{name} {fmt}: W's pages by island {held}")
    reports = []
    for shapes in ([rows] for rows in row_list):  # a program a shape
        xs = {r: activations(r, shape["k"], seed) for r in shapes}
        txs = {
            r: session.tensor(xs[r], "bfloat16", "tile", **hints["x"]) for r in shapes
        }
        tys = {
            r: session.empty((r * 32, shape["n"]), "bfloat16", "tile", **hints["y"])
            for r in shapes
        }
        biggest = max(shapes)
        try:
            program = made(
                session,
                name,
                biggest,
                {"x": txs[biggest], "w": tw, "y": tys[biggest]},
                chunk,
            )
            run = session.load(program)
        except km.KomorebiError as error:  # a shape that does not fit: its refusal
            refused = {"product": name, "format": fmt, "rows": shapes}
            refused["refused"] = f"{type(error).__name__}: {error}"[:600]
            log(f"{name} {fmt} rows {shapes}: {json.dumps(refused)}")
            reports.append(refused)
            for tensor in (*txs.values(), *tys.values()):
                tensor.free()
            continue
        try:
            done = [
                run_shape(
                    session, run, program, (txs[r], tw, tys[r]), name, fmt, r, repeat
                )
                for r in shapes
            ]
        finally:
            run.stop()
        for report in done:  # Y's first and last row tile on the checked columns
            rows = report["rows"]
            got = np.concatenate(
                [
                    row_tile(session, tys[rows], r, shape["n"])
                    for r in sorted({0, rows - 1})
                ]
            )[:, cols]
            x = (
                xs[rows]
                .reshape(rows, 32, -1)[sorted({0, rows - 1})]
                .reshape(-1, shape["k"])
            )
            want = x.astype(np.float64) @ wcols.astype(np.float64)
            report["relative_rms"] = rms(got, want)
            report["chunk"] = chunk
            log(f"{name} {fmt} rows {rows}: {json.dumps(report, default=float)}")
            reports.append(report)
        for tensor in (*txs.values(), *tys.values()):
            tensor.free()
    tw.free()
    return reports


def run_shape(session, run, program, args, name, fmt, rows, repeat):
    """`repeat` rounds of a shape's commands; a round's time is each island's sum of its
    commands' microseconds, the larger island's (the median of the rounds)."""
    graph = islands_of(session)
    cards = len(graph.islands) if graph else 1
    rounds = []
    for _ in range(repeat):
        per_island = {}
        for handler, group, col0, island in commands(session, name, rows):
            event = run.wait(
                run.submit(getattr(program, handler), group, args=(*args, col0)),
                timeout=60.0,
            )
            each = getattr(event, "parts", (event,))
            if any(part.kind != "done" for part in each):
                raise km.KomorebiError(f"{name} {handler} {group} {col0}: {event}")
            for at, part in enumerate(each):
                key = at if island is None else island
                per_island[key] = per_island.get(key, 0.0) + part.microseconds
        rounds.append(max(per_island.values()))
    bounds = bounds_of(name, fmt, rows, cards)
    us = float(np.median(rounds))
    return {
        "product": name,
        "format": fmt,
        "cards": cards,
        "rows": rows,
        "m_k_n": [rows * 32, PRODUCTS[name]["k"], PRODUCTS[name]["n"]],
        "commands": len(commands(session, name, rows)),
        "us": us,
        "rounds": rounds,
        **bounds,
        "memory_ratio": us / bounds["memory_us"],
        "compute_ratio": us / bounds["compute_us"],
        "kernel_text": program.sizes.text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0", help="0, or 0,1 (the output sharded)")
    parser.add_argument("--format", nargs="+", default=["bfp8_b"])
    parser.add_argument("--products", default=",".join(PRODUCTS))
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--check-cols", type=int, default=256)
    parser.add_argument("--kernel-bytes", type=int)
    parser.add_argument("--chunk", type=int, default=2, help="decode: X tiles a block")
    args = parser.parse_args()
    cards = [int(card) for card in args.cards.split(",")]
    opened = {"cards": cards} if len(cards) > 1 else {"card": cards[0]}
    if args.kernel_bytes:
        opened["kernel_bytes"] = args.kernel_bytes
    reports = []
    with km.Session(log=log, **opened) as session:
        log(f"open: grid {session.grid}, products {args.products}, rows {args.rows}")
        for fmt in args.format:
            for name in args.products.split(","):
                reports += measure(
                    session,
                    name,
                    fmt,
                    args.rows,
                    args.repeat,
                    args.seed,
                    args.check_cols,
                    args.chunk,
                )
    for report in reports:
        print(json.dumps(report, default=float), flush=True)
    clock = power.clocked({"cards": cards}, card=cards[0])
    print(json.dumps(clock), flush=True)  # the AI clock the times were taken at
    return 0


if __name__ == "__main__":
    sys.exit(main())
