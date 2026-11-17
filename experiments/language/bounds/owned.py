"""bounds.py's prefill products with every core reading its own W: a 10 x 12 rectangle,
H groups of rows of cores, N in bands, S cores of a row splitting K (a tree meets the
sums), X down each column (README.md, "Every core reads its own W").
Example: python owned.py --format bfp8_b --products gate --rows 8"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi.bench import power

STARTED = time.monotonic()
RESULTS = Path(__file__).resolve().parent / "results"  # bounds.py's encoded W
HIDDEN, MLP, HEADS, KV_HEADS, HEAD_DIM = 5376, 21504, 32, 16, 128
PRODUCTS = {  # K, N (bounds.py's shapes)
    "qkv": (HIDDEN, (HEADS + 2 * KV_HEADS) * HEAD_DIM),
    "o": (HEADS * HEAD_DIM, HIDDEN),
    "gate": (HIDDEN, MLP),
    "up": (HIDDEN, MLP),
    "down": (MLP, HIDDEN),
}
COLUMNS, ROWS = 12, 10  # the rectangle: the whole grid of one card
# The plan a product, format and M (row tiles): H, S, KC, B (row tiles a command), from
# the arithmetic of P1 (one card, a lone sender at 10.3 GB/s).
PLANS = {
    ("qkv", "bfp8_b", 8): (1, 2, 2, 8),
    ("o", "bfp8_b", 8): (1, 4, 2, 8),
    ("gate", "bfp8_b", 8): (1, 1, 2, 8),
    ("up", "bfp8_b", 8): (1, 1, 2, 8),
    ("down", "bfp8_b", 8): (1, 4, 2, 8),
    ("qkv", "bfp8_b", 32): (2, 2, 2, 16),
    ("o", "bfp8_b", 32): (2, 2, 2, 32),
    ("gate", "bfp8_b", 32): (2, 1, 4, 32),
    ("up", "bfp8_b", 32): (2, 1, 4, 32),
    ("down", "bfp8_b", 32): (2, 2, 2, 32),
    ("qkv", "bfp4_b", 8): (1, 3, 4, 8),
    ("o", "bfp4_b", 8): (1, 4, 2, 8),
    ("gate", "bfp4_b", 8): (1, 2, 4, 8),
    ("up", "bfp4_b", 8): (1, 2, 4, 8),
    ("down", "bfp4_b", 8): (1, 4, 4, 8),
    ("qkv", "bfp4_b", 32): (2, 2, 4, 16),
    ("o", "bfp4_b", 32): (2, 2, 2, 32),
    ("gate", "bfp4_b", 32): (2, 1, 8, 16),
    ("up", "bfp4_b", 32): (2, 1, 8, 16),
    ("down", "bfp4_b", 32): (2, 3, 8, 16),
}
# The plans the measured rates pick (README, "X's sender"), where they differ from
# PLANS: X's sender reads on mover B at about 6.2 GB/s, so its bytes (B/H)(K/S) are cut.
PLANS_X = {
    ("qkv", "bfp8_b", 8): (1, 4, 2, 8),
    ("gate", "bfp8_b", 8): (1, 2, 2, 8),
    ("up", "bfp8_b", 8): (1, 2, 2, 8),
    ("down", "bfp8_b", 8): (1, 6, 2, 8),
    ("qkv", "bfp8_b", 32): (2, 2, 1, 32),
    ("o", "bfp8_b", 32): (2, 4, 1, 16),
    ("down", "bfp8_b", 32): (2, 3, 2, 16),
    ("qkv", "bfp4_b", 8): (1, 4, 2, 8),
    ("down", "bfp4_b", 8): (1, 6, 2, 8),
    ("qkv", "bfp4_b", 32): (2, 2, 1, 32),
    ("o", "bfp4_b", 32): (2, 4, 1, 16),
    ("gate", "bfp4_b", 32): (2, 1, 2, 32),
    ("up", "bfp4_b", 32): (2, 1, 2, 32),
}
# Two cards (an island a card, 12 x 8 cores of each, N's half a card), by M exactly.
PLANS_2 = {
    ("qkv", "bfp8_b", 8): (1, 4, 1, 8),
    ("o", "bfp8_b", 8): (2, 4, 1, 8),
    ("gate", "bfp8_b", 8): (1, 2, 1, 8),
    ("up", "bfp8_b", 8): (1, 2, 1, 8),
    ("down", "bfp8_b", 8): (1, 6, 1, 8),
    ("qkv", "bfp8_b", 32): (2, 3, 1, 16),
    ("o", "bfp8_b", 32): (4, 2, 1, 32),
    ("gate", "bfp8_b", 32): (4, 1, 1, 32),
    ("up", "bfp8_b", 32): (4, 1, 1, 32),
    ("down", "bfp8_b", 32): (4, 2, 1, 32),
    ("qkv", "bfp8_b", 128): (8, 1, 1, 64),
    ("o", "bfp8_b", 128): (4, 2, 1, 32),
    ("gate", "bfp8_b", 128): (4, 1, 2, 64),
    ("up", "bfp8_b", 128): (4, 1, 2, 64),
    ("down", "bfp8_b", 128): (4, 2, 1, 32),
    ("qkv", "bfp4_b", 8): (1, 4, 1, 8),
    ("o", "bfp4_b", 8): (2, 4, 1, 8),
    ("gate", "bfp4_b", 8): (1, 2, 1, 8),
    ("up", "bfp4_b", 8): (1, 2, 1, 8),
    ("down", "bfp4_b", 8): (1, 6, 1, 8),
    ("qkv", "bfp4_b", 32): (4, 2, 1, 32),
    ("o", "bfp4_b", 32): (4, 2, 1, 32),
    ("gate", "bfp4_b", 32): (4, 1, 1, 32),
    ("up", "bfp4_b", 32): (4, 1, 1, 32),
    ("down", "bfp4_b", 32): (4, 2, 1, 32),
    ("qkv", "bfp4_b", 128): (8, 1, 1, 64),
    ("o", "bfp4_b", 128): (4, 2, 1, 32),
    ("gate", "bfp4_b", 128): (4, 1, 2, 64),
    ("up", "bfp4_b", 128): (4, 1, 2, 64),
    ("down", "bfp4_b", 128): (4, 2, 1, 32),
}
ROWS_2 = 8  # rows of cores an island's rectangle on two cards (119 or 118 ranks)
PAGE = {"bfp8_b": 1088, "bfp4_b": 576}
DRAM_GBS, TILE_PRODUCTS = 495.6, 4.70e9  # bounds.py's ceilings


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class Owned(km.Program):
    """Y[top0 : top0 + B rows] = X W, no split of K: core (x, y) of the rectangle
    makes its RB rows of its group's block and its band's OW columns of Y."""

    K_TILES = km.constant(168)
    N_TILES = km.constant(672)
    GROUP_ROWS = km.constant(10)  # rows of cores a group (ROWS / H)
    KC = km.constant(2)
    RB = km.constant(8)
    OW = km.constant(6)
    WINDOW = km.constant(2)  # blocks in flight of each broadcast
    part: km.L1[km.fp32, RB, OW, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def owned(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, top0: int, n0: int):
        down = km.Broadcast(
            self, along="column", length=self.GROUP_ROWS, window=self.WINDOW
        )
        own = km.Broadcast(self, along="row", length=1, window=self.WINDOW)
        top = top0 + self.y // self.GROUP_ROWS * self.RB
        band = self.y % self.GROUP_ROWS * 12 + self.x
        left = n0 + min(band * self.OW, self.N_TILES - self.OW)
        for chunk in range(self.K_TILES // self.KC):
            k0 = chunk * self.KC
            if down.sender():
                xs = down.cast(x.tiles[top : top + self.RB, k0 : k0 + self.KC].read())
            else:
                xs = down.arrive()
            if own.sender():
                ws = own.cast(w.tiles[k0 : k0 + self.KC, left : left + self.OW].read())
            else:
                ws = own.arrive()
            self.part += self.project(xs, ws)
        y.tiles[top : top + self.RB, left : left + self.OW].write(self.part)


class OwnedSwap(Owned):
    """Owned with the movers swapped: X down the columns on mover A, W on mover B."""

    @km.handler
    def owned(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, top0: int, n0: int):
        down = km.Broadcast(
            self, along="column", length=self.GROUP_ROWS, window=self.WINDOW, mover="A"
        )
        own = km.Broadcast(self, along="row", length=1, window=self.WINDOW, mover="B")
        top = top0 + self.y // self.GROUP_ROWS * self.RB
        band = self.y % self.GROUP_ROWS * 12 + self.x
        left = n0 + min(band * self.OW, self.N_TILES - self.OW)
        for chunk in range(self.K_TILES // self.KC):
            k0 = chunk * self.KC
            if down.sender():
                xs = down.cast(x.tiles[top : top + self.RB, k0 : k0 + self.KC].read())
            else:
                xs = down.arrive()
            if own.sender():
                ws = own.cast(w.tiles[k0 : k0 + self.KC, left : left + self.OW].read())
            else:
                ws = own.arrive()
            self.part += self.project(xs, ws)
        y.tiles[top : top + self.RB, left : left + self.OW].write(self.part)


class Casts(km.Program):
    """X's column broadcast of Owned alone: the senders read and cast X's blocks, every
    core's compute drops them. casts_b on mover B (Owned's), casts_a on mover A."""

    K_TILES = km.constant(168)
    GROUP_ROWS = km.constant(5)
    KC = km.constant(4)
    RB = km.constant(16)
    WINDOW = km.constant(2)

    @km.handler
    def casts_b(self, x: km.Tensor, top0: int):
        down = km.Broadcast(
            self, along="column", length=self.GROUP_ROWS, window=self.WINDOW, mover="B"
        )
        top = top0 + self.y // self.GROUP_ROWS * self.RB
        for chunk in range(self.K_TILES // self.KC):
            k0 = chunk * self.KC
            if down.sender():
                xs = down.cast(x.tiles[top : top + self.RB, k0 : k0 + self.KC].read())
            else:
                xs = down.arrive()
            for _tile in xs:
                pass

    @km.handler
    def casts_a(self, x: km.Tensor, top0: int):
        down = km.Broadcast(
            self, along="column", length=self.GROUP_ROWS, window=self.WINDOW, mover="A"
        )
        top = top0 + self.y // self.GROUP_ROWS * self.RB
        for chunk in range(self.K_TILES // self.KC):
            k0 = chunk * self.KC
            if down.sender():
                xs = down.cast(x.tiles[top : top + self.RB, k0 : k0 + self.KC].read())
            else:
                xs = down.arrive()
            for _tile in xs:
                pass


class OwnedSplit(km.Program):
    """Owned with K split over S consecutive cores of a row: each its K share, the
    group's sums met by a tree at its first core, which writes Y."""

    K_TILES = km.constant(168)
    N_TILES = km.constant(672)
    GROUP_ROWS = km.constant(10)
    S = km.constant(2)
    BANDS = km.constant(6)  # bands a row of cores (12 / S)
    KC = km.constant(2)
    RB = km.constant(8)
    OW = km.constant(12)
    part: km.L1[km.fp32, RB, OW, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def owned(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, top0: int, n0: int):
        down = km.Broadcast(self, along="column", length=self.GROUP_ROWS, window=2)
        own = km.Broadcast(self, along="row", length=1, window=2)
        top = top0 + self.y // self.GROUP_ROWS * self.RB
        band = self.y % self.GROUP_ROWS * self.BANDS + self.x // self.S
        left = n0 + min(band * self.OW, self.N_TILES - self.OW)
        ks = self.K_TILES // self.S
        first = self.x % self.S * ks
        for chunk in range(ks // self.KC):
            k0 = first + chunk * self.KC
            if down.sender():
                xs = down.cast(x.tiles[top : top + self.RB, k0 : k0 + self.KC].read())
            else:
                xs = down.arrive()
            if own.sender():
                ws = own.cast(w.tiles[k0 : k0 + self.KC, left : left + self.OW].read())
            else:
                ws = own.arrive()
            self.part += self.project(xs, ws)
        whole = km.reduce(self.part, form="tree", group=self.S)
        if self.rank == self.leader:
            y.tiles[top : top + self.RB, left : left + self.OW].write(whole)


def plan_of(plans, name, fmt, rows):
    """The plan (H, S, KC, B) of a shape: by its M, else by min(M, 32); None."""
    return plans.get((name, fmt, rows), plans.get((name, fmt, min(rows, 32))))


def islands_of(session):
    """The session's graph, or None on one card."""
    return getattr(session, "graph", None)


def hints_of(session, name):
    """bounds.py's placement. Two cards: W and Y split by tile columns over a group
    centred on island 1's first rank, each island exactly its half; X a copy."""
    graph = islands_of(session)
    if graph is None:
        return {"x": {}, "w": {}, "y": {}}
    nt, edge = PRODUCTS[name][1] // 32, graph.first(1)
    each = next(
        e
        for e in range(1, nt + 1)
        if nt % e == 0 and nt // e % 2 == 0 and nt // e <= 2 * edge
    )
    split = {"split": 1, "group": (edge - nt // each // 2, nt // each)}
    return {"x": {"copy": True}, "w": split, "y": split}


def made(session, name, fmt, rows, tensors, plans=PLANS):
    """The class for `rows` row tiles of product `name` in `fmt` (made with the
    example `tensors`), its B and its rows of cores; on two cards a card's half of N."""
    k, n = PRODUCTS[name]
    h, s, kc, b = plan_of(plans, name, fmt, rows)
    graph = islands_of(session)
    cores = ROWS if graph is None else ROWS_2
    kt, nt = k // 32, n // 32 // (1 if graph is None else len(graph.islands))
    bands = COLUMNS // s * (cores // h)
    ow = -(-nt // bands)
    common = {
        "K_TILES": kt,
        "N_TILES": nt,
        "GROUP_ROWS": cores // h,
        "KC": kc,
        "RB": b // h,
        "OW": ow,
        "rect": (0, 0, COLUMNS, cores),
        "log": log,
        "session": session,
        **tensors,
    }
    if s == 1:
        return Owned(**common), b, cores
    return OwnedSplit(S=s, BANDS=COLUMNS // s, **common), b, cores


def weights(folder, name, fmt):
    """bounds.py's encoded W (pages, checked columns, their float32 values), read."""
    held = np.load(Path(folder) / f"{name}-{fmt}-s1-c256.npz")
    return held["pages"], held["cols"], held["wcols"]


def activations(rows, k, seed=1):
    """X as bounds.py makes it: bfloat16 values (normal) as float32."""
    rng = np.random.default_rng(seed + 1)
    bits = rng.standard_normal((rows * 32, k), dtype=np.float32).view(np.uint32)
    return ((bits + 0x8000) & 0xFFFF0000).view(np.float32)


def row_tile(session, tensor, row, n):
    """Row tile `row` of a [M, n] bfloat16 tile tensor as [32, n] float64 values."""
    nt = n // 32
    (raw,) = session.read(tensor, first=row * nt, pages=nt)
    values = formats.from_bytes(
        np.asarray(raw).reshape(-1), (32, n), "bfloat16", "tile"
    )
    return np.asarray(values, np.float64).reshape(32, n)


def bounds_of(name, fmt, rows, cards=1):
    """The two bounds of a product on `cards` cards (each its half), microseconds."""
    k, n = PRODUCTS[name]
    kt, nt = k // 32, n // 32
    return {
        "memory_us": kt * nt * PAGE[fmt] / (DRAM_GBS * 1e3) / cards,
        "compute_us": rows * kt * nt / TILE_PRODUCTS * 1e6 / cards,
    }


def measure(session, name, fmt, row_list, repeat, folder, plans=PLANS):
    """Each M of `row_list` (row tiles) of `name` in `fmt` that `plans` holds: its
    commands' summed microseconds (median of `repeat`), Y's first and last row tile
    against float64. On two cards a command a card on its half of N, a round's time
    the larger card's sum."""
    row_list = [rows for rows in row_list if plan_of(plans, name, fmt, rows)]
    if not row_list:
        return []
    k, n = PRODUCTS[name]
    graph = islands_of(session)
    islands = 1 if graph is None else len(graph.islands)
    hints = hints_of(session, name)
    pages, cols, wcols = weights(folder, name, fmt)
    tw = session.empty((k, n), fmt, "tile", **hints["w"])
    session.write(tw, pages)
    if graph is not None:  # each island's pages of W: its half, or crossings
        held = [tw.placement.part_pages(i) for i in range(len(tw.placement.parts))]
        log(f"{name} {fmt}: W's pages by island {held}")
    reports = []
    for rows in row_list:
        x = activations(rows, k)
        tx = session.tensor(x, "bfloat16", "tile", **hints["x"])
        ty = session.empty((rows * 32, n), "bfloat16", "tile", **hints["y"])
        tensors = {"x": tx, "w": tw, "y": ty}
        program, b, cores = made(session, name, fmt, rows, tensors, plans)
        half = n // 32 // islands
        run = session.load(program)
        rounds, singles = [], []
        try:
            for _ in range(repeat):
                each = [[] for _ in range(islands)]
                for top0 in range(0, rows, b):
                    for island in range(islands):
                        first = 0 if graph is None else graph.first(island)
                        event = run.wait(
                            run.submit(
                                program.owned,
                                (first, COLUMNS * cores),
                                args=(tx, tw, ty, top0, island * half),
                            ),
                            timeout=60.0,
                        )
                        if event.kind != "done":
                            raise km.KomorebiError(
                                f"{name} {fmt} {rows} {top0} {island}: {event}"
                            )
                        each[island].append(event.microseconds)
                rounds.append(max(sum(times) for times in each))
                singles.append(each if islands > 1 else each[0])
        finally:
            run.stop()
        picked = sorted({0, rows - 1})
        got = np.concatenate([row_tile(session, ty, r, n) for r in picked])[:, cols]
        want = x.reshape(rows, 32, -1)[picked].reshape(-1, k).astype(np.float64)
        want = want @ wcols.astype(np.float64)
        rms = float(np.sqrt(((got - want) ** 2).mean()) / np.sqrt((want**2).mean()))
        us = float(np.median(rounds))
        bounds = bounds_of(name, fmt, rows, islands)
        report = {
            "product": name,
            "format": fmt,
            "rows": rows,
            "cards": islands,
            "plan_h_s_kc_b": plan_of(plans, name, fmt, rows),
            "commands": rows // b * islands,
            "us": us,
            "rounds": rounds,
            "commands_us": singles,
            **bounds,
            "ratio": us / max(bounds.values()),
            "relative_rms": rms,
            "kernel_text": program.sizes.text,
        }
        log(f"{name} {fmt} rows {rows}: {json.dumps(report, default=float)}")
        reports.append(report)
        tx.free()
        ty.free()
    tw.free()
    return reports


PROBES = {  # the two rates P1 estimated, on gate's tensors in bfp8_b at M 1,024
    # a command's fixed part: one K block of one tile into one tile of Y a core
    "fixed": (Owned, {"K_TILES": 2, "KC": 2, "RB": 1, "OW": 1, "GROUP_ROWS": 10}),
    "fixed_split": (
        OwnedSplit,
        {"K_TILES": 4, "KC": 2, "RB": 1, "OW": 1, "GROUP_ROWS": 10, "S": 2, "BANDS": 6},
    ),
    # the field's pack a K block: M 1,024's plan at KC 4 and at KC 2 (twice the packs)
    "kc4": (Owned, {"K_TILES": 168, "KC": 4, "RB": 16, "OW": 12, "GROUP_ROWS": 5}),
    "kc2": (Owned, {"K_TILES": 168, "KC": 2, "RB": 16, "OW": 12, "GROUP_ROWS": 5}),
    # X's column sender (M 1,024's gate plan: 24 senders, 5.51 MB each): casts alone
    # (no product) at KC 4 and 2, window 2 and 8, on mover B and A; reads alone (bands
    # of 1 core on 12 x 2 cores: 24 readers, no receivers); Owned at window 4, swapped
    "x_cast_b_w2": (Casts, {}, "casts_b"),
    "x_cast_b_kc2_w2": (Casts, {"KC": 2}, "casts_b"),
    "x_cast_b_kc2_w8": (Casts, {"KC": 2, "WINDOW": 8}, "casts_b"),
    "x_cast_a_w2": (Casts, {}, "casts_a"),
    "x_read_w2": (Casts, {"GROUP_ROWS": 1}, "casts_b", (0, 0, COLUMNS, 2)),
    "x_owned_kc2_w4": (
        Owned,
        {"K_TILES": 168, "KC": 2, "RB": 16, "OW": 12, "GROUP_ROWS": 5, "WINDOW": 4},
    ),
    "x_owned_kc4_swap": (
        OwnedSwap,
        {"K_TILES": 168, "KC": 4, "RB": 16, "OW": 12, "GROUP_ROWS": 5},
    ),
}


def probe(session, folder, repeat, names):
    """Each of PROBES named (every one when none): `repeat` single commands at top0 0,
    their microseconds. An entry: class, constants, handler, rectangle."""
    k, n = PRODUCTS["gate"]
    pages, _, _ = weights(folder, "gate", "bfp8_b")
    tw = session.empty((k, n), "bfp8_b", "tile")
    session.write(tw, pages)
    tx = session.tensor(activations(32, k), "bfloat16", "tile")
    ty = session.empty((32 * 32, n), "bfloat16", "tile")
    reports = []
    for name in names or PROBES:
        cls, given, handler, rect = (
            PROBES[name]
            + (
                "owned",
                (0, 0, COLUMNS, ROWS),
            )[len(PROBES[name]) - 2 :]
        )
        tensors = {"x": tx} if cls is Casts else {"x": tx, "w": tw, "y": ty}
        args = (tx, 0) if cls is Casts else (tx, tw, ty, 0, 0)
        if cls is not Casts:
            given = {"N_TILES": n // 32, **given}
        program = cls(rect=rect, log=log, session=session, **tensors, **given)
        run = session.load(program)
        times = []
        try:
            for _ in range(repeat):
                event = run.wait(
                    run.submit(
                        getattr(program, handler), (0, rect[2] * rect[3]), args=args
                    ),
                    timeout=60.0,
                )
                if event.kind != "done":
                    raise km.KomorebiError(f"probe {name}: {event}")
                times.append(event.microseconds)
        finally:
            run.stop()
        report = {"probe": name, "us": float(np.median(times)), "times": times}
        log(f"probe {json.dumps(report)}")
        reports.append(report)
    for tensor in (tw, tx, ty):
        tensor.free()
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0", help="0, or 0,1 (N split by card)")
    parser.add_argument("--format", nargs="+", default=["bfp8_b"])
    parser.add_argument("--products", default=",".join(PRODUCTS))
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--weights", type=Path, default=RESULTS)
    parser.add_argument(
        "--plans",
        choices=["p1", "x", "two"],
        default="p1",
        help="PLANS, PLANS_X alone, or PLANS_2 (two cards)",
    )
    parser.add_argument(
        "--probe", nargs="*", help="the PROBES named (all when none) and nothing else"
    )
    args = parser.parse_args()
    reports = []
    cards = [int(card) for card in args.cards.split(",")]
    opened = {"cards": cards} if len(cards) > 1 else {"card": cards[0]}
    plans = {"p1": PLANS, "x": PLANS_X, "two": PLANS_2}[args.plans]
    with km.Session(log=log, **opened) as session:
        if args.probe is not None:
            reports = probe(session, args.weights, max(args.repeat, 10), args.probe)
            args.format = []
        log(f"open: grid {session.grid}, products {args.products}, rows {args.rows}")
        for fmt in args.format:
            for name in args.products.split(","):
                reports += measure(
                    session,
                    name,
                    fmt,
                    args.rows,
                    args.repeat,
                    args.weights,
                    plans,
                )
    for report in reports:
        print(json.dumps(report, default=float), flush=True)
    clock = power.clocked({"cards": cards}, card=cards[0])
    print(json.dumps(clock), flush=True)  # the AI clock the times were taken at
    return 0


if __name__ == "__main__":
    sys.exit(main())
