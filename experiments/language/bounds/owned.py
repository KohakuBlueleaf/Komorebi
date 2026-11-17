"""The prefill products of bounds.py with every core reading its own share of W: a
10 x 12 rectangle, H groups of rows of cores (each its block of M's rows), N in bands
across, S consecutive cores of a row splitting K (their sums met by a tree); X down each
column in bands, one sender a band; W by a row broadcast of one core (each core its own
sender). README.md, "Every core reads its own W".
Example: python owned.py --format bfp8_b --products gate --rows 8"""

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
    part: km.L1[km.fp32, RB, OW, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def owned(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, top0: int):
        down = km.Broadcast(self, along="column", length=self.GROUP_ROWS, window=2)
        own = km.Broadcast(self, along="row", length=1, window=2)
        top = top0 + self.y // self.GROUP_ROWS * self.RB
        band = self.y % self.GROUP_ROWS * 12 + self.x
        left = min(band * self.OW, self.N_TILES - self.OW)
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
    def owned(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, top0: int):
        down = km.Broadcast(self, along="column", length=self.GROUP_ROWS, window=2)
        own = km.Broadcast(self, along="row", length=1, window=2)
        top = top0 + self.y // self.GROUP_ROWS * self.RB
        band = self.y % self.GROUP_ROWS * self.BANDS + self.x // self.S
        left = min(band * self.OW, self.N_TILES - self.OW)
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


def made(session, name, fmt, rows, tensors):
    """The class for `rows` row tiles of product `name` in `fmt` (made with the
    example `tensors`), and its B."""
    k, n = PRODUCTS[name]
    h, s, kc, b = PLANS[(name, fmt, min(rows, 32))]
    kt, nt = k // 32, n // 32
    bands = COLUMNS // s * (ROWS // h)
    ow = -(-nt // bands)
    common = {
        "K_TILES": kt,
        "N_TILES": nt,
        "GROUP_ROWS": ROWS // h,
        "KC": kc,
        "RB": b // h,
        "OW": ow,
        "rect": (0, 0, COLUMNS, ROWS),
        "log": log,
        "session": session,
        **tensors,
    }
    if s == 1:
        return Owned(**common), b
    return OwnedSplit(S=s, BANDS=COLUMNS // s, **common), b


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


def bounds_of(name, fmt, rows):
    """The two bounds of a product on one card, microseconds."""
    k, n = PRODUCTS[name]
    kt, nt = k // 32, n // 32
    return {
        "memory_us": kt * nt * PAGE[fmt] / (DRAM_GBS * 1e3),
        "compute_us": rows * kt * nt / TILE_PRODUCTS * 1e6,
    }


def measure(session, name, fmt, row_list, repeat, folder):
    """Each M of `row_list` (row tiles) of `name` in `fmt`: its commands' summed
    microseconds (median of `repeat`), Y's first and last row tile against float64."""
    k, n = PRODUCTS[name]
    pages, cols, wcols = weights(folder, name, fmt)
    tw = session.empty((k, n), fmt, "tile")
    session.write(tw, pages)
    reports = []
    for rows in row_list:
        x = activations(rows, k)
        tx = session.tensor(x, "bfloat16", "tile")
        ty = session.empty((rows * 32, n), "bfloat16", "tile")
        program, b = made(session, name, fmt, rows, {"x": tx, "w": tw, "y": ty})
        run = session.load(program)
        rounds, singles = [], []
        try:
            for _ in range(repeat):
                each = []
                for top0 in range(0, rows, b):
                    event = run.wait(
                        run.submit(
                            program.owned,
                            (0, COLUMNS * ROWS),
                            args=(tx, tw, ty, top0),
                        ),
                        timeout=60.0,
                    )
                    if event.kind != "done":
                        raise km.KomorebiError(f"{name} {fmt} {rows} {top0}: {event}")
                    each.append(event.microseconds)
                rounds.append(sum(each))
                singles.append(each)
        finally:
            run.stop()
        picked = sorted({0, rows - 1})
        got = np.concatenate([row_tile(session, ty, r, n) for r in picked])[:, cols]
        want = x.reshape(rows, 32, -1)[picked].reshape(-1, k).astype(np.float64)
        want = want @ wcols.astype(np.float64)
        rms = float(np.sqrt(((got - want) ** 2).mean()) / np.sqrt((want**2).mean()))
        us = float(np.median(rounds))
        bounds = bounds_of(name, fmt, rows)
        report = {
            "product": name,
            "format": fmt,
            "rows": rows,
            "plan_h_s_kc_b": PLANS[(name, fmt, min(rows, 32))],
            "commands": len(singles[0]),
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
}


def probe(session, folder, repeat):
    """Each of PROBES: `repeat` single commands at top0 0, their microseconds."""
    k, n = PRODUCTS["gate"]
    pages, _, _ = weights(folder, "gate", "bfp8_b")
    tw = session.empty((k, n), "bfp8_b", "tile")
    session.write(tw, pages)
    tx = session.tensor(activations(32, k), "bfloat16", "tile")
    ty = session.empty((32 * 32, n), "bfloat16", "tile")
    reports = []
    for name, (cls, given) in PROBES.items():
        program = cls(
            N_TILES=n // 32,
            rect=(0, 0, COLUMNS, ROWS),
            log=log,
            session=session,
            x=tx,
            w=tw,
            y=ty,
            **given,
        )
        run = session.load(program)
        times = []
        try:
            for _ in range(repeat):
                event = run.wait(
                    run.submit(
                        program.owned, (0, COLUMNS * ROWS), args=(tx, tw, ty, 0)
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
    parser.add_argument("--format", nargs="+", default=["bfp8_b"])
    parser.add_argument("--products", default=",".join(PRODUCTS))
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--weights", type=Path, default=RESULTS)
    parser.add_argument("--probe", action="store_true", help="the PROBES only")
    args = parser.parse_args()
    reports = []
    with km.Session(log=log) as session:  # the first visible card
        if args.probe:
            reports = probe(session, args.weights, max(args.repeat, 10))
            args.format = []
        log(f"open: grid {session.grid}, products {args.products}, rows {args.rows}")
        for fmt in args.format:
            for name in args.products.split(","):
                reports += measure(
                    session, name, fmt, args.rows, args.repeat, args.weights
                )
    for report in reports:
        print(json.dumps(report, default=float), flush=True)
    card = int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0])
    clock = power.clocked({"cards": [card]}, card=card)
    print(json.dumps(clock), flush=True)  # the AI clock the times were taken at
    return 0


if __name__ == "__main__":
    sys.exit(main())
