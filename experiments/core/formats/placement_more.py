"""The turned product's other forms (README.md): Y = X W of float16 tiles (the A
family, a 16-bit Dst, its sums in DST) at LoFi, HiFi2 and HiFi4, and of two
broadcasts' blocks (bfloat16 X, bfp8_b W, HiFi4, a 2 x 2 rectangle), each with no
marker and with W marked km.srcb, against float64. `build` on a mock device, `card`."""

import argparse
import json
import os
import sys
import time

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi import session as km_session
from komorebi.device import native

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from placement import (
    CHUNK,
    CORES,
    DESCRIPTOR,
    JIT_CACHE,
    K_TILES,
    N_TILES,
    REPEAT,
    RESULTS,
    WIDTH,
    build_only,
    errors,
    log,
)

FIDELITIES = ("LoFi", "HiFi2", "HiFi4")
COLUMNS = ROWS = 2  # the broadcasts' rectangle
OH = OW = 2  # a core's block of Y's tiles
KC = 2  # K tiles a cast
STARTED = time.monotonic()


def half_class(fidelity):
    """Float16 tiles at `fidelity`: each core WIDTH columns of Y, the sums in DST."""

    class Half(km.Program):
        @km.tile_function(fidelity=fidelity)
        def plain(x, w):
            return x @ w

        @km.tile_function(fidelity=fidelity)
        def turned(x, w):
            return x @ km.srcb(w)

        @km.handler
        def unmarked(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
            first = self.rank * WIDTH
            acc = km.tile.zeros(km.fp32, WIDTH)
            for xb, wb in zip(
                km.Stream(x.tiles[0:1, 0:K_TILES], block=CHUNK),
                km.Stream(w.tiles[0:K_TILES, first : first + WIDTH], block=8),
            ):
                acc += self.plain(xb, wb)
            y.tiles[0:1, first : first + WIDTH].write(acc)

        @km.handler
        def marked(self, x: km.Tensor, t: km.Tensor, y: km.Tensor):
            first = self.rank * WIDTH
            acc = km.tile.zeros(km.fp32, WIDTH)
            for xb, wb in zip(
                km.Stream(x.tiles[0:1, 0:K_TILES], block=CHUNK),
                km.Stream(t.tiles[0:K_TILES, first : first + WIDTH], block=8),
            ):
                acc += self.turned(xb, wb)
            y.tiles[0:1, first : first + WIDTH].write(acc)

    Half.__name__ = Half.__qualname__ = f"Half{fidelity}"
    return Half


def grid_class(turned):
    """Y = X W of two broadcasts' blocks on a COLUMNS x ROWS rectangle: X's OH x KC
    tiles along a row, W's KC x OW down a column, the sums in an fp32 field."""

    class Grid(km.Program):
        part: km.L1[km.fp32, OH, OW, 32, 32]

        if turned:

            @km.tile_function(fidelity="HiFi4")
            def product(x, w):
                return x @ km.srcb(w)

        else:

            @km.tile_function(fidelity="HiFi4")
            def product(x, w):
                return x @ w

        @km.handler
        def grid(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
            along = km.Broadcast(self, along="row", window=2)
            down = km.Broadcast(self, along="column", window=2)
            top = self.y * OH
            left = self.x * OW
            for chunk in range(K_TILES // KC):
                k0 = chunk * KC
                if along.sender():
                    xs = along.cast(x.tiles[top : top + OH, k0 : k0 + KC].read())
                else:
                    xs = along.arrive()
                if down.sender():
                    ws = down.cast(w.tiles[k0 : k0 + KC, left : left + OW].read())
                else:
                    ws = down.arrive()
                self.part += self.product(xs, ws)
            y.tiles[top : top + OH, left : left + OW].write(self.part)

    Grid.__name__ = Grid.__qualname__ = "GridTurned" if turned else "GridPlain"
    return Grid


def inputs(seed=2):
    """X float16 values (32 x K) and the grid's X (ROWS * OH tiles of rows), W normal
    / 32 (float32)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((32, K_TILES * 32)).astype(np.float16).astype(np.float32)
    gx = rng.standard_normal((ROWS * OH * 32, K_TILES * 32)).astype(np.float32)
    gx = formats.bf16_values(formats.bf16_bits(gx))
    w = (rng.standard_normal((K_TILES * 32, N_TILES * 32)) / 32).astype(np.float32)
    return x, gx, w[:, : COLUMNS * OW * 32].copy(), w


def made(session, data, empty):
    """The tensors by program: the half programs' (x, w, t, y) and the grid's."""
    x, gx, gw, w = data

    def tensor(values, dtype, turned=False):
        if empty:
            made = session.empty(values.shape, dtype, "tile")
            made.transposed_tiles = turned
            return made
        return session.tensor(values, dtype, "tile", transposed_tiles=turned)

    half = {
        "x": tensor(x, "float16"),
        "w": tensor(w, "float16"),
        "t": tensor(w, "float16", True),
        "y": session.empty((32, N_TILES * 32), "float32", "tile"),
    }
    grid = {
        "x": tensor(gx, "bfloat16"),
        "w": tensor(gw, "bfp8_b"),
        "y": session.empty((gx.shape[0], gw.shape[1]), "float32", "tile"),
    }
    return half, grid, tensor(gw, "bfp8_b", True)


def build(args):
    """Mock device: the three half programs and the two grid programs built."""
    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / DESCRIPTOR)
    km_session.chip_holder = lambda card, path=None: None  # a mock device has no chip
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    km_session.Session.load = build_only()
    with km.Session(log=log) as session:
        half, grid, turned_w = made(session, inputs(), empty=True)
        for fidelity in FIDELITIES:
            session.load(half_class(fidelity)(log=log, session=session, **half))
        rect = (0, 0, COLUMNS, ROWS)
        for turned in (False, True):
            tensors = {**grid, "w": turned_w} if turned else grid
            program = grid_class(turned)(log=log, session=session, rect=rect, **tensors)
            session.load(program)
    return 0


def timed(run, handler, group, args):
    """REPEAT commands; their microseconds."""
    times = []
    for _ in range(REPEAT):
        event = run.wait(run.submit(handler, group, args=args), timeout=30.0)
        if event.kind != "done":
            raise RuntimeError(f"{handler.name}: {event}")
        times.append(event.microseconds)
    return times


def card(args):
    """One card: each half program, then each grid program, loaded in turn."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    data = inputs()
    x, gx, gw, w = data
    rows = []
    with km.Session(log=log) as session:
        half, grid, turned_w = made(session, data, empty=False)
        want = x.astype(np.float64) @ w.astype(np.float64)
        for fidelity in FIDELITIES:
            cls = half_class(fidelity)
            run = session.load(cls(log=log, session=session, **half))
            outs = []
            try:
                for form, weight in (("unmarked", half["w"]), ("marked", half["t"])):
                    y = session.empty((32, N_TILES * 32), "float32", "tile")
                    handler = getattr(cls, form)
                    times = timed(run, handler, (0, CORES), (half["x"], weight, y))
                    outs.append((form, y, times))
            finally:
                run.stop()
            for form, y, times in outs:
                row = {"form": "float16 " + form, "fidelity": fidelity}
                row |= errors(y.read().astype(np.float64), want)
                row["microseconds"] = times
                rows.append(row)
                log(json.dumps(row))
        want = gx.astype(np.float64) @ gw.astype(np.float64)
        rect = (0, 0, COLUMNS, ROWS)
        for turned in (False, True):
            cls = grid_class(turned)
            tensors = {**grid, "w": turned_w} if turned else grid
            program = cls(log=log, session=session, rect=rect, **tensors)
            run = session.load(program)
            try:
                y = session.empty(grid["y"].shape, "float32", "tile")
                group = program.rectangle["group"]
                times = timed(run, cls.grid, group, (tensors["x"], tensors["w"], y))
            finally:
                run.stop()
            row = {"form": f"broadcasts {'marked' if turned else 'unmarked'}"}
            row |= {"fidelity": "HiFi4", **errors(y.read().astype(np.float64), want)}
            row["microseconds"] = times
            rows.append(row)
            log(json.dumps(row))
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "placement_more.json", "w") as f:
        json.dump({"rows": rows, "wall_s": round(time.monotonic() - STARTED, 1)}, f)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["build", "card"])
    args = p.parse_args()
    return build(args) if args.what == "build" else card(args)


if __name__ == "__main__":
    sys.exit(main())
