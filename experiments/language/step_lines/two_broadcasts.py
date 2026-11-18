"""Two handlers' broadcasts on one mover in one class (README.md): broadcast_product's
`down` and `window`, both a column's broadcast on mover B, each reading its own
arguments; their commands alternate, each Y against float64 and the two against
each other. `build` on a mock device, `card`."""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.device import native
from komorebi.session import session as km_session

HERE = Path(__file__).resolve().parent
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def product_module():
    """broadcast_product.py beside this file: its tile function, sizes and data."""
    spec = importlib.util.spec_from_file_location(
        "broadcast_product", HERE / "broadcast_product.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bp = product_module()
COLUMNS, ROWS, K, N, N2 = bp.COLUMNS, bp.ROWS, bp.K, bp.N, bp.N2
project = bp.project


class Two(km.Program):
    """broadcast_product's Down.down and Window.window in one class."""

    @km.handler
    def down(self, x: km.Tensor, w: km.Tensor, wt: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        left = self.x * 2 * N + N
        if column.sender():
            ws = column.cast(w.tiles[0:K, left : left + N].read())
        else:
            ws = column.arrive()
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2, pattern="A"):
            acc += project(xb, ws)
        y.pages(self.y * COLUMNS * N + self.x * N, N).write(acc)

    @km.handler
    def window(self, x: km.Tensor, w: km.Tensor, wt: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        left = self.x * 2 * N
        if column.sender():
            ws = column.cast(w.tiles[0:K, left : left + N2].read())
        else:
            ws = column.arrive()
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2, pattern="A"):
            acc += project(xb, ws[:, N : N + N])
        y.pages(self.y * COLUMNS * N + self.x * N, N).write(acc)


def made(session, tensors):
    return Two(log=log, session=session, rect=(0, 0, COLUMNS, ROWS), **tensors)


def build(args):
    """Mock device: the class built."""
    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / bp.DESCRIPTOR)
    os.environ["TT_METAL_CACHE"] = str(bp.JIT_CACHE)
    km_session.Session.load = bp.build_only()
    with km.Session(log=log) as session:
        session.load(made(session, bp.tensors_of(session, bp.inputs(), empty=True)))
    return 0


def card(args):
    """One card: the class loaded once; down and window in turn, REPEAT commands of
    each, each into its own Y, read after the run's stop."""
    os.environ["TT_METAL_CACHE"] = str(bp.JIT_CACHE)
    x, w = bp.inputs()
    want = bp.wanted(x, w)
    with km.Session(log=log) as session:
        tensors = bp.tensors_of(session, (x, w), empty=False)
        run = session.load(made(session, tensors))
        outputs = {
            name: (session.empty(tensors["y"].shape, "float32", "tile"), [])
            for name in ("down", "window")
        }
        try:
            for _ in range(bp.REPEAT):
                for name, (y, times) in outputs.items():
                    words = (tensors["x"], tensors["w"], tensors["wt"], y)
                    sent = run.submit(getattr(Two, name), bp.GROUP, args=words)
                    event = run.wait(sent, timeout=30.0)
                    if event.kind != "done":
                        raise RuntimeError(f"{name}: {event}")
                    times.append(event.microseconds)
        finally:
            run.stop()
        got = {name: y.read().astype(np.float64) for name, (y, _) in outputs.items()}
    rows = []
    for name, (_, times) in outputs.items():
        row = {"handler": name, **bp.errors(got[name], want), "microseconds": times}
        row["equal_to_down"] = bool(np.array_equal(got[name], got["down"]))
        rows.append(row)
        log(json.dumps(row))
    bp.RESULTS.mkdir(exist_ok=True)
    with open(bp.RESULTS / "two_broadcasts.json", "w") as f:
        json.dump({"rows": rows, "wall_s": round(time.monotonic() - STARTED, 1)}, f)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["build", "card"])
    args = p.parse_args()
    return build(args) if args.what == "build" else card(args)


if __name__ == "__main__":
    sys.exit(main())
