"""A product of a stream's tile with a broadcast's block (README.md): Y = X W on a
COLUMNS x ROWS rectangle, W's block cast down each column once and multiplied by each
core's own row of X, plain, a window of a wider block and a transposed part, beside
the two streams' form, against float64. `build` on a mock device, `card`."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.device import native
from komorebi.session import session as km_session
from komorebi.tools import kernel_sizes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
JIT_CACHE = ROOT / "build" / "ttcache-formats"  # the jobs' cache (job.sh reads it)
DESCRIPTOR = (
    "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"
)
COLUMNS, ROWS = 12, 4  # the rectangle: a column's ROWS cores share W's block
K, N = 8, 4  # K tiles; Y's tiles a core
N2, W_BLOCK = 2 * N, 2 * N  # a column block's W columns; W's tiles a stream block
REPEAT = 5
GROUP = (0, COLUMNS * ROWS)  # the rectangle's cores, fixed at the load
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function
def project(x, w):
    return x @ w


class Down(km.Program):
    """Core (x, y): Y's tiles (y, x N .. x N + N - 1) = X's row y by W's columns from
    x 2 N + N, N of them (W is COLUMNS x 2 N tiles wide). A class runs one broadcast a
    mover (its arguments are the core's), so each broadcast handler is a class's."""

    @km.handler
    def streams(self, x: km.Tensor, w: km.Tensor, wt: km.Tensor, y: km.Tensor):
        left = self.x * 2 * N + N
        acc = km.tile.zeros(km.fp32, N)
        for xb, wb in zip(
            km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2),
            km.Stream(w.tiles[0:K, left : left + N], block=W_BLOCK),
        ):
            acc += project(xb, wb)
        y.pages(self.y * COLUMNS * N + self.x * N, N).write(acc)

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


class Window(km.Program):
    """Down's Y from a window of a block twice as wide: W's whole column block cast."""

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


class Turned(km.Program):
    """Down's Y from W transposed (wt), the cast's part turned back."""

    @km.handler
    def turned(self, x: km.Tensor, w: km.Tensor, wt: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        left = self.x * 2 * N + N
        if column.sender():
            ws = column.cast(wt.tiles[left : left + N, 0:K].T.read())
        else:
            ws = column.arrive()
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2, pattern="A"):
            acc += project(xb, ws)
        y.pages(self.y * COLUMNS * N + self.x * N, N).write(acc)


CLASSES = ((Down, ("streams", "down")), (Window, ("window",)), (Turned, ("turned",)))


def bf16(values):
    """float32 values rounded to bfloat16 (to nearest even), as float32."""
    bits = np.asarray(values, np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)


def inputs(seed=3):
    """x [ROWS x 32, K x 32] and W [K x 32, COLUMNS x 2 N x 32], bf16, normal / sqrt(K
    x 32) for W."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((ROWS * 32, K * 32))
    w = rng.standard_normal((K * 32, COLUMNS * 2 * N * 32)) / np.sqrt(K * 32)
    return bf16(x), bf16(w)


def wanted(x, w):
    """Y [ROWS x 32, COLUMNS x N x 32]: each column block's second half of W."""
    full = x.astype(np.float64) @ w.astype(np.float64)
    blocks = full.reshape(ROWS * 32, COLUMNS, 2, N * 32)[:, :, 1, :]
    return blocks.reshape(ROWS * 32, COLUMNS * N * 32)


def errors(got, want):
    """The relative RMS error and the largest error over the largest value."""
    return {
        "rms_rel": float(np.sqrt(((got - want) ** 2).mean() / (want**2).mean())),
        "max_rel": float(np.abs(got - want).max() / np.abs(want).max()),
    }


def tensors_of(session, data, empty):
    x, w = data
    shaped = {"x": x, "w": w, "wt": np.ascontiguousarray(w.T)}
    made = {
        name: (
            session.empty(value.shape, "bfloat16", "tile")
            if empty
            else session.tensor(value, "bfloat16", "tile")
        )
        for name, value in shaped.items()
    }
    made["y"] = session.empty((ROWS * 32, COLUMNS * N * 32), "float32", "tile")
    return made


def made(cls, session, tensors):
    return cls(log=log, session=session, rect=(0, 0, COLUMNS, ROWS), **tensors)


def build_only():
    """Session.load cut down to the runtime's kernel build (tools.kernel_sizes)."""
    child = kernel_sizes

    def load(self, program, **kwargs):
        try:
            child.build_only(self, program, **kwargs)
        except child.Built:
            return None
        raise RuntimeError("no build")

    return load


def build(args):
    """Mock device: the classes built, in the card's order."""
    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / DESCRIPTOR)
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    km_session.Session.load = build_only()
    with km.Session(log=log) as session:
        tensors = tensors_of(session, inputs(), empty=True)
        for cls, _ in CLASSES:
            session.load(made(cls, session, tensors))
    return 0


def card(args):
    """One card: each class loaded in turn, each handler REPEAT commands into its own
    Y, each read after its run's stop."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    x, w = inputs()
    want = wanted(x, w)
    rows, outputs = [], {}
    with km.Session(log=log) as session:
        tensors = tensors_of(session, (x, w), empty=False)
        got = {}
        for cls, names in CLASSES:
            run = session.load(made(cls, session, tensors))
            try:
                for name in names:
                    y = session.empty(tensors["y"].shape, "float32", "tile")
                    words = (tensors["x"], tensors["w"], tensors["wt"], y)
                    times = []
                    for _ in range(REPEAT):
                        sent = run.submit(getattr(cls, name), GROUP, args=words)
                        event = run.wait(sent, timeout=30.0)
                        if event.kind != "done":
                            raise RuntimeError(f"{name}: {event}")
                        times.append(event.microseconds)
                    outputs[name] = (y, times)
            finally:
                run.stop()
            for name in names:
                got[name] = outputs[name][0].read().astype(np.float64)
    for name, (_, times) in outputs.items():
        row = {"handler": name, **errors(got[name], want), "microseconds": times}
        row["equal_to_streams"] = bool(np.array_equal(got[name], got["streams"]))
        rows.append(row)
        log(json.dumps(row))
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "broadcast_product.json", "w") as f:
        json.dump({"rows": rows, "wall_s": round(time.monotonic() - STARTED, 1)}, f)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["build", "card"])
    args = p.parse_args()
    return build(args) if args.what == "build" else card(args)


if __name__ == "__main__":
    sys.exit(main())
