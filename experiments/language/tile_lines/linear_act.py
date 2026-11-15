"""A product and a tile function in one body (README.md): y = silu(x W) with x W's sums
in DST (`hidden.pages(...).write(act(acc))`) and in a field of sums
(`y.tiles[...].write(act(self.part))`), beside y = x W alone, against float64.
`build` on a mock device, `card`."""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import native
from komorebi import session as km_session

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
JIT_CACHE = ROOT / "build" / "ttcache-formats"  # the jobs' cache (job.sh reads it)
DESCRIPTOR = (
    "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"
)
CORES, K, WIDTH, REPEAT = 12, 32, 4, 5  # cores; K tiles; y's tiles a core; commands
W_BLOCK = 2 * WIDTH  # W tiles a block: 2 K rows of WIDTH
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function
def act(h):
    return km.silu(h)


class LinearAct(km.Program):
    """Each core: WIDTH columns of y = x W from rank x WIDTH, K tiles of x's row, 2 K
    tiles a block."""

    part: km.L1[km.fp32, 1, WIDTH, 32, 32]

    @km.tile_function
    def project(x, w):
        return x @ w

    @km.handler
    def plain(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        first = self.rank * WIDTH
        acc = km.tile.zeros(km.fp32, WIDTH)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:K], block=2),
            km.Stream(w.tiles[0:K, first : first + WIDTH], block=W_BLOCK),
        ):
            acc += self.project(xb, wb)
        y.pages(first, WIDTH).write(acc)

    @km.handler
    def in_dst(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        first = self.rank * WIDTH
        acc = km.tile.zeros(km.fp32, WIDTH)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:K], block=2),
            km.Stream(w.tiles[0:K, first : first + WIDTH], block=W_BLOCK),
        ):
            acc += self.project(xb, wb)
        y.pages(first, WIDTH).write(act(acc))

    @km.handler
    def in_field(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        first = self.rank * WIDTH
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:K], block=2),
            km.Stream(w.tiles[0:K, first : first + WIDTH], block=W_BLOCK),
        ):
            self.part += self.project(xb, wb)
        y.tiles[0:1, first : first + WIDTH].write(act(self.part))


def bf16(values):
    """float32 values rounded to bfloat16 (to nearest even), as float32."""
    bits = np.asarray(values, np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)


def inputs(seed=7):
    """x uniform in [-1, 1), W uniform in [-1, 1) / sqrt(K x 32): bf16."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (32, K * 32))
    w = rng.uniform(-1, 1, (K * 32, CORES * WIDTH * 32)) / np.sqrt(K * 32)
    return bf16(x), bf16(w)


def errors(got, want):
    """The relative RMS error and the largest error over the largest value."""
    return {
        "rms_rel": float(np.sqrt(((got - want) ** 2).mean() / (want**2).mean())),
        "max_rel": float(np.abs(got - want).max() / np.abs(want).max()),
    }


def tensors_of(session, data, empty):
    x, w = data
    if empty:
        made = {"x": session.empty(x.shape, "bfloat16", "tile")}
        made["w"] = session.empty(w.shape, "bfloat16", "tile")
    else:
        made = {"x": session.tensor(x, "bfloat16", "tile")}
        made["w"] = session.tensor(w, "bfloat16", "tile")
    made["y"] = session.empty((32, w.shape[1]), "float32", "tile")
    return made


def build_only():
    """Session.load cut down to the runtime's kernel build (the sizes test's child)."""
    path = ROOT / "tests" / "projects" / "kernel_sizes_child.py"
    spec = importlib.util.spec_from_file_location("kernel_sizes_child", path)
    child = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(child)

    def load(self, program, **kwargs):
        try:
            child.build_only(self, program, **kwargs)
        except child.Built:
            return None
        raise RuntimeError("no build")

    return load


def build(args):
    """Mock device: the class built."""
    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / DESCRIPTOR)
    km_session.chip_holder = lambda card, path=None: None  # a mock device has no chip
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    km_session.Session.load = build_only()
    with km.Session(log=log) as session:
        tensors = tensors_of(session, inputs(), empty=True)
        session.load(LinearAct(log=log, session=session, **tensors))
    return 0


def card(args):
    """One card: the class loaded once, each handler REPEAT commands, each output read
    after the run's stop (a read waits for the cores), before the session closes."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    x, w = inputs()
    plain = x.astype(np.float64) @ w.astype(np.float64)
    wants = {"plain": plain, "in_dst": plain / (1 + np.exp(-plain))}
    wants["in_field"] = wants["in_dst"]
    rows, outputs = [], {}
    with km.Session(log=log) as session:
        tensors = tensors_of(session, (x, w), empty=False)
        run = session.load(LinearAct(log=log, session=session, **tensors))
        try:
            for name in wants:
                y = session.empty(tensors["y"].shape, "float32", "tile")
                times = []
                for _ in range(REPEAT):
                    words = (tensors["x"], tensors["w"], y)
                    sent = run.submit(getattr(LinearAct, name), (0, CORES), args=words)
                    event = run.wait(sent, timeout=30.0)
                    if event.kind != "done":
                        raise RuntimeError(f"{name}: {event}")
                    times.append(event.microseconds)
                outputs[name] = (y, times)
        finally:
            run.stop()
        for name, (y, times) in outputs.items():  # read in the session, after stop
            got = y.read().astype(np.float64)
            row = {"handler": name, **errors(got, wants[name]), "microseconds": times}
            rows.append(row)
            log(json.dumps(row))
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "linear_act.json", "w") as f:
        json.dump({"rows": rows, "wall_s": round(time.monotonic() - STARTED, 1)}, f)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["build", "card"])
    args = p.parse_args()
    return build(args) if args.what == "build" else card(args)


if __name__ == "__main__":
    sys.exit(main())
