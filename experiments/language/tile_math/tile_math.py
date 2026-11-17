"""Tile math (README.md): tanh, erf and log in a handler's tile line, and GELU and
softplus written with them, each against float64 on float32 tiles. `build` on a mock
device, `card`."""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.device import native
from komorebi.session import session as km_session

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
JIT_CACHE = ROOT / "build" / "ttcache-formats"  # the jobs' cache (job.sh reads it)
DESCRIPTOR = (
    "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"
)
CORES, TILES, REPEAT = 12, 16, 3  # cores, tiles a core, commands a handler
SQRT_HALF = 0.7071067811865476
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function
def gelu(x):
    return 0.5 * x * (1.0 + km.erf(x * SQRT_HALF))


@km.tile_function
def softplus(x):
    return km.maximum(x, 0.0) + km.log(1.0 + km.exp(-abs(x)))


class Math(km.Program):
    """Each handler: each core's TILES tiles from rank x TILES of its input, one line
    a tile into the same tile of out."""

    @km.handler
    def tanh_line(self, a: km.Tensor, out: km.Tensor):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x in km.Stream(a.pages(first, TILES), block=8):
            results.push(km.tanh(x))

    @km.handler
    def erf_line(self, a: km.Tensor, out: km.Tensor):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x in km.Stream(a.pages(first, TILES), block=8):
            results.push(km.erf(x))

    @km.handler
    def log_line(self, b: km.Tensor, out: km.Tensor):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x in km.Stream(b.pages(first, TILES), block=8):
            results.push(km.log(x))

    @km.handler
    def gelu_line(self, a: km.Tensor, out: km.Tensor):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x in km.Stream(a.pages(first, TILES), block=8):
            results.push(gelu(x))

    @km.handler
    def softplus_line(self, a: km.Tensor, out: km.Tensor):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x in km.Stream(a.pages(first, TILES), block=8):
            results.push(softplus(x))


def inputs(seed=7):
    """a: normal times 3 (both sides of tanh's 0.625 and erf's tails); b: e^u, u
    uniform on [-20, 20] (log's input). float32 values."""
    rng = np.random.default_rng(seed)
    shape = (CORES * TILES * 32, 32)
    a = 3.0 * rng.standard_normal(shape)
    b = np.exp(rng.uniform(-20.0, 20.0, shape))
    return [m.astype(np.float32) for m in (a, b)]


def wants(a, b):
    """Each handler's input and its float64 result."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    erf = np.vectorize(math.erf)
    return {
        "tanh_line": ("a", np.tanh(a)),
        "erf_line": ("a", erf(a)),
        "log_line": ("b", np.log(b)),
        "gelu_line": ("a", 0.5 * a * (1.0 + erf(a * SQRT_HALF))),
        "softplus_line": ("a", np.logaddexp(a, 0.0)),
    }


def errors(got, want):
    """The relative RMS error, the largest error over the RMS, the largest relative."""
    scale = np.sqrt((want**2).mean())
    nonzero = want != 0
    return {
        "rms_rel": float(np.sqrt(((got - want) ** 2).mean()) / scale),
        "max_over_rms": float(np.abs(got - want).max() / scale),
        "max_rel": float((np.abs(got - want)[nonzero] / np.abs(want[nonzero])).max()),
    }


def tensors_of(session, data, empty):
    """a and b in float32."""
    if empty:
        return {
            n: session.empty(v.shape, "float32", "tile") for n, v in zip("ab", data)
        }
    return {n: session.tensor(v, "float32", "tile") for n, v in zip("ab", data)}


def build_only():
    """Session.load cut down to the runtime's kernel build (the sizes test's child)."""
    path = ROOT / "tests" / "projects" / "sizes" / "kernel_sizes_child.py"
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
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    km_session.Session.load = build_only()
    with km.Session(log=log) as session:
        tensors = tensors_of(session, inputs(), empty=True)
        out = session.empty(tensors["a"].shape, "float32", "tile")
        session.load(Math(log=log, session=session, **tensors, out=out))
    return 0


def check(rows):
    """The rows' shape and prints, as `card` writes them (no card: a made-up run)."""
    for row in rows:
        log(json.dumps(row))
    return {"rows": rows, "wall_s": round(time.monotonic() - STARTED, 1)}


def card(args):
    """One card: the class loaded once, each handler REPEAT commands."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    data = inputs()
    with km.Session(log=log) as session:
        tensors = tensors_of(session, data, empty=False)
        out = session.empty(data[0].shape, "float32", "tile")
        run = session.load(Math(log=log, session=session, **tensors, out=out))
        results = {}
        try:
            for name, (given, _) in wants(*data).items():
                y = session.empty(data[0].shape, "float32", "tile")
                times = []
                for _ in range(REPEAT):
                    sent = run.submit(
                        getattr(Math, name), (0, CORES), args=(tensors[given], y)
                    )
                    event = run.wait(sent, timeout=30.0)
                    if event.kind != "done":
                        raise RuntimeError(f"{name}: {event}")
                    times.append(event.microseconds)
                log(f"{name}: {times}")
                results[name] = (y, times)
        finally:
            run.stop()
        # read after the stop: a read waits for the cores, which a run holds
        got = {}
        for name, (y, times) in results.items():
            got[name] = (y.read().astype(np.float64), times)
            log(f"{name} read")
    rows = [
        {"handler": name, **errors(got[name][0], want), "microseconds": got[name][1]}
        for name, (_, want) in wants(*data).items()
    ]
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "tile_math.json", "w") as f:
        json.dump(check(rows), f)
    return 0


def made_up(args):
    """The host side after the card: float64 results moved by 1e-7 through `check`."""
    data = inputs()
    rows = [
        {"handler": n, **errors(w * (1 + 1e-7), w), "microseconds": [1.0]}
        for n, (_, w) in wants(*data).items()
    ]
    check(rows)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["build", "card", "made_up"])
    args = p.parse_args()
    return {"build": build, "card": card, "made_up": made_up}[args.what](args)


if __name__ == "__main__":
    sys.exit(main())
