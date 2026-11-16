"""Tile lines of a handler (README.md): a tile expression pushed (`out.push(x / y)`),
the vector functions x / y, sqrt, abs, minimum and reciprocal, and four operand tiles,
each against float64; x / y, sqrt and silu(x) y beside their inline-body or
tile-function form. `build` on a mock device, `card`."""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi import session as km_session
from komorebi.device import native
from komorebi.session import process

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
JIT_CACHE = ROOT / "build" / "ttcache-formats"  # the jobs' cache (job.sh reads it)
DESCRIPTOR = (
    "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"
)
CORES, TILES, REPEAT = 12, 64, 3  # cores, tiles a core, commands a handler
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class Lines(km.Program):
    """Each handler: each core's TILES tiles from rank x TILES of its inputs, one line
    a tile into the same tile of out."""

    @km.tile_function
    def gate(x, y):
        return km.silu(x) * y

    @km.handler
    def divide(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(first, TILES), block=8),
            km.Stream(b.pages(first, TILES), block=8),
        ):
            results.push(x / y)

    @km.handler
    def divide_body(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(first, TILES), block=8),
            km.Stream(b.pages(first, TILES), block=8),
        ):
            q = km.cxx(
                """
                #include "api/compute/eltwise_binary_sfpu.h"
                copy_init(x_buffer);
                copy_tile(x_buffer, x_index, 0);
                copy_init(y_buffer);
                copy_tile(y_buffer, y_index, 1);
                div_binary_tile_init();
                div_binary_tile(0, 1, 0);
                """,
                takes=[x, y],
                gives=km.Tile,
            )
            results.push(q)

    @km.handler
    def root(self, a: km.Tensor, out: km.Tensor):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x in km.Stream(a.pages(first, TILES), block=8):
            results.push(km.sqrt(x))

    @km.handler
    def root_body(self, a: km.Tensor, out: km.Tensor):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x in km.Stream(a.pages(first, TILES), block=8):
            r = km.cxx(
                """
                #include "api/compute/eltwise_unary/sqrt.h"
                copy_init(x_buffer);
                copy_tile(x_buffer, x_index, 0);
                sqrt_tile_init();
                sqrt_tile(0);
                """,
                takes=[x],
                gives=km.Tile,
            )
            results.push(r)

    @km.handler
    def smaller(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(first, TILES), block=8),
            km.Stream(b.pages(first, TILES), block=8),
        ):
            results.push(km.minimum(abs(x), y))

    @km.handler
    def inverse(self, b: km.Tensor, out: km.Tensor):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for y in km.Stream(b.pages(first, TILES), block=8):
            results.push(km.reciprocal(y))

    @km.handler
    def four(
        self, a: km.Tensor, b: km.Tensor, c: km.Tensor, d: km.Tensor, out: km.Tensor
    ):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x, y, z, w in zip(
            km.Stream(a.pages(first, TILES), block=8),
            km.Stream(b.pages(first, TILES), block=8),
            km.Stream(c.pages(first, TILES), block=8),
            km.Stream(d.pages(first, TILES), block=8),
        ):
            results.push(x * y + z * w)

    @km.handler
    def gated(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(first, TILES), block=8),
            km.Stream(b.pages(first, TILES), block=8),
        ):
            results.push(km.silu(x) * y)

    @km.handler
    def gated_function(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(first, TILES), block=8),
            km.Stream(b.pages(first, TILES), block=8),
        ):
            results.push(self.gate(x, y))


def inputs(seed=5):
    """a, d normal; b normal with its magnitude moved to [0.5, ...); c the absolute
    values of a normal (the root's input): bf16 values."""
    rng = np.random.default_rng(seed)
    shape = (CORES * TILES * 32, 32)
    a, b, c, d = (rng.standard_normal(shape) for _ in range(4))
    b, c = np.sign(b) * (0.5 + np.abs(b)), np.abs(c)
    return [stored(m.astype(np.float32)) for m in (a, b, c, d)]


def stored(values):
    """The values a bfloat16 tensor holds (the host's encode and decode)."""
    data = formats.to_bytes(values, "bfloat16", "tile")
    return formats.from_bytes(*data, "bfloat16", "tile")


def wants(a, b, c, d):
    """Each handler's inputs and its float64 result."""
    a, b, c, d = (m.astype(np.float64) for m in (a, b, c, d))
    return {
        "divide": (("a", "b"), a / b),
        "divide_body": (("a", "b"), a / b),
        "root": (("c",), np.sqrt(c)),
        "root_body": (("c",), np.sqrt(c)),
        "smaller": (("a", "b"), np.minimum(np.abs(a), b)),
        "inverse": (("b",), 1 / b),
        "four": (("a", "b", "c", "d"), a * b + c * d),
        "gated": (("a", "b"), a / (1 + np.exp(-a)) * b),
        "gated_function": (("a", "b"), a / (1 + np.exp(-a)) * b),
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
    """a, b, c and d in bfloat16."""
    if empty:
        return {
            n: session.empty(v.shape, "bfloat16", "tile") for n, v in zip("abcd", data)
        }
    return {n: session.tensor(v, "bfloat16", "tile") for n, v in zip("abcd", data)}


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
    process.chip_holder = lambda card, path=None: None  # a mock device has no chip
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    km_session.Session.load = build_only()
    with km.Session(log=log) as session:
        tensors = tensors_of(session, inputs(), empty=True)
        out = session.empty(tensors["a"].shape, "float32", "tile")
        session.load(Lines(log=log, session=session, **tensors, out=out))
    return 0


def card(args):
    """One card: the class loaded once, each handler REPEAT commands."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    data = inputs()
    rows = []
    with km.Session(log=log) as session:
        tensors = tensors_of(session, data, empty=False)
        out = session.empty(data[0].shape, "float32", "tile")
        run = session.load(Lines(log=log, session=session, **tensors, out=out))
        results = {}
        try:
            for name, (names, _) in wants(*data).items():
                if args.only and name not in args.only:
                    continue
                y = session.empty(data[0].shape, "float32", "tile")
                times = []
                for _ in range(REPEAT):
                    words = (*[tensors[n] for n in names], y)
                    sent = run.submit(getattr(Lines, name), (0, CORES), args=words)
                    event = run.wait(sent, timeout=30.0)
                    if event.kind != "done":
                        raise RuntimeError(f"{name}: {event}")
                    times.append(event.microseconds)
                results[name] = (y, times)
        finally:
            run.stop()
        # read after the stop: a read waits for the cores, which a run holds
        results = {k: (y.read().astype(np.float64), t) for k, (y, t) in results.items()}
    for name, (_, want) in wants(*data).items():
        if name not in results:
            continue
        got, times = results[name]
        row = {"handler": name, **errors(got, want), "microseconds": times}
        rows.append(row)
        log(json.dumps(row))
    pairs = (("divide", "divide_body"), ("root", "root_body"))
    pairs += (("gated", "gated_function"),)
    for line, other in pairs:  # the two forms' words, datum by datum
        if line not in results or other not in results:
            continue
        same = float((results[line][0] == results[other][0]).mean())
        rows.append({"pair": [line, other], "equal_datums": same})
        log(json.dumps(rows[-1]))
    RESULTS.mkdir(exist_ok=True)
    name = (
        "tile_lines.json" if not args.only else f"tile_lines_{'_'.join(args.only)}.json"
    )
    with open(RESULTS / name, "w") as f:
        json.dump({"rows": rows, "wall_s": round(time.monotonic() - STARTED, 1)}, f)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["build", "card"])
    p.add_argument("--only", nargs="*", default=[], help="these handlers alone")
    args = p.parse_args()
    return build(args) if args.what == "build" else card(args)


if __name__ == "__main__":
    sys.exit(main())
