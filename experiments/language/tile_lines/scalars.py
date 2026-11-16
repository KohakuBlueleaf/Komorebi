"""Run-time scalars in a tile function (README.md): a float word in a tile expression
(`results.push(x * scale)`) and a tile function's call (`affine(x, scale, shift)`),
beside the inline body of tt-metal's `mul_unary_tile`, each command its own scale,
against float64. `build` on a mock device, `card`."""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import session as km_session
from komorebi.device import native

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
JIT_CACHE = ROOT / "build" / "ttcache-formats"  # the jobs' cache (job.sh reads it)
DESCRIPTOR = (
    "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"
)
CORES, TILES = 12, 64  # cores, tiles a core
SCALES = [(2.0, 0.5), (-0.75, 3.0), (1.0e-3, -1.0), (1.5e4, 0.0)]  # (scale, shift)
STARTED = time.monotonic()

SCALE = """
#include "api/compute/eltwise_unary/binop_with_scalar.h"
copy_init(tile_buffer);
copy_tile(tile_buffer, tile_index, 0);
binop_with_scalar_tile_init();
mul_unary_tile(0, __builtin_bit_cast(uint32_t, scale));
"""


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function
def affine(x, scale, shift):
    return x * scale + shift


class Scalars(km.Program):
    """Each handler: each core's TILES tiles from rank x TILES of a, one line a tile
    into the same tile of out, scale (and shift) words of the command."""

    @km.handler
    def line(self, a: km.Tensor, out: km.Tensor, scale: float):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x in km.Stream(a.pages(first, TILES), block=8):
            results.push(x * scale)

    @km.handler
    def called(self, a: km.Tensor, out: km.Tensor, scale: float, shift: float):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x in km.Stream(a.pages(first, TILES), block=8):
            results.push(affine(x, scale, shift))

    @km.handler
    def body(self, a: km.Tensor, out: km.Tensor, scale: float):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True)
        for tile in km.Stream(a.pages(first, TILES)):
            scaled = km.cxx(SCALE, takes=[tile, scale], gives=km.Tile)
            results.push(scaled)


def inputs(seed=11):
    """a: bf16 normals, 32 x (CORES x TILES x 32)."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((32, CORES * TILES * 32)).astype(np.float32)
    bits = a.view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)


def errors(got, want):
    """The relative RMS error and the largest error over the largest value."""
    return {
        "rms_rel": float(np.sqrt(((got - want) ** 2).mean() / (want**2).mean())),
        "max_rel": float(np.abs(got - want).max() / np.abs(want).max()),
    }


def tensors_of(session, a, empty):
    if empty:
        made = {"a": session.empty(a.shape, "bfloat16", "tile")}
    else:
        made = {"a": session.tensor(a, "bfloat16", "tile")}
    made["out"] = session.empty(a.shape, "float32", "tile")
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
        session.load(Scalars(log=log, session=session, **tensors))
    return 0


def card(args):
    """One card: the class loaded once; each handler a command a (scale, shift) of
    SCALES, each into its own output, read after the run's stop."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    a = inputs()
    x = a.astype(np.float64)
    rows, outputs = [], []
    with km.Session(log=log) as session:
        tensors = tensors_of(session, a, empty=False)
        run = session.load(Scalars(log=log, session=session, **tensors))
        try:
            for scale, shift in SCALES:
                for name in ("line", "called", "body"):
                    out = session.empty(a.shape, "float32", "tile")
                    words = (scale, shift) if name == "called" else (scale,)
                    handler = getattr(Scalars, name)
                    sent = run.submit(
                        handler, (0, CORES), args=(tensors["a"], out, *words)
                    )
                    event = run.wait(sent, timeout=30.0)
                    if event.kind != "done":
                        raise RuntimeError(f"{name} {scale}: {event}")
                    want = x * np.float32(scale) + (
                        np.float32(shift) if name == "called" else 0
                    )
                    outputs.append((name, scale, shift, out, want, event.microseconds))
        finally:
            run.stop()
        for name, scale, shift, out, want, microseconds in outputs:
            got = out.read().astype(np.float64)
            row = {
                "handler": name,
                "scale": scale,
                "shift": shift if name == "called" else None,
            }
            row.update(errors(got, want), microseconds=microseconds)
            row["body_equal"] = None
            rows.append((row, got))
    by = {(r["handler"], r["scale"]): got for r, got in rows}
    for row, got in rows:
        if row["handler"] == "line":
            row["body_equal"] = bool(np.array_equal(got, by[("body", row["scale"])]))
        log(json.dumps(row))
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "scalars.json", "w") as f:
        json.dump(
            {
                "rows": [r for r, _ in rows],
                "wall_s": round(time.monotonic() - STARTED, 1),
            },
            f,
        )
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["build", "card"])
    args = p.parse_args()
    return build(args) if args.what == "build" else card(args)


if __name__ == "__main__":
    sys.exit(main())
