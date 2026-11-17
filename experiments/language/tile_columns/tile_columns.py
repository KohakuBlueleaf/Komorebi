"""Tile columns (README.md): a per-column operand in a call on a stream's tiles, a
field of one plain row read by `.row()` (a bias, a per-channel scale), against float64
on float32 and bfloat16 tiles. `build` on a mock device, `card`."""

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
from komorebi.session import process
from komorebi.session import session as km_session

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
JIT_CACHE = ROOT / "build" / "ttcache-formats"  # the jobs' cache (job.sh reads it)
DESCRIPTOR = (
    "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"
)
CORES, TILES, REPEAT = 12, 16, 3  # cores, tiles a core, commands a handler
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function
def affine(x, scale, shift):
    return x * scale.row() + shift.row()


@km.tile_function
def biased(x, bias):
    return km.silu(x + bias.row())


class Columns(km.Program):
    """affine_line: each core's TILES float32 tiles of a, x scale + shift, a value a
    column from the fields. bias_line: bfloat16 tiles of c, silu(x + bias)."""

    scale: km.L1[km.fp32, 1, 32, 32]
    shift: km.L1[km.fp32, 1, 32, 32]
    bias: km.L1[km.bf16, 1, 32, 32]

    @km.handler
    def affine_line(self, a: km.Tensor, s: km.Tensor, b: km.Tensor, out: km.Tensor):
        first = self.rank * TILES
        self.scale.fill(km.Stream(s.pages(0, 1)))
        self.shift.fill(km.Stream(b.pages(0, 1)))
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x in km.Stream(a.pages(first, TILES), block=8):
            results.push(affine(x, self.scale, self.shift))

    @km.handler
    def bias_line(self, c: km.Tensor, d: km.Tensor, out16: km.Tensor):
        first = self.rank * TILES
        self.bias.fill(km.Stream(d.pages(0, 1)))
        results = km.Stream(out16.pages(first, TILES), write=True, block=8)
        for x in km.Stream(c.pages(first, TILES), block=8):
            results.push(biased(x, self.bias))


def bf16(v):
    """float32 values rounded to bfloat16 (nearest even), as float32."""
    bits = np.asarray(v, np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16 << 16
    return bits.astype(np.uint32).view(np.float32)


def row_tile(values):
    """A tile whose row 0 holds 32 values (a plain row), the other rows 0."""
    tile = np.zeros((32, 32), np.float32)
    tile[0] = values
    return tile


def inputs(seed=7):
    """a: normals; s, b: 32 normals each (a scale and a shift a column); c: normals
    times 2 and d: 32 normals, both rounded to bfloat16."""
    rng = np.random.default_rng(seed)
    shape = (CORES * TILES * 32, 32)
    a = rng.standard_normal(shape).astype(np.float32)
    s, b, d = rng.standard_normal((3, 32)).astype(np.float32)
    c = bf16(2.0 * rng.standard_normal(shape))
    return {"a": a, "s": s, "b": b, "c": c, "d": bf16(d)}


def wants(data):
    """Each handler's float64 result (bias_line's of the bfloat16 values)."""
    a, s, b, c, d = (data[n].astype(np.float64) for n in "asbcd")
    z = c + d
    return {"affine_line": a * s + b, "bias_line": z / (1.0 + np.exp(-z))}


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
    """a, s, b float32 tiles; c, d bfloat16; s, b, d a tile each, the row in row 0."""
    made = {}
    for n in "asbcd":
        dtype = "bfloat16" if n in "cd" else "float32"
        v = data[n] if data[n].ndim == 2 else row_tile(data[n])
        made[n] = (
            session.empty(v.shape, dtype, "tile")
            if empty
            else session.tensor(v, dtype, "tile")
        )
    return made


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


def program_of(session, tensors):
    return Columns(
        log=log, session=session, **tensors, out=tensors["a"], out16=tensors["c"]
    )


def build(args):
    """Mock device: the class built."""
    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / DESCRIPTOR)
    process.chip_holder = lambda card, path=None: None  # a mock device has no chip
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    km_session.Session.load = build_only()
    with km.Session(log=log) as session:
        tensors = tensors_of(session, inputs(), empty=True)
        session.load(program_of(session, tensors))
    return 0


def check(rows):
    """The rows' shape and prints, as `card` writes them (no card: a made-up run)."""
    for row in rows:
        log(json.dumps(row))
    return {"rows": rows, "wall_s": round(time.monotonic() - STARTED, 1)}


def timed(run, name, args):
    """REPEAT commands of handler `name` on every core; their microseconds."""
    times = []
    for _ in range(REPEAT):
        sent = run.submit(getattr(Columns, name), (0, CORES), args=args)
        event = run.wait(sent, timeout=30.0)
        if event.kind != "done":
            raise RuntimeError(f"{name}: {event}")
        times.append(event.microseconds)
    log(f"{name}: {times}")
    return times


def card(args):
    """One card: the class loaded once, each handler REPEAT commands."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    data = inputs()
    with km.Session(log=log) as session:
        tensors = tensors_of(session, data, empty=False)
        out = session.empty(data["a"].shape, "float32", "tile")
        out16 = session.empty(data["c"].shape, "bfloat16", "tile")
        run = session.load(program_of(session, tensors))
        try:
            words = (tensors["a"], tensors["s"], tensors["b"], out)
            times = {"affine_line": timed(run, "affine_line", words)}
            words = (tensors["c"], tensors["d"], out16)
            times["bias_line"] = timed(run, "bias_line", words)
        finally:
            run.stop()
        # read after the stop: a read waits for the cores, which a run holds
        got = {
            "affine_line": out.read().astype(np.float64),
            "bias_line": out16.read().astype(np.float64),
        }
    rows = [
        {"handler": n, **errors(got[n], w), "microseconds": times[n]}
        for n, w in wants(data).items()
    ]
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "tile_columns.json", "w") as f:
        json.dump(check(rows), f)
    return 0


class MadeUp:
    """A run that answers every command done in 1 us (made_up's)."""

    def submit(self, handler, cores, args):
        return handler

    def wait(self, sent, timeout):
        return type("Event", (), {"kind": "done", "microseconds": 1.0})


def made_up(args):
    """The host side after the card: the commands' loop on a made-up run, float64
    results moved by 1e-7 through `check`."""
    timed(MadeUp(), "affine_line", ())
    timed(MadeUp(), "bias_line", ())
    want = wants(inputs())
    rows = [
        {"handler": n, **errors(w * (1 + 1e-7), w), "microseconds": [1.0]}
        for n, w in want.items()
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
