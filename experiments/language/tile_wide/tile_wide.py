"""Tile wide rows (README.md): the RMS norm and the layer norm over rows of 736
values (23 tiles: 11 chunks of 2 and a last of 1), the weight and bias fields a
chunk's per-column tile each, against float64 on float32 tiles. `build` on a mock
device, `card`."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.device import native
from komorebi.dsl.tile import row_operand
from komorebi.session import session as km_session
from komorebi.tools import kernel_sizes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
JIT_CACHE = ROOT / "build" / "ttcache-formats"  # the jobs' cache (job.sh reads it)
DESCRIPTOR = (
    "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"
)
CORES, ROWS, REPEAT = 12, 4, 3  # cores, rows a core, commands a handler
WIDE = 23  # tiles a row: 736 values (11 chunks of 2, a last of 1)
CHUNKS = 12  # chunks a row: the fields' tiles
EPS_RMS, EPS = 1e-6, 1e-5
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function
def rms_norm(row, weight):
    ms = km.reduce_sum(row * row, exact=True) * (1.0 / (32 * WIDE))
    return row * km.rsqrt(ms + EPS_RMS).col() * weight.row()


@km.tile_function
def layer_norm(row, weight, bias):
    mean = km.reduce_sum(row, exact=True) * (1.0 / (32 * WIDE))
    d = row - mean.col()
    var = km.reduce_sum(d * d, exact=True) * (1.0 / (32 * WIDE))
    return d * km.rsqrt(var + EPS).col() * weight.row() + bias.row()


class Wide(km.Program):
    """rms_rows and layer_rows: each core's `rows` rows of WIDE tiles from rank x
    rows, normalized into the same pages of out with the weight (and bias) fields."""

    BLOCK = km.constant(WIDE)  # tiles a block: one row
    weight: km.L1[km.fp32, CHUNKS, 32, 32]
    bias: km.L1[km.fp32, CHUNKS, 32, 32]

    @km.handler
    def rms_rows(self, x: km.Tensor, w: km.Tensor, out: km.Tensor, rows: int):
        first = self.rank * rows * WIDE
        self.weight.fill(km.Stream(w.pages(0, CHUNKS)))
        results = km.Stream(out.pages(first, rows * WIDE), write=True, block=self.BLOCK)
        for row in km.Stream(x.pages(first, rows * WIDE), block=self.BLOCK, row=WIDE):
            results.push(rms_norm(row, self.weight))

    @km.handler
    def layer_rows(
        self, x: km.Tensor, w: km.Tensor, b: km.Tensor, out: km.Tensor, rows: int
    ):
        first = self.rank * rows * WIDE
        self.weight.fill(km.Stream(w.pages(0, CHUNKS)))
        self.bias.fill(km.Stream(b.pages(0, CHUNKS)))
        results = km.Stream(out.pages(first, rows * WIDE), write=True, block=self.BLOCK)
        for row in km.Stream(x.pages(first, rows * WIDE), block=self.BLOCK, row=WIDE):
            results.push(layer_norm(row, self.weight, self.bias))


def inputs(seed=7):
    """x: rows of 736 values, normal times 2 plus 0.7; w, b: 736 normals each."""
    rng = np.random.default_rng(seed)
    x = 2.0 * rng.standard_normal((CORES * ROWS * 32, 32 * WIDE)) + 0.7
    w, b = rng.standard_normal((2, 32 * WIDE))
    return {n: v.astype(np.float32) for n, v in zip("xwb", (x, w, b))}


def wants(data):
    """Each handler's float64 result."""
    x, w, b = (data[n].astype(np.float64) for n in "xwb")
    rms = x / np.sqrt((x * x).mean(1, keepdims=True) + EPS_RMS) * w
    d = x - x.mean(1, keepdims=True)
    layer = d / np.sqrt((d * d).mean(1, keepdims=True) + EPS) * w + b
    return {"rms_rows": rms, "layer_rows": layer}


def errors(got, want):
    """The relative RMS error, the largest error over the RMS, the largest relative."""
    scale = np.sqrt((want**2).mean())
    nonzero = want != 0
    return {
        "rms_rel": float(np.sqrt(((got - want) ** 2).mean()) / scale),
        "max_over_rms": float(np.abs(got - want).max() / scale),
        "max_rel": float((np.abs(got - want)[nonzero] / np.abs(want[nonzero])).max()),
    }


def chunks(values):
    """A field's pages from a row of WIDE tiles of values: chunk c's per-column tile
    (`row_operand` of its 64 values, the last chunk's 32) in page c."""
    pages = [
        row_operand(values[64 * c : 64 * c + 64], min(2, WIDE - 2 * c))
        for c in range(CHUNKS)
    ]
    return np.asarray(pages, np.float32)


def tensors_of(session, data, empty):
    """x as float32 tiles; w and b as the fields' pages."""
    made = {
        "x": (
            session.empty(data["x"].shape, "float32", "tile")
            if empty
            else session.tensor(data["x"], "float32", "tile")
        )
    }
    for n in "wb":
        made[n] = session.empty((CHUNKS, 1024), "float32")
        if not empty:
            session.write(made[n], chunks(data[n]).view(np.uint8).reshape(CHUNKS, -1))
    return made


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


def program_of(session, tensors):
    return Wide(log=log, session=session, **tensors, out=tensors["x"])


def build(args):
    """Mock device: the class built."""
    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / DESCRIPTOR)
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
        sent = run.submit(getattr(Wide, name), (0, CORES), args=args)
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
        outs = {
            n: session.empty(data["x"].shape, "float32", "tile") for n in wants(data)
        }
        run = session.load(program_of(session, tensors))
        try:
            x, w, b = (tensors[n] for n in "xwb")
            times = {"rms_rows": timed(run, "rms_rows", (x, w, outs["rms_rows"], ROWS))}
            words = (x, w, b, outs["layer_rows"], ROWS)
            times["layer_rows"] = timed(run, "layer_rows", words)
        finally:
            run.stop()
        # read after the stop: a read waits for the cores, which a run holds
        got = {n: t.read().astype(np.float64) for n, t in outs.items()}
    rows = [
        {"handler": n, **errors(got[n], w), "microseconds": times[n]}
        for n, w in wants(data).items()
    ]
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "tile_wide.json", "w") as f:
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
    timed(MadeUp(), "rms_rows", ())
    timed(MadeUp(), "layer_rows", ())
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
