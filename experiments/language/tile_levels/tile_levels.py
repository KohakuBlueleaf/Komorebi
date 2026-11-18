"""Tile levels (README.md): a tile function of two results (a state and an output)
and a layer norm (two sums in turn over the row, a weight and a bias row), each
against float64 on float32 tiles. `build` on a mock device, `card`."""

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
CORES, TILES, REPEAT = 12, 16, 3  # cores, tiles a core, commands a handler
WIDTH, ROWS = 2, 16  # tiles a row, rows a core (layer norm)
DECAY, EPS = 0.9, 1e-5
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function
def decay(x, h):
    h = DECAY * h + x
    return h, km.silu(h)


@km.tile_function
def layer_norm(row, weight, bias):
    mean = km.reduce_sum(row, exact=True) * (1.0 / (32 * WIDTH))
    d = row - mean.col()
    var = km.reduce_sum(d * d, exact=True) * (1.0 / (32 * WIDTH))
    return d * km.rsqrt(var + EPS).col() * weight.row() + bias.row()


class Levels(km.Program):
    """decay_line: each core's TILES tiles of a and h, h' = 0.9 h + x into state and
    silu(h') into out. layer_rows: each core's ROWS rows of WIDTH tiles, layer norm
    with the weight and bias rows the fields hold."""

    BLOCK = km.constant(2 * WIDTH)  # tiles a block: two rows
    weight: km.L1[km.fp32, 1, 32, 32]
    bias: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def decay_line(self, a: km.Tensor, h: km.Tensor, out: km.Tensor, state: km.Tensor):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        states = km.Stream(state.pages(first, TILES), write=True, block=8)
        for x, s in zip(
            km.Stream(a.pages(first, TILES), block=8),
            km.Stream(h.pages(first, TILES), block=8),
        ):
            s2, y = decay(x, s)
            states.push(s2)
            results.push(y)

    @km.handler
    def layer_rows(
        self, x: km.Tensor, w: km.Tensor, b: km.Tensor, out: km.Tensor, rows: int
    ):
        first = self.rank * rows * WIDTH
        self.weight.fill(km.Stream(w.pages(0, 1)))
        self.bias.fill(km.Stream(b.pages(0, 1)))
        results = km.Stream(
            out.pages(first, rows * WIDTH), write=True, block=self.BLOCK
        )
        for row in km.Stream(x.pages(first, rows * WIDTH), block=self.BLOCK, row=WIDTH):
            results.push(layer_norm(row, self.weight, self.bias))


def inputs(seed=7):
    """a, h: normals (decay); x: rows of 64 values, normal times 2 plus 0.7 (a mean
    far from 0, the variance's test); w, b: rows of 64 normals."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((CORES * TILES * 32, 32))
    h = rng.standard_normal((CORES * TILES * 32, 32))
    x = 2.0 * rng.standard_normal((CORES * ROWS * 32, 32 * WIDTH)) + 0.7
    w, b = rng.standard_normal((2, 32 * WIDTH))
    return {n: v.astype(np.float32) for n, v in zip("ahxwb", (a, h, x, w, b))}


def wants(data):
    """Each result's float64 value."""
    a, h, x, w, b = (data[n].astype(np.float64) for n in "ahxwb")
    s = DECAY * h + a
    d = x - x.mean(1, keepdims=True)
    norm = d / np.sqrt((d * d).mean(1, keepdims=True) + EPS) * w + b
    return {"state": s, "out": s / (1.0 + np.exp(-s)), "layer": norm}


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
    """a, h, x as float32 tiles; w and b as the fields' pages (`row_operand`)."""
    made = {}
    for n in "ahx":
        v = data[n]
        made[n] = (
            session.empty(v.shape, "float32", "tile")
            if empty
            else session.tensor(v, "float32", "tile")
        )
    for n in "wb":
        made[n] = session.empty((1, 1024), "float32")
        if not empty:
            page = np.asarray(row_operand(data[n], WIDTH), np.float32)
            session.write(made[n], page.view(np.uint8).reshape(1, -1))
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
    example = tensors["a"]  # out and state: float32 tiles, as a
    return Levels(log=log, session=session, **tensors, out=example, state=example)


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
        sent = run.submit(getattr(Levels, name), (0, CORES), args=args)
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
        state = session.empty(data["a"].shape, "float32", "tile")
        out = session.empty(data["a"].shape, "float32", "tile")
        layer = session.empty(data["x"].shape, "float32", "tile")
        run = session.load(program_of(session, tensors))
        try:
            words = (tensors["a"], tensors["h"], out, state)
            t_decay = timed(run, "decay_line", words)
            words = (tensors["x"], tensors["w"], tensors["b"], layer, ROWS)
            t_layer = timed(run, "layer_rows", words)
        finally:
            run.stop()
        # read after the stop: a read waits for the cores, which a run holds
        got = {
            n: t.read().astype(np.float64)
            for n, t in (("state", state), ("out", out), ("layer", layer))
        }
    want = wants(data)
    times = {"state": t_decay, "out": t_decay, "layer": t_layer}
    rows = [
        {"result": n, **errors(got[n], want[n]), "microseconds": times[n]}
        for n in ("state", "out", "layer")
    ]
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "tile_levels.json", "w") as f:
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
    timed(MadeUp(), "decay_line", ())
    timed(MadeUp(), "layer_rows", ())
    want = wants(inputs())
    rows = [
        {"result": n, **errors(w * (1 + 1e-7), w), "microseconds": [1.0]}
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
