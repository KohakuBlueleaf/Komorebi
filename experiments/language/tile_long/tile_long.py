"""Tile long rows (README.md): the RMS norm and the layer norm over rows of 1,024
and 4,096 values, and the RMS norm over 2,048, each row longer than its stream's
block of 8 tiles (read once a level of sums, then once for the writes), against
float64 on float32 tiles. `build` on a mock device, `card`."""

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
CORES, REPEAT = 12, 3  # cores (a row each), commands a handler
BLOCK = 8  # tiles a stream block: a part of every row here
EPS_RMS, EPS = 1e-6, 1e-5
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function
def rms_1024(row, weight):
    ms = km.reduce_sum(row * row, exact=True) * (1.0 / 1024)
    return row * km.rsqrt(ms + EPS_RMS).col() * weight.row()


@km.tile_function
def layer_1024(row, weight, bias):
    mean = km.reduce_sum(row, exact=True) * (1.0 / 1024)
    d = row - mean.col()
    var = km.reduce_sum(d * d, exact=True) * (1.0 / 1024)
    return d * km.rsqrt(var + EPS).col() * weight.row() + bias.row()


@km.tile_function
def rms_2048(row, weight):
    ms = km.reduce_sum(row * row, exact=True) * (1.0 / 2048)
    return row * km.rsqrt(ms + EPS_RMS).col() * weight.row()


@km.tile_function
def rms_4096(row, weight):
    ms = km.reduce_sum(row * row, exact=True) * (1.0 / 4096)
    return row * km.rsqrt(ms + EPS_RMS).col() * weight.row()


@km.tile_function
def layer_4096(row, weight, bias):
    mean = km.reduce_sum(row, exact=True) * (1.0 / 4096)
    d = row - mean.col()
    var = km.reduce_sum(d * d, exact=True) * (1.0 / 4096)
    return d * km.rsqrt(var + EPS).col() * weight.row() + bias.row()


class Long1024(km.Program):
    """rms and layer: each core's row of 32 tiles (its rank's), normalized into the
    same pages of out with the weight (and bias) fields."""

    BLOCK = km.constant(BLOCK)
    weight: km.L1[km.fp32, 16, 32, 32]
    bias: km.L1[km.fp32, 16, 32, 32]

    @km.handler
    def rms(self, x: km.Tensor, w: km.Tensor, out: km.Tensor):
        first = self.rank * 32
        self.weight.fill(km.Stream(w.pages(0, 16)))
        results = km.Stream(out.pages(first, 32), write=True, block=self.BLOCK)
        for row in km.Stream(x.pages(first, 32), block=self.BLOCK, row=32):
            results.push(rms_1024(row, self.weight))

    @km.handler
    def layer(self, x: km.Tensor, w: km.Tensor, b: km.Tensor, out: km.Tensor):
        first = self.rank * 32
        self.weight.fill(km.Stream(w.pages(0, 16)))
        self.bias.fill(km.Stream(b.pages(0, 16)))
        results = km.Stream(out.pages(first, 32), write=True, block=self.BLOCK)
        for row in km.Stream(x.pages(first, 32), block=self.BLOCK, row=32):
            results.push(layer_1024(row, self.weight, self.bias))


class Long2048(km.Program):
    """rms: as Long1024's, rows of 64 tiles."""

    BLOCK = km.constant(BLOCK)
    weight: km.L1[km.fp32, 32, 32, 32]

    @km.handler
    def rms(self, x: km.Tensor, w: km.Tensor, out: km.Tensor):
        first = self.rank * 64
        self.weight.fill(km.Stream(w.pages(0, 32)))
        results = km.Stream(out.pages(first, 64), write=True, block=self.BLOCK)
        for row in km.Stream(x.pages(first, 64), block=self.BLOCK, row=64):
            results.push(rms_2048(row, self.weight))


class Long4096(km.Program):
    """rms and layer: as Long1024's, rows of 128 tiles."""

    BLOCK = km.constant(BLOCK)
    weight: km.L1[km.fp32, 64, 32, 32]
    bias: km.L1[km.fp32, 64, 32, 32]

    @km.handler
    def rms(self, x: km.Tensor, w: km.Tensor, out: km.Tensor):
        first = self.rank * 128
        self.weight.fill(km.Stream(w.pages(0, 64)))
        results = km.Stream(out.pages(first, 128), write=True, block=self.BLOCK)
        for row in km.Stream(x.pages(first, 128), block=self.BLOCK, row=128):
            results.push(rms_4096(row, self.weight))

    @km.handler
    def layer(self, x: km.Tensor, w: km.Tensor, b: km.Tensor, out: km.Tensor):
        first = self.rank * 128
        self.weight.fill(km.Stream(w.pages(0, 64)))
        self.bias.fill(km.Stream(b.pages(0, 64)))
        results = km.Stream(out.pages(first, 128), write=True, block=self.BLOCK)
        for row in km.Stream(x.pages(first, 128), block=self.BLOCK, row=128):
            results.push(layer_4096(row, self.weight, self.bias))


@km.tile_function
def rms_apply_2048(x, w, s):
    return x * km.rsqrt(s.col() * (1.0 / 2048) + EPS_RMS) * w.row()


class Split2048(km.Program):
    """apply: the hand decode norm's last step on a row of 2,048 split over SPLIT
    cores (4 tiles each): x times 1 / rms from the row's sum of squares (field s,
    column 0, as a reduce over the cores would leave it) times the weight row."""

    s: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, s: km.Tensor, out: km.Tensor):
        first = self.rank * 4
        self.s.fill(km.Stream(s.pages(0, 1)))
        results = km.Stream(out.pages(first, 4), write=True, block=2)
        for xt, wt in zip(  # blocks of 2: both halves of each pair
            km.Stream(x.pages(first, 4), block=2), km.Stream(w.pages(first, 4), block=2)
        ):
            results.push(rms_apply_2048(xt, wt, self.s))


SPLIT = 16  # cores a row of 2,048 is split over (the hand's)


# {class: (tiles a row, its handlers)}
CLASSES = {Long1024: (32, ("rms", "layer")), Long2048: (64, ("rms",))}
CLASSES[Long4096] = (128, ("rms", "layer"))


def inputs(tiles, seed=7):
    """x: rows of 32 x tiles values, normal times 2 plus 0.7; w, b: normals."""
    rng = np.random.default_rng(seed + tiles)
    x = 2.0 * rng.standard_normal((CORES * 32, 32 * tiles)) + 0.7
    w, b = rng.standard_normal((2, 32 * tiles))
    return {n: v.astype(np.float32) for n, v in zip("xwb", (x, w, b))}


def wants(data):
    """Each handler's float64 result."""
    x, w, b = (data[n].astype(np.float64) for n in "xwb")
    rms = x / np.sqrt((x * x).mean(1, keepdims=True) + EPS_RMS) * w
    d = x - x.mean(1, keepdims=True)
    layer = d / np.sqrt((d * d).mean(1, keepdims=True) + EPS) * w + b
    return {"rms": rms, "layer": layer}


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
    """A field's pages from a row of values: chunk c's per-column tile (`row_operand`
    of its 64 values) in page c."""
    pages = [
        row_operand(values[64 * c : 64 * c + 64], 2) for c in range(len(values) // 64)
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
    count = data["w"].size // 64  # the fields' tiles
    for n in "wb":
        made[n] = session.empty((count, 1024), "float32")
        if not empty:
            session.write(made[n], chunks(data[n]).view(np.uint8).reshape(count, -1))
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


def program_of(cls, session, tensors):
    return cls(log=log, session=session, **tensors, out=tensors["x"])


def build(args):
    """Mock device: each class built."""
    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / DESCRIPTOR)
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    km_session.Session.load = build_only()
    with km.Session(log=log) as session:
        for cls, (tiles, _) in CLASSES.items():
            tensors = tensors_of(session, inputs(tiles), empty=True)
            session.load(program_of(cls, session, tensors))
        tensors = split_tensors(session, split_data()[0], empty=True)
        out = session.empty((32, 2048), "float32", "tile")
        session.load(Split2048(log=log, session=session, **tensors, out=out))
    return 0


def check(rows):
    """The rows' shape and prints, as `card` writes them (no card: a made-up run)."""
    for row in rows:
        log(json.dumps(row))
    return {"rows": rows, "wall_s": round(time.monotonic() - STARTED, 1)}


def timed(run, handler, cores, args):
    """REPEAT commands of `handler` on `cores`; their microseconds."""
    times = []
    for _ in range(REPEAT):
        sent = run.submit(handler, cores, args=args)
        event = run.wait(sent, timeout=30.0)
        if event.kind != "done":
            raise RuntimeError(f"{handler.__name__}: {event}")
        times.append(event.microseconds)
    return times


def card_class(session, cls, tiles, names):
    """One class loaded: each handler REPEAT commands on every core (a row each),
    then on one core (one row); the rows of its results."""
    data = inputs(tiles)
    tensors = tensors_of(session, data, empty=False)
    outs = {n: session.empty(data["x"].shape, "float32", "tile") for n in names}
    run = session.load(program_of(cls, session, tensors))
    times, ones = {}, {}
    try:
        x, w, b = (tensors[n] for n in "xwb")
        for n in names:
            words = (x, w, outs[n]) if n == "rms" else (x, w, b, outs[n])
            handler = getattr(cls, n)
            times[n] = timed(run, handler, (0, CORES), words)
            ones[n] = timed(run, handler, (0, 1), words)
            log(f"{cls.__name__}.{n}: {times[n]}; one row, one core: {ones[n]}")
    finally:
        run.stop()
    # read after the stop: a read waits for the cores, which a run holds
    got = {n: t.read().astype(np.float64) for n, t in outs.items()}
    want = wants(data)
    return [
        {
            "values": 32 * tiles,
            "handler": n,
            **errors(got[n], want[n]),
            "microseconds": times[n],
            "one_row_microseconds": ones[n],
        }
        for n in names
    ]


def split_data(seed=11):
    """A row of 2,048 (one tile row), its weight in row 0 of w, and s: each row's sum
    of squares in column 0 (the host's float64, rounded to float32)."""
    rng = np.random.default_rng(seed)
    x = (2.0 * rng.standard_normal((32, 2048)) + 0.7).astype(np.float32)
    w = np.zeros((32, 2048), np.float32)
    w[0] = rng.standard_normal(2048)
    s = np.zeros((32, 32), np.float32)
    s[:, 0] = (x.astype(np.float64) ** 2).sum(1)
    want = x / np.sqrt(s[:, :1].astype(np.float64) / 2048 + EPS_RMS) * w[:1]
    return {"x": x, "w": w, "s": s}, want


def split_tensors(session, data, empty):
    """The split apply's tensors as float32 tiles (empty: the build's)."""
    if empty:
        return {n: session.empty(v.shape, "float32", "tile") for n, v in data.items()}
    return {n: session.tensor(v, "float32", "tile") for n, v in data.items()}


def card_split(session):
    """Split2048.apply on SPLIT cores, REPEAT commands; its row."""
    data, want = split_data()
    tensors = split_tensors(session, data, empty=False)
    out = session.empty((32, 2048), "float32", "tile")
    run = session.load(Split2048(log=log, session=session, **tensors, out=out))
    try:
        words = (tensors["x"], tensors["w"], tensors["s"], out)
        times = timed(run, Split2048.apply, (0, SPLIT), words)
        log(f"Split2048.apply on {SPLIT} cores: {times}")
    finally:
        run.stop()
    got = out.read().astype(np.float64)
    return {
        "values": 2048,
        "handler": f"split apply ({SPLIT} cores)",
        **errors(got, want),
        "microseconds": times,
    }


def card(args):
    """One card: each class loaded in turn, then the split apply."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    rows = []
    with km.Session(log=log) as session:
        for cls, (tiles, names) in CLASSES.items():
            rows += card_class(session, cls, tiles, names)
        rows.append(card_split(session))
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "tile_long.json", "w") as f:
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
    rows = []
    for cls, (tiles, names) in CLASSES.items():
        want = wants(inputs(tiles))
        for n in names:
            times = timed(MadeUp(), getattr(cls, n), (0, CORES), ())
            rows.append(
                {
                    "values": 32 * tiles,
                    "handler": n,
                    **errors(want[n] * (1 + 1e-7), want[n]),
                    "microseconds": times,
                }
            )
    check(rows)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["build", "card", "made_up"])
    args = p.parse_args()
    return {"build": build, "card": card, "made_up": made_up}[args.what](args)


if __name__ == "__main__":
    sys.exit(main())
