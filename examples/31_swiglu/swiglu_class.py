"""The README's SwiGLU layer, y = (silu(x W1) * (x W3)) W2, as a class (README.md): the
class is the README's block as printed; the driver below runs it as a command a step
and as its chain, checks hidden and y against float64 of the stored values and prints
one JSON line. `build` builds it on a mock device; `card` runs it."""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.bench import power


@km.tile_function
def gate(h, g):  # the vector unit
    return km.silu(h) * g


class SwiGLU(km.Program):
    """x is [32, D], W1 and W3 are [D, H], W2 is [H, D]; every size in tiles of 32."""

    D = km.constant(32)  # x's and y's width: 1,024
    H = km.constant(96)  # the hidden width: 3,072
    WIDTH = km.constant(8)  # the columns one core makes
    CHUNK = km.constant(2)  # K tiles a block
    W_BLOCK = CHUNK * WIDTH

    h: km.L1[km.fp32, 1, WIDTH, 32, 32]  # x W1, this core's columns: a field in L1
    g: km.L1[km.fp32, 1, WIDTH, 32, 32]  # x W3
    out: km.L1[km.fp32, 1, WIDTH, 32, 32]  # hidden W2

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):  # the matrix unit
        return x @ w

    @km.handler
    def up(self, x: km.Tensor, w1: km.Tensor, w3: km.Tensor, hidden: km.Tensor):
        first = self.rank * self.WIDTH  # this core's columns of the hidden
        for xb, w1b, w3b in zip(
            km.Stream(x.tiles[0:1, 0 : self.D], block=self.CHUNK),
            km.Stream(
                w1.tiles[0 : self.D, first : first + self.WIDTH], block=self.W_BLOCK
            ),
            km.Stream(
                w3.tiles[0 : self.D, first : first + self.WIDTH], block=self.W_BLOCK
            ),
        ):
            self.h += self.project(xb, w1b)
            self.g += self.project(xb, w3b)
        hidden.tiles[0:1, first : first + self.WIDTH].write(gate(self.h, self.g))

    @km.handler
    def down(self, hidden: km.Tensor, w2: km.Tensor, y: km.Tensor):
        first = self.rank * self.WIDTH  # this core's columns of y
        for hb, wb in zip(
            km.Stream(hidden.tiles[0:1, 0 : self.H], block=self.CHUNK),
            km.Stream(
                w2.tiles[0 : self.H, first : first + self.WIDTH], block=self.W_BLOCK
            ),
        ):
            self.out += self.project(hb, wb)
        y.tiles[0:1, first : first + self.WIDTH].write(self.out)

    @km.chain
    def forward(self, x, w1, w3, w2, hidden, y):
        self.up(x, w1, w3, hidden, cores=(0, 12))  # 96 columns of the hidden, 8 a core
        self.down(hidden, w2, y, cores=(0, 4))  # 32 columns of y


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DESCRIPTOR = (
    "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"
)
JIT_CACHE = ROOT / "build" / "ttcache-formats"  # the jobs' cache: mock's and card's
RUNS = 5  # each form's timed runs
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def bf16(values):
    """float32 values rounded to bfloat16 (to nearest even), as float32."""
    bits = np.asarray(values, np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)


def inputs(seed=1):
    """x uniform in [-1, 1), each W uniform in [-1, 1) / sqrt(its fan in); bf16."""
    rng = np.random.default_rng(seed)
    d, h = 32 * 32, 96 * 32
    x = rng.uniform(-1, 1, (32, d))
    w1, w3 = (rng.uniform(-1, 1, (d, h)) / np.sqrt(d) for _ in range(2))
    w2 = rng.uniform(-1, 1, (h, d)) / np.sqrt(h)
    return {n: bf16(m) for n, m in zip(("x", "w1", "w3", "w2"), (x, w1, w3, w2))}


def wanted(data, hidden):
    """float64: the hidden of the stored inputs, y of the card's own hidden, y whole."""
    x, w1, w3, w2 = (data[n].astype(np.float64) for n in ("x", "w1", "w3", "w2"))
    p, q = x @ w1, x @ w3
    gate_ = p / (1 + np.exp(-p)) * q
    return gate_, hidden.astype(np.float64) @ w2, gate_ @ w2


def errors(got, want):
    """The relative RMS error and the largest error over the largest value."""
    return {
        "rms_rel": float(np.sqrt(((got - want) ** 2).mean() / (want**2).mean())),
        "max_rel": float(np.abs(got - want).max() / np.abs(want).max()),
    }


def tensors_of(session, data, empty):
    """x, w1, w3, w2 in bf16, hidden bf16, y float32: tiles."""
    out = {}
    for name, values in data.items():
        if empty:
            out[name] = session.empty(values.shape, "bfloat16", "tile")
        else:
            out[name] = session.tensor(values, "bfloat16", "tile")
    out["hidden"] = session.empty((32, 96 * 32), "bfloat16", "tile")
    out["y"] = session.empty((32, 32 * 32), "float32", "tile")
    return out


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


def build():
    """Mock device: the class built."""
    # here, not at the top: the class's block keeps its imports as printed
    from komorebi import session as km_session  # noqa: PLC0415
    from komorebi.device import native  # noqa: PLC0415
    from komorebi.session import process  # noqa: PLC0415

    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / DESCRIPTOR)
    process.chip_holder = lambda card, path=None: None  # a mock device has no chip
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    km_session.Session.load = build_only()
    with km.Session(log=log) as session:
        t = tensors_of(session, inputs(), empty=True)
        session.load(SwiGLU(log=log, session=session, **t))
    return 0


def card():
    """One card: the class loaded for each form in turn (a command a step, then the
    chain), run once and read after its stop (a read waits for the cores), then
    loaded again and timed."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    data = inputs()
    result = {"shape": [32, 1024, 3072], "cores": {"up": 12, "down": 4}}
    with km.Session(log=log) as session:
        t = tensors_of(session, data, empty=False)
        names = ("x", "w1", "w3", "w2", "hidden", "y")
        x, w1, w3, w2, hidden, y = (t[n] for n in names)
        program = SwiGLU(log=log, session=session, **t)  # compiled here

        def steps(run):
            first = run.wait(run.submit(SwiGLU.up, (0, 12), args=(x, w1, w3, hidden)))
            second = run.wait(run.submit(SwiGLU.down, (0, 4), args=(hidden, w2, y)))
            for event in (first, second):
                if event.kind != "done":
                    raise RuntimeError(f"steps: {event}")
            return first.microseconds + second.microseconds

        layers = {}

        def chained(run):
            if run not in layers:  # stated once a load
                layers[run] = run.chain(program.forward(x, w1, w3, w2, hidden, y))
            event = run.wait(run.submit(layers[run]))  # the layer is one command
            if event.kind != "done":
                raise RuntimeError(f"chain: {event}")
            return event.microseconds

        times, outputs = {}, {}
        for form, once in (("steps", steps), ("chain", chained)):
            hidden.write(np.zeros((32, 96 * 32), np.float32))
            y.write(np.zeros((32, 32 * 32), np.float32))
            run = session.load(program)
            try:
                times[form] = [once(run) for _ in range(1 + RUNS)][1:]
            finally:
                run.stop()
            outputs[form] = (hidden.read(), y.read())
    for form, (got_hidden, got_y) in outputs.items():
        want_hidden, want_y_card, want_y = wanted(data, got_hidden)
        result[form] = {
            "hidden": errors(got_hidden.astype(np.float64), want_hidden),
            "y_of_its_hidden": errors(got_y.astype(np.float64), want_y_card),
            "y": errors(got_y.astype(np.float64), want_y),
            "microseconds": times[form],
            "median_us": float(np.median(times[form])),
        }
    print(json.dumps(power.clocked(result)), flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / "swiglu_class.json", "w") as f:
        json.dump(result, f)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=["build", "card"])
    args = parser.parse_args()
    return build() if args.what == "build" else card()


if __name__ == "__main__":
    sys.exit(main())
