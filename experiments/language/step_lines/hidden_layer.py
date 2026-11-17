"""A hidden layer kept in L1 (README.md): Y = relu(X W1) W2 on CORES cores, R rows of
tiles a core; relu(X W1)'s row of KH tiles written into a field in bfloat16 and
taken as the left side of the product by W2's rows; against float64. `build` on a
mock device, `card`."""

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
CORES, R, K, KH, N = 12, 2, 8, 4, 4  # cores; rows a core; X's, H's and Y's tiles
H_BLOCK = 2 * KH  # W1's tiles a stream block: 2 K rows
O_BLOCK = 2 * N  # W2's tiles a stream block: 2 of H's rows
REPEAT = 5
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function
def project(x, w):
    return x @ w


@km.tile_function
def act(t):
    return km.relu(t)


class Hidden(km.Program):
    """Core c, row r: Y's tile row r CORES + c = relu(X's row W1) W2."""

    h: km.L1[km.bf16, KH, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w1: km.Tensor, w2: km.Tensor, y: km.Tensor):
        for r in range(R):
            row = r * CORES + self.rank
            acc = km.tile.zeros(km.fp32, KH)
            for xb, wb in zip(
                km.Stream(x.tiles[row : row + 1, 0:K], block=2),
                km.Stream(w1.tiles[0:K, 0:KH], block=H_BLOCK),
            ):
                acc += project(xb, wb)
            self.h.write(act(acc))
            out = km.tile.zeros(km.fp32, N)
            for wb in km.Stream(w2.tiles[0:KH, 0:N], block=O_BLOCK):
                out += project(self.h, wb)
            y.pages(row * N, N).write(out)


def bf16(values):
    """float32 values rounded to bfloat16 (to nearest even), as float32."""
    bits = np.asarray(values, np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)


def inputs(seed=7):
    """X [R CORES 32, K 32], W1 [K 32, KH 32], W2 [KH 32, N 32], bf16."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((R * CORES * 32, K * 32))
    w1 = rng.standard_normal((K * 32, KH * 32)) / np.sqrt(K * 32)
    w2 = rng.standard_normal((KH * 32, N * 32)) / np.sqrt(KH * 32)
    return bf16(x), bf16(w1), bf16(w2)


def wanted(x, w1, w2):
    """(Y with H in float64, Y with H rounded to bfloat16 as the field holds it)."""
    h = np.maximum(x.astype(np.float64) @ w1.astype(np.float64), 0.0)
    w2 = w2.astype(np.float64)
    return h @ w2, bf16(h).astype(np.float64) @ w2


def tensors_of(session, data, empty):
    made = {
        name: (
            session.empty(value.shape, "bfloat16", "tile")
            if empty
            else session.tensor(value, "bfloat16", "tile")
        )
        for name, value in zip(("x", "w1", "w2"), data)
    }
    made["y"] = session.empty((R * CORES * 32, N * 32), "float32", "tile")
    return made


def made(session, tensors):
    return Hidden(log=log, session=session, **tensors)


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
    process.chip_holder = lambda card, path=None: None  # a mock device has no chip
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    km_session.Session.load = build_only()
    with km.Session(log=log) as session:
        session.load(made(session, tensors_of(session, inputs(), empty=True)))
    return 0


def errors(got, want):
    diff = got - want
    return {
        "rms_rel": float(np.sqrt((diff**2).mean() / (want**2).mean())),
        "max_abs_rel": float(np.abs(diff).max() / np.abs(want).max()),
    }


def card(args):
    """One card: the class loaded once, REPEAT commands, Y read after the run's
    stop."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    data = inputs()
    exact, rounded = wanted(*data)
    with km.Session(log=log) as session:
        tensors = tensors_of(session, data, empty=False)
        run = session.load(made(session, tensors))
        times = []
        try:
            words = tuple(tensors[n] for n in ("x", "w1", "w2", "y"))
            for _ in range(REPEAT):
                sent = run.submit(Hidden.run, (0, CORES), args=words)
                event = run.wait(sent, timeout=30.0)
                if event.kind != "done":
                    raise RuntimeError(f"run: {event}")
                times.append(event.microseconds)
        finally:
            run.stop()
        got = tensors["y"].read().astype(np.float64)
    (HERE / "build").mkdir(exist_ok=True)  # the raw output, beside the builds
    np.save(HERE / "build" / "hidden_layer_y.npy", got)
    row = {
        "against_h_bf16": errors(got, rounded),
        "against_float64": errors(got, exact),
        "microseconds": times,
    }
    log(json.dumps(row))
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "hidden_layer.json", "w") as f:
        json.dump({"row": row, "wall_s": round(time.monotonic() - STARTED, 1)}, f)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["build", "card"])
    args = p.parse_args()
    return build(args) if args.what == "build" else card(args)


if __name__ == "__main__":
    sys.exit(main())
