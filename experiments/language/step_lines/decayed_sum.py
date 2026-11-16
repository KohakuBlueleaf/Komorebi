"""Sums started from a field (README.md): o = o d + X_c W_c over C chunks on CORES
cores, d a per-row decay in slot 0 of a state tile the host writes, o an fp32 field
of N tiles; Y = o d after the last chunk; against float64. `build` on a mock device,
`card`."""

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
CORES, C, K, N = 12, 4, 4, 2  # cores; chunks; a chunk's K tiles; o's tiles
W_BLOCK = 2 * N  # W's tiles a stream block: 2 K rows
REPEAT = 5
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def rules_module():
    """experiments/language/tile_rules/rules.py: the state tile's host layout."""
    path = ROOT / "experiments" / "language" / "tile_rules" / "rules.py"
    spec = importlib.util.spec_from_file_location("tile_rules_rules", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@km.tile_function
def project(x, w):
    return x @ w


@km.tile_function
def decay(row, state):
    return row * state[0].col(), (state[0],)


class Decayed(km.Program):
    """Core r: Y's tile row r = d (sum over c of d^(C-1-c) X_rc W_c), d its rows'."""

    state: km.L1[km.fp32, 1, 32, 32]
    o: km.L1[km.fp32, N, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, init: km.Tensor, y: km.Tensor):
        self.state.fill(km.Stream(init.pages(self.rank, 1)))
        zero = km.tile.zeros(km.fp32, N)
        self.o.write(zero)
        for c in range(C):
            acc = decay(self.o, self.state[0])
            for xb, wb in zip(
                km.Stream(
                    x.tiles[self.rank : self.rank + 1, c * K : c * K + K], block=2
                ),
                km.Stream(w.tiles[c * K : c * K + K, 0:N], block=W_BLOCK),
            ):
                acc += project(xb, wb)
            self.o.write(acc)
        last = decay(self.o, self.state[0])
        y.pages(self.rank * N, N).write(last)


def bf16(values):
    """float32 values rounded to bfloat16 (to nearest even), as float32."""
    bits = np.asarray(values, np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)


def inputs(seed=11):
    """X [CORES 32, C K 32], W [C K 32, N 32] bf16, and each row's decay in [0.5, 1)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((CORES * 32, C * K * 32))
    w = rng.standard_normal((C * K * 32, N * 32)) / np.sqrt(K * 32)
    d = rng.uniform(0.5, 1.0, CORES * 32).astype(np.float32)
    return bf16(x), bf16(w), d


def wanted(x, w, d):
    """Y = d (sum over c of d^(C-1-c) X_c W_c), in float64."""
    x, w, d = x.astype(np.float64), w.astype(np.float64), d.astype(np.float64)
    o = np.zeros((CORES * 32, N * 32))
    for c in range(C):
        cols = slice(c * K * 32, (c + 1) * K * 32)
        o = o * d[:, None] + x[:, cols] @ w[cols]
    return o * d[:, None]


def tensors_of(session, data, empty):
    x, w, d = data
    rules = rules_module()
    states = np.concatenate(
        [
            rules.matrix_of(rules.state_tile({0: d[32 * r : 32 * r + 32]}))
            for r in range(CORES)
        ],
        axis=1,
    )
    made = {
        name: (
            session.empty(value.shape, "bfloat16", "tile")
            if empty
            else session.tensor(value, "bfloat16", "tile")
        )
        for name, value in (("x", x), ("w", w))
    }
    made["init"] = session.tensor(states, "float32", "tile")
    made["y"] = session.empty((32, CORES * N * 32), "float32", "tile")
    return made


def made(session, tensors):
    return Decayed(log=log, session=session, **tensors)


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
    want = wanted(*data)
    with km.Session(log=log) as session:
        tensors = tensors_of(session, data, empty=False)
        run = session.load(made(session, tensors))
        times = []
        try:
            words = tuple(tensors[n] for n in ("x", "w", "init", "y"))
            for _ in range(REPEAT):
                sent = run.submit(Decayed.run, (0, CORES), args=words)
                event = run.wait(sent, timeout=30.0)
                if event.kind != "done":
                    raise RuntimeError(f"run: {event}")
                times.append(event.microseconds)
        finally:
            run.stop()
        y = tensors["y"].read().astype(np.float64)
    got = np.concatenate(
        [y[:, r * N * 32 : (r + 1) * N * 32] for r in range(CORES)]
    )  # core r's tiles are its rows
    (HERE / "build").mkdir(exist_ok=True)  # the raw output, beside the builds
    np.save(HERE / "build" / "decayed_sum_y.npy", got)
    undecayed = errors(got, want / data[2].astype(np.float64)[:, None])
    row = {**errors(got, want), "against_y_without_the_last_d": undecayed}
    row["microseconds"] = times
    log(json.dumps(row))
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "decayed_sum.json", "w") as f:
        json.dump({"row": row, "wall_s": round(time.monotonic() - STARTED, 1)}, f)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["build", "card"])
    args = p.parse_args()
    return build(args) if args.what == "build" else card(args)


if __name__ == "__main__":
    sys.exit(main())
