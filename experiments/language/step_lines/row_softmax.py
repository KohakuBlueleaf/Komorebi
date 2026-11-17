"""A row softmax as a class (README.md): Z = X W on CORES cores, a core's 32 rows in C
chunks of N tiles, each chunk's step updating a row's reference m and sum l in a
state tile and writing P = 2^(x - m), x = Z log2 e; m and l written after the last;
against float64. `build` on a mock device, `card`."""

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
CORES, K, N, C = 12, 8, 2, 4  # cores; K tiles; a chunk's tiles; chunks a row
LOG2E = 1.0 / math.log(2.0)
W_BLOCK = 2 * N  # W's tiles a stream block: 2 K rows of a chunk
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
def softmax_step(row, state):
    x = row * LOG2E
    m_old, l_old = state[0], state[1]
    m_new = km.lagging_max(m_old, km.reduce_max(x, exact=True), 0.0)
    alpha = km.exp2(m_old - m_new, error=2e-3, over=(-math.inf, 0.0))
    p = km.exp2(x - m_new.col(), error=2e-3, over=(-math.inf, 0.0))
    l_new = alpha * l_old + km.reduce_sum(p, exact=True)
    return p, (m_new, l_new)


class RowSoftmax(km.Program):
    """Core r: rows 32 r .. 32 r + 31 of Z = X W, C chunks of N tiles; P's chunk c
    into pages r C N + c N of p, the state into page r of st."""

    state: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def run(
        self, x: km.Tensor, w: km.Tensor, init: km.Tensor, p: km.Tensor, st: km.Tensor
    ):
        self.state.fill(km.Stream(init.pages(0, 1)))
        for c in range(C):
            acc = km.tile.zeros(km.fp32, N)
            for xb, wb in zip(
                km.Stream(x.tiles[self.rank : self.rank + 1, 0:K], block=2),
                km.Stream(w.tiles[0:K, c * N : c * N + N], block=W_BLOCK),
            ):
                acc += project(xb, wb)
            p.pages(self.rank * C * N + c * N, N).write(
                softmax_step(acc, self.state[0])
            )
        st.pages(self.rank, 1).write(self.state)


def bf16(values):
    """float32 values rounded to bfloat16 (to nearest even), as float32."""
    bits = np.asarray(values, np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)


def inputs(seed=5):
    """x [CORES x 32, K x 32] and W [K x 32, C N x 32], bf16, Z's rows of spread ~3."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((CORES * 32, K * 32))
    w = rng.standard_normal((K * 32, C * N * 32)) * 3.0 / np.sqrt(K * 32)
    return bf16(x), bf16(w)


def wanted(x, w):
    """(x = Z log2 e by core, chunk; each chunk's reference m_c; the final l)."""
    z = x.astype(np.float64) @ w.astype(np.float64) * LOG2E
    chunks = z.reshape(CORES * 32, C, N * 32)
    refs = np.maximum.accumulate(
        np.maximum(chunks.max(axis=2), 0.0), axis=1
    )  # the state starts at m = 0
    final = (np.exp2(z - refs[:, -1:])).sum(axis=1)
    return chunks, refs, final


def tensors_of(session, data, empty):
    x, w = data
    made = {
        name: (
            session.empty(value.shape, "bfloat16", "tile")
            if empty
            else session.tensor(value, "bfloat16", "tile")
        )
        for name, value in (("x", x), ("w", w))
    }
    made["init"] = session.tensor(np.zeros((32, 32), np.float32), "float32", "tile")
    made["p"] = session.empty((32, CORES * C * N * 32), "float32", "tile")
    made["st"] = session.empty((32, CORES * 32), "float32", "tile")
    return made


def made(session, tensors):
    return RowSoftmax(log=log, session=session, **tensors)


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
        session.load(made(session, tensors_of(session, inputs(), empty=True)))
    return 0


def checked(p, st, data):
    """P's, m's and l's errors against float64, and the softmax P makes."""
    rules = rules_module()
    chunks, refs, final = wanted(*data)
    got_p = p.reshape(CORES, 32, C, N * 32).transpose(0, 1, 2, 3)
    got_p = got_p.reshape(CORES * 32, C, N * 32)
    want_p = np.exp2(chunks - refs[:, :, None])
    m = np.zeros(CORES * 32)
    l_sum = np.zeros(CORES * 32)
    spread = 0.0
    for core in range(CORES):
        tile = rules.tiles_of(st[:, 32 * core : 32 * core + 32].astype(np.float32))[0]
        m[32 * core : 32 * core + 32], s0 = rules.state_slot(tile, 0)
        l_sum[32 * core : 32 * core + 32], s1 = rules.state_slot(tile, 1)
        spread = max(spread, s0, s1)
    scaled = got_p * np.exp2(refs[:, :, None] - m[:, None, None]) / l_sum[:, None, None]
    z = chunks.reshape(CORES * 32, C * N * 32)
    soft = np.exp2(z - z.max(axis=1, keepdims=True))
    soft /= soft.sum(axis=1, keepdims=True)
    return {
        "p_rms_rel": float(
            np.sqrt(((got_p - want_p) ** 2).mean() / (want_p**2).mean())
        ),
        "p_max_abs": float(np.abs(got_p - want_p).max()),
        "m_max_abs": float(np.abs(m - refs[:, -1]).max()),
        "l_max_rel": float(np.abs(l_sum / final - 1.0).max()),
        "lane_spread": spread,
        "softmax_max_abs": float(np.abs(scaled.reshape(CORES * 32, -1) - soft).max()),
    }


def card(args):
    """One card: the class loaded once, REPEAT commands, the last's outputs read
    after the run's stop."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    data = inputs()
    with km.Session(log=log) as session:
        tensors = tensors_of(session, data, empty=False)
        run = session.load(made(session, tensors))
        times = []
        try:
            words = tuple(tensors[n] for n in ("x", "w", "init", "p", "st"))
            for _ in range(REPEAT):
                sent = run.submit(RowSoftmax.run, (0, CORES), args=words)
                event = run.wait(sent, timeout=30.0)
                if event.kind != "done":
                    raise RuntimeError(f"run: {event}")
                times.append(event.microseconds)
        finally:
            run.stop()
        p = tensors["p"].read().astype(np.float64)
        st = tensors["st"].read()
    p_rows = np.concatenate(
        [p[:, (core * C * N) * 32 : (core + 1) * C * N * 32] for core in range(CORES)]
    )
    (HERE / "build").mkdir(exist_ok=True)  # the raw outputs, beside the builds
    np.savez(HERE / "build" / "row_softmax_raw.npz", p=p_rows, st=st)
    row = {**checked(p_rows, st, data), "microseconds": times}
    log(json.dumps(row))
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "row_softmax.json", "w") as f:
        json.dump({"row": row, "wall_s": round(time.monotonic() - STARTED, 1)}, f)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["build", "card"])
    args = p.parse_args()
    return build(args) if args.what == "build" else card(args)


if __name__ == "__main__":
    sys.exit(main())
