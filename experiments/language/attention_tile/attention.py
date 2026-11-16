"""The attention project's ATTEND as a class of tile functions and lines, beside the
hand program on the same tensors: no inline C++. README.md says what each line
lowers to and what it measured.
Example: python attention.py card --shape t1; python attention.py build --shape t1"""

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
from komorebi import native
from komorebi import session as km_session

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from projects.attention import card  # noqa: E402
from projects.attention import plan as pl  # noqa: E402
from projects.attention.reference import sdpa  # noqa: E402

JIT_CACHE = ROOT / "build" / "ttcache-formats"  # the jobs' cache (job.sh reads it)
DESCRIPTOR = (
    "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"
)
STARTED = time.monotonic()
# The plan's sizes for a head dim of 64 (t1, causal64): Q / K and V's tiles a row, the
# rows R a job, the keys KB a block, the window, the broadcast's length.
DT, DV, R, KB, WINDOW, LENGTH = 2, 2, 13, 12, 2, 10
SUB = 2  # key tiles a step (the hand's 3 cuts t1's 4 keys only with a mask)
QB = R * DT  # a job's block of Q
STATE_PAGES, O_PAGES = R, R * DV  # init's pages: R states, then R rows of O
SCALE = 1.0 / math.sqrt(64.0)
LOG2E = 1.0 / math.log(2.0)


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class Job(km.Entry):
    """A job of the table (attention.h kJob*): first Q, K, V and O pages, its rows, the
    first row in its head, key tiles, the row step."""

    q: int
    k: int
    v: int
    o: int
    rows: int
    start: int
    keys: int
    stride: int


@km.tile_function(fidelity="HiFi2")
def score(q, kt):
    return q @ kt


@km.tile_function(fidelity="HiFi2")
def weigh(p, v):
    return p @ v


@km.tile_function
def softmax_step(row, state):
    x = row * (SCALE * LOG2E)
    m_old, l_old = state[0], state[1]
    m_new = km.lagging_max(m_old, km.reduce_max(x, exact=True), 0.0)
    alpha = km.exp2(m_old - m_new, error=2e-3, over=(-math.inf, 0.0))
    p = km.exp2(x - m_new.col(), error=2e-3, over=(-math.inf, 0.0))
    l_new = alpha * l_old + km.reduce_sum(p, exact=True)
    return p, (m_new, l_new, alpha)


@km.tile_function
def rescale(row, state):
    return row * state[2].col(), (state[0], state[1], state[2])


@km.tile_function
def norm(row, state):
    return row / state[1].col(), (state[0], state[1], state[2])


class Attention(km.Program):
    """attend: each core's job (one a core), Q's rows against the job's K / V blocks
    from its band's broadcast; each row's softmax state and O row kept in L1 across
    the blocks, O written after the last."""

    state: km.L1[km.fp32, R, 32, 32]
    o: km.L1[km.fp32, R, DV, 32, 32]
    p: km.L1[km.bf16, SUB, 32, 32]

    @km.handler
    def attend(
        self, q: km.Tensor, k: km.Tensor, v: km.Tensor, o: km.Tensor,
        tables: km.Tensor, init: km.Tensor,
    ):  # fmt: skip
        self.state.fill(km.Stream(init.pages(0, STATE_PAGES)))
        self.o.fill(km.Stream(init.pages(STATE_PAGES, O_PAGES)))
        down = km.Broadcast(
            self, along="column", length=LENGTH, mover="A", window=WINDOW
        )
        jobs = tables.pages(self.rank, 1).entries(Job, first=8, size=8, count=0)
        for job in jobs:
            r0 = job.q // DT
            qs = km.Stream(
                q.tiles[r0 : min(r0 + R * job.stride, r0 + job.rows * job.stride)
                        : job.stride, 0:DT],
                block=QB,
            )  # fmt: skip
            k0 = job.k // DT
            v0 = job.v // DV
            for b in range((job.keys + KB - 1) // KB):
                kr = k0 + b * KB
                vr = v0 + b * KB
                if down.sender():
                    kv = down.cast(
                        k.tiles[kr : min(kr + KB, k0 + job.keys), 0:DT].T,
                        v.tiles[vr : min(vr + KB, v0 + job.keys), 0:DV],
                    )
                else:
                    kv = down.arrive()
                for r in range(job.rows):
                    for c in range(min(KB, job.keys - b * KB) // SUB):
                        s = km.tile.zeros(km.fp32, SUB)
                        s += score(qs[r], kv[0][:, c * SUB : c * SUB + SUB])
                        self.p.write(softmax_step(s, self.state[r]))
                        acc = rescale(self.o[r], self.state[r])
                        acc += weigh(self.p, kv[1][c * SUB : c * SUB + SUB, :])
                        self.o[r].write(acc)
            for r in range(job.rows):
                last = norm(self.o[r], self.state[r])
                o.pages(job.o + r * job.stride * DV, DV).write(last)


def inputs(shape, seed):
    """The call's dims, whether causal, and Q, K, V (the project's run's)."""
    batches, heads, kv_heads, sequence_length, head_dim, causal = pl.SHAPES[shape]
    sizes = (batches, heads, kv_heads, sequence_length, sequence_length, head_dim)
    call = pl.dims(*sizes, head_dim, causal)
    q, k, v = sdpa.inputs(*sizes, head_dim, seed)
    return call, causal, q, k, v


def wanted(q, k, v, causal):
    """O in float64, [batches heads rows, value dim] (O's tensor order)."""
    group = q.shape[1] // k.shape[1]
    rows = np.arange(q.shape[2])
    return np.concatenate(
        [
            sdpa.attention(q[b, h], k[b, h // group], v[b, h // group], rows, causal)
            for b in range(q.shape[0])
            for h in range(q.shape[1])
        ]
    )


def errors(got, want):
    diff = np.asarray(got, np.float64) - want
    return {
        "rms_rel": float(np.sqrt((diff**2).mean() / (want**2).mean())),
        "max_abs": float(np.abs(diff).max()),
    }


def init_values():
    """The fills' pages: R zero states (m 0, l 0, the rescale 0), R zero rows of O."""
    return np.zeros((32, 32 * (STATE_PAGES + O_PAGES)), np.float32)


def checked_plan(call, grid):
    plan = pl.make(call, grid)
    made = (plan.r, plan.kb, plan.window, plan.length, call.dt, call.dvt)
    if made != (R, KB, WINDOW, LENGTH, DT, DV):
        raise SystemExit(f"plan {made}: the class is of {(R, KB, WINDOW, LENGTH)}")
    if max(len(jobs) for jobs in plan.jobs) > 1:
        raise SystemExit("the class keeps one job's states a core; the plan gives more")
    return plan


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
    """Mock device: the class built (its kernel text)."""
    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / DESCRIPTOR)
    km_session.chip_holder = lambda card, path=None: None  # a mock device has no chip
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    km_session.Session.load = build_only()
    call, _, q, _, v = inputs(args.shape, args.seed)
    with km.Session(log=log) as session:
        plan = checked_plan(call, session.nocs().grid)
        zeros = np.zeros((*q.shape[:-1], v.shape[-1]), np.float32)
        tensors = {
            "q": session.empty((q.size // 64, 64), "bfloat16", "tile"),
            "k": session.empty((32 * call.skt * call.hk, 64), "bfloat16", "tile"),
            "v": session.empty((32 * call.skt * call.hk, 64), "bfloat16", "tile"),
            "o": session.empty(zeros.reshape(-1, zeros.shape[-1]).shape, "bfloat16",
                               "tile"),
            "tables": session.empty(pl.tables(plan).shape, "uint32"),
            "init": session.empty(init_values().shape, "float32", "tile"),
        }  # fmt: skip
        program = Attention(session=session, log=log, **tensors)
        session.load(program)
        print(json.dumps({"class_text": program.sizes.text}), flush=True)
    return 0


def hand_attention(session, plan, tensors):
    """The hand program loaded (card.Attention) and its kernel text by kernel."""
    built, load = [], session.load

    def kept(program, **options):
        built.append(program)
        return load(program, **options)

    session.load = kept
    try:
        hand = card.Attention(session, plan, tensors, SCALE)
    finally:
        session.load = load
    return hand, built[0].sizes.text


def run_card(args):
    """One card: the hand's commands, then the class's on the same Q, K, V (its own
    O); both O's against float64 and each other."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    call, causal, q, k, v = inputs(args.shape, args.seed)
    report = {"shape": args.shape, "sub": SUB}
    with km.Session(log=log) as session:
        plan = checked_plan(call, session.nocs().grid)
        report["plan"] = plan.summary()
        zeros = np.zeros((*q.shape[:-1], v.shape[-1]), np.float32)
        tensors = {
            "q": card.bf16_tensor(session, q),
            "k": card.bf16_tensor(session, k),
            "v": card.bf16_tensor(session, v),
            "o": card.bf16_tensor(session, zeros),
            "tables": card.table_tensor(session, pl.tables(plan)),
        }
        hand, report["hand_text"] = hand_attention(session, plan, tensors)
        _, hand_us = hand.calls(args.calls)
        report["hand_run"] = hand.stop()
        hand_o = tensors["o"].read()
        class_o = card.bf16_tensor(session, zeros)
        init = session.tensor(init_values(), "float32", "tile")
        program = Attention(
            q=tensors["q"], k=tensors["k"], v=tensors["v"], o=class_o,
            tables=tensors["tables"], init=init, session=session, log=log,
        )  # fmt: skip
        loaded = session.load(program)
        words = (tensors["q"], tensors["k"], tensors["v"], class_o,
                 tensors["tables"], init)  # fmt: skip
        try:
            handles = [
                loaded.submit(Attention.attend, (0, len(plan.jobs)), args=words,
                              exclusive=True)
                for _ in range(args.calls)
            ]  # fmt: skip
            events = loaded.wait(handles, timeout=60.0)
        finally:
            loaded.stop()
        kinds = sorted({event.kind for event in events})
        report["class_events"] = kinds
        report["class_text"] = program.sizes.text
        made_o = class_o.read()
    want = wanted(q, k, v, causal)
    hand_o = hand_o.reshape(want.shape).astype(np.float64)
    made_o = made_o.reshape(want.shape).astype(np.float64)
    (HERE / "build").mkdir(exist_ok=True)  # the raw outputs, beside the builds
    np.savez(HERE / "build" / f"{args.shape}_o.npz", hand=hand_o, made=made_o)
    report["hand_vs_float64"] = errors(hand_o, want)
    report["class_vs_float64"] = errors(made_o, want)
    report["class_vs_hand"] = errors(made_o, hand_o)
    report["hand_us"] = round(float(np.median(hand_us)), 2)
    if kinds == ["done"]:
        report["class_us"] = round(
            float(np.median([e.microseconds for e in events])), 2
        )
        report["class_us_range"] = [
            round(float(min(e.microseconds for e in events)), 2),
            round(float(max(e.microseconds for e in events)), 2),
        ]
    report["ok"] = (
        kinds == ["done"]
        and report["hand_run"]["ok"]
        and report["class_vs_float64"]["rms_rel"] < 2e-2
    )
    print(json.dumps(report), flush=True)
    return 0 if report["ok"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("build", "card"))
    parser.add_argument("--shape", choices=("t1",), default="t1")
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    sys.exit(build(args) if args.mode == "build" else run_card(args))


if __name__ == "__main__":
    main()
