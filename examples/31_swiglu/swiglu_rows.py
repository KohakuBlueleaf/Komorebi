"""The README's SwiGLU layer over 128 rows of x (4 tile rows) as one chain: each root
writes its columns of every row, and the chain's records are stated in tile columns
(docs/reference/chain.md, "More than one tile row"). The driver runs the three calls as
commands and as the chain, compares every word of h, g, hidden and y, checks y against
float64, times both and reads the chain's stamps' terms. `build` builds it on a mock
device; `card` runs it."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import power
from komorebi.dsl import chain_stamps


@km.tile_function
def gate(h, g):  # the vector unit
    return km.silu(h) * g


class SwiGLURows(km.Program):
    """x is [128, D], W1 and W3 are [D, H], W2 is [H, D]; every size in tiles of 32."""

    ROWS = km.constant(4)  # x's tile rows: 128 rows
    D = km.constant(32)  # x's and y's width: 1,024
    H = km.constant(96)  # the hidden width: 3,072
    UP = km.constant(4)  # the hidden's columns an up core makes (two sums in DST)
    WIDTH = km.constant(8)  # y's columns a down core makes
    CHUNK = km.constant(2)  # K tiles a block
    U_BLOCK = CHUNK * UP
    W_BLOCK = CHUNK * WIDTH

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):  # the matrix unit
        return x @ w

    @km.handler
    def up(
        self, x: km.Tensor, w1: km.Tensor, w3: km.Tensor, h: km.Tensor, g: km.Tensor
    ):
        first = self.rank * self.UP  # this core's columns of h and g, every row
        for r in range(self.ROWS):
            acc = km.tile.zeros(km.fp32, self.UP)
            for xb, wb in zip(
                km.Stream(x.tiles[r : r + 1, 0 : self.D], block=self.CHUNK),
                km.Stream(
                    w1.tiles[0 : self.D, first : first + self.UP], block=self.U_BLOCK
                ),
            ):
                acc += self.project(xb, wb)
            h.tiles[r : r + 1, first : first + self.UP].write(acc)
            acc = km.tile.zeros(km.fp32, self.UP)
            for xb, wb in zip(
                km.Stream(x.tiles[r : r + 1, 0 : self.D], block=self.CHUNK),
                km.Stream(
                    w3.tiles[0 : self.D, first : first + self.UP], block=self.U_BLOCK
                ),
            ):
                acc += self.project(xb, wb)
            g.tiles[r : r + 1, first : first + self.UP].write(acc)

    @km.handler
    def mix(self, h: km.Tensor, g: km.Tensor, hidden: km.Tensor):
        first = self.rank * self.UP  # the columns up's core of this rank wrote
        out = km.Stream(
            hidden.tiles[0 : self.ROWS, first : first + self.UP],
            block=self.UP,
            write=True,
        )
        for hb, gb in zip(
            km.Stream(h.tiles[0 : self.ROWS, first : first + self.UP], block=self.UP),
            km.Stream(g.tiles[0 : self.ROWS, first : first + self.UP], block=self.UP),
        ):
            out.push(gate(hb, gb))

    @km.handler
    def down(self, hidden: km.Tensor, w2: km.Tensor, y: km.Tensor):
        first = self.rank * self.WIDTH  # this core's columns of y, every row
        for r in range(self.ROWS):
            acc = km.tile.zeros(km.fp32, self.WIDTH)
            for hb, wb in zip(
                km.Stream(hidden.tiles[r : r + 1, 0 : self.H], block=self.CHUNK),
                km.Stream(
                    w2.tiles[0 : self.H, first : first + self.WIDTH], block=self.W_BLOCK
                ),
            ):
                acc += self.project(hb, wb)
            y.tiles[r : r + 1, first : first + self.WIDTH].write(acc)

    @km.chain
    def forward(self, x, w1, w3, w2, h, g, hidden, y):
        self.up(x, w1, w3, h, g, cores=(0, 24))  # 96 columns of h and g, 4 a core
        self.mix(h, g, hidden, cores=(0, 24))  # reads the columns its rank's up wrote
        self.down(hidden, w2, y, cores=(0, 4))  # 32 columns of y, every row of hidden


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DESCRIPTOR = (
    "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"
)
JIT_CACHE = ROOT / "build" / "ttcache-formats"  # the jobs' cache: mock's and card's
NAMES = ("x", "w1", "w3", "w2", "h", "g", "hidden", "y")
OUTPUTS = ("h", "g", "hidden", "y")
CORES = 24  # the chain's largest call: its stamps' rows
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
    x = rng.uniform(-1, 1, (128, d))
    w1, w3 = (rng.uniform(-1, 1, (d, h)) / np.sqrt(d) for _ in range(2))
    w2 = rng.uniform(-1, 1, (h, d)) / np.sqrt(h)
    return {n: bf16(m) for n, m in zip(("x", "w1", "w3", "w2"), (x, w1, w3, w2))}


def wanted(data, hidden):
    """float64: the hidden of the stored inputs, y of the card's own hidden."""
    x, w1, w3, w2 = (data[n].astype(np.float64) for n in ("x", "w1", "w3", "w2"))
    p, q = x @ w1, x @ w3
    return p / (1 + np.exp(-p)) * q, hidden.astype(np.float64) @ w2


def errors(got, want):
    """The relative RMS error and the largest error over the largest value."""
    got = got.astype(np.float64)
    return {
        "rms_rel": float(np.sqrt(((got - want) ** 2).mean() / (want**2).mean())),
        "max_rel": float(np.abs(got - want).max() / np.abs(want).max()),
    }


def tensors_of(session, data, empty):
    """x, w1, w3, w2, h, g and hidden in bf16, y float32: tiles."""
    out = {}
    for name, values in data.items():
        if empty:
            out[name] = session.empty(values.shape, "bfloat16", "tile")
        else:
            out[name] = session.tensor(values, "bfloat16", "tile")
    for name in ("h", "g", "hidden"):
        out[name] = session.empty((128, 96 * 32), "bfloat16", "tile")
    out["y"] = session.empty((128, 32 * 32), "float32", "tile")
    return out


def cleared(t):
    for name in OUTPUTS:
        t[name].write(np.zeros(t[name].shape, np.float32))


def words(t):
    """Each output's words as read (uint32 of its float32 values)."""
    return {
        n: np.ascontiguousarray(t[n].read(), np.float32).view(np.uint32)
        for n in OUTPUTS
    }


def done(event, what):
    if event.kind != "done":
        raise RuntimeError(f"{what}: {event}")
    return event.microseconds


def steps(run, t):
    """The three calls as three commands: their card us summed."""
    x, w1, w3, w2, h, g, hidden, y = (t[n] for n in NAMES)
    us = done(
        run.wait(run.submit(SwiGLURows.up, (0, 24), args=(x, w1, w3, h, g))), "up"
    )
    us += done(
        run.wait(run.submit(SwiGLURows.mix, (0, 24), args=(h, g, hidden))), "mix"
    )
    return us + done(
        run.wait(run.submit(SwiGLURows.down, (0, 4), args=(hidden, w2, y))), "down"
    )


def form(session, program, t, chained, runs, deadline):
    """A load, the form once and `runs` times more: (its outputs' words after the
    first, the timed us, the run's error)."""
    cleared(t)
    run = session.load(program, wait=deadline, deadline=deadline)
    try:
        if chained:
            chain = run.chain(program.forward(*(t[n] for n in NAMES)))
            once = lambda: done(run.wait(run.submit(chain)), "chain")  # noqa: E731
        else:
            once = lambda: steps(run, t)  # noqa: E731
        once()
        got = words(t)
        times = [once() for _ in range(runs)]
    finally:
        run.stop()
    return got, times, run.result()["error_name"]


def stamped(session, t, count, deadline):
    """The class made with the chain's stamps: a chain once (its stamps read and
    dropped), then `count` chains: each record's terms (us)."""
    program = SwiGLURows(log=log, session=session, chain_stamps=True, **t)
    run = session.load(program, wait=deadline, deadline=deadline)
    rows_tensor = chain_stamps.tensor_for(session, CORES)
    try:
        chain = run.chain(program.forward(*(t[n] for n in NAMES)))
        for number in (1, count):
            for _ in range(number):
                done(run.wait(run.submit(chain)), "stamped chain")
            rows = chain_stamps.read(session, run, program, rows_tensor, (0, CORES))
        terms = chain_stamps.terms(rows, power.aiclk_mhz(), 3)
    finally:
        run.stop()
    return terms, run.result()["error_name"]


def card_result(session, runs=50, count=20, deadline=0.5):
    """On an open session: the commands and the chain (every word), their times, the
    chain's stamps' terms."""
    data = inputs()
    t = tensors_of(session, data, empty=False)
    program = SwiGLURows(log=log, session=session, **t)
    result = {"shape": [128, 1024, 3072], "cores": {"up": 24, "mix": 24, "down": 4}}
    got, us, kernels = {}, {}, []
    for name, chained in (("steps", False), ("chain", True)):
        got[name], times, error = form(session, program, t, chained, runs, deadline)
        us[name] = round(float(np.median(times)), 2)
        kernels.append(error)
    result["words_differ"] = {
        n: int(np.sum(got["chain"][n] != got["steps"][n])) for n in OUTPUTS
    }
    hidden = got["chain"]["hidden"].view(np.float32)
    want_hidden, want_y = wanted(data, hidden)
    result["hidden_err"] = errors(hidden, want_hidden)
    result["y_err"] = errors(got["chain"]["y"].view(np.float32), want_y)
    result["median_us"] = us
    result["terms"], error = stamped(session, t, count, deadline)
    result["kernel"] = [*kernels, error]
    result["ok"] = all(e == "none" for e in result["kernel"]) and not any(
        result["words_differ"].values()
    )
    return result


def build():
    """Mock device: the class built (the jobs' cache)."""
    # here, not at the top: a mock device's settings before the session's import
    from komorebi import native  # noqa: PLC0415
    from komorebi import session as km_session  # noqa: PLC0415

    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / DESCRIPTOR)
    km_session.chip_holder = lambda card, path=None: None  # a mock device has no chip
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    with km.Session(log=log) as session:
        t = tensors_of(session, inputs(), empty=True)
        SwiGLURows(log=log, session=session, **t)
    return 0


def card():
    os.environ.setdefault("TT_METAL_CACHE", str(JIT_CACHE))
    with km.Session(log=log) as session:
        result = card_result(session)
        power.clocked(result)
    print(json.dumps(result), flush=True)
    return 0 if result["ok"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=["build", "card"])
    args = parser.parse_args()
    return build() if args.what == "build" else card()


if __name__ == "__main__":
    sys.exit(main())
