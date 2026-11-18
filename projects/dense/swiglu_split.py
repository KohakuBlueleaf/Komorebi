"""SwiGLU, y = (silu(x W1) * (x W3)) W2, as a class whose call is one chain command:
`up` on 64 cores (a row tile of x, two hidden columns, W1 and W3 held in fields) writes
the gate, `down` on 8 (a row tile, W2 held) reads the row's hidden. Checked against
float64 on every row; prints one JSON line; exits 1 when the check fails."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi.bench import power  # noqa: E402
from reference import dense as ref  # noqa: E402

# Bound of the check on these inputs (SWIGLU.md, "The check's bound").
RMS_BOUND, MAX_BOUND = 2e-2, 3e-2
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function
def gate(h, g):  # the vector unit
    return km.silu(h) * g


class SwiGLUSplit(km.Program):
    """x is [32 ROWS, 32 D], W1 and W3 [32 D, 32 H], W2 [32 H, 32 D]; SPLIT up cores a
    row tile of x, an up core H / SPLIT hidden columns, a down core a row tile."""

    ROWS = km.constant(8)
    D = km.constant(8)
    H = km.constant(16)
    SPLIT = km.constant(8)
    HW = H // SPLIT  # the hidden's columns a core makes
    w1: km.L1[km.bf16, D, HW, 32, 32]
    w3: km.L1[km.bf16, D, HW, 32, 32]
    YW = D // 2  # y's columns a DST sum holds
    w2a: km.L1[km.bf16, H, YW, 32, 32]  # a down core's: W2's left columns
    w2b: km.L1[km.bf16, H, YW, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):  # the matrix unit
        return x @ w

    @km.handler
    def load(self, w1: km.Tensor, w3: km.Tensor, w2: km.Tensor):
        first = self.rank % self.SPLIT * self.HW  # this core's columns of the hidden
        self.w1.fill(km.Stream(w1.tiles[0 : self.D, first : first + self.HW]))
        self.w3.fill(km.Stream(w3.tiles[0 : self.D, first : first + self.HW]))
        self.w2a.fill(km.Stream(w2.tiles[0 : self.H, 0 : self.YW]))
        self.w2b.fill(km.Stream(w2.tiles[0 : self.H, self.YW : self.D]))

    @km.handler
    def up(self, x: km.Tensor, hidden: km.Tensor):
        row = self.rank // self.SPLIT  # this group's row tile of x
        h = km.tile.zeros(km.fp32, self.HW)
        for xb in km.Stream(x.tiles[row : row + 1, 0 : self.D], block=self.D):
            h += self.project(xb, self.w1)
        g = km.tile.zeros(km.fp32, self.HW)
        for xb in km.Stream(x.tiles[row : row + 1, 0 : self.D], block=self.D):
            g += self.project(xb, self.w3)
        first = self.rank * self.HW  # the hidden as one row of tiles, a core's in turn
        hidden.tiles[0:1, first : first + self.HW].write(gate(h, g))

    @km.handler
    def down(self, hidden: km.Tensor, y: km.Tensor):
        first = self.rank * self.H  # this row's hidden, from its group's up cores
        left = km.tile.zeros(km.fp32, self.YW)
        for hb in km.Stream(hidden.tiles[0:1, first : first + self.H], block=self.H):
            left += self.project(hb, self.w2a)
        y.pages(self.rank * self.D, self.YW).write(left)  # y's row tile, in pages
        right = km.tile.zeros(km.fp32, self.YW)
        for hb in km.Stream(hidden.tiles[0:1, first : first + self.H], block=self.H):
            right += self.project(hb, self.w2b)
        y.pages(self.rank * self.D + self.YW, self.YW).write(right)

    @km.chain
    def forward(self, x, hidden, y):
        cores = self.ROWS * self.SPLIT
        self.up(x, hidden, cores=(0, cores))
        self.down(hidden, y, cores=(0, self.ROWS))


class Layer:
    """The class loaded with W1, W3 and W2 filled once; calls of its chain."""

    def __init__(self, session, program, tensors):
        self.session, self.tensors = session, tensors
        self.group = (0, SwiGLUSplit.ROWS.default * SwiGLUSplit.SPLIT.default)
        self.run = session.load(program)
        fill = (tensors["w1"], tensors["w3"], tensors["w2"])
        event = self.run.wait(self.run.submit(SwiGLUSplit.load, self.group, args=fill))
        if event.kind != "done":
            raise km.KomorebiError(f"load: {event}")
        self.load_microseconds = event.microseconds
        names = ("x", "hidden", "y")
        self.chain = self.run.chain(program.forward(*(tensors[n] for n in names)))
        self.handler, self.group = self.chain.handler, self.chain.group
        self.words = [[int(word) for word in self.chain.args({}, 0)]]

    def calls(self, count, timeout=60.0):
        """`count` calls queued at once and waited once: host seconds and each call's
        card microseconds [count, 1]."""
        rows = np.tile(np.array(self.words, np.int64), (count, 1))
        started = time.perf_counter()
        handles = self.run.submit_many(
            self.handler, self.group[0], self.group[1], rows, exclusive=True
        )
        events = self.run.wait(list(handles), timeout=timeout)
        host_s = time.perf_counter() - started
        unfinished = [event for event in events if event.kind != "done"]
        if unfinished:
            raise km.KomorebiError(f"{len(unfinished)} calls not done: {unfinished[0]}")
        return host_s, np.array([[event.microseconds] for event in events])


def timed(layer, args, index, report):
    """Host-queued calls: one at a time, then 200 back to back."""
    single = [layer.calls(1)[1].sum() for _ in range(20)]
    host_s, _ = layer.calls(200)
    report["single_card_us"] = float(np.median(single))
    report["loop200_us_per_call"] = host_s * 1e6 / 200


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--calls", type=int, default=2, help="calls before the check")
    parser.add_argument("--seconds", type=float, default=0.0, help="then timed calls")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", default=None)
    return parser.parse_args()


def main():
    args = parse()
    rows, d, h = (
        SwiGLUSplit.ROWS.default * 32,
        SwiGLUSplit.D.default * 32,
        SwiGLUSplit.H.default * 32,
    )
    report = {"m": rows, "d": d, "h": h, "split": SwiGLUSplit.SPLIT.default}
    x, w1, w3, w2 = ref.swiglu_inputs(rows, d, h, args.seed)
    index = power.visible_card()
    with km.Session(log=log) as session:
        tensors = {
            name: session.tensor(value, "bfloat16", "tile")
            for name, value in zip(("x", "w1", "w3", "w2"), (x, w1, w3, w2))
        }
        tensors["hidden"] = session.empty((32, rows // 32 * h), "bfloat16", "tile")
        tensors["y"] = session.empty((rows, d), "float32", "tile")
        program = SwiGLUSplit(log=log, session=session, **tensors)
        layer = Layer(session, program, tensors)
        report["load_microseconds"] = layer.load_microseconds
        host_s, _ = layer.calls(args.calls)
        report["first_calls_microseconds"] = round(host_s * 1e6 / args.calls, 3)
        log(f"{args.calls} calls in {host_s:.4f} s")
        if args.seconds:
            timed(layer, args, index, report)
        layer.run.stop()
        report["kernel"] = layer.run.result()["error_name"]
        got = tensors["y"].read()
    report["check"] = ref.errors(got, ref.swiglu(x, w1, w3, w2))
    check = report["check"]
    check["ok"] = check["rms_rel"] <= RMS_BOUND and check["max_rel"] <= MAX_BOUND
    report["ok"] = check["ok"] and report["kernel"] == "none"
    report["load_avg"] = Path("/proc/loadavg").read_text().split()[:3]
    print(json.dumps(power.clocked(report), default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
