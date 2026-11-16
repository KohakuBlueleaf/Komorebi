"""swiglu_rows.py's layer with the gate inside the handler that makes the sums: up_gate
holds h's and g's sums in DST together and writes gate(h, g), so the chain is two calls
(no mix, no h and g in DRAM); on the card beside the three calls, every word of hidden
and y against the commands', both against float64, timed. `build` / `card`."""

import argparse
import json
import os
import sys

import numpy as np
import swiglu_rows as rows
from swiglu_rows import gate, log

import komorebi as km
from komorebi.bench import power


class SwiGLUGated(km.Program):
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
    def up_gate(self, x: km.Tensor, w1: km.Tensor, w3: km.Tensor, hidden: km.Tensor):
        first = self.rank * self.UP  # this core's columns of the hidden, every row
        for r in range(self.ROWS):
            h = km.tile.zeros(km.fp32, self.UP)
            for xb, wb in zip(
                km.Stream(x.tiles[r : r + 1, 0 : self.D], block=self.CHUNK),
                km.Stream(
                    w1.tiles[0 : self.D, first : first + self.UP], block=self.U_BLOCK
                ),
            ):
                h += self.project(xb, wb)
            g = km.tile.zeros(km.fp32, self.UP)  # DST 4 to 7, beside h's 0 to 3
            for xb, wb in zip(
                km.Stream(x.tiles[r : r + 1, 0 : self.D], block=self.CHUNK),
                km.Stream(
                    w3.tiles[0 : self.D, first : first + self.UP], block=self.U_BLOCK
                ),
            ):
                g += self.project(xb, wb)
            hidden.tiles[r : r + 1, first : first + self.UP].write(gate(h, g))

    down = km.handler(rows.SwiGLURows.down.function)

    @km.chain
    def forward(self, x, w1, w3, w2, hidden, y):
        self.up_gate(x, w1, w3, hidden, cores=(0, 24))  # 4 hidden columns a core
        self.down(hidden, w2, y, cores=(0, 4))  # 32 columns of y, every row


NAMES = ("x", "w1", "w3", "w2", "hidden", "y")


def gated_form(session, program, t, runs, deadline):
    """The gated chain once on a load (its words read after the stop), then a second
    load and `runs` chains: (the words, the times, the kernels' ends)."""
    rows.cleared(t)
    times, ends, got = None, [], None
    for count in (1, runs):
        run = session.load(program, wait=deadline, deadline=deadline)
        try:
            chain = run.chain(program.forward(*(t[n] for n in NAMES)))
            times = [
                rows.done(run.wait(run.submit(chain)), "gated chain")
                for _ in range(count)
            ]
        finally:
            run.stop()
        ends.append(run.result()["error_name"])
        if count == 1:
            got = rows.words(t)
    return got, times, ends


def where(gated, steps, columns):
    """Where gated's words differ from the steps': by 32-row tile, by block of
    `columns` (a core's), and the largest difference."""
    a, b = gated.view(np.float32), steps.view(np.float32)
    wrong = a != b
    return {
        "by_row_tile": wrong.reshape(-1, 32, a.shape[1]).sum(axis=(1, 2)).tolist(),
        "by_core": wrong.reshape(a.shape[0], -1, columns).sum(axis=(0, 2)).tolist(),
        "largest": float(np.abs(a.astype(np.float64) - b).max()),
    }


def card_result(session, runs=50, deadline=0.5, dump=None):
    data = rows.inputs()
    t = rows.tensors_of(session, data, empty=False)
    three = rows.SwiGLURows(log=log, session=session, **t)
    gated = SwiGLUGated(log=log, session=session, **t)
    result = {"shape": [128, 1024, 3072], "cores": {"up_gate": 24, "down": 4}}
    got, us, kernels = {}, {}, []
    for name, chained in (("steps", False), ("chain", True)):
        got[name], times, ends = rows.form(session, three, t, chained, runs, deadline)
        us[name] = round(float(np.median(times)), 2)
        kernels += ends
    got["gated"], times, ends = gated_form(session, gated, t, runs, deadline)
    us["gated"] = round(float(np.median(times)), 2)
    kernels += ends
    result["words_differ_from_steps"] = {
        n: int(np.sum(got["gated"][n] != got["steps"][n])) for n in ("hidden", "y")
    }
    result["words"] = {n: int(got["steps"][n].size) for n in ("hidden", "y")}
    result["where"] = {
        n: where(got["gated"][n], got["steps"][n], c)
        for n, c in (("hidden", 128), ("y", 256))
    }
    if dump:
        np.savez_compressed(
            dump, **{f"{f}_{n}": got[f][n] for f in got for n in ("hidden", "y")}
        )
    for name in ("steps", "gated"):
        hidden = got[name]["hidden"].view(np.float32)
        want_hidden, want_y = rows.wanted(data, hidden)
        result[f"{name}_hidden_err"] = rows.errors(hidden, want_hidden)
        result[f"{name}_y_err"] = rows.errors(got[name]["y"].view(np.float32), want_y)
    result["median_us"] = us
    result["kernel"] = kernels
    result["ok"] = all(e == "none" for e in kernels)
    return result


def build():
    """Mock device: the class built (the jobs' cache)."""
    # here, not at the top: a mock device's settings before the session's import
    from komorebi import session as km_session  # noqa: PLC0415
    from komorebi.device import native  # noqa: PLC0415

    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / rows.DESCRIPTOR)
    km_session.chip_holder = lambda card, path=None: None  # a mock device has no chip
    os.environ["TT_METAL_CACHE"] = str(rows.JIT_CACHE)
    with km.Session(log=log) as session:
        t = rows.tensors_of(session, rows.inputs(), empty=True)
        SwiGLUGated(log=log, session=session, **t)
    return 0


def card(dump=None):
    os.environ.setdefault("TT_METAL_CACHE", str(rows.JIT_CACHE))
    with km.Session(log=log) as session:
        result = card_result(session, dump=dump)
        power.clocked(result)
    print(json.dumps(result), flush=True)
    return 0 if result["ok"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=["build", "card"])
    parser.add_argument("--dump", help="card: every form's hidden and y to this .npz")
    args = parser.parse_args()
    return build() if args.what == "build" else card(args.dump)


if __name__ == "__main__":
    sys.exit(main())
