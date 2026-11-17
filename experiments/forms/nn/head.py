"""The head of a language model on one token row, against numpy: logits = x W in
chunks of 3 tiles with a running per-row maximum by the step form (H1), then the
largest logit of row 0 and its index by C++ bodies on the pack CPU, a core's tiles
each, and the leader's combine over the cores' pages; a second row has a tie."""

import argparse
import sys

import numpy as np
from forms import error, finish, log, random, read, tiles, timed, words

import komorebi as km

E = 4  # hidden tiles (128)
N = 3  # logit tiles a chunk (the step form's row: 1 to 3 tiles of DST)
CHUNKS = 4
W_BLOCK = 2 * N  # W tiles a stream block: 2 rows of a chunk (a block is a constant)
V = N * CHUNKS  # logit tiles a core
SLOT = (0, 2, 32, 34)  # a state slot's DST rows (tile.md, "Step form")
NEGATIVE = -1e30  # the state's maximum before the first chunk


@km.tile_function
def max_step(row, state):
    """The step form with a lag of 0: the running maximum of a row of chunks."""
    m_old = state[0]
    m_new = km.lagging_max(m_old, km.reduce_max(row, exact=True), 0.0)
    return row, (m_new, m_old)


class HeadMax(km.Program):
    """H1: core r takes logit tiles r V to r V + V - 1 in CHUNKS chunks of N."""

    state: km.L1[km.fp32, 1, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def product(a, b):
        return a @ b

    @km.handler
    def head(self, x: km.Tensor, w: km.Tensor, logits: km.Tensor, st: km.Tensor):
        self.state.fill(km.Stream(st.pages(self.rank, 1)))
        for c in range(CHUNKS):
            v = self.rank * V + c * N
            acc = km.tile.zeros(km.fp32, N)
            for xb, wb in zip(
                km.Stream(x.tiles[0:1, 0:E], block=2),
                km.Stream(w.tiles[0:E, v : v + N], block=W_BLOCK),
            ):
                acc += self.product(xb, wb)
            logits.pages(v, N).write(max_step(acc, self.state[0]))
        st.pages(self.rank, 1).write(self.state)


SETUP = """
float scan_best = -3.4e38f;
uint32_t scan_index = 0;
uint32_t scan_tile = first;
"""

SCAN = """
{
    using Floats = volatile tt_l1_ptr float*;
    Floats values = reinterpret_cast<Floats>(tile);
    for (uint32_t i = 0; i < 16; ++i) {
        if (values[i] > scan_best) {
            scan_best = values[i];
            scan_index = scan_tile * 32 + i;
        }
        if (values[256 + i] > scan_best) {
            scan_best = values[256 + i];
            scan_index = scan_tile * 32 + 16 + i;
        }
    }
    scan_tile += 1;
}
"""

GIVE = """
using Words = volatile tt_l1_ptr uint32_t*;
Words words = reinterpret_cast<Words>(out);
words[0] = __builtin_bit_cast(uint32_t, scan_best);
words[1] = scan_index;
"""

COMBINE = """
{
    using Words = volatile tt_l1_ptr uint32_t*;
    Words words = reinterpret_cast<Words>(tile);
    float v = __builtin_bit_cast(float, words[0]);
    if (v > scan_best) { scan_best = v; scan_index = words[1]; }
}
"""


class Argmax(km.Program):
    """Row 0's largest value and its first index: each core scans its tiles (face 0
    words 0 to 15, face 1 words 256 to 271 of a float32 tile) into one page of
    (value, index); the leader then keeps the largest of the cores' pages in order."""

    @km.handler
    def scan(self, logits: km.Tensor, result: km.Tensor, tiles: int):
        first = self.rank * tiles
        km.cxx(SETUP, takes=[first], on="pack")
        for tile in km.Stream(logits.pages(first, tiles), block=4):
            km.cxx(SCAN, takes=[tile], on="pack")
        out = km.Stream(result.pages(self.rank, 1), write=True, block=1)
        km.cxx(GIVE, takes=[out], on="pack")

    @km.handler
    def combine(self, result: km.Tensor, final: km.Tensor, pages: int):
        if self.rank == self.leader:
            first = 0
            km.cxx(SETUP, takes=[first], on="pack")
            for tile in km.Stream(result.pages(0, pages), block=4):
                km.cxx(COMBINE, takes=[tile], on="pack")
            out = km.Stream(final.pages(0, 1), write=True, block=1)
            km.cxx(GIVE, takes=[out], on="pack")


def state_slot(page, k):
    """Slot k's 32 per-row values of a state page (fp32 words in face order)."""
    dst = page.view(np.float32).reshape(64, 16).astype(np.float64)
    lanes, out = np.arange(32), np.zeros(32)
    for h in range(2):
        for q in range(4):
            address = 16 * h + 4 * q + SLOT[k]
            rows = (address & ~3) + lanes // 8
            columns = 2 * (lanes % 8) + (1 if address & 2 else 0)
            out[16 * h + 4 * q : 16 * h + 4 * q + 4] = dst[rows, columns][::8]
    return out


def tied(rng, columns, places):
    """A [32, columns] float32 matrix whose row 0 has its largest value, 4.0, at each
    of `places` (the first is the argmax), the other values in (-1, 1)."""
    m = rng.uniform(-1.0, 1.0, (32, columns)).astype(np.float32)
    m[0, places] = 4.0
    return m.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cores = args.cores
    columns = cores * V * 32
    x = random(rng, (32, E * 32))
    w = random(rng, (E * 32, columns), 1.0 / np.sqrt(E * 32))
    places = [2 * V * 32 + 40, 2 * V * 32 + 300, 5 * V * 32 + 7]  # core 2 twice, core 5
    tie = tied(rng, columns, [p for p in places if p < columns])
    report = {"cores": cores, "vocabulary": columns}
    with km.Session(log=log) as session:
        tx, tw = tiles(session, x), tiles(session, w)
        logits = tiles(session, np.zeros((32, columns)), "float32")
        st = tiles(session, np.full((32, cores * 32), NEGATIVE), "float32")
        ties = tiles(session, tie, "float32")
        result = tiles(session, np.zeros((cores * 32, 32)), "float32")
        final = tiles(session, np.zeros((32, 32)), "float32")
        program = HeadMax(x=tx, w=tw, logits=logits, st=st, session=session, log=log)
        run = session.load(program)
        try:
            report["head_us"] = timed(
                run, HeadMax.head, (0, cores), (tx, tw, logits, st)
            )
        finally:
            run.stop()
        got_logits, got_state = read(session, logits), words(session, st)
        program = Argmax(
            logits=logits, result=result, final=final, session=session, log=log
        )
        run = session.load(program)
        picks = {}
        try:
            for name, source in (("argmax", logits), ("argmax_tie", ties)):
                us = timed(run, Argmax.scan, (0, cores), (source, result, V))
                us2 = timed(run, Argmax.combine, (0, cores), (result, final, cores))
                (page,) = words(session, final)[:1]
                picks[name] = (
                    int(page[1]),
                    float(page[:1].view(np.float32)[0]),
                    [us, us2],
                )
        finally:
            run.stop()
    want = x @ w
    report["logits"] = error(got_logits, want)
    m = np.array([state_slot(got_state[c], 0) for c in range(cores)])  # [core, row]
    blocks = got_logits.reshape(32, cores, V * 32)
    report["running_max"] = error(m.T, blocks.max(axis=2))  # against what was written
    report["running_max_vs_numpy"] = error(m.T, want.reshape(32, cores, -1).max(axis=2))
    for name, values in (("argmax", got_logits[0]), ("argmax_tie", tie[0])):
        index, value, us = picks[name]
        first = int(np.argmax(values))  # numpy's argmax: the first of a tie
        report[name] = {"index": index, "want": first, "value": value,
                        "want_value": float(values[first]), "us": us}  # fmt: skip
        log(f"{name}: {report[name]}")
    log(f"logits {report['logits']}, running max {report['running_max']}")
    report["argmax_ok"] = all(
        report[n]["index"] == report[n]["want"]
        and report[n]["value"] == report[n]["want_value"]
        for n in ("argmax", "argmax_tie")
    )
    report["argmax_error"] = {"max": 0.0 if report["argmax_ok"] else 1.0}
    finish(
        report,
        [("logits", "rms", 0.01), ("running_max", "max", 0.0),
         ("running_max_vs_numpy", "rms", 0.01), ("argmax_error", "max", 0.0)],
    )  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
