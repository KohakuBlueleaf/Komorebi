"""Tile row count (README.md): the RMS norm over rows of 2,048 values (64 tiles, read
in blocks of 8), the count of rows a core a command word under a stated most, run
with the counts 1, 7 and 64 from one build and one load, against float64; the rows
past the count left as they were. `card`, `made_up`."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.dsl.tile import row_operand

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
JIT_CACHE = ROOT / "build" / "ttcache-formats"  # the jobs' cache (job.sh reads it)
CORES, MOST, REPEAT = 4, 64, 3  # cores, rows a core at most, commands a count
COUNTS = (1, 7, 64)  # rows a core, one command word
EPS_RMS = 1e-6
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function
def rms_2048(row, weight):
    ms = km.reduce_sum(row * row, exact=True) * (1.0 / 2048)
    return row * km.rsqrt(ms + EPS_RMS).col() * weight.row()


class Counted(km.Program):
    """rms: `rows` rows of 64 tiles (at most MOST) from rank x MOST, normalized into
    the same pages of out with the weight field."""

    weight: km.L1[km.fp32, 32, 32, 32]

    @km.handler
    def rms(self, x: km.Tensor, w: km.Tensor, out: km.Tensor, rows: int):
        r0 = self.rank * MOST
        self.weight.fill(km.Stream(w.pages(0, 32)))
        results = km.Stream(
            out.tiles[r0 : min(r0 + MOST, r0 + rows), 0:64], write=True, block=8
        )
        for row in km.Stream(
            x.tiles[r0 : min(r0 + MOST, r0 + rows), 0:64], block=8, row=64
        ):
            results.push(rms_2048(row, self.weight))


def inputs(seed=17):
    """x: CORES x MOST tile rows of 2,048 values; w: 2,048 normals."""
    rng = np.random.default_rng(seed)
    x = (2.0 * rng.standard_normal((CORES * MOST * 32, 2048)) + 0.7).astype(np.float32)
    return x, rng.standard_normal(2048).astype(np.float32)


def wanted(x, w, count):
    """The float64 result where a core's first `count` rows are normalized, the
    rest as out started (zeros)."""
    x64 = x.astype(np.float64)
    want = np.zeros_like(x64)
    norm = x64 / np.sqrt((x64 * x64).mean(1, keepdims=True) + EPS_RMS) * w
    for core in range(CORES):
        rows = slice(core * MOST * 32, (core * MOST + count) * 32)
        want[rows] = norm[rows]
    return want


def errors(got, want):
    """The relative RMS error, the largest error over the RMS, the largest relative;
    `untouched`: whether the rows past the count stayed zero."""
    done = np.any(want != 0, axis=1)
    g, w = got[done], want[done]
    scale = np.sqrt((w**2).mean())
    return {
        "rms_rel": float(np.sqrt(((g - w) ** 2).mean()) / scale),
        "max_over_rms": float(np.abs(g - w).max() / scale),
        "max_rel": float((np.abs(g - w)[w != 0] / np.abs(w[w != 0])).max()),
        "untouched": bool(np.all(got[~done] == 0)),
    }


def weight_pages(w):
    """The weight field's pages: chunk c's per-column tile in page c."""
    pages = [row_operand(w[64 * c : 64 * c + 64], 2) for c in range(32)]
    return np.asarray(pages, np.float32).view(np.uint8).reshape(32, -1)


def check(rows):
    """The rows' shape and prints, as `card` writes them (no card: a made-up run)."""
    for row in rows:
        log(json.dumps(row))
    return {"rows": rows, "wall_s": round(time.monotonic() - STARTED, 1)}


def card(args):
    """One card, one load: each count REPEAT commands on CORES cores into its own
    out, which starts at zeros."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    x_host, w_host = inputs()
    rows = []
    with km.Session(log=log) as session:
        x = session.tensor(x_host, "float32", "tile")
        w = session.empty((32, 1024), "float32")
        session.write(w, weight_pages(w_host))
        zeros = np.zeros_like(x_host)
        outs = {n: session.tensor(zeros, "float32", "tile") for n in COUNTS}
        run = session.load(Counted(log=log, session=session, x=x, w=w, out=outs[1]))
        times = {}
        try:
            for n in COUNTS:
                times[n] = []
                for _ in range(REPEAT):
                    sent = run.submit(Counted.rms, (0, CORES), args=(x, w, outs[n], n))
                    event = run.wait(sent, timeout=30.0)
                    if event.kind != "done":
                        raise RuntimeError(f"count {n}: {event}")
                    times[n].append(event.microseconds)
                log(f"count {n}: {times[n]}")
        finally:
            run.stop()
        for n in COUNTS:
            got = outs[n].read().astype(np.float64)
            rows.append(
                {"count": n, **errors(got, wanted(x_host, w_host, n))}
                | {"microseconds": times[n]}
            )
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "tile_count.json", "w") as f:
        json.dump(check(rows), f)
    return 0


def made_up(args):
    """The host side after the card: each count's reference through `errors`."""
    x, w = inputs()
    rows = []
    for n in COUNTS:
        want = wanted(x, w, n)
        rows.append({"count": n, **errors(want * (1 + 1e-7), want)})
    check(rows)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["card", "made_up"])
    args = p.parse_args()
    return {"card": card, "made_up": made_up}[args.what](args)


if __name__ == "__main__":
    sys.exit(main())
