"""Tile split norm (README.md): the RMS norm of one row of 2,048 values split over 16
cores as the hand decode norm is (each core reads the whole row for its sums of
squares, then writes its 4 tiles), on bf16 tiles; and the layer norm over rows of
4,096 with bf16 weight and bias fields. Against float64. `card`, `made_up`."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.dsl.tile import row_operand
from komorebi.formats import bf16_bits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
JIT_CACHE = ROOT / "build" / "ttcache-formats"  # the jobs' cache (job.sh reads it)
SPLIT, REPEAT = 16, 3  # cores the row is split over, commands a handler
CORES = 12  # the layer norm's cores, a row each
EPS_RMS, EPS = 1e-6, 1e-5
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function
def square_sums(row):
    return km.reduce_sum(row * row, exact=True).col()


@km.tile_function
def rms_apply(x, w, s):
    return x * km.rsqrt(s.col() * (1.0 / 2048) + EPS_RMS) * w.row()


class SplitNorm(km.Program):
    """norm: each core's sums of squares over the whole row of 64 tiles into its own
    field, then its 4 tiles (its rank's) normalized with them and the weight row."""

    part: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def norm(self, x: km.Tensor, w: km.Tensor, out: km.Tensor):
        for row in km.Stream(x.pages(0, 64), block=8, row=64):
            self.part.write(square_sums(row))
        first = self.rank * 4
        results = km.Stream(out.pages(first, 4), write=True, block=2)
        for xt, wt in zip(
            km.Stream(x.pages(first, 4), block=2), km.Stream(w.pages(first, 4), block=2)
        ):
            results.push(rms_apply(xt, wt, self.part))


class SplitNormAll(km.Program):
    """norm: each core's sums of squares over its own 4 tiles, summed over the group
    by km.all_reduce so every core's field holds the whole row's in the one command,
    then its 4 tiles normalized with them and the weight row."""

    part: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def norm(self, x: km.Tensor, w: km.Tensor, out: km.Tensor):
        first = self.rank * 4
        for row in km.Stream(x.pages(first, 4), block=4, row=4):
            self.part.write(square_sums(row))
        km.all_reduce(self.part)
        results = km.Stream(out.pages(first, 4), write=True, block=2)
        for xt, wt in zip(
            km.Stream(x.pages(first, 4), block=2), km.Stream(w.pages(first, 4), block=2)
        ):
            results.push(rms_apply(xt, wt, self.part))


@km.tile_function
def layer_4096(row, weight, bias):
    mean = km.reduce_sum(row, exact=True) * (1.0 / 4096)
    d = row - mean.col()
    var = km.reduce_sum(d * d, exact=True) * (1.0 / 4096)
    return d * km.rsqrt(var + EPS).col() * weight.row() + bias.row()


class HalfFields(km.Program):
    """layer: each core's row of 128 tiles (its rank's), normalized into the same
    pages of out with bf16 weight and bias fields (half the fields' L1)."""

    weight: km.L1[km.bf16, 64, 32, 32]
    bias: km.L1[km.bf16, 64, 32, 32]

    @km.handler
    def layer(self, x: km.Tensor, w: km.Tensor, b: km.Tensor, out: km.Tensor):
        first = self.rank * 128
        self.weight.fill(km.Stream(w.pages(0, 64)))
        self.bias.fill(km.Stream(b.pages(0, 64)))
        results = km.Stream(out.pages(first, 128), write=True, block=8)
        for row in km.Stream(x.pages(first, 128), block=8, row=128):
            results.push(layer_4096(row, self.weight, self.bias))


def as_bf16(values):
    """float32 values rounded to bf16, as float32."""
    bits = bf16_bits(values).astype(np.uint32) << 16
    return bits.view(np.float32).reshape(np.shape(values))


def split_data(seed=11):
    """A bf16 row of 2,048 (one tile row) and its weight (row 0 of w), bf16."""
    rng = np.random.default_rng(seed)
    x = as_bf16(2.0 * rng.standard_normal((32, 2048)) + 0.7)
    w = np.zeros((32, 2048), np.float32)
    w[0] = as_bf16(rng.uniform(0.5, 1.5, 2048))  # a norm's weight, as model.py's
    want = x / np.sqrt((x.astype(np.float64) ** 2).mean(1, keepdims=True) + EPS_RMS)
    return {"x": x, "w": w}, want * w[:1]


def layer_data(seed=13):
    """x: CORES rows of 4,096, float32; w, b: bf16-rounded normals."""
    rng = np.random.default_rng(seed)
    x = (2.0 * rng.standard_normal((CORES * 32, 4096)) + 0.7).astype(np.float32)
    w, b = as_bf16(rng.standard_normal((2, 4096)))
    x64 = x.astype(np.float64)
    d = x64 - x64.mean(1, keepdims=True)
    want = d / np.sqrt((d * d).mean(1, keepdims=True) + EPS) * w + b
    return {"x": x, "w": w, "b": b}, want


def bf16_chunks(values):
    """A bf16 field's pages from a row of values: chunk c's per-column tile
    (`row_operand` of its 64 values) as bf16 bits in page c."""
    pages = [row_operand(values[64 * c : 64 * c + 64], 2) for c in range(64)]
    return bf16_bits(np.asarray(pages, np.float32)).view(np.uint8).reshape(64, -1)


def errors(got, want):
    """The relative RMS error, the largest error over the RMS, the largest relative."""
    scale = np.sqrt((want**2).mean())
    nonzero = want != 0
    return {
        "rms_rel": float(np.sqrt(((got - want) ** 2).mean()) / scale),
        "max_over_rms": float(np.abs(got - want).max() / scale),
        "max_rel": float((np.abs(got - want)[nonzero] / np.abs(want[nonzero])).max()),
    }


def timed(run, handler, cores, args):
    """REPEAT commands of `handler` on `cores`; their microseconds."""
    times = []
    for _ in range(REPEAT):
        event = run.wait(run.submit(handler, cores, args=args), timeout=30.0)
        if event.kind != "done":
            raise RuntimeError(f"{handler.__name__}: {event}")
        times.append(event.microseconds)
    return times


def card_split(
    session, program=SplitNorm, what="split RMS norm, 2,048, bf16, 16 cores"
):
    """`program`.norm (SplitNorm's or SplitNormAll's) on SPLIT cores, REPEAT
    commands; its row."""
    data, want = split_data()
    t = {n: session.tensor(v, "bfloat16", "tile") for n, v in data.items()}
    out = session.empty((32, 2048), "bfloat16", "tile")
    run = session.load(program(log=log, session=session, **t, out=out))
    try:
        times = timed(run, program.norm, (0, SPLIT), (t["x"], t["w"], out))
        log(f"{program.__name__}.norm on {SPLIT} cores: {times}")
    finally:
        run.stop()
    got = out.read().astype(np.float64)
    return {"what": what, **errors(got, want)} | {"microseconds": times}


def card_layer(session):
    """HalfFields.layer on CORES cores, REPEAT commands; its row."""
    data, want = layer_data()
    x = session.tensor(data["x"], "float32", "tile")
    fields = {}
    for n in "wb":
        fields[n] = session.empty((64, 1024), "bfloat16")
        session.write(fields[n], bf16_chunks(data[n]))
    out = session.empty(data["x"].shape, "float32", "tile")
    run = session.load(HalfFields(log=log, session=session, x=x, **fields, out=out))
    try:
        times = timed(
            run, HalfFields.layer, (0, CORES), (x, fields["w"], fields["b"], out)
        )
        log(f"HalfFields.layer on {CORES} cores: {times}")
    finally:
        run.stop()
    got = out.read().astype(np.float64)
    return {"what": "layer norm, 4,096, bf16 fields", **errors(got, want)} | {
        "microseconds": times
    }


def check(rows):
    """The rows' shape and prints, as `card` writes them (no card: a made-up run)."""
    for row in rows:
        log(json.dumps(row))
    return {"rows": rows, "wall_s": round(time.monotonic() - STARTED, 1)}


def card(args):
    """One card: the split norm, then the layer norm with bf16 fields."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    with km.Session(log=log) as session:
        rows = [card_split(session), card_layer(session)]
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "tile_split.json", "w") as f:
        json.dump(check(rows), f)
    return 0


def card_all(args):
    """One card: the split norm as above, then with each core's sums of its own 4
    tiles all-reduced (SplitNormAll), side by side."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    with km.Session(log=log) as session:
        rows = [
            card_split(session),
            card_split(
                session,
                SplitNormAll,
                "split RMS norm, 2,048, bf16, 16 cores, sums all-reduced",
            ),
        ]
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "tile_split_all.json", "w") as f:
        json.dump(check(rows), f)
    return 0


def card_islands(args):
    """Card 0 split in 2: SplitNormAll on SPLIT cores, half on each island, so its
    all-reduce crosses; x and w a copy on each island, out placed by the session."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    data, want = split_data()
    # the movers' page calls across islands and the cast: more than the default 160 KiB
    with km.Session(card=0, islands=2, kernel_bytes=192 << 10, log=log) as session:
        t = {
            n: session.tensor(v, "bfloat16", "tile", copy=True) for n, v in data.items()
        }
        out = session.tensor(np.zeros((32, 2048), np.float32), "bfloat16", "tile")
        run = session.load(SplitNormAll(log=log, session=session, **t, out=out))
        cores = (session.graph.first(1) - SPLIT // 2, SPLIT)
        try:
            times = timed(run, SplitNormAll.norm, cores, (t["x"], t["w"], out))
            log(f"SplitNormAll.norm on {cores} over 2 islands: {times}")
        finally:
            run.stop()
        got = out.read().astype(np.float64)
    what = "split RMS norm, 2,048, bf16, 16 cores over 2 islands, sums all-reduced"
    row = {"what": what, "cores": cores, **errors(got, want), "microseconds": times}
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "tile_split_islands.json", "w") as f:
        json.dump(check([row]), f)
    return 0


def made_up(args):
    """The host side after the card: the references through `check`, moved by 1e-7."""
    rows = []
    for what, (_, want) in (("split", split_data()), ("layer", layer_data())):
        rows.append({"what": what, **errors(want * (1 + 1e-7), want)})
    check(rows)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["card", "card_all", "card_islands", "made_up"])
    args = p.parse_args()
    calls = {"card": card, "card_all": card_all, "card_islands": card_islands}
    return {**calls, "made_up": made_up}[args.what](args)


if __name__ == "__main__":
    sys.exit(main())
