"""The RMS norms of DeepSeek-V4.1-Flash on one chip: a row of 5,120, 1,280 or 512
values split over cores, the squares' sums all-reduced (README.md). `python norms.py
card --card N` checks both widths against float64."""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi.formats import bf16_bits

EPS = 1e-20  # config.json rms_norm_eps
EACH = 4  # tiles a core
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


# one set of compute options for every class of a load
COMPUTE = {"math_fidelity": "HiFi2", "dst_full_sync_en": True, "fp32_dest_acc_en": True}


@km.tile_function
def square_sums(row):
    return km.reduce_sum(row * row, exact=True).col()


class RmsNorm(km.Program, compute=COMPUTE):
    """norm, every width in one class: the cores' own EACH tiles of x's tile row 0,
    their squares' sums all-reduced over the group (width / (32 EACH) cores), then
    x rsqrt(sum + e) w, w = weight sqrt(width) (row 0), e = width eps (README.md)."""

    EACH = km.constant(EACH)
    part: km.L1[km.fp32, 1, 32, 32]

    @km.tile_function
    def apply(x, w, s, e):
        return x * km.rsqrt(s.col() + e) * w.row()

    @km.handler
    def norm(self, x: km.Tensor, w: km.Tensor, e: km.Tensor, out: km.Tensor):
        first = self.rank * self.EACH
        for r in km.Stream(x.pages(first, self.EACH), block=self.EACH, row=self.EACH):
            self.part.write(square_sums(r))
        km.all_reduce(self.part, local=True)
        results = km.Stream(out.pages(first, self.EACH), write=True, block=2)
        for xt, wt, et in zip(
            km.Stream(x.pages(first, self.EACH), block=2),
            km.Stream(w.pages(first, self.EACH), block=2),
            km.Stream(e.pages(0, self.EACH), block=2),
        ):
            results.push(self.apply(xt, wt, self.part, et))


def cores(width):
    return width // 32 // EACH


def as_bf16(values):
    """float32 values rounded to bf16, as float32."""
    bits = bf16_bits(np.asarray(values, np.float32)).astype(np.uint32) << 16
    return bits.view(np.float32).reshape(np.shape(values))


def weight_rows(weight):
    """A norm weight (n values) as the [32, n] tile tensor's values: row 0 the weight
    times sqrt(n) (the mean's 1 / n moved out of the root)."""
    w = np.zeros((32, len(weight)), np.float32)
    w[0] = as_bf16(weight).astype(np.float64) * np.sqrt(len(weight))
    return w


def eps_tiles(width):
    """The EACH tiles of e for a width: every value width x eps."""
    return np.full((32, 32 * EACH), width * EPS, np.float32)


def reference(x, weight):
    """float64 of the stored values: x [rows, n] and the weight, both bf16-rounded."""
    x = np.asarray(x, np.float64)
    return (
        x
        / np.sqrt((x * x).mean(1, keepdims=True) + EPS)
        * np.asarray(weight, np.float64)
    )


class Norm:
    """One width's norm loaded on a session: `__call__(x, out)` runs one command on the
    group (first, cores) and returns its microseconds."""

    def __init__(self, session, weight, x, out, first=0):
        n = len(weight)
        self.w = session.tensor(weight_rows(weight), "float32", "tile")
        self.e = session.tensor(eps_tiles(n), "float32", "tile")
        self.group, self.cores = (first, cores(n)), cores(n)
        self.program = RmsNorm(
            log=log, session=session, form="runtime", x=x, w=self.w, e=self.e, out=out
        )
        self.run = session.load(self.program)

    def __call__(self, x, out, timeout=30.0):
        event = self.run.wait(
            self.run.submit(RmsNorm.norm, self.group, args=(x, self.w, self.e, out)),
            timeout=timeout,
        )
        if event.kind != "done":
            raise km.KomorebiError(f"norm: {event}")
        return event.microseconds

    def stop(self):
        self.run.stop()


def errors(got, want):
    """The relative RMS error, the largest error over the RMS, the largest relative."""
    scale = np.sqrt((want**2).mean())
    nonzero = want != 0
    return {
        "rms_rel": float(np.sqrt(((got - want) ** 2).mean()) / scale),
        "max_over_rms": float(np.abs(got - want).max() / scale),
        "max_rel": float((np.abs(got - want)[nonzero] / np.abs(want[nonzero])).max()),
    }


def card(args):
    """Each width on one chip: random bf16 rows (a scale of 3 and a mean of 0.5, as the
    model's residual), a weight uniform in [0.5, 1.5), `--repeat` commands."""
    rng = np.random.default_rng(args.seed)
    rows = []
    with km.Session(log=log, card=args.card) as session:
        for n in (5120, 1280, 512):
            x = as_bf16(3.0 * rng.standard_normal((32, n)) + 0.5)
            weight = as_bf16(rng.uniform(0.5, 1.5, n))
            tx = session.tensor(x, "bfloat16", "tile")
            out = session.empty((32, n), "bfloat16", "tile")
            norm = Norm(session, weight, tx, out)
            try:
                times = [norm(tx, out) for _ in range(args.repeat)]
            finally:
                norm.stop()
            got = out.read().astype(np.float64)
            row = {
                "width": n,
                "cores": norm.cores,
                **errors(got, reference(x, weight)),
            }
            row["row0"] = errors(got[:1], reference(x[:1], weight))
            row["us"] = times
            log(json.dumps(row))
            rows.append(row)
    for row in rows:
        print(json.dumps(row), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["card"])
    p.add_argument("--card", type=int, default=0)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args()
    return {"card": card}[args.what](args)


if __name__ == "__main__":
    sys.exit(main())
