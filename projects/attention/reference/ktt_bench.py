"""Kohaku-TT's ktt_flash (forced by `sdpa_kernel("flash")`) on run.py's inputs: the
same sampled rows checked against float64, synced calls, then calls back to back for
about `--seconds` under the power sampler with AICLK sampled. Runs in Kohaku-TT's
environment (fast dispatch). Usage in ../README.md.
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import kohakutt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from komorebi import power  # noqa: E402
from projects.attention import plan as pl  # noqa: E402
from projects.attention.reference import sdpa  # noqa: E402

T0 = time.monotonic()


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def sample_clock(card, samples, stop):
    """AICLK (MHz) every 20 ms into `samples` until `stop` is set."""
    path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
    while not stop.wait(0.02):
        samples.append(int(path.read_text().split()[0]))


def check(y, q, k, v, s, causal, count):
    """The sampled query tile rows of y (on the card) against float64."""
    group = s.h // s.hk
    got, want = [], []
    for bi, hi, row in sdpa.sample_rows(s.b, s.h, s.sqt, count):
        r0 = row * pl.TILE
        got.append(y[bi, hi, r0 : r0 + pl.TILE].cpu().float().numpy())
        rows = np.arange(r0, r0 + pl.TILE)
        kv = hi // group
        want.append(sdpa.attention(q[bi, hi], k[bi, kv], v[bi, kv], rows, causal))
    return sdpa.errors(np.concatenate(got), np.concatenate(want))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shape", choices=pl.SHAPES, required=True)
    p.add_argument("--approx", action="store_true", help="approximate exponential")
    p.add_argument("--calls", type=int, default=20)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--rows", type=int, default=16, help="query tile rows checked")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    b, h, hk, sq, d, causal = pl.SHAPES[args.shape]
    s = pl.dims(b, h, hk, sq, sq, d, d, causal)
    q, k, v = sdpa.inputs(b, h, hk, sq, sq, d, d, args.seed)
    dev = kohakutt.device(0)
    tq, tk, tv = (torch.from_numpy(x).bfloat16().to(dev) for x in (q, k, v))
    out = {"shape": args.shape, "approx": args.approx}
    with kohakutt.sdpa_kernel("flash"), kohakutt.numerics(approx=args.approx):
        out["numerics"] = str(kohakutt.current_numerics())

        def call():
            return kohakutt.sdpa(tq, tk, tv, is_causal=causal)

        y = call()
        kohakutt.synchronize(dev)
        log("first call")
        out["check"] = check(y, q, k, v, s, causal, args.rows)
        log(f"checked {out['check']}")
        synced = []
        for _ in range(args.calls):
            t0 = time.perf_counter()
            call()
            kohakutt.synchronize(dev)
            synced.append(time.perf_counter() - t0)
        out["synced_us_median"] = statistics.median(synced) * 1e6
        # The loop's length from calls back to back (a synced call is much slower).
        t0 = time.perf_counter()
        for _ in range(args.calls):
            call()
        kohakutt.synchronize(dev)
        back_to_back = (time.perf_counter() - t0) / args.calls
        reps = max(10, int(args.seconds / back_to_back))
        card = int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0])
        clock, stop = [], threading.Event()
        meter = power.PowerSampler(card)
        sampler = threading.Thread(target=sample_clock, args=(card, clock, stop))
        log(f"loop of {reps} calls")
        with meter:
            meter.mark("open")
            time.sleep(1.0)
            meter.mark("loop")
            sampler.start()
            t0 = time.perf_counter()
            for _ in range(reps):
                call()
            kohakutt.synchronize(dev)
            loop_s = time.perf_counter() - t0
            meter.mark("after")
            stop.set()
            sampler.join()
        log("loop done")
    phase = meter.result.phases["loop"]
    out["loop"] = {
        "reps": reps,
        "us_per_call": loop_s / reps * 1e6,
        "mean_w": phase.mean_w,
        "peak_w": phase.peak_w,
        "open_w": meter.result.phases["open"].mean_w,
        "temp_c": phase.mean_temp_c,
        "aiclk_mhz": [min(clock), round(statistics.mean(clock), 1), max(clock)],
    }
    out["counters"] = {
        n: c for n, c in kohakutt.counters().items() if n.startswith("attention")
    }
    out["load_avg"] = os.getloadavg()[0]
    print(json.dumps(out, default=str), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
