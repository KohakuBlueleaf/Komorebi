"""Kohaku-TT's matrix product on the same inputs as run.py: synced calls, then calls
back to back for about `--seconds` with one synchronize, under the power sampler with
AICLK sampled. Runs in Kohaku-TT's environment; its route comes from the caller's
KTT_GEMM and KTT_OWN_MATMUL_MIN_ROWS. Usage in ../README.md.
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
from projects.dense import plan as pl  # noqa: E402
from ref import dense as ref  # noqa: E402

ROUTE_VARIABLES = ("KTT_GEMM", "KTT_OWN_MATMUL_MIN_ROWS")


def sample_clock(card, samples, stop):
    """AICLK (MHz) every 20 ms into `samples` until `stop` is set."""
    path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
    while not stop.wait(0.02):
        samples.append(int(path.read_text().split()[0]))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shape", choices=pl.SHAPES, required=True)
    p.add_argument("--calls", type=int, default=20)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    m, k, n = pl.SHAPES[args.shape]
    a, w = ref.inputs(m, k, n)
    dev = kohakutt.device(0)
    ta = torch.from_numpy(a).bfloat16().to(dev)
    tw = torch.from_numpy(w).bfloat16().to(dev)
    y = ta @ tw
    kohakutt.synchronize(dev)
    out = {"shape": args.shape, "m": m, "k": k, "n": n}
    out["env"] = {v: os.environ.get(v) for v in ROUTE_VARIABLES}
    rows = np.sort(np.random.default_rng(2).permutation(m)[:256])
    got = y.cpu().float().numpy()
    out["check"] = ref.errors(got[rows], ref.matmul(a[rows], w))
    synced = []
    for _ in range(args.calls):
        t0 = time.perf_counter()
        y = ta @ tw
        kohakutt.synchronize(dev)
        synced.append(time.perf_counter() - t0)
    out["synced_us_median"] = statistics.median(synced) * 1e6
    reps = max(10, int(args.seconds / statistics.median(synced)))
    card = int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0])
    clock, stop = [], threading.Event()
    meter = power.PowerSampler(card)
    sampler = threading.Thread(target=sample_clock, args=(card, clock, stop))
    with meter:
        meter.mark("open")
        time.sleep(1.0)
        meter.mark("loop")
        sampler.start()
        t0 = time.perf_counter()
        for _ in range(reps):
            y = ta @ tw
        kohakutt.synchronize(dev)
        loop_s = time.perf_counter() - t0
        meter.mark("after")
        stop.set()
        sampler.join()
    phase = meter.result.phases["loop"]
    out["loop"] = {
        "reps": reps,
        "us_per_call": loop_s / reps * 1e6,
        "mean_w": phase.mean_w,
        "peak_w": phase.peak_w,
        "open_w": meter.result.phases["open"].mean_w,
        "aiclk_mhz": [min(clock), round(statistics.mean(clock), 1), max(clock)],
    }
    out["load_avg"] = os.getloadavg()[0]
    print(json.dumps(out), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
