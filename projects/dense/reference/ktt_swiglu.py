"""Kohaku-TT's SwiGLU on the same inputs as swiglu.py: (silu(x W1) * (x W3)) W2 as its
device operators, synced calls, then calls back to back for about `--seconds` with one
synchronize, under the power sampler with AICLK sampled; the same rows checked against
float64. Runs in Kohaku-TT's environment (no ttnn). Usage in ../SWIGLU.md.
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
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from komorebi import power  # noqa: E402
from projects.dense import swiglu_plan as sp  # noqa: E402
from projects.dense.reference.ktt_bench import (  # noqa: E402
    read_rows,
    sample_clock,
    version,
)
from ref import dense as ref  # noqa: E402

VARIABLES = ("KTT_GEMM", "KTT_OWN_MATMUL_MIN_ROWS", "KTT_FIDELITY", "KTT_FUSION")


def log(msg):
    print(msg, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shape", choices=sp.SHAPES, required=True)
    p.add_argument("--calls", type=int, default=20)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    m, d, h = sp.SHAPES[args.shape]
    x, w1, w3, w2 = ref.swiglu_inputs(m, d, h)
    dev = kohakutt.device(0)
    log("device open")
    tx, t1, t3, t2 = (torch.from_numpy(v).bfloat16().to(dev) for v in (x, w1, w3, w2))

    def layer():
        return (torch.nn.functional.silu(tx @ t1) * (tx @ t3)) @ t2

    t0 = time.perf_counter()
    y = layer()
    kohakutt.synchronize(dev)
    out = {"shape": args.shape, "m": m, "d": d, "h": h}
    out["first_call_ms"] = (time.perf_counter() - t0) * 1e3
    out["env"] = {v: os.environ.get(v) for v in VARIABLES}
    out["kohakutt"] = version()
    log(f"first call {out['first_call_ms']:.1f} ms")
    rows = ref.sampled_rows(m)
    out["check"] = ref.errors(read_rows(y, rows), ref.swiglu(x[rows], w1, w3, w2))
    synced = []
    for _ in range(args.calls):
        t0 = time.perf_counter()
        y = layer()
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
            y = layer()
        kohakutt.synchronize(dev)
        loop_s = time.perf_counter() - t0
        meter.mark("after")
        stop.set()
        sampler.join()
    log(f"loop of {reps} calls in {loop_s:.2f} s")
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
