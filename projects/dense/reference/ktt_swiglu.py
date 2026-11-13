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
from reference import dense as ref  # noqa: E402

VARIABLES = ("KTT_GEMM", "KTT_OWN_MATMUL_MIN_ROWS", "KTT_FIDELITY", "KTT_FUSION")


def log(message):
    print(message, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shape", choices=sp.SHAPES, required=True)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    m_rows, model_width, hidden_width = sp.SHAPES[args.shape]
    x, w1, w3, w2 = ref.swiglu_inputs(m_rows, model_width, hidden_width)
    device = kohakutt.device(0)
    log("device open")
    x_tensor, w1_tensor, w3_tensor, w2_tensor = (
        torch.from_numpy(array).bfloat16().to(device) for array in (x, w1, w3, w2)
    )

    def layer():
        return (
            torch.nn.functional.silu(x_tensor @ w1_tensor) * (x_tensor @ w3_tensor)
        ) @ w2_tensor

    started = time.perf_counter()
    y = layer()
    kohakutt.synchronize(device)
    report = {"shape": args.shape, "m": m_rows, "d": model_width, "h": hidden_width}
    report["first_call_ms"] = (time.perf_counter() - started) * 1e3
    report["env"] = {variable: os.environ.get(variable) for variable in VARIABLES}
    report["kohakutt"] = version()
    log(f"first call {report['first_call_ms']:.1f} ms")
    rows = ref.sampled_rows(m_rows)
    report["check"] = ref.errors(read_rows(y, rows), ref.swiglu(x[rows], w1, w3, w2))
    synced = []
    for _ in range(args.calls):
        started = time.perf_counter()
        y = layer()
        kohakutt.synchronize(device)
        synced.append(time.perf_counter() - started)
    report["synced_microseconds_median"] = statistics.median(synced) * 1e6
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
        started = time.perf_counter()
        for _ in range(reps):
            y = layer()
        kohakutt.synchronize(device)
        loop_s = time.perf_counter() - started
        meter.mark("after")
        stop.set()
        sampler.join()
    log(f"loop of {reps} calls in {loop_s:.2f} s")
    phase = meter.result.phases["loop"]
    report["loop"] = {
        "reps": reps,
        "microseconds_per_call": loop_s / reps * 1e6,
        "mean_w": phase.mean_w,
        "peak_w": phase.peak_w,
        "open_w": meter.result.phases["open"].mean_w,
        "aiclk_mhz": [min(clock), round(statistics.mean(clock), 1), max(clock)],
    }
    report["load_avg"] = os.getloadavg()[0]
    print(json.dumps(report), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report))


if __name__ == "__main__":
    main()
