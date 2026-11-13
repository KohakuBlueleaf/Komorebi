"""Kohaku-TT's matrix product on the same inputs as run.py: synced calls, then calls
back to back for about `--seconds` with one synchronize, under the power sampler with
AICLK sampled. Runs in Kohaku-TT's environment; its route comes from the caller's
KTT_GEMM and KTT_OWN_MATMUL_MIN_ROWS. Usage in ../README.md.
"""

import argparse
import json
import os
import statistics
import subprocess
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


def version():
    """The commit of the Kohaku-TT checkout kohakutt was imported from, if any."""
    git_command = [
        "git",
        "-C",
        str(Path(kohakutt.__file__).parent),
        "rev-parse",
        "--short",
    ]
    done = subprocess.run(
        [*git_command, "HEAD"], capture_output=True, text=True, timeout=10
    )
    return done.stdout.strip() or None


def read_rows(y, rows):
    """Rows `rows` of the device tensor y as float32, one 32-row slice per read (at most
    1 MB each, a progress line each)."""
    got = {}
    for tile in np.unique(rows // ref.TILE):
        first_row = int(tile) * ref.TILE
        got[first_row] = y[first_row : first_row + ref.TILE].cpu().float().numpy()
        print(f"rows {first_row} to {first_row + ref.TILE} read", flush=True)
    return np.stack([got[row // ref.TILE * ref.TILE][row % ref.TILE] for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shape", choices=pl.SHAPES, required=True)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    m_rows, k_depth, n_columns = pl.SHAPES[args.shape]
    print("building the inputs", flush=True)
    a_matrix, w_matrix = ref.inputs(m_rows, k_depth, n_columns)
    print("inputs built", flush=True)
    device = kohakutt.device(0)
    a_tensor = torch.from_numpy(a_matrix).bfloat16().to(device)
    print("A uploaded", flush=True)
    w_tensor = torch.from_numpy(w_matrix).bfloat16().to(device)
    print("W uploaded", flush=True)
    y = a_tensor @ w_tensor
    kohakutt.synchronize(device)
    print("first product done", flush=True)
    report = {"shape": args.shape, "m": m_rows, "k": k_depth, "n": n_columns}
    report["env"] = {variable: os.environ.get(variable) for variable in ROUTE_VARIABLES}
    report["kohakutt"] = version()
    rows = ref.sampled_rows(m_rows)
    got = read_rows(y, rows)
    print("checking the sampled rows", flush=True)
    report["check"] = ref.errors(got, ref.matmul(a_matrix[rows], w_matrix))
    synced = []
    for _ in range(args.calls):
        started = time.perf_counter()
        y = a_tensor @ w_tensor
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
            y = a_tensor @ w_tensor
        kohakutt.synchronize(device)
        loop_s = time.perf_counter() - started
        meter.mark("after")
        stop.set()
        sampler.join()
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
