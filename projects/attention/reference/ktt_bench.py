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

from komorebi.bench import power  # noqa: E402
from projects.attention import plan as pl  # noqa: E402
from projects.attention.reference import sdpa  # noqa: E402

STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def sample_clock(card_index, samples, stop):
    """AICLK (MHz) every 20 ms into `samples` until `stop` is set."""
    path = Path(f"/sys/class/tenstorrent/tenstorrent!{card_index}/tt_aiclk")
    while not stop.wait(0.02):
        samples.append(int(path.read_text().split()[0]))


def check(output, queries, keys, values, call, causal, count):
    """The sampled query tile rows of y (on the card) against float64."""
    group = call.h // call.hk
    got, want = [], []
    for batch, head, row in sdpa.sample_rows(call.b, call.h, call.sqt, count):
        first_row = row * pl.TILE
        got.append(
            output[batch, head, first_row : first_row + pl.TILE].cpu().float().numpy()
        )
        rows = np.arange(first_row, first_row + pl.TILE)
        kv_head = head // group
        want.append(
            sdpa.attention(
                queries[batch, head],
                keys[batch, kv_head],
                values[batch, kv_head],
                rows,
                causal,
            )
        )
    return sdpa.errors(np.concatenate(got), np.concatenate(want))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shape", choices=pl.SHAPES, required=True)
    parser.add_argument("--approx", action="store_true", help="approximate exponential")
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--rows", type=int, default=16, help="query tile rows checked")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    batches, heads, kv_heads, sequence_length, head_dim, causal = pl.SHAPES[
        arguments.shape
    ]
    call = pl.dims(
        batches,
        heads,
        kv_heads,
        sequence_length,
        sequence_length,
        head_dim,
        head_dim,
        causal,
    )
    queries, keys, values = sdpa.inputs(
        batches,
        heads,
        kv_heads,
        sequence_length,
        sequence_length,
        head_dim,
        head_dim,
        arguments.seed,
    )
    device = kohakutt.device(0)
    query_tensor, key_tensor, value_tensor = (
        torch.from_numpy(array).bfloat16().to(device)
        for array in (queries, keys, values)
    )
    report = {"shape": arguments.shape, "approx": arguments.approx}
    with kohakutt.sdpa_kernel("flash"), kohakutt.numerics(approx=arguments.approx):
        report["numerics"] = str(kohakutt.current_numerics())

        def call():
            return kohakutt.sdpa(
                query_tensor, key_tensor, value_tensor, is_causal=causal
            )

        y = call()
        kohakutt.synchronize(device)
        log("first call")
        report["check"] = check(y, queries, keys, values, call, causal, arguments.rows)
        log(f"checked {report['check']}")
        synced = []
        for _ in range(arguments.calls):
            started = time.perf_counter()
            call()
            kohakutt.synchronize(device)
            synced.append(time.perf_counter() - started)
        report["synced_microseconds_median"] = statistics.median(synced) * 1e6
        # The loop's length from calls back to back (a synced call is much slower).
        started = time.perf_counter()
        for _ in range(arguments.calls):
            call()
        kohakutt.synchronize(device)
        back_to_back = (time.perf_counter() - started) / arguments.calls
        reps = max(10, int(arguments.seconds / back_to_back))
        card_index = int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0])
        clock, stop = [], threading.Event()
        meter = power.PowerSampler(card_index)
        sampler = threading.Thread(
            target=sample_clock, arguments=(card_index, clock, stop)
        )
        log(f"loop of {reps} calls")
        with meter:
            meter.mark("open")
            time.sleep(1.0)
            meter.mark("loop")
            sampler.start()
            started = time.perf_counter()
            for _ in range(reps):
                call()
            kohakutt.synchronize(device)
            loop_s = time.perf_counter() - started
            meter.mark("after")
            stop.set()
            sampler.join()
        log("loop done")
    phase = meter.result.phases["loop"]
    report["loop"] = {
        "reps": reps,
        "microseconds_per_call": loop_s / reps * 1e6,
        "mean_w": phase.mean_w,
        "peak_w": phase.peak_w,
        "open_w": meter.result.phases["open"].mean_w,
        "temp_c": phase.mean_temp_c,
        "aiclk_mhz": [min(clock), round(statistics.mean(clock), 1), max(clock)],
    }
    report["counters"] = {
        name: value
        for name, value in kohakutt.counters().items()
        if name.startswith("attention")
    }
    report["load_avg"] = os.getloadavg()[0]
    print(json.dumps(power.clocked(report), default=str), flush=True)
    if arguments.json:
        Path(arguments.json).write_text(json.dumps(report, default=str))


if __name__ == "__main__":
    main()
