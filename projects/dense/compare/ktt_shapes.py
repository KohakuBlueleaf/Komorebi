"""Kohaku-TT's matrix product over several shapes in one process, on one card or on the
1 x 2 mesh of cards 0 and 1: synced calls (median), then calls back to back for about
`--seconds` with AICLK of every card sampled, a cool-down between points. Runs in
Kohaku-TT's environment; one JSON a shape and variant. Usage in ../README.md."""

import argparse
import contextlib
import gc
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
from projects.dense.compare import shapes  # noqa: E402
from projects.dense.reference import ktt_bench as base  # noqa: E402
from reference import dense as ref  # noqa: E402

SINGLES = 3
VARIANTS = ("default", "ksplit", "colwise")


def say(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def operands(variant, x, w):
    """(x, w, product) of `variant` from the uploaded `x` and `w`. `default`: plain
    `x @ w`, the placement Kohaku-TT picks. `ksplit`: X split along K, W along K, the
    partial completed to Y split along N, as Linear does. `colwise`: X on every chip,
    W and Y split along N (nothing crosses a call)."""
    if variant == "ksplit":
        x, w = kohakutt.shard(x, 1), kohakutt.shard(w, 0)
        return x, w, lambda a, b: kohakutt.shard(a @ b, 1)
    if variant == "colwise":
        x, w = kohakutt.replicate(x), kohakutt.shard(w, 1)
    return x, w, lambda a, b: a @ b


def describe(mesh, x, w, y):
    """The placements of the operands and the product and the engine's last decision."""
    if not mesh:
        return None
    last = kohakutt.placement_log()[-1:] or [None]
    return {
        "x": kohakutt.placement(x),
        "w": kohakutt.placement(w),
        "y": kohakutt.placement(y),
        "decision": last[0],
    }


def measure(product, device, cards, seconds):
    """Synced calls, then calls back to back with every card's AICLK sampled."""
    synced = []
    for _ in range(SINGLES):
        started = time.perf_counter()
        product()
        kohakutt.synchronize(device)
        synced.append(time.perf_counter() - started)
    median = statistics.median(synced)
    say(f"single calls {[round(value * 1e6, 1) for value in synced]} us")
    out = {"synced_microseconds": [value * 1e6 for value in synced]}
    out["synced_microseconds_median"] = median * 1e6
    if seconds <= 0:
        return out
    reps = max(3, int(seconds / median))
    clocks = {card: [] for card in cards}
    stop = threading.Event()
    threads = [
        threading.Thread(target=base.sample_clock, args=(card, clocks[card], stop))
        for card in cards
    ]
    with contextlib.ExitStack() as stack:
        meters = {card: power.PowerSampler(card) for card in cards}
        for meter in meters.values():
            stack.enter_context(meter).mark("open")
        time.sleep(0.5)
        for meter in meters.values():
            meter.mark("loop")
        for thread in threads:
            thread.start()
        say(f"loop of {reps} calls")
        started = time.perf_counter()
        for _ in range(reps):
            product()
        kohakutt.synchronize(device)
        loop_s = time.perf_counter() - started
        for meter in meters.values():
            meter.mark("after")
        stop.set()
        for thread in threads:
            thread.join()
    phases = {}
    for card, meter in meters.items():
        phase = meter.result.phases["loop"]
        phases[card] = {"mean_w": phase.mean_w, "peak_w": phase.peak_w}
        phases[card]["mean_temp_c"] = phase.mean_temp_c
    out["loop"] = {
        "reps": reps,
        "microseconds_per_call": loop_s / reps * 1e6,
        "power": phases,
        "aiclk_mhz": {
            card: [min(c), round(statistics.mean(c), 1), max(c)]
            for card, c in clocks.items()
            if c
        },
    }
    say(f"loop {out['loop']['microseconds_per_call']:.1f} us a call")
    return out


def one_point(args, device, cards, shape, variant, uploaded, matrices):
    """Runs, checks and times one variant on the uploaded operands; its report."""
    m_rows, k_depth, n_columns = shape
    x, w, run = operands(variant, *uploaded)
    product = lambda: run(x, w)  # noqa: E731
    report = {"m": m_rows, "k": k_depth, "n": n_columns, "cards": cards}
    report |= {"variant": variant, "kohakutt": base.version()}
    report["env"] = {name: os.environ.get(name) for name in base.ROUTE_VARIABLES}
    with kohakutt.record_placements():
        y = product()
        kohakutt.synchronize(device)
    say("first product done")
    report["placements"] = describe(len(cards) > 1, x, w, y)
    rows = ref.sampled_rows(m_rows)
    x_matrix, w_matrix = matrices
    got = base.read_rows(y, rows)
    say("checking the sampled rows")
    report["check"] = ref.errors(got, ref.matmul(x_matrix[rows], w_matrix))
    del y
    report |= measure(product, device, cards, args.seconds)
    report["load_avg"] = os.getloadavg()[0]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0")
    parser.add_argument("--shapes", required=True, help="M x K x N, comma separated")
    parser.add_argument("--variants", default="default")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--out", required=True, help="folder of the JSON files")
    args = parser.parse_args()
    cards = [int(value) for value in args.cards.split(",")]
    variants = args.variants.split(",")
    if not set(variants) <= set(VARIANTS) or (
        len(cards) == 1 and variants != ["default"]
    ):
        raise SystemExit(f"variants {variants}: one card has only 'default'")
    device = kohakutt.mesh(cards) if len(cards) > 1 else kohakutt.device(cards[0])
    say(f"opened {cards}")
    for name in args.shapes.split(","):
        shape = shapes.parse(name)
        say(f"building the inputs of {name}")
        matrices = ref.inputs(*shape)
        say("inputs built")
        uploaded = [
            torch.from_numpy(matrix).bfloat16().to(device) for matrix in matrices
        ]
        say("X and W uploaded")
        for variant in variants:
            report = one_point(args, device, cards, shape, variant, uploaded, matrices)
            path = shapes.result_path(args.out, shape, "ktt", cards, variant)
            Path(path).write_text(json.dumps(report, indent=1, default=float))
            say(f"{name} {variant}: {json.dumps(report['check'])} {path}")
            time.sleep(args.seconds)
        del uploaded
        gc.collect()
    kohakutt.close()
    say("closed")


if __name__ == "__main__":
    main()
