"""DOT commands that carry a block: every core reads its command's table of 32-bit
weights from DRAM (more than a command's ten words), and the leader's result is checked
against numpy; then the commands are timed with and without a block. Prints one JSON
line; see README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
TILE = 32
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", type=int, default=200)
    parser.add_argument("--words", type=int, default=64, help="weights per block")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--count", type=int, default=12, help="group size")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def dot(weights, x0):
    """sum of w[i] * (x0 + i), mod 2^32, as the cores compute it."""
    x = (x0 + np.arange(len(weights), dtype=np.uint64)) & 0xFFFFFFFF
    return int((weights.astype(np.uint64) * x & 0xFFFFFFFF).sum() & 0xFFFFFFFF)


def timed(run, group, jobs, alone=False):
    """Submits `jobs` [(args, block)] and waits for all (or, `alone`, for each before
    the next): host microseconds per submit and per job, the mean of the events'."""
    started = time.perf_counter()
    handles, events = [], []
    for words, block in jobs:
        handles.append(run.submit("dot", group, words, block=block))
        if alone:
            events.append(run.wait(handles.pop(), timeout=60.0))
    submit_s = time.perf_counter() - started
    events += run.wait(handles, timeout=60.0) if handles else []
    wall_s = time.perf_counter() - started
    return {
        "submit_microseconds": round(submit_s * 1e6 / len(jobs), 2),
        "job_microseconds": round(wall_s * 1e6 / len(jobs), 2),
        "event_microseconds": round(
            float(np.mean([event.microseconds for event in events])), 3
        ),
    }


def main():
    args = parse()
    generator = np.random.default_rng(args.seed)
    tables = generator.integers(0, 1 << 32, (args.jobs, args.words), dtype=np.uint64)
    tables = tables.astype(np.uint32)
    x0s = generator.integers(0, 1 << 32, args.jobs, dtype=np.uint64).astype(np.uint32)
    sums = [int(table.astype(np.uint64).sum() & 0xFFFFFFFF) for table in tables]
    report = {"jobs": args.jobs, "words": args.words, "group": [args.first, args.count]}
    with km.Session(log=log) as session:
        tiles = session.tensor(np.zeros((TILE, TILE), np.float32), "bfloat16", "tile")
        program = km.Program(
            "mapcollect",
            kernels=KERNELS,
            buffers=[
                km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
                km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
            ],
            compile_args={
                "reader": tiles.accessor_args() + 2 * tiles.placement_args(),
                "writer": tiles.accessor_args() + 2 * tiles.placement_args(),
            },
            compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
            handlers=KERNELS / "mapcollect.h",
        )
        run = session.load(program)
        group = (args.first, args.count)
        log(f"loaded, DOT on cores {group}")
        handles = [
            run.submit(
                "dot", group, [int(x0s[job]), sums[job]], result=True,
                block=tables[job],
            )  # fmt: skip
            for job in range(args.jobs)
        ]
        events = run.wait(handles, timeout=60.0)
        wrong = [
            job
            for job, event in enumerate(events)
            if event.kind != "done" or event.ticks != dot(tables[job], int(x0s[job]))
        ]
        log(f"checked {len(events)} results, {len(wrong)} wrong")
        report["checked"], report["wrong"], report["first_wrong"] = (
            len(events),
            len(wrong),
            wrong[:5],
        )
        blocks = [([int(x0s[job]), sums[job]], tables[job]) for job in range(args.jobs)]
        bare = [([0, 0], None)] * args.jobs
        report["with_block"] = timed(run, group, blocks)
        report["without_block"] = timed(run, group, bare)
        report["alone_with_block"] = timed(run, group, blocks[:50], alone=True)
        report["alone_without_block"] = timed(run, group, bare[:50], alone=True)
        log(f"timed: {report['with_block']} / {report['without_block']}")
        report["blocks_kept"] = len(run._kept)
        run.stop()
        states = run.state()
        report["gave_up_cores"] = sum(
            1 for state in states if state["reader"] or state["writer"]
        )
    report["ok"] = not wrong and report["gave_up_cores"] == 0
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
