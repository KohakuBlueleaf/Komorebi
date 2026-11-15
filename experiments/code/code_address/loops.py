"""Cycles an iteration of a fixed loop of machine code run from chosen L1 addresses.
Runs are 'blob@code', 'blob@code:data' or 'blob@code:data/fill' (hex addresses; code 0
runs the loop in place, where the kernel's text is); blobs: see BLOBS. One JSON line.
Example: python loops.py --runs tiny@0,tiny@0x90000,tiny@0x120000 --card 0"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

KERNEL = Path(__file__).resolve().parent / "kernels" / "loops.cpp"
BLOBS = [
    "tiny",
    "small",
    "medium",
    "large",
    "memory",
    "jumps",
    "straight",
    "untaken",
    "far",
    "indirect",
]
ITERATIONS = {
    "tiny": 20000,
    "small": 5000,
    "medium": 1000,
    "large": 100,
    "memory": 4000,
    "jumps": 4000,
    "straight": 4000,
    "untaken": 4000,
    "far": 1000,
    "indirect": 4000,
}
# Instructions an iteration runs.
INSTRUCTIONS = {
    "tiny": 2,
    "small": 14,
    "medium": 62,
    "large": 602,
    "memory": 26,
    "jumps": 18,
    "straight": 19,
    "untaken": 18,
    "far": 3,
    "indirect": 26,
}
NOP = 0x00000013  # addi x0, x0, 0
REGION = 16384  # bytes reserved from 4 KiB below an address, the kernel fills them
REPORT, WORDS = 16, 512
T0 = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - T0:7.2f}] {message}", flush=True)


def parse(text):
    """[(blob, code address, data address)] from 'blob@code[:data]' items."""
    runs = []
    for item in text.split(","):
        blob, _, where = item.partition("@")
        code, _, data = where.partition(":")
        data, _, fill = data.partition("/")
        if blob not in BLOBS:
            sys.exit(f"stop: no blob {blob!r}")
        runs.append(
            (
                blob,
                int(code, 0),
                int(data, 0) if data else 0,
                int(fill, 0) if fill else NOP,
            )
        )
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--processor", type=int, default=0, choices=[0, 1])
    parser.add_argument("--card", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("TT_VISIBLE_DEVICES") != str(args.card):
        sys.exit(f"stop: TT_VISIBLE_DEVICES must be {args.card}")
    runs = parse(args.runs)
    # Each address is held as a region that starts 4 KiB below it; overlaps merge.
    spans = []
    for base in sorted(
        {(a - 4096) & ~4095 for _, c, d, _ in runs for a in (c, d) if a}
    ):
        if spans and base < spans[-1][1]:
            spans[-1][1] = base + REGION
        else:
            spans.append([base, base + REGION])
    assert len(runs) * args.repeats + REPORT <= WORDS
    with km.Session() as session:
        log("device open")
        flags = session.tensor(
            np.zeros((1, WORDS), np.uint32), "uint32", shard_grid=(1, 1)
        )
        if any(
            flags.address < end and start < flags.address + WORDS * 4
            for start, end in spans
        ):
            sys.exit(f"stop: the report at {flags.address:#x} overlaps an address")
        held = [
            session.empty(
                (1, (end - start) // 4), "uint32", shard_grid=(1, 1), address=start
            )
            for start, end in spans
        ]
        held_text = [(hex(a), hex(b)) for a, b in spans]
        log(f"report at {flags.address:#x}; reserved {held_text}")
        words = [flags.address, len(runs), args.repeats]
        for blob, code, data, fill in runs:
            words += [BLOBS.index(blob), code, ITERATIONS[blob], data, fill]
        launch = session.launch()
        launch.kernel(
            str(KERNEL), [(0, 0, 0, 0)], args.processor, [], [], [], [(0, 0, words)]
        )
        log("launch")
        launch.run()
        session.device.finish()
        report = flags.read().astype(np.int64).reshape(-1)
        log("done")
        del held
    if report[0] != 1:
        sys.exit(
            f"stop: kernel status {report[0]}, runs done {report[1]}, "
            f"refused at {[hex(int(w)) for w in report[4:7]]}"
        )
    rows = []
    for i, (blob, code, data, fill) in enumerate(runs):
        cycles = report[REPORT + i * args.repeats : REPORT + (i + 1) * args.repeats]
        per = [round(c / ITERATIONS[blob], 3) for c in cycles]
        rows.append(
            {
                "blob": blob,
                "code": hex(code),
                "data": hex(data),
                "fill": hex(fill),
                "cycles": [int(c) for c in cycles],
                "per_iteration": per,
                "per_instruction": [round(p / INSTRUCTIONS[blob], 3) for p in per],
            }
        )
    print(
        json.dumps(
            {"text": hex(int(report[2])), "tiny_bytes": int(report[3]), "runs": rows}
        )
    )


if __name__ == "__main__":
    main()
