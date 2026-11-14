"""Where a run's overlay slots land when its circular buffers take most of L1, and
what the service overlay's PUSH costs there: example 01's program with one more buffer
of BUFFER KiB, its DOT check, then READS reads of a DRAM tensor while the run is live
(each a PUSH served from the slots). One JSON line; see README.md."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "examples" / "01_mapcollect" / "kernels"
TILE = 32
SLOTS = re.compile(r"slots(?::| from) (?:[\d,]+ B from )?(0x[0-9a-f]+)")
T0 = time.monotonic()
seen = []


def log(message):
    seen.append(message)
    print(f"[{time.monotonic() - T0:7.2f}] {message}", flush=True)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--buffer", type=int, default=960, help="KiB, multiple of 64")
    parser.add_argument("--reads", type=int, default=5)
    parser.add_argument("--pages", type=int, default=256, help="4 KiB pages a read")
    parser.add_argument("--jobs", type=int, default=200)
    return parser.parse_args(argv)


def dot(weights, x0):
    """sum of w[i] * (x0 + i), mod 2^32, as example 01's cores compute it."""
    x = (x0 + np.arange(len(weights), dtype=np.uint64)) & 0xFFFFFFFF
    return int((weights.astype(np.uint64) * x & 0xFFFFFFFF).sum() & 0xFFFFFFFF)


def main():
    args = parse()
    generator = np.random.default_rng(1)
    tables = generator.integers(0, 1 << 32, (args.jobs, 64), dtype=np.uint64)
    tables = tables.astype(np.uint32)
    x0s = generator.integers(0, 1 << 32, args.jobs, dtype=np.uint64).astype(np.uint32)
    sums = [int(table.astype(np.uint64).sum() & 0xFFFFFFFF) for table in tables]
    data = np.arange(args.pages * 1024, dtype=np.uint32).reshape(args.pages, 1024)
    report = {"buffer_kib": args.buffer, "pages": args.pages}
    with km.Session(card=args.card, log=log) as session:
        tiles = session.tensor(np.zeros((TILE, TILE), np.float32), "bfloat16", "tile")
        source = session.tensor(data, "uint32")
        program = km.Program(
            "placement",
            kernels=KERNELS,
            buffers=[
                km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
                km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
                km.Buffer(index=2, format="uint32", page_bytes=65536,
                          pages=args.buffer // 64),
            ],  # fmt: skip
            compile_args={
                "reader": tiles.accessor_args(),
                "writer": tiles.accessor_args(),
            },
            compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
            handlers=KERNELS / "mapcollect.h",
        )
        run = session.load(program)
        found = [SLOTS.search(line) for line in seen]
        report["slots"] = [match.group(1) for match in found if match]
        log(f"loaded, slots from {report['slots']}")
        handles = [
            run.submit(
                "dot", (0, 12), [int(x0s[job]), sums[job]], result=True,
                block=tables[job],
            )  # fmt: skip
            for job in range(args.jobs)
        ]
        events = run.wait(handles, timeout=60.0)
        report["wrong"] = sum(
            event.kind != "done" or event.ticks != dot(tables[job], int(x0s[job]))
            for job, event in enumerate(events)
        )
        log(f"checked {len(events)} DOT results, {report['wrong']} wrong")
        report["push_cycles"], report["push_wrong"] = [], 0
        for read in range(args.reads):
            (pages,) = session.read(source)
            report["push_wrong"] += int(
                not np.array_equal(pages.view(np.uint32).reshape(data.shape), data)
            )
            report["push_cycles"].append(session.transfer_cycles)
            log(f"read {read + 1} of {args.reads}: {session.transfer_cycles} cycles")
        run.stop()
    report["ok"] = report["wrong"] == 0 and report["push_wrong"] == 0
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
