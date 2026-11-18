"""DRAM reads by route: a reader kernel (mover A's RISC-V and NoC 0, or mover B's and
NoC 1) on N cores in a row or a column reads an interleaved DRAM tensor into its L1;
each core's rate while all its readers ran, one JSON line a run (README.md).
Example: python routes.py"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

import komorebi as km  # noqa: E402
from komorebi.tools.feeder import commands as rec  # noqa: E402
from komorebi.tools.feeder import fed  # noqa: E402

PAGE = 2048  # bytes a page: a bfloat16 tile
PAGES = 32768  # the source tensor: 64 MB
AREA = 1024  # words a kernel's area (RISC-V 1's at AREA)
COUNTERS = 2 * AREA  # the meeting counters: word COUNTERS + 16 * run + 15
LANDING = COUNTERS + 512  # words before the landing pages
SLOTS = 16  # landing pages
ROW = LANDING + SLOTS * PAGE // 4  # words a core's shard
NO_ROWS_NO_X = [0] * 13  # the reader's last words: no rows, no X, no consumer


def run_list(gx, gy):
    """Every run: (name, cores, (risc, noc), bank or None). Mover A is RISC-V 0 on
    NoC 0, mover B RISC-V 1 on NoC 1; a run's cores read 2,048 pages each."""
    lines = {  # the same N readers along a row and down a column
        "row0_10": [(x, 0) for x in range(10)],
        "column0_10": [(0, y) for y in range(10)],
        "rows0and5_20": [(x, y) for y in (0, 5) for x in range(10)],
        "columns0and6_20": [(x, y) for x in (0, 6) for y in range(10)],
        "row0_12": [(x, 0) for x in range(gx)],  # comp's: 12 senders a row
        "rows0and5_24": [(x, y) for y in (0, 5) for x in range(gx)],
    }
    out = []
    for name, cores in lines.items():
        for mover, kernel in (("A", (0, 0)), ("B", (1, 1))):
            out.append((f"{name}_{mover}", cores, kernel, None))
    for mover, kernel in (("A", (0, 0)), ("B", (1, 1))):
        out.append((f"row0_12_bank0_{mover}", lines["row0_12"], kernel, 0))
    return out


def one(session, shard, source, index, run, mhz):
    """Launches `run`; its record: per kernel the rate while every kernel ran."""
    name, cores, kernel, bank = run
    kernels, block, count = [kernel], 16, 2048
    step = 1 if bank is None else 8
    total = len(cores) * len(kernels)
    lx, ly = session.device.worker_core(*cores[0])
    counter = shard.address + 4 * (COUNTERS + 16 * index + 15)
    blocks = -(-count // block)
    every = -(-blocks // 900)
    launch = session.launch()
    order = 0
    for risc, noc in kernels:
        args = []
        for x, y in cores:
            if step == 1:
                first = (order * count) % (PAGES - count)
            else:
                first = (order * count * step) % (PAGES - count * step)
                first = first - first % 8 + bank
            order += 1
            tag = 0xD7A00000 | x | y << 8 | risc << 16
            args.append(
                (
                    x,
                    y,
                    [
                        shard.address + 4 * AREA * risc,
                        shard.address + 4 * LANDING,
                        SLOTS,
                        source.address,
                        PAGE,
                        first,
                        step,
                        count,
                        block,
                        every,
                        lx | ly << 16,
                        counter,
                        total,
                        tag,
                        1,  # one block a pattern, every block this kernel's
                        1,
                        0xFF,  # every bank
                        *NO_ROWS_NO_X,
                    ],
                )
            )
        launch.kernel(
            str(HERE / "reader.cpp"),
            fed.core_set(cores),
            risc,
            [],
            [],
            [str(rec.INCLUDE)],
            args,
            noc=noc,
        )
    launch.build()  # found in the cache when prebuilt
    t0 = time.perf_counter()
    launch.run()
    session.device.finish()
    host_s = time.perf_counter() - t0
    rows = rec.as_uint32(fed.read_int32(shard)).astype(np.int64)
    found = []
    for r in rows:
        for risc in (0, 1):
            area = r[AREA * risc : AREA * (risc + 1)]
            tag = int(area[0])
            if tag >> 20 != 0xD7A:
                continue
            x, y = tag & 0xFF, tag >> 8 & 0xFF
            if (x, y) not in cores or (tag >> 16 & 1) != risc:
                continue
            if risc not in [k[0] for k in kernels]:
                continue
            samples = int(area[6])
            found.append(
                {
                    "core": [x, y],
                    "risc": risc,
                    "noc": int(area[1]),
                    "start": int(area[2]) | int(area[3]) << 32,
                    "cycles": int(area[4]),
                    "done": int(area[5]),
                    "ok": int(area[8]) == 1 and int(area[7]) == 0,
                    "trace": [int(v) for v in area[64 : 64 + samples]],
                    "every_pages": every * block,
                }
            )
    return summarize(name, cores, kernels, found, total, mhz, host_s)


def summarize(name, cores, kernels, found, total, mhz, host_s):
    """Rates in GB/s: each kernel alone over its run, and while every kernel ran."""
    record = {"run": name, "cores": len(cores), "kernels": total, "found": len(found)}
    if not found:
        record["ok"] = False
        return record
    starts = [f["start"] for f in found]
    ends = [f["start"] + f["cycles"] for f in found]
    both_from, both_to = max(starts), min(ends)
    rates, in_window = [], []
    for f in found:
        rates.append(f["done"] * PAGE / (f["cycles"] / mhz) / 1e3)
        if both_to > both_from:
            pages = pages_between(f, both_from, both_to)
            in_window.append(pages * PAGE / ((both_to - both_from) / mhz) / 1e3)
    record.update(
        {
            "ok": all(f["ok"] for f in found) and len(found) == total,
            "skew_us": round((max(starts) - min(starts)) / mhz, 2),
            "alone_gbs": stats(rates),
            "together_us": round(max(0, both_to - both_from) / mhz, 1),
            "together_gbs": stats(in_window),
            "together_sum_gbs": round(sum(in_window), 1) if in_window else None,
            "span_sum_gbs": round(
                sum(f["done"] for f in found)
                * PAGE
                / ((max(ends) - min(starts)) / mhz)
                / 1e3,
                1,
            ),
            "host_s": round(host_s, 3),
        }
    )
    if len(found) <= 12:
        record["per_kernel"] = [
            [*f["core"], f["risc"], f["noc"], round(r, 2)] for f, r in zip(found, rates)
        ]
    else:
        record["slowest"] = sorted(
            [[*f["core"], f["risc"], round(r, 2)] for f, r in zip(found, in_window)],
            key=lambda item: item[-1],
        )[:6]
        record["by_row"] = by_line(found, in_window, 1)
        record["by_column"] = by_line(found, in_window, 0)
    return record


def pages_between(f, t_from, t_to):
    """Pages kernel `f` read between wall clock t_from and t_to (from its trace)."""
    times = [f["start"]] + [f["start"] + t for t in f["trace"]]
    pages = [0] + [f["every_pages"] * (i + 1) for i in range(len(f["trace"]))]
    pages = [min(p, f["done"]) for p in pages]
    return float(np.interp(t_to, times, pages) - np.interp(t_from, times, pages))


def by_line(found, rates, axis):
    """Mean rate a line (row: axis 1, column: axis 0) of cores while all ran."""
    lines = {}
    for f, r in zip(found, rates):
        lines.setdefault(f["core"][axis], []).append(r)
    return {k: round(float(np.mean(v)), 2) for k, v in sorted(lines.items())}


def stats(values):
    if not values:
        return None
    v = np.array(values)
    return [
        round(float(v.min()), 2),
        round(float(np.median(v)), 2),
        round(float(v.max()), 2),
    ]


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    with km.Session() as session:
        mhz = fed.aiclk_mhz()
        gx, gy = session.grid
        fed.log(f"device open, AICLK {mhz} MHz, grid {gx} x {gy}")
        corners = {
            f"{x},{y}": list(session.device.worker_core(x, y))
            for x, y in ((0, 0), (gx - 1, 0), (0, gy - 1), (gx // 2 - 1, gy // 2 - 1))
        }
        fed.log(f"worker cores (NoC 0): {json.dumps(corners)}")
        source = session.empty((PAGES, PAGE // 4), "int32")
        shard = fed.int32_tensor(session, np.zeros((gx * gy, ROW)), (gx, gy))
        fed.log(f"source at {source.address:#x}, shard at {shard.address:#x}")
        chosen = run_list(gx, gy)
        records = []
        for index, run in enumerate(chosen):
            record = one(session, shard, source, index, run, mhz)
            fed.log(json.dumps(record))
            records.append(record)
    ok = all(r["ok"] for r in records)
    print(json.dumps({"runs": len(records), "ok": ok, "aiclk_mhz": mhz}), flush=True)
    if not any(r["found"] for r in records):  # no kernel's record: every area zeros
        raise SystemExit("routes.py: no kernel's record came back (reads were zeros)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
