"""The decode's reads: every core of the grid reads its own run of bfp8_b-sized pages,
its blocks shared by the two movers by a pattern (AAAB: mover B every fourth), or one
mover reading two banks only. Per run the card's GB/s and when each core ends.
Example: python decode.py"""

import argparse
import json
import sys
import time

import numpy as np
import routes
from routes import AREA, COUNTERS, HERE, LANDING, SLOTS, fed, km, rec

PAGE = 1088  # bytes a page: a bfp8_b tile
COUNT = 1024  # pages a core
BLOCK = 16  # pages a block
TOKEN_GB = 27.2  # bytes a token of the bfp8_b decode (62 layers, products only)
MOVERS = {"A": (0, 0), "B": (1, 1)}  # RISC-V and NoC of each mover


def run_list():
    """Every run: (name, pattern letters, banks mask)."""
    out = [(f"pattern_{p}", p, 0xFF) for p in ("A", "B", "AB", "AAB", "AAAB")]
    out.append(("pattern_AAAAAAAB", "AAAAAAAB", 0xFF))
    for mover in "AB":
        out.append((f"banks04_{mover}", mover, 0x11))
        out.append((f"banks15_{mover}", mover, 0x22))
    return out


def one(session, shard, source, index, run, cores, mhz):
    """Launches `run` on `cores`; its record: the card's rate and each core's end."""
    name, letters, banks = run
    movers = sorted(set(letters))
    lx, ly = session.device.worker_core(*cores[0])
    counter = shard.address + 4 * (COUNTERS + 16 * index + 15)
    launch = session.launch()
    for mover in movers:
        risc, noc = MOVERS[mover]
        takes = sum(1 << b for b, letter in enumerate(letters) if letter == mover)
        args = []
        for order, (x, y) in enumerate(cores):
            words = [
                shard.address + 4 * AREA * risc,
                shard.address + 4 * LANDING,
                SLOTS,
                source.address,
                PAGE,
                order * COUNT * 8 // bin(banks).count("1"),  # its own pages
                1,
                COUNT,
                BLOCK,
                1,
                lx | ly << 16,
                counter,
                len(cores) * len(movers),
                0xD7A00000 | x | y << 8 | risc << 16,
                len(letters),
                takes,
                banks,
            ]
            args.append((x, y, words))
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
    kernels = []
    for r in rows:
        for risc in {MOVERS[m][0] for m in movers}:
            area = r[AREA * risc : AREA * (risc + 1)]
            tag = int(area[0])
            core = (tag & 0xFF, tag >> 8 & 0xFF)
            if tag >> 20 != 0xD7A or (tag >> 16 & 1) != risc or core not in cores:
                continue
            kernels.append(
                {
                    "core": core,
                    "start": int(area[2]) | int(area[3]) << 32,
                    "cycles": int(area[4]),
                    "done": int(area[5]),
                    "ok": int(area[8]) == 1 and int(area[7]) == 0,
                }
            )
    record = summarize(name, cores, movers, kernels, mhz)
    record["host_s"] = round(host_s, 3)
    return record


def summarize(name, cores, movers, kernels, mhz):
    """The card's GB/s (every byte over first start to last end), each core's end
    after the first start (us) and its own rate (GB/s), by row and by column."""
    record = {"run": name, "kernels": len(kernels)}
    if len(kernels) != len(cores) * len(movers):
        record["ok"] = False
        return record
    t0 = min(k["start"] for k in kernels)
    by_core = {}
    for k in kernels:
        held = by_core.setdefault(
            k["core"], {"start": k["start"], "end": 0, "pages": 0}
        )
        held["start"] = min(held["start"], k["start"])
        held["end"] = max(held["end"], k["start"] + k["cycles"])
        held["pages"] += k["done"]
    ends = {c: (h["end"] - t0) / mhz for c, h in by_core.items()}
    rates = {
        c: h["pages"] * PAGE / ((h["end"] - h["start"]) / mhz) / 1e3
        for c, h in by_core.items()
    }
    total = sum(h["pages"] for h in by_core.values()) * PAGE
    card = total / (max(ends.values()) * 1e3)
    record.update(
        {
            "ok": all(k["ok"] for k in kernels)
            and all(h["pages"] == COUNT for h in by_core.values()),
            "card_gbs": round(card, 1),
            "tok_s": round(card / TOKEN_GB, 2),
            "skew_us": round((max(k["start"] for k in kernels) - t0) / mhz, 2),
            "end_us": routes.stats(list(ends.values())),
            "core_gbs": routes.stats(list(rates.values())),
            "end_by_row": line_means(ends, 1),
            "end_by_column": line_means(ends, 0),
            "latest": [
                [*c, round(e, 1)]
                for c, e in sorted(ends.items(), key=lambda item: -item[1])[:8]
            ],
        }
    )
    return record


def line_means(values, axis):
    """Mean of `values` (by core) a row (axis 1) or a column (axis 0)."""
    lines = {}
    for core, value in values.items():
        lines.setdefault(core[axis], []).append(value)
    return {k: round(float(np.mean(v)), 1) for k, v in sorted(lines.items())}


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    with km.Session() as session:
        mhz = fed.aiclk_mhz()
        gx, gy = session.grid
        cores = [(x, y) for y in range(gy) for x in range(gx)]
        fed.log(f"device open, AICLK {mhz} MHz, grid {gx} x {gy}")
        source = session.empty((len(cores) * COUNT * 4, PAGE // 4), "int32")
        shard = fed.int32_tensor(
            session, np.zeros((gx * gy, LANDING + SLOTS * PAGE // 4)), (gx, gy)
        )
        records = []
        for index, run in enumerate(run_list()):
            record = one(session, shard, source, index, run, cores, mhz)
            fed.log(json.dumps(record))
            records.append(record)
    ok = all(r["ok"] for r in records)
    print(json.dumps({"runs": len(records), "ok": ok, "aiclk_mhz": mhz}), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
