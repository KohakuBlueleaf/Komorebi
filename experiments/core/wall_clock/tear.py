"""Wall-clock tear check: both movers of each core read km::wall64() in a tight loop for
--window-s across a carry of the low word; a read that went back or jumped by 2^31 or
more is a tear. One JSON line; exit 1 on a tear. Example: python tear.py --runs 3"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

import komorebi as km  # noqa: E402
from komorebi.tools.feeder import commands as rec  # noqa: E402
from komorebi.tools.feeder import fed  # noqa: E402

CORES = [(0, 0), (1, 0), (0, 1), (5, 4)]
WORDS = 32  # a core's row: 16 words a mover
MAGIC = 0x7EA20000


def run(session, cores, lead, window):
    """One run on `cores`: per core and mover (reads, back, jumps, largest step,
    first, last, first bad pair) as a dict."""
    gx, gy = session.grid
    out = fed.int32_tensor(session, np.zeros((gx * gy, WORDS)), (gx, gy))
    launch = session.launch()
    for mover in (0, 1):
        args = [(x, y, [out.address, mover, lead, window]) for x, y in cores]
        launch.kernel(
            str(HERE / "tear.cpp"),
            fed.core_set(cores),
            mover,
            [],
            [],
            [str(rec.INCLUDE)],
            args,
        )
    started = time.perf_counter()
    launch.run()
    session.device.finish()
    host_s = time.perf_counter() - started
    rows = rec.as_uint32(fed.read_int32(out)).astype(np.int64)
    found = []
    for x, y in cores:
        row = rows[y * gx + x]
        for mover in (0, 1):
            w = row[16 * mover : 16 * mover + 16]
            if int(w[0]) != MAGIC | mover:
                found.append({"core": [x, y], "mover": mover, "record": "missing"})
                continue

            def wide(i, w=w):
                return int(w[i]) | int(w[i + 1]) << 32

            found.append(
                {
                    "core": [x, y],
                    "mover": mover,
                    "reads": int(w[1]),
                    "back": int(w[2]),
                    "jumps": int(w[3]),
                    "largest_step": int(w[4]),
                    "first": wide(5),
                    "last": wide(7),
                    "carry_crossed": wide(5) >> 32 != wide(7) >> 32,
                    "bad": [wide(9), wide(11)] if wide(9) else None,
                }
            )
    return host_s, found


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--window-s", type=float, default=1.0)
    p.add_argument("--all", action="store_true", help="every core of the grid")
    args = p.parse_args()
    with km.Session() as session:
        gx, gy = session.grid
        cores = [(x, y) for y in range(gy) for x in range(gx)] if args.all else CORES
        mhz = fed.aiclk_mhz()
        window = int(args.window_s * mhz * 1e6)
        lead = window // 2  # the loop starts this long before a carry
        fed.log(f"device open, AICLK {mhz} MHz; {args.runs} runs of {window} cycles")
        runs = []
        for index in range(args.runs):
            host_s, found = run(session, cores, lead, window)
            fed.log(f"run {index}: {host_s:.2f} s")
            runs.append(found)
    records = [r for found in runs for r in found]
    missing = sum("record" in r for r in records)
    good = [r for r in records if "record" not in r]
    torn = sum(r["back"] + r["jumps"] for r in good)
    out = {
        "aiclk_mhz": mhz,
        "runs": args.runs,
        "cores": len(cores),
        "records": len(good),
        "missing": missing,
        "carries_crossed": sum(r["carry_crossed"] for r in good),
        "reads": sum(r["reads"] for r in good),
        "back": sum(r["back"] for r in good),
        "jumps": sum(r["jumps"] for r in good),
        "largest_step": max((r["largest_step"] for r in good), default=None),
        "first_bad": next((r for r in good if r["bad"]), None),
        "ok": not missing and not torn and bool(good),
    }
    print(json.dumps(out), flush=True)
    raise SystemExit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
