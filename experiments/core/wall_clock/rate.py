"""The wall clock's ticks a cycle of the clock the chip reports (tt_aiclk), against the
host's clock: core (0, 0) spins for about 0.5 s and then for 2.5 s of ticks; the 2 s
between them over the host's time between them (the launch's cost cancels). One JSON
line, the ratio beside device/chips.py's. Example: python rate.py --runs 3"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

import komorebi as km  # noqa: E402
from komorebi.device import chips  # noqa: E402
from komorebi.tools.feeder import commands as rec  # noqa: E402
from komorebi.tools.feeder import fed  # noqa: E402


def spin(session, ticks):
    """Core (0, 0) spins `ticks`; returns (host seconds, ticks the core counted)."""
    gx, gy = session.grid
    out = fed.int32_tensor(session, np.zeros((gx * gy, 16)), (gx, gy))
    launch = session.launch()
    args = [(0, 0, [out.address, ticks & 0xFFFFFFFF, ticks >> 32])]
    launch.kernel(
        str(HERE / "rate.cpp"),
        fed.core_set([(0, 0)]),
        0,
        [],
        [],
        [str(rec.INCLUDE)],
        args,
    )
    started = time.perf_counter()
    launch.run()
    session.device.finish()
    host_s = time.perf_counter() - started
    w = rec.as_uint32(fed.read_int32(out)).astype(np.int64)[0]
    return host_s, (int(w[2]) | int(w[3]) << 32) - (int(w[0]) | int(w[1]) << 32)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args()
    with km.Session() as session:
        arch = session.device.arch()
        aiclk = fed.aiclk_mhz(session.device)
        short, long_ = int(0.5 * aiclk * 1e6), int(2.5 * aiclk * 1e6)
        spin(session, short)  # the kernel built and cached
        ratios = []
        for _ in range(args.runs):
            host_a, ticks_a = spin(session, short)
            host_b, ticks_b = spin(session, long_)
            if host_b <= host_a or ticks_b <= ticks_a:  # the stand-in card runs nothing
                continue
            ratio = (ticks_b - ticks_a) / (host_b - host_a) / 1e6 / aiclk
            fed.log(
                f"{ticks_b - ticks_a} ticks in {host_b - host_a:.4f} s: {ratio:.5f}"
            )
            ratios.append(ratio)
        aiclk_after = fed.aiclk_mhz(session.device)
    out = {
        "arch": arch,
        "aiclk_mhz": [aiclk, aiclk_after],
        "ratios": [round(r, 5) for r in ratios],
        "ratio": round(float(np.median(ratios)), 4) if ratios else None,
        "table": chips.chip(arch).tick_ratio,
        "mticks_per_s": round(float(np.median(ratios)) * aiclk, 2) if ratios else None,
    }
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
