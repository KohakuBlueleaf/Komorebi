"""The host's read of a region the thread wrote: the thread fills a region of --mb MB
with known words, the run stops, run.thread.read times it and checks it word for word.
Prints one JSON line. Usage: python region_read.py [--mb 1]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "tests" / "hardware" / "thread" / "kernels"  # every record a NOP
SPIN = ROOT / "tests" / "hardware" / "thread" / "spin.c"  # fills region 1, waits
T0 = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mb", type=float, default=1.0, help="region size, MB")
    args = p.parse_args()
    size = int(args.mb * (1 << 20)) // 4 * 4
    prog = km.Program("region_read", kernels=KERNELS)
    prog.thread(sources=[SPIN], regions={"unused": 64, "out": size}, deadline=5.0)
    with km.Session(log=log) as session:
        run = session.load(prog)
        result = run.thread_result(timeout=0.5)
        run.stop()
        log(f"stopped, thread at stamp {result.where}; reading {size} bytes")
        t = time.monotonic()
        got = run.thread.read("out")
        read_s = time.monotonic() - t
        log(f"read in {read_s:.2f} s")
    want = (np.arange(1, size // 4 + 1, dtype=np.uint32) * 2654435761).view(np.uint8)
    out = {"bytes": size, "read_s": round(read_s, 3), "where": result.where}
    out["exact"] = bool(got.size == size and (got == want).all())
    print(json.dumps(out), flush=True)
    sys.exit(0 if out["exact"] and result.where == 2 else 1)


if __name__ == "__main__":
    main()
