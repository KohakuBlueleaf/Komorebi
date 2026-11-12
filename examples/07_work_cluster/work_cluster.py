"""Work cluster, first form: an X280 thread on cluster 1 computes prefix sums over a
cores x bins table of counts the host filled, on four harts. The same function runs
on the host as the thread's fallback; the two results must agree bit for bit. Prints
one JSON line. See README.md.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
T0 = time.monotonic()
M32 = 0xFFFFFFFF


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def prefix_offsets(regions, words):
    """The thread's function on the host: offsets[c][b] = the counts of every core in
    bins before b, plus bin b on the cores before c. Returns (code, [sum])."""
    cores, bins = words
    counts = regions["counts"][: cores * bins].reshape(cores, bins).astype(np.int64)
    totals = counts.sum(0)
    before = np.cumsum(totals) - totals
    offsets = before[None, :] + np.cumsum(counts, 0) - counts
    regions["offsets"][: cores * bins] = (offsets & M32).astype(np.uint32).ravel()
    return 0, [int(counts.sum()) & M32]


def program(cores, bins, seed):
    """An idle grid and the thread, with random counts in [0, 64)."""
    counts = np.random.default_rng(seed).integers(0, 64, (cores, bins), np.uint32)
    prog = km.Program("work_cluster", kernels=HERE / "kernels")
    prog.thread(
        cluster=1,
        sources=[HERE / "thread" / "prefix.c"],
        regions={"counts": counts, "offsets": counts.nbytes},
        words=[cores, bins],
        host_fallback=prefix_offsets,
        deadline=5.0,
    )
    return prog


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cores", type=int, default=119)
    p.add_argument("--bins", type=int, default=1024, help="a multiple of 64")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    prog = program(args.cores, args.bins, args.seed)
    got = {}
    with km.Session(log=log) as session:
        log("device open")
        for where in ("card", "host"):
            run = session.load(prog, thread=where)
            log(f"loaded, thread on the {where}")
            result = run.thread_result(timeout=10.0)
            log(f"thread returned: {result.state}, code {result.code}")
            run.stop()
            log("run stopped")
            offsets = run.thread.read("offsets").view(np.uint32)
            got[where] = (result, offsets)
    card, host = got["card"], got["host"]
    out = {
        "cores": args.cores,
        "bins": args.bins,
        "card": card[0]._asdict(),
        "host_code": host[0].code,
        "host_words": host[0].words,
        "same_result": card[0].code == host[0].code and card[0].words == host[0].words,
        "same_offsets": bool(np.array_equal(card[1], host[1])),
        "wall_s": round(time.monotonic() - T0, 1),
    }
    out["ok"] = card[0].ran and card[0].state == "done" and out["same_result"]
    out["ok"] = out["ok"] and out["same_offsets"]
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
