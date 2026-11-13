"""X280 ports: X280 cores and Tensix cores talk without the dispatch kernel. Per
round each Tensix core waits for its X280 core's salt, asks it `--asks` values, hashes
them and tells it the hash; the hashes are checked against numpy and the asks timed.
One JSON line; see README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.session import aiclk_mhz

HERE = Path(__file__).resolve().parent
T0 = time.monotonic()
TICKS_PER_US = 50  # kmx_time() counts the 50 MHz timer
LINE = 16  # words of a core's line in region `out` (x280_ports.h)
HASH, ROUNDS, CALLS, SUM_LO, SUM_HI, MAX, WHO = range(7)
CODES = {0: "ok", 1: "bad words or region", 2: "late or stopped"}
M32 = 0xFFFFFFFF


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def expected(cores, rounds, asks):
    """Each core's hash of the last round (x280_ports.h in numpy)."""
    c = np.arange(cores, dtype=np.uint64)
    acc = ((rounds * 0x9E3779B1) & M32) ^ ((c * 0x85EBCA77) & M32)
    for k in range(1, asks + 1):
        v = ((k * 0xC2B2AE3D) & M32) ^ ((c << 16) & M32) ^ 0x27D4EB2F
        acc = ((acc ^ v) * 0x01000193) & M32
    return acc.astype(np.uint32)


def program(clusters, cores, rounds, asks):
    """The grid (the reader asks), the thread on `clusters` and every core's port."""
    prog = km.Program("x280_ports", kernels=HERE / "kernels")
    prog.thread(
        clusters=clusters,
        sources=[HERE / "thread" / "ports.c"],
        includes=[HERE],
        regions={"out": 4 * LINE * 128},
        words=[rounds, cores, asks],
        deadline=30.0,
    )
    prog.ports()  # every core of the grid, dealt over the X280 cores in grid order
    return prog


def rounds(args):
    """Loads, waits for the thread, reads every cluster's lines."""
    with km.Session(log=log) as session:
        prog = program(args.clusters, args.cores, args.rounds, args.asks)
        run = session.load(prog)
        mhz = aiclk_mhz()  # read with the grid running
        log(f"loaded, aiclk {mhz} MHz")
        try:
            results = run.thread_results(timeout=60.0)
        except km.ThreadFailed as e:
            results = e.results
        for c, r in results.items():
            log(f"cluster {c}: {r.state}, code {r.code}, error {r.error}")
        run.stop()
        got = np.zeros((128, LINE), np.uint32)
        for c, t in run.threads.items():
            lines = t.read("out").view(np.uint32)[: 128 * LINE].reshape(128, LINE)
            mine = (lines[:, WHO] != 0) & ((lines[:, WHO] - 1) >> 2 == c)
            got[mine] = lines[mine]
    return results, got, mhz


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--clusters", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--cores", type=int, default=0, help="0: every core of the grid")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--asks", type=int, default=100)
    args = p.parse_args()
    results, got, mhz = rounds(args)
    lead = results[min(results)]
    w = list(lead.words) + [0] * 6
    served = got[:, WHO] != 0
    cores = args.cores or int(served.sum())
    lines = got[:cores]
    calls = lines[:, CALLS].astype(np.int64)
    cycles = (lines[:, SUM_HI].astype(np.int64) << 32) + lines[:, SUM_LO]
    cpus = sorted({int(v) - 1 for v in lines[:, WHO] if v})
    out = {
        "clusters": args.clusters,
        "cores": cores,
        "x280_cores": len(cpus),
        "rounds": args.rounds,
        "asks": args.asks,
        "aiclk_mhz": mhz,
        "codes": {c: CODES.get(r.code, r.code) for c, r in results.items()},
        "rounds_done": w[0],
        "timeouts": w[1],
        "exact": bool(
            (lines[:, HASH] == expected(cores, args.rounds, args.asks)).all()
            and (lines[:, ROUNDS] == args.rounds).all()
        ),
        "ask_us_mean": round(float(cycles.sum()) / max(calls.sum(), 1) / mhz, 3),
        "ask_us_max": round(float(lines[:, MAX].max(initial=0)) / mhz, 3),
        "round_us_mean": round(((w[3] << 32) + w[2]) / max(w[0], 1) / TICKS_PER_US, 1),
        "round_us_best": round(w[4] / TICKS_PER_US, 1),
        "asks_per_s": 0,
    }
    if w[2] or w[3]:
        total_s = ((w[3] << 32) + w[2]) / (TICKS_PER_US * 1e6)
        out["asks_per_s"] = round(int(calls.sum()) / total_s)
    out["ok"] = (
        all(r.ran and r.code == 0 and r.error == "none" for r in results.values())
        and out["rounds_done"] == args.rounds
        and out["exact"]
    )
    print(json.dumps(out), flush=True)
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
