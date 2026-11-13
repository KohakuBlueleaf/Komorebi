"""Atomic answers probe: every asking core reserves K positions with non-posted atomic
adds on random cores, one at a time, and waits (bounded) for each returned old value.
The tails are on the asking cores themselves, or on counter tiles that only answer.
One JSON line. python stress.py [--counters 1] [--fixed-return]"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
W = 1280  # stress.h: kWords
TAIL, STAT, SENT, LOSTN, TABLE, TAILARR = 0, 16, 112, 232, 480, 640
MHZ = 1350
T0 = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cores", type=int, default=119, help="asking cores 0 to cores - 1")
    p.add_argument("--k", type=int, default=20000, help="reservations per core")
    p.add_argument("--timeout", type=int, default=3000000, help="answer wait, cycles")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--counters", type=int, default=0, help="tiles after the askers holding tails"
    )
    p.add_argument(
        "--fixed-return", action="store_true", help="one return line for every add"
    )
    return p.parse_args()


def main():
    a = parse()
    with km.Session() as session:
        log("device open")
        gx, gy = session.grid
        total, n = gx * gy, a.cores
        if n < 1 or n + a.counters > total:
            raise SystemExit(f"{n} askers and {a.counters} counters: {total} cores")
        noc = []
        for c in range(total):
            x, y = session.device.worker_core(c % gx, c // gx)
            noc.append(x | y << 16)
        init = np.zeros((total, W), dtype=np.uint32)
        init[:, TABLE : TABLE + total] = np.array(noc, dtype=np.uint32)
        blk = session.tensor(init, "uint32", shard_grid=(gx, gy))
        rng = np.random.default_rng(a.seed)
        args, cores = [], []
        for c in range(n):
            x, y = c % gx, c // gx
            seed = int(rng.integers(1, 1 << 31))
            ctr = [n, a.counters, int(a.fixed_return)]
            args.append((x, y, [blk.address, n, a.k, c, seed, a.timeout, *ctr]))
            cores.append((x, y, x, y))
        launch = session.launch()
        launch.kernel(str(HERE / "writer.cpp"), cores, 1, [], [], [str(HERE)], args)
        log("launch")
        t = time.perf_counter()
        launch.run()
        session.device.finish()
        host_s = time.perf_counter() - t
        log(f"done in {host_s:.2f} s")
        rows = blk.read().astype(np.int64)
        log("read back")
    print(json.dumps(report(a, rows, host_s)), flush=True)


def report(a, rows, host_s):
    """Reservations, lost answers, whether every tail is exact, latency and rate."""
    n = a.cores
    st = rows[:n, STAT : STAT + 10]
    if a.counters:
        tails = np.array([rows[n + d % a.counters, TAILARR + 4 * d] for d in range(n)])
    else:
        tails = rows[:n, TAIL]
    added = rows[:n, SENT : SENT + n].sum(0) + rows[:n, LOSTN : LOSTN + n].sum(0)
    resv = int(st[:, 2].sum())
    cycles = int((st[:, 8] | st[:, 9] << 32).max())
    return {
        "askers": n,
        "counters": a.counters,
        "fixed_return": a.fixed_return,
        "done": int((st[:, 0] == 1).sum()),
        "fail": sorted({int(v) for v in st[:, 1] if v}),
        "reservations": resv,
        "lost": int(st[:, 3].sum()),
        "lost_cores": [int(c) for c in np.nonzero(st[:, 3])[0]],
        "older_answers": int(st[:, 7].sum()),
        "tails_exact": bool((tails == added).all()),
        "latency_mean": round(float((st[:, 5] | st[:, 6] << 32).sum()) / max(resv, 1)),
        "latency_max": int(st[:, 4].max()),
        "per_second_M": round(resv / max(cycles, 1) * MHZ, 1),
        "host_s": round(host_s, 2),
    }


if __name__ == "__main__":
    main()
