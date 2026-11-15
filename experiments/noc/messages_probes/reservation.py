"""Reservation probe: do non-posted atomic adds return the value before the add, and do
concurrent adders get disjoint positions covering the range exactly. JSON line.

Example: python reservation.py --adders 2 --k 4"""

import argparse
import json
import time

import numpy as np

import komorebi as km

WORDS = 512  # per core
TARGET, READY, STAT, POS = 0, 4, 16, 32
X0 = 1000  # initial value of the reserved word
T0 = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def check(rows, adders, k):
    """Positions per adder against the expected increments; the summary dict."""
    intervals, stats = [], []
    for c in range(1, adders + 1):
        r = rows[c]
        st = [int(v) for v in r[STAT : STAT + 9]]
        stats.append(st)
        intervals.extend((int(r[POS + j]), 1 + (7 * c + j) % 32) for j in range(k))
    total = sum(n for _, n in intervals)
    intervals.sort()
    cover, at = True, X0
    for p, n in intervals:
        if p != at:
            cover = False
            break
        at += n
    cyc_sum = sum(s[6] | s[7] << 32 for s in stats)
    return {
        "adders": adders,
        "k": k,
        "total": total,
        "final": int(rows[0][TARGET]),
        "expected_final": X0 + total,
        "ready": int(rows[0][READY]),
        "disjoint_cover": cover and at == X0 + total,
        "first_bad": None if cover else [p, n, at],
        "done": all(s[0] == 1 for s in stats),
        "fail": {f: sum(s[1] == f for s in stats) for f in {s[1] for s in stats}},
        "late": sum(s[2] for s in stats),
        "ret_mask": sorted({s[3] for s in stats}),
        "ids_ok": all(s[8] == c for c, s in enumerate(stats, 1)),
        "cycles_min": min(s[4] for s in stats),
        "cycles_max": max(s[5] for s in stats),
        "cycles_mean": cyc_sum / (adders * k),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--adders", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    a = p.parse_args()
    assert a.k <= WORDS - POS
    with km.Session() as session:
        log("device open")
        gx, gy = session.grid
        cores = gx * gy
        assert 1 <= a.adders < cores
        init = np.zeros((cores, WORDS), dtype=np.uint32)
        init[0, TARGET] = X0
        out = session.tensor(init, "uint32", shard_grid=(gx, gy))
        tx, ty = session.device.worker_core(0, 0)
        args, ranges = [], []
        for c in range(1, a.adders + 1):
            x, y = c % gx, c // gx
            args.append((x, y, [out.address, tx | ty << 16, a.adders, a.k, c]))
            ranges.append((x, y, x, y))
        source = str(__file__).replace("reservation.py", "reservation.cpp")
        launch = session.launch()
        launch.kernel(source, ranges, 0, [], [], [], args)
        log("launch")
        launch.run()
        session.device.finish()
        log("kernel done")
        rows = out.read().astype(np.int64)
        log("read back")
    print(json.dumps(check(rows, a.adders, a.k)), flush=True)


if __name__ == "__main__":
    main()
