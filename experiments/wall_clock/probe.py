"""Tensix wall clock: its rate against host time, and one core's clock against another's
(round trips from core (0, 0) to up to six cores, twice, a spin apart). One JSON line.

Example: python probe.py --cores 2 --smoke; python probe.py"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tests" / "hardware" / "feeder"))

import commands as rec  # noqa: E402
import fed  # noqa: E402

import komorebi as km  # noqa: E402

CORES = [(0, 0), (11, 0), (0, 9), (11, 9), (5, 4), (6, 5), (1, 0)]  # master first
WORDS = 64
POLLS = (
    48  # peer words: polls until each ping was seen (1: it came before the peer looked)
)


def run(session, spin, cores):
    """Runs the probe once on `cores`; returns (host seconds, rows by core index)."""
    xy = []
    for x, y in cores:
        cx, cy = session.device.worker_core(x, y)
        xy.append(cx | cy << 16)
    # A row in every core of the grid at one L1 address; the probe's cores fill theirs.
    gx, gy = session.grid
    out = fed.int32_tensor(session, np.zeros((gx * gy, WORDS)), (gx, gy))
    args = []
    peers = len(cores) - 1
    for i, (x, y) in enumerate(cores):
        tail = xy[1:] if i == 0 else [xy[0]]
        args.append((x, y, [out.address, i, peers, spin, *tail]))
    launch = session.launch()
    launch.kernel(
        str(HERE / "probe.cpp"),
        fed.core_set(cores),
        0,
        [],
        [],
        [str(rec.INCLUDE)],
        args,
    )
    fed.log(f"launch, spin {spin} cycles")
    t0 = time.perf_counter()
    launch.run()
    session.device.finish()
    host = time.perf_counter() - t0
    fed.log(f"run of {spin} cycles: {host:.3f} s")
    rows = rec.as_uint32(fed.read_int32(out)).astype(np.int64)
    fed.log("read back")
    by_index = {int(r[6]) & 0xFFFF: r for r in rows if int(r[6]) >> 16 == 0xC10C}
    return host, [by_index[i] for i in range(len(cores))]


def wrap(d):
    return int(d) % (1 << 32)


def peer_table(runs, cores):
    """Per peer over `runs`: offset from the master's clock at the round trip's midpoint
    (cycles), the round trip, and its ticks per master tick between the two rounds.
    A round whose ping came before the peer's kernel looked is counted, not used."""
    peers = []
    for p, core in enumerate(cores[1:]):
        offsets, rtts, ratios, early = [], [], [], 0
        for _, rows in runs:
            m = rows[0]
            mid, arrival, ok = [], [], True
            for r in range(2):
                sent, got, reply = (int(v) for v in m[8 + 4 * (p * 2 + r) :][:3])
                if int(rows[p + 1][POLLS + r]) <= 1:
                    early += 1
                    ok = False
                    continue
                rtt = wrap(reply - sent)
                mid.append(sent + rtt / 2)
                arrival.append(got)
                rtts.append(rtt)
                d = wrap(got - sent) - rtt / 2
                offsets.append(d if d < 1 << 31 else d - (1 << 32))
            if ok:
                ratios.append(
                    wrap(arrival[1] - arrival[0]) / ((mid[1] - mid[0]) % (1 << 32))
                )
        rate = round((float(np.mean(ratios)) - 1) * 1e6, 3) if ratios else None
        peers.append(
            {
                "core": list(core),
                "offset_cycles": [round(o, 1) for o in offsets],
                "rtt_cycles": [min(rtts), max(rtts)] if rtts else [],
                "rate_vs_master_ppm": rate,
                "early_rounds": early,
            }
        )
    return peers


def missed(runs):
    """Pings or replies that did not come, and flushes that did not end, in `runs`."""
    return sum(int(r[7]) for _, rows in runs for r in rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cores", type=int, default=len(CORES), help="master and peers")
    p.add_argument("--smoke", action="store_true", help="one run of 0.1 s only")
    args = p.parse_args()
    cores = CORES[: args.cores]
    with km.Session() as session:
        # After the open: the clock idles lower while no device is open.
        mhz = fed.aiclk_mhz()
        fed.log(f"device open, AICLK {mhz} MHz")
        first = run(session, mhz * 10**5, cores)  # compile and warm up (0.1 s)
        spins = {0.5: [], 2.0: []}
        for _ in range(0 if args.smoke else 3):
            for s in spins:
                spins[s].append(run(session, int(s * mhz * 10**6), cores))
    if args.smoke:
        out = {"missed": missed([first]), "peers": peer_table([first], cores)}
        print(json.dumps(out), flush=True)
        return

    # Rate: master ticks against host seconds, from the difference of the two spins.
    def ticks(rows):
        return int(rows[0][4]) | int(rows[0][5]) << 32

    short, long_ = spins[0.5], spins[2.0]
    dh = np.mean([h for h, _ in long_]) - np.mean([h for h, _ in short])
    dt = np.mean([ticks(r) for _, r in long_]) - np.mean([ticks(r) for _, r in short])
    out = {
        "missed": missed([first, *short, *long_]),
        "sysfs_mhz": mhz,
        "measured_mhz": round(dt / dh / 1e6, 2),
        "host_s": {s: [round(h, 4) for h, _ in v] for s, v in spins.items()},
        "peers": peer_table(long_, cores),
    }
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
