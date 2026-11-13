"""What a message to one core costs: a put by distance (hop table) and many senders
bumping one port (bump table). One JSON line.

Example: python message_cost.py hops --reps 1000; python message_cost.py bumps"""

import argparse
import json
import os
import random
import threading
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
NOC_X, NOC_Y = 17, 12  # the NoC torus of the chip, in NoC0 coordinates
SENDER = (6, 5)  # hop table: logical core, inside the grid
CB_BLOCK, CB_TIMES, CB_SLOT = 0, 1, 2
SLOT_BYTES = 64 + 16 * 128  # kSlotBytes
FAILED = 0xFFFFFFFF
T0 = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


class Clock:
    """AICLK (MHz) and card power (W) sampled every 50 ms while it runs."""

    def __init__(self):
        card = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
        base = Path(f"/sys/class/tenstorrent/tenstorrent!{card}")
        self.aiclk = base / "tt_aiclk"
        self.power = next(iter(sorted((base / "device" / "hwmon").glob("hwmon*"))))
        self.power = self.power / "power1_input"
        self.samples, self.stop = [], threading.Event()

    def read(self):
        mhz = int(self.aiclk.read_text().split()[0])
        return mhz, int(self.power.read_text()) / 1e6

    def run(self):
        while not self.stop.wait(0.05):
            self.samples.append(self.read())

    def __enter__(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()

    def summary(self):
        mhz = np.array([s[0] for s in self.samples] or [self.read()[0]])
        watts = np.array([s[1] for s in self.samples] or [self.read()[1]])
        return {
            "aiclk_mhz_min": int(mhz.min()),
            "aiclk_mhz_mean": round(float(mhz.mean()), 1),
            "power_w_min": round(float(watts.min()), 1),
            "power_w_mean": round(float(watts.mean()), 1),
            "samples": len(self.samples),
        }


def cores(cs):
    return [(x, y, x, y) for x, y in cs]


def physical(logical):
    """NoC0 coordinates of a logical core on this card (Tensix columns 7 and 10 are
    harvested: logical columns 0-5 are NoC x 1-6, 6-11 are 11-16; rows 2-11)."""
    x, y = logical
    return (x + 1 if x < 6 else x + 5), y + 2


def hops(src, dst, noc):
    """Router hops of a unicast: NoC0 goes +x then +y, NoC1 -x then -y (torus)."""
    (sx, sy), (dx, dy) = physical(src), physical(dst)
    if noc == 0:
        return (dx - sx) % NOC_X + (dy - sy) % NOC_Y
    return (sx - dx) % NOC_X + (sy - dy) % NOC_Y


def stats(cycles, mhz):
    c = np.asarray(cycles, dtype=np.float64)
    q = np.percentile(c, [0, 50, 99])
    return {
        "cycles": [int(v) for v in q],
        "microseconds": [round(v / mhz, 3) for v in q],
        "n": len(c),
    }


def uint32_rows(session, rows, cols):
    return session.tensor(np.zeros((rows, cols), np.uint32), "uint32")


def cb(index, size, where):
    return (where, index, "uint32", size, size)


def dm(name, where, args, riscv):
    """A data movement kernel on RISC-V `riscv` and its NoC; args [(x, y, words)]."""
    return (str(KERNELS / name), where, riscv, [], [], [str(KERNELS)], args)


def launch(session, kernels, cbs):
    """Runs one program of `kernels` and `cbs` and waits for it."""
    program = session.launch()
    for c in cbs:
        program.cb(*c)
    for k in kernels:
        program.kernel(*k)
    program.run()
    session.device.finish()


def noc_xy(session, logical):
    x, y = session.device.worker_core(*logical)
    return x | y << 16


def run_hops(session, grid, args, mhz):
    gx, gy = grid
    sx, sy = SENDER
    targets = [(x, sy) for x in range(gx) if x != sx]
    targets += [(sx, y) for y in range(gy) if y != sy]
    n, reps = len(targets), args.reps
    limit = int(mhz * 1e6 * 0.01)  # 10 ms per wait
    out = uint32_rows(session, 4 * n, reps)
    tag = random.getrandbits(15) | 1
    xy = [noc_xy(session, t) for t in targets]
    sync_limit = int(mhz * 1e6 * 2.0)  # the NoC1 kernel waits for the NoC0 one
    common = [out.address, 0, n, reps, 1, limit, sync_limit, tag]
    everyone = cores([(x, y) for y in range(gy) for x in range(gx)])
    sender = cores([SENDER])
    rt0 = [(sx, sy, common + xy)]
    rt1 = [(sx, sy, [common[0], 2 * n, n, reps, 0, *common[5:]] + xy)]
    log(f"hops: launch, {n} targets, {reps} repetitions")
    launch(
        session,
        [dm("hop_send.cpp", sender, rt0, 0), dm("hop_send.cpp", sender, rt1, 1)],
        [
            cb(CB_BLOCK, 2048, everyone),
            cb(CB_SLOT, SLOT_BYTES, everyone),
            cb(CB_TIMES, reps * 4, sender),
        ],
    )
    log("hops: done")
    got = out.read().astype(np.int64)
    log("hops: read back")
    rows, fits = [], {}
    for noc in (0, 1):
        for s, size in enumerate((64, 2048)):
            med, hop = [], []
            for k, t in enumerate(targets):
                c = got[noc * 2 * n + s * n + k]
                failed = int((c == FAILED).sum())
                ok = c[c != FAILED]
                h = hops(SENDER, t, noc)
                row = {"noc": noc, "bytes": size, "target": list(t), "hops": h}
                row["dx"], row["dy"] = t[0] - sx, t[1] - sy
                row.update(stats(ok, mhz) if len(ok) else {"n": 0})
                row["failed"] = failed
                rows.append(row)
                if len(ok):
                    med.append(np.median(ok))
                    hop.append(h)
            b, a = np.polyfit(hop, med, 1)
            fits[f"noc{noc}_{size}"] = {
                "fixed_cycles": round(float(a), 1),
                "per_hop_cycles": round(float(b), 2),
                "fixed_microseconds": round(float(a) / mhz, 3),
                "per_hop_ns": round(float(b) / mhz * 1e3, 2),
            }
    return {"sender": list(SENDER), "fits": fits, "rows": rows}


def run_bumps(session, grid, args, mhz):
    gx, gy = grid
    logical = [(i % gx, i // gx) for i in range(gx * gy)]
    reps = args.reps
    limit = int(mhz * 1e6 * 0.01)
    lead = int(mhz * args.lead_microseconds)
    results = {}
    for n in args.group_sizes:
        out = uint32_rows(session, 2, reps)
        tag = random.getrandbits(15) | 1
        recv, senders = logical[0], logical[1 : n + 1]
        sync = int(mhz * 1e6 * 2.0)  # cores start at different times
        head = [out.address, n, reps, lead, limit, tag, sync]
        rt_recv = [(*recv, head + [noc_xy(session, s) for s in senders])]
        rt_send = [
            (x, y, [noc_xy(session, recv), reps, limit, lead, tag, k, sync])
            for k, (x, y) in enumerate(senders)
        ]
        group = cores([recv, *senders])
        log(f"bumps: launch, {n} senders")
        launch(
            session,
            [
                dm("bump_recv.cpp", cores([recv]), rt_recv, 1),
                dm("bump_send.cpp", cores(senders), rt_send, 1),
            ],
            [cb(CB_SLOT, SLOT_BYTES, group), cb(CB_TIMES, 2 * reps * 4, cores([recv]))],
        )
        got = out.read().astype(np.int64)
        log(f"bumps: {n} senders read back")
        times, late = got[0], got[1]
        ok = times[times != FAILED]
        results[n] = {
            **(stats(ok, mhz) if len(ok) else {"n": 0}),
            "failed": int((times == FAILED).sum()),
            "late_bumps": int(late[times != FAILED].sum()),
        }
    return {
        "receiver": list(logical[0]),
        "lead_microseconds": args.lead_microseconds,
        "group_sizes": results,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["hops", "bumps"])
    p.add_argument("--reps", type=int, default=1024, help="a multiple of 16")
    p.add_argument("--group-sizes", type=int, nargs="+", default=[1, 8, 32, 60, 118])
    p.add_argument(
        "--lead-microseconds", type=float, default=40.0, help="bumps: release lead"
    )
    args = p.parse_args()
    if args.reps % 16:  # a result row is whole 64-byte DRAM lines
        p.error("--reps must be a multiple of 16")
    with km.Session() as session:
        grid = session.grid
        virtual = [noc_xy(session, (x, 0)) & 0xFFFF for x in range(grid[0])]
        log(f"device open, grid {grid}, NoC x of row 0: {virtual}")
        with Clock() as clock:
            mhz = clock.read()[0]
            run = run_hops if args.what == "hops" else run_bumps
            out = run(session, grid, args, mhz)
        out["clock"] = clock.summary()
        out["mhz_used"] = mhz
        out["noc_x_row0"] = virtual
        out["loadavg"] = Path("/proc/loadavg").read_text().split()[:3]
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
