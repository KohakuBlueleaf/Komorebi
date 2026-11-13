"""Landing experiment: which protocol lands a 64-byte multicast record safely. One
JSON line per run: the X280's time per record and the summed core counters.

    python landing.py --proto a --depth 1 --records 100 [--stamp]
"""

import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
sys.path.insert(0, BUILD)

import _landing_x280 as x280mod  # noqa: E402
import numpy as np  # noqa: E402

import komorebi as km  # noqa: E402

KERNEL = os.path.join(HERE, "kernels", "worker.cpp")
IMAGE = os.path.join(BUILD, "landing-seq.bin")
CTL_CB, CTL_BYTES, CNT_CB = 8, 1024, 9  # template kCtlCb; worker.cpp kCntCb
N_CORES = 120

# Counter word indices (kernels/landing.h LC_*).
LC_NAMES = [
    "magic",
    "accepted",
    "torn",
    "early",
    "idle_lo",
    "idle_hi",
    "check_lo",
    "check_hi",
    "check_max",
    "check_min",
    "checks",
    "torn_seq",
    "torn_word",
    "torn_value",
    "healed",
    "heal_polls",
    "lost",
    "lost_seq",
    "stamp_cost",
    "tag",
    "last_seq",
]
LC = {name: i for i, name in enumerate(LC_NAMES)}
LC_WORDS = 32
LANDING_MAGIC = 0x1A4D0000


def log(msg):
    print(msg, flush=True)


def loadavg():
    with open("/proc/loadavg") as f:
        return f.read().split()[0]


def program(card, plan, cnt_addr, n, defines):
    """One data-movement kernel on every core; its args are built after plan.start()."""
    gx, gy = plan.grid_x, plan.grid_y
    cores = [(0, 0, gx - 1, gy - 1)]
    args = [
        (x, y, [cnt_addr, n, y * gx + x] + list(plan.worker_args(x, y)))
        for y in range(gy)
        for x in range(gx)
    ]
    launch = card.launch()
    launch.kernel(KERNEL, cores, 0, [], defines, [], args)
    launch.cb(cores, CTL_CB, "uint32", CTL_BYTES, CTL_BYTES)
    launch.cb(cores, CNT_CB, "uint32", LC_WORDS * 4, LC_WORDS * 4)
    return launch


def summarize(cnt, n, mhz):
    """Sum the per-core counters ([cores, 32] uint32) into one record."""
    c = {k: cnt[:, i] for k, i in LC.items()}
    idle = c["idle_lo"] + (c["idle_hi"] << 32)
    check_sum = c["check_lo"] + (c["check_hi"] << 32)
    checks = int(c["checks"].sum())
    torn_cores = [
        {
            "core": i,
            "torn": int(c["torn"][i]),
            "first_seq": int(c["torn_seq"][i]),
            "word": int(c["torn_word"][i]),
            "value": hex(int(c["torn_value"][i])),
            "healed": int(c["healed"][i]),
            "heal_polls_max": int(c["heal_polls"][i]),
        }
        for i in range(cnt.shape[0])
        if c["torn"][i]
    ]
    lost_cores = [
        {"core": i, "seq": int(c["lost_seq"][i])}
        for i in range(cnt.shape[0])
        if c["lost"][i]
    ]
    magic_ok = int(
        sum(int(c["magic"][i]) == (LANDING_MAGIC | i) for i in range(cnt.shape[0]))
    )
    out = {
        "cores_finished": magic_ok,
        "cores_all_accepted": int((c["accepted"] == n).sum()),
        "accepted": int(c["accepted"].sum()),
        "torn": int(c["torn"].sum()),
        "healed": int(c["healed"].sum()),
        "early": int(c["early"].sum()),
        "idle_polls": int(idle.sum()),
        "lost": int(c["lost"].sum()),
        "tag_ok": bool((c["tag"] == c["tag"][0]).all()),
        "stamp_cost_cycles": int(c["stamp_cost"].max()),
    }
    if checks:
        out["check_cycles_mean"] = round(float(check_sum.sum()) / checks, 2)
        out["check_cycles_min"] = int(c["check_min"][c["checks"] > 0].min())
        out["check_cycles_max"] = int(c["check_max"].max())
        out["check_ns_mean"] = round(out["check_cycles_mean"] / mhz * 1e3, 2)
    if torn_cores:
        out["torn_cores"] = len(torn_cores)
        out["first_torn"] = torn_cores[:8]
    if lost_cores:
        out["lost_cores"] = lost_cores[:8]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--proto", choices=["a", "b"], required=True)
    p.add_argument("--depth", type=int, choices=[1, 4], required=True)
    p.add_argument("--records", type=int, required=True)
    p.add_argument("--stamp", action="store_true", help="time the acceptance check")
    p.add_argument("--pll", type=int, default=1750)
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args()
    n = args.records
    record = {"proto": args.proto, "depth": args.depth, "records": n}
    record["stamp"] = args.stamp
    record["load_start"] = loadavg()
    t_start = time.time()

    with km.Session() as card:
        log(f"device open {time.time() - t_start:.1f}s")
        record.update(run(card, args, n))
    record["load_end"] = loadavg()
    record["wall_s"] = round(time.time() - t_start, 1)
    print(json.dumps(record), flush=True)


def run(card, args, n):
    runtime = x280mod.Runtime(card.device.mesh_capsule(), pll_mhz=args.pll)
    plan = runtime.plan(IMAGE, args.depth, 1 if args.proto == "b" else 0)
    gx, gy = plan.grid_x, plan.grid_y
    assert gx * gy == N_CORES, (gx, gy)
    log(f"plan: grid {gx}x{gy} panels {plan.panels} depth {plan.depth}")

    cnt = card.tensor(np.zeros((N_CORES, LC_WORDS), np.uint32), "uint32")
    defines = [("PROTO_B", "1")] if args.proto == "b" else []
    if args.stamp:
        defines.append(("STAMP", "1"))

    # The X280 starts first: it picks the tag the worker args carry.
    session = plan.start(n)
    finished = False
    log("launch")
    t0 = time.perf_counter()
    try:
        program(card, plan, cnt.address, n, defines).run()
        ticks = plan.finish(session, args.timeout)
        finished = True
    finally:
        if not finished:  # stop the X280 image before the device closes
            plan.finish(session, 5.0)
    wall = time.perf_counter() - t0
    mhz = plan.aiclk_mhz()
    log(f"done: {ticks} ticks, host wall {wall:.2f}s")

    # Close the X280 runtime before reading anything back.
    del plan, runtime
    gc.collect()
    log("x280 runtime closed")

    counters = cnt.read().astype(np.int64)
    log("counters read back")
    out = {
        "ticks": ticks,
        "us_per_record": (
            round(ticks / x280mod.MTIME_HZ * 1e6 / n, 4) if ticks else None
        ),
        "host_wall_s": round(wall, 2),
        "aiclk_mhz": mhz,
    }
    out.update(summarize(counters, n, mhz))
    out["ok"] = bool(
        ticks
        and out["cores_finished"] == N_CORES
        and out["cores_all_accepted"] == N_CORES
        and out["lost"] == 0
        and out["tag_ok"]
    )
    return out


if __name__ == "__main__":
    main()
