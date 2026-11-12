"""The dispatch kernel's idle pass period without a thread and with one whose empty ring
core 0 looks at every fourth pass: load, one record, an idle wait, stop; ticks over
passes. Prints one JSON line. Usage: python ring_cost.py [--idle 1.0] [--reps 3]
"""

import argparse
import json
import time
from pathlib import Path

import komorebi as km

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "tests" / "hardware" / "thread" / "kernels"  # every record a NOP
SPIN = ROOT / "tests" / "hardware" / "thread" / "spin.c"  # waits for its stop
T0 = time.monotonic()
NS_PER_TICK = 20.0  # the X280 timer runs at 50 MHz


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def program(ring):
    prog = km.Program("ring_cost", kernels=KERNELS)
    if ring:
        prog.thread(sources=[SPIN], regions={"unused": 64}, deadline=30.0)
    return prog


def one(session, ring, idle):
    """The pass period (ns) and passes of one idle run."""
    run = session.load(program(ring))
    run.wait(run.issue(1, (0, 1)), timeout=5.0)
    time.sleep(idle)
    if ring:
        run.thread_result(timeout=0.5)
    r = run.stop()
    return round(r["ticks"] * NS_PER_TICK / r["passes"], 1), r["passes"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--idle", type=float, default=1.0, help="seconds per run")
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()
    out = {"idle_s": args.idle, "no_ring_ns": [], "ring_ns": []}
    with km.Session(log=log) as session:
        for _ in range(args.reps):
            for ring in (False, True):
                ns, passes = one(session, ring, args.idle)
                log(f"ring {ring}: {ns} ns per pass, {passes} passes")
                out["ring_ns" if ring else "no_ring_ns"].append(ns)
    lo = min(out["no_ring_ns"])
    out["ratio"] = round(min(out["ring_ns"]) / lo, 2)
    out["wall_s"] = round(time.monotonic() - T0, 1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
