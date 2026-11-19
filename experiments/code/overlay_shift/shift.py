"""Two programs with overlays of their own in one run, first (times 1, xor 2) and
second (plus 1: the run's 3), and an added overlay (4); each TAG names its overlays in
its own numbers, every core's result, number and shift checked. One JSON line.
Example: python shift.py --rounds 4"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.program import several as placed

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
M32 = 0xFFFFFFFF
STARTED = time.monotonic()
OVERLAYS = {  # name: (program, its own number, the host's arithmetic)
    "times": ("first", 1, lambda v: (v * 3 + 1) & M32),
    "xor": ("first", 2, lambda v: v ^ 0xA5A5A5A5),
    "plus": ("second", 1, lambda v: (v + 1000) & M32),
    "added": ("first", 4, lambda v: v ^ 0x00FF00FF),  # the run's 4, added after
}


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def programs(lines):
    """first and second in the runtime's form, the same kernels and buffer; each
    with its own overlays, numbered from 1."""
    made = {}
    for name in ("first", "second"):
        made[name] = km.Program(
            name,
            kernels=KERNELS / "tag",
            buffers=[km.Buffer(0, "uint32", 64, 1)],
            compile_args={"reader": lines.tensor_args()},
            handlers={"tag": 1},
        )
    for overlay, (program, number, _) in OVERLAYS.items():
        if overlay != "added":
            made[program].overlay(overlay, KERNELS / overlay, number=number)
    # the first program's slots are the run's; every program asks for run.addon's path
    made["first"].slots(count={"reader": 3}, bytes={})
    made["second"].slots(bytes={})
    return made


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=0, help="0: the whole grid")
    parser.add_argument("--rounds", type=int, default=4, help="commands an overlay")
    args = parser.parse_args()
    report = {"commands": 0, "wrong": 0, "times": {}}
    with km.Session(log=log) as session:
        cores = args.cores or session.grid[0] * session.grid[1]
        lines = session.empty((cores, 16), "uint32")
        made = programs(lines)
        run = session.load([made["first"], made["second"]])  # every core holds both
        rng = np.random.default_rng(1)
        try:
            report["numbers"] = {  # each program's own overlays as the run numbers them
                name: {o.name: o.number for o in placed.own_overlays(program)}
                for name, program in made.items()
            }
            report["added"] = run.addon("added", KERNELS / "added")
            for _ in range(args.rounds):
                for overlay, (owner, number, host) in OVERLAYS.items():
                    seed = int(rng.integers(0, M32))
                    handler = made[owner].handler("tag")
                    words = [number, seed, lines.address, 0]
                    event = run.wait(run.submit(handler, (0, cores), words), 30.0)
                    if event.kind != "done":
                        raise km.KomorebiError(f"{overlay}: {event}")
                    report["times"].setdefault(overlay, []).append(event.microseconds)
                    (pages,) = session.read(lines)
                    got = pages.reshape(cores, -1).view(np.uint32).astype(np.int64)
                    want = [host((seed ^ core) & M32) for core in range(cores)]
                    shift = made[owner].overlay_shift
                    wrong = int((got[:, 1] != want).sum())
                    wrong += int((got[:, 2] != number).sum())
                    wrong += int((got[:, 3] != shift).sum())
                    report["wrong"] += wrong
                    report["commands"] += 1
                    log(f"{overlay} ({owner}'s {number}): {event}, wrong {wrong}")
        finally:
            run.stop()
    report["ok"] = report["wrong"] == 0 and report["commands"] == 4 * args.rounds
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
