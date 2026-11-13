"""X280 ports: X280 cores and Tensix cores talk without the dispatcher. Per
round each Tensix core waits for its X280 core's salt, asks it `--asks` values, hashes
them and tells it the hash; the hashes are checked against numpy and the asks timed.
`--pull`: a pass over uneven units, equal ranges against pulled ones. See README.md."""

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
PULL_LINE = 8  # words of a core's line per kind of pass (x280_ports.h, HP_*)
PULL_NAMES = ["equal", "pull", "log", "replay", "pick"]
PULL_KINDS = len(PULL_NAMES)
P_UNITS, P_WORK, P_SPAN, P_HASH, P_PULLS, P_WAITED, P_WHO = range(7)


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


def costs(units, base, extra, centre, half):
    """Each unit's cycles (HX_COST in numpy)."""
    d = np.abs(np.arange(units, dtype=np.int64) - centre)
    return base + np.where(d < half, extra * (half - d) // half, 0)


def marks(first, count):
    """HX_MARK of units first to first + count - 1, summed mod 2^32."""
    u = np.arange(first, first + count, dtype=np.uint64)
    x = (u * 0x9E3779B1) & M32
    return int((((x ^ (x >> 15)) * 0x85EBCA77) & M32).sum() & M32)


def pull_program(args, cores):
    """The grid, the pull thread and the ports: every core served by the first
    `--x280-cores` X280 cores (1, 4 or 12), dealt in grid order."""
    clusters = [1] if args.x280_cores <= 4 else [1, 2, 3]
    prog = km.Program("x280_pull", kernels=HERE / "kernels")
    ranges = -(-args.units // args.per)
    prog.thread(
        clusters=clusters,
        sources=[HERE / "thread" / "pull.c"],
        includes=[HERE],
        regions={
            "out": 4 * PULL_LINE * PULL_KINDS * 128,
            "log": 4 * 128 * (1 + ranges),
        },
        words=[args.units, args.per, args.base, args.extra, args.centre, args.half],
        deadline=60.0,
    )
    cpus = [(c, h) for c in clusters for h in range(4)][: args.x280_cores]
    prog.ports({i: cpus[i * len(cpus) // cores] for i in range(cores)})
    return prog


def pull_pass(args):
    """Runs the pull thread; its result and every core's line of every kind."""
    with km.Session(log=log) as session:
        gx, gy = session.grid  # the kernels' grid
        cores = gx * gy
        run = session.load(pull_program(args, cores))
        mhz = aiclk_mhz()
        log(f"loaded, {cores} cores, aiclk {mhz} MHz")
        try:
            results = run.thread_results(timeout=120.0)
        except km.ThreadFailed as e:
            results = e.results
        for c, r in results.items():
            log(f"cluster {c}: {r.state}, code {r.code}, error {r.error}")
        run.stop()
        got = np.zeros((PULL_KINDS, cores, PULL_LINE), np.uint32)
        for c, t in run.threads.items():
            n = PULL_KINDS * 128 * PULL_LINE
            lines = (
                t.read("out").view(np.uint32)[:n].reshape(PULL_KINDS, 128, PULL_LINE)
            )
            mine = (lines[:, :cores, P_WHO] != 0) & (
                (lines[:, :cores, P_WHO] - 1) >> 2 == c
            )
            got[mine] = lines[:, :cores][mine]
    return results, got, mhz


def pull_main(args):
    """The pull mode: one JSON line with each kind's pass time, slowest over mean, the
    pulls and their cost, and whether every unit ran exactly once."""
    results, got, mhz = pull_pass(args)
    cores = got.shape[1]
    lead = results[min(results)]
    ticks = list(lead.words) + [0] * PULL_KINDS
    ranges = -(-args.units // args.per)
    total = marks(0, args.units)
    unit_cost = costs(args.units, args.base, args.extra, args.centre, args.half)
    starts = [
        i * (args.units // cores) + min(i, args.units % cores) for i in range(cores)
    ]
    lens = [args.units // cores + (i < args.units % cores) for i in range(cores)]
    model = np.array([unit_cost[a : a + n].sum() for a, n in zip(starts, lens)])
    out = {
        "units": args.units,
        "per": args.per,
        "cores": cores,
        "x280_cores": args.x280_cores,
        "aiclk_mhz": mhz,
        "unit_us_mean": round(float(unit_cost.mean()) / mhz, 2),
        "range_us_mean": round(float(unit_cost.mean()) * args.per / mhz, 1),
        "equal_model_slowest_over_mean": round(float(model.max() / model.mean()), 3),
        "codes": {c: r.code for c, r in results.items()},
    }
    exact = True
    for k, name in enumerate(PULL_NAMES):
        lines = got[k].astype(np.int64)
        work = lines[:, P_WORK]
        pulls = int(lines[:, P_PULLS].sum())
        ok = (
            int(lines[:, P_UNITS].sum()) == args.units
            and (int(lines[:, P_HASH].sum()) & M32) == total
        )
        if name == "equal":
            ok = ok and all(
                lines[i, P_HASH] == marks(starts[i], lens[i]) for i in range(cores)
            )
        if name == "pick":
            ok = ok and all(
                lines[i, P_PULLS] == len(range(i, ranges, cores)) for i in range(cores)
            )
        if name == "replay":
            same = [P_UNITS, P_HASH, P_PULLS]
            ok = ok and bool(
                (got[k][:, same] == got[PULL_NAMES.index("log")][:, same]).all()
            )
        exact = exact and ok
        out[name] = {
            "pass_us": round(ticks[k] / TICKS_PER_US, 1),
            "slowest_over_mean": round(float(work.max() / max(work.mean(), 1)), 3),
            "span_us_max": round(float(lines[:, P_SPAN].max()) / mhz, 1),
            "pulls": pulls,
            "wait_us_per_pull": round(
                float(lines[:, P_WAITED].sum()) / max(pulls + cores, 1) / mhz, 2
            ),
            "exact": bool(ok),
        }
    out["exact"] = exact
    out["ok"] = exact and all(
        r.ran and r.code == 0 and r.error == "none" for r in results.values()
    )
    print(json.dumps(out), flush=True)
    sys.exit(0 if out["ok"] else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--clusters", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--cores", type=int, default=0, help="0: every core of the grid")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--asks", type=int, default=100)
    p.add_argument("--pull", action="store_true", help="the pull mode (README.md)")
    p.add_argument("--x280-cores", type=int, choices=[1, 4, 12], default=12)
    p.add_argument("--units", type=int, default=31000)
    p.add_argument("--per", type=int, default=16, help="units per pulled range")
    p.add_argument("--base", type=int, default=20250, help="a unit's cycles")
    p.add_argument("--extra", type=int, default=16200, help="added at the peak")
    p.add_argument("--centre", type=int, default=9300, help="the peak's unit")
    p.add_argument("--half", type=int, default=6200, help="the peak's half-width")
    args = p.parse_args()
    if args.pull:
        pull_main(args)
        return
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
