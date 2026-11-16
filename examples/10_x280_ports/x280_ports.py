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
from komorebi.bench import power
from komorebi.session.process import aiclk_mhz

HERE = Path(__file__).resolve().parent
STARTED = time.monotonic()
TICKS_PER_MICROSECOND = 50  # kmx_time() counts the 50 MHz timer
LINE = 16  # words of a core's line in region `lines` (x280_ports.h)
HASH, ROUNDS, CALLS, SUM_LOW, SUM_HIGH, LARGEST, SERVED_BY = range(7)
CODES = {0: "ok", 1: "bad words or region", 2: "late or stopped"}
MASK_32_BITS = 0xFFFFFFFF
PULL_LINE = 8  # words of a core's line per kind of pass (x280_ports.h, HP_*)
PULL_NAMES = ["equal", "pull", "log", "replay", "pick"]
PULL_KINDS = len(PULL_NAMES)
(
    PASS_UNITS, PASS_WORK, PASS_SPAN, PASS_HASH, PASS_PULLS, PASS_WAITED,
    PASS_SERVED_BY,
) = range(7)  # fmt: skip


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def expected(cores, rounds, asks):
    """Each core's hash of the last round (x280_ports.h in numpy)."""
    core = np.arange(cores, dtype=np.uint64)
    running_hash = ((rounds * 0x9E3779B1) & MASK_32_BITS) ^ (
        (core * 0x85EBCA77) & MASK_32_BITS
    )
    for ask in range(1, asks + 1):
        value = (
            ((ask * 0xC2B2AE3D) & MASK_32_BITS)
            ^ ((core << 16) & MASK_32_BITS)
            ^ 0x27D4EB2F
        )
        running_hash = ((running_hash ^ value) * 0x01000193) & MASK_32_BITS
    return running_hash.astype(np.uint32)


def program(clusters, cores, rounds, asks):
    """The grid (the reader asks), the thread on `clusters` and every core's port."""
    ports_program = km.Program("x280_ports", kernels=HERE / "kernels")
    ports_program.thread(
        clusters=clusters,
        sources=[HERE / "thread" / "ports.c"],
        includes=[HERE],
        regions={"lines": 4 * LINE * 128},
        words=[rounds, cores, asks],
        deadline=30.0,
    )
    # every core of the grid, dealt over the X280 cores in grid order
    ports_program.ports()
    return ports_program


def rounds(args):
    """Loads, waits for the thread, reads every cluster's lines."""
    with km.Session(log=log) as session:
        ports_program = program(args.clusters, args.cores, args.rounds, args.asks)
        run = session.load(ports_program)
        megahertz = aiclk_mhz()  # read with the grid running
        log(f"loaded, aiclk {megahertz} MHz")
        try:
            results = run.thread_results(timeout=60.0)
        except km.ThreadFailed as failure:
            results = failure.results
        for cluster, result in results.items():
            log(
                f"cluster {cluster}: {result.state}, code {result.code}, "
                f"error {result.error}"
            )
        run.stop()
        got = np.zeros((128, LINE), np.uint32)
        for cluster, thread in run.threads.items():
            lines = (
                thread.read("lines").view(np.uint32)[: 128 * LINE].reshape(128, LINE)
            )
            mine = (lines[:, SERVED_BY] != 0) & (
                (lines[:, SERVED_BY] - 1) >> 2 == cluster
            )
            got[mine] = lines[mine]
    return results, got, megahertz


def costs(units, base, extra, centre, half):
    """Each unit's cycles (HX_COST in numpy)."""
    distance = np.abs(np.arange(units, dtype=np.int64) - centre)
    return base + np.where(distance < half, extra * (half - distance) // half, 0)


def marks(first, count):
    """HX_MARK of units first to first + count - 1, summed mod 2^32."""
    units = np.arange(first, first + count, dtype=np.uint64)
    x = (units * 0x9E3779B1) & MASK_32_BITS
    return int((((x ^ (x >> 15)) * 0x85EBCA77) & MASK_32_BITS).sum() & MASK_32_BITS)


def pull_program(args, cores):
    """The grid, the pull thread and the ports: every core served by the first
    `--x280-cores` X280 cores (1, 4 or 12), dealt in grid order."""
    clusters = [1] if args.x280_cores <= 4 else [1, 2, 3]
    pull_thread_program = km.Program("x280_pull", kernels=HERE / "kernels")
    ranges = -(-args.units // args.per)
    pull_thread_program.thread(
        clusters=clusters,
        sources=[HERE / "thread" / "pull.c"],
        includes=[HERE],
        regions={
            "lines": 4 * PULL_LINE * PULL_KINDS * 128,
            "log": 4 * 128 * (1 + ranges),
        },
        words=[args.units, args.per, args.base, args.extra, args.centre, args.half],
        deadline=60.0,
    )
    serving = [(cluster, x280_core) for cluster in clusters for x280_core in range(4)][
        : args.x280_cores
    ]
    pull_thread_program.ports(
        {core: serving[core * len(serving) // cores] for core in range(cores)}
    )
    return pull_thread_program


def pull_pass(args):
    """Runs the pull thread; its result and every core's line of every kind."""
    with km.Session(log=log) as session:
        grid_x, grid_y = session.grid  # the kernels' grid
        cores = grid_x * grid_y
        run = session.load(pull_program(args, cores))
        megahertz = aiclk_mhz()
        log(f"loaded, {cores} cores, aiclk {megahertz} MHz")
        try:
            results = run.thread_results(timeout=120.0)
        except km.ThreadFailed as failure:
            results = failure.results
        for cluster, result in results.items():
            log(
                f"cluster {cluster}: {result.state}, code {result.code}, "
                f"error {result.error}"
            )
        run.stop()
        got = np.zeros((PULL_KINDS, cores, PULL_LINE), np.uint32)
        for cluster, thread in run.threads.items():
            words = PULL_KINDS * 128 * PULL_LINE
            lines = (
                thread.read("lines")
                .view(np.uint32)[:words]
                .reshape(PULL_KINDS, 128, PULL_LINE)
            )
            mine = (lines[:, :cores, PASS_SERVED_BY] != 0) & (
                (lines[:, :cores, PASS_SERVED_BY] - 1) >> 2 == cluster
            )
            got[mine] = lines[:, :cores][mine]
    return results, got, megahertz


def pull_main(args):
    """The pull mode: one JSON line with each kind's pass time, slowest over mean, the
    pulls and their cost, and whether every unit ran exactly once."""
    results, got, megahertz = pull_pass(args)
    cores = got.shape[1]
    lead = results[min(results)]
    ticks = list(lead.words) + [0] * PULL_KINDS
    ranges = -(-args.units // args.per)
    total = marks(0, args.units)
    unit_cost = costs(args.units, args.base, args.extra, args.centre, args.half)
    starts = [
        core * (args.units // cores) + min(core, args.units % cores)
        for core in range(cores)
    ]
    lengths = [
        args.units // cores + (core < args.units % cores) for core in range(cores)
    ]
    model = np.array(
        [
            unit_cost[start : start + length].sum()
            for start, length in zip(starts, lengths)
        ]
    )
    report = {
        "units": args.units,
        "per": args.per,
        "cores": cores,
        "x280_cores": args.x280_cores,
        "aiclk_mhz": megahertz,
        "unit_microseconds_mean": round(float(unit_cost.mean()) / megahertz, 2),
        "range_microseconds_mean": round(
            float(unit_cost.mean()) * args.per / megahertz, 1
        ),
        "equal_model_slowest_over_mean": round(float(model.max() / model.mean()), 3),
        "codes": {cluster: result.code for cluster, result in results.items()},
    }
    exact = True
    for kind, name in enumerate(PULL_NAMES):
        lines = got[kind].astype(np.int64)
        work = lines[:, PASS_WORK]
        pulls = int(lines[:, PASS_PULLS].sum())
        ok = (
            int(lines[:, PASS_UNITS].sum()) == args.units
            and (int(lines[:, PASS_HASH].sum()) & MASK_32_BITS) == total
        )
        if name == "equal":
            ok = ok and all(
                lines[core, PASS_HASH] == marks(starts[core], lengths[core])
                for core in range(cores)
            )
        if name == "pick":
            ok = ok and all(
                lines[core, PASS_PULLS] == len(range(core, ranges, cores))
                for core in range(cores)
            )
        if name == "replay":
            same = [PASS_UNITS, PASS_HASH, PASS_PULLS]
            ok = ok and bool(
                (got[kind][:, same] == got[PULL_NAMES.index("log")][:, same]).all()
            )
        exact = exact and ok
        report[name] = {
            "pass_microseconds": round(ticks[kind] / TICKS_PER_MICROSECOND, 1),
            "slowest_over_mean": round(float(work.max() / max(work.mean(), 1)), 3),
            "span_microseconds_max": round(
                float(lines[:, PASS_SPAN].max()) / megahertz, 1
            ),
            "pulls": pulls,
            "wait_microseconds_per_pull": round(
                float(lines[:, PASS_WAITED].sum()) / max(pulls + cores, 1) / megahertz,
                2,
            ),
            "exact": bool(ok),
        }
    report["exact"] = exact
    report["ok"] = exact and all(
        result.ran and result.code == 0 and result.error == "none"
        for result in results.values()
    )
    print(json.dumps(power.clocked(report)), flush=True)
    sys.exit(0 if report["ok"] else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clusters", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--cores", type=int, default=0, help="0: every core of the grid"
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--asks", type=int, default=100)
    parser.add_argument("--pull", action="store_true", help="the pull mode (README.md)")
    parser.add_argument("--x280-cores", type=int, choices=[1, 4, 12], default=12)
    parser.add_argument("--units", type=int, default=31000)
    parser.add_argument("--per", type=int, default=16, help="units per pulled range")
    parser.add_argument("--base", type=int, default=20250, help="a unit's cycles")
    parser.add_argument("--extra", type=int, default=16200, help="added at the peak")
    parser.add_argument("--centre", type=int, default=9300, help="the peak's unit")
    parser.add_argument("--half", type=int, default=6200, help="the peak's half-width")
    args = parser.parse_args()
    if args.pull:
        pull_main(args)
        return
    results, got, megahertz = rounds(args)
    lead = results[min(results)]
    result_words = list(lead.words) + [0] * 6
    served = got[:, SERVED_BY] != 0
    cores = args.cores or int(served.sum())
    lines = got[:cores]
    calls = lines[:, CALLS].astype(np.int64)
    cycles = (lines[:, SUM_HIGH].astype(np.int64) << 32) + lines[:, SUM_LOW]
    x280_cores_used = sorted(
        {int(served_by) - 1 for served_by in lines[:, SERVED_BY] if served_by}
    )
    report = {
        "clusters": args.clusters,
        "cores": cores,
        "x280_cores": len(x280_cores_used),
        "rounds": args.rounds,
        "asks": args.asks,
        "aiclk_mhz": megahertz,
        "codes": {
            cluster: CODES.get(result.code, result.code)
            for cluster, result in results.items()
        },
        "rounds_done": result_words[0],
        "timeouts": result_words[1],
        "exact": bool(
            (lines[:, HASH] == expected(cores, args.rounds, args.asks)).all()
            and (lines[:, ROUNDS] == args.rounds).all()
        ),
        "ask_microseconds_mean": round(
            float(cycles.sum()) / max(calls.sum(), 1) / megahertz, 3
        ),
        "ask_microseconds_max": round(
            float(lines[:, LARGEST].max(initial=0)) / megahertz, 3
        ),
        "round_microseconds_mean": round(
            ((result_words[3] << 32) + result_words[2])
            / max(result_words[0], 1)
            / TICKS_PER_MICROSECOND,
            1,
        ),
        "round_microseconds_best": round(result_words[4] / TICKS_PER_MICROSECOND, 1),
        "asks_per_s": 0,
    }
    if result_words[2] or result_words[3]:
        total_seconds = ((result_words[3] << 32) + result_words[2]) / (
            TICKS_PER_MICROSECOND * 1e6
        )
        report["asks_per_s"] = round(int(calls.sum()) / total_seconds)
    report["ok"] = (
        all(
            result.ran and result.code == 0 and result.error == "none"
            for result in results.values()
        )
        and report["rounds_done"] == args.rounds
        and report["exact"]
    )
    print(json.dumps(power.clocked(report)), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
