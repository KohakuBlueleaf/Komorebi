"""X280 cores as ranks of a dynamic run: the cores' messages go to the X280 ranks and
come back, each stepped at every visit, until their hops are used up; the dispatcher
ends the run once every line is quiet. Every end against a numpy replay; one JSON
line. See README.md."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.bench import power

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parents[1] / "src" / "komorebi_native" / "kernels"
RANK = {
    name: int(value)
    for name, value in re.findall(
        r"#define (k\w+) (\d+)u", (HERE / "kernels" / "rank.h").read_text()
    )
}
STARTED = time.monotonic()
M32 = 0xFFFFFFFF
CLUSTER = 1
COUNTERS = 1


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def grid_step(value):
    value ^= (value << 13) & M32
    value ^= value >> 17
    value ^= (value << 5) & M32
    return value


def x280_step(value, rank):
    return (value * 0x2C1B3C6D + 0x297A2D39 + rank) & M32


def seed_value(id_):
    value = ((id_ + 1) * 0x9E3779B1) & M32
    return value or 1


def replay(cores, x280, seeds, hops):
    """Each consumer's (consumed, sum, xor): cores by rank, then the X280 ranks."""
    totals = [[0, 0, 0] for _ in range(cores + x280)]
    for id_ in range(cores * seeds):
        value, left = seed_value(id_), hops + (id_ & 1)
        where = cores + value % x280  # the seed goes to an X280 rank
        while True:
            at_x280 = where >= cores
            value = x280_step(value, where) if at_x280 else grid_step(value)
            if left == 0:
                break
            left -= 1
            where = value % cores if at_x280 else cores + value % x280
        end = totals[where]
        end[0] += 1
        end[1] += value
        end[2] ^= (id_ * 0x9E3779B1 + value) & M32
    return totals


def program(args, results):
    """The cores' compute kernel under the runtime's reader and writer, both handlers
    dynamic, the X280 ranks on cluster 1's thread."""
    made = km.Program(
        "thread_rank",
        kernels={
            "reader": RUNTIME / "messages_reader.cpp",
            "compute": HERE / "kernels" / "compute.cpp",
            "writer": RUNTIME / "messages_writer.cpp",
        },
        args=lambda core: ([], [], [results.address]),
        handlers={"grid": RANK["kHandlerGrid"], "x280": RANK["kHandlerX280"]},
    ).dynamic(
        "grid",
        "x280",
        message_bytes=64,
        max_live=args.live,
        queue_messages=args.live,
        bins=4,
        counters=COUNTERS,
        x280_ranks=args.x280,
        x280_cluster=CLUSTER,
    )
    made.thread(
        cluster=CLUSTER,
        sources=[HERE / "thread" / "rank.c"],
        includes=[HERE / "kernels"],
        regions={"totals": 4 * RANK["kX280Words"] * 4},
        deadline=args.deadline + 10.0,
    )
    return made


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=12, help="core ranks")
    parser.add_argument("--x280", type=int, default=4, help="X280 ranks, 1 to 4")
    parser.add_argument("--seeds", type=int, default=64, help="messages a core seeds")
    parser.add_argument("--hops", type=int, default=100, help="visits a message makes")
    parser.add_argument("--live", type=int, default=1 << 13, help="max_live")
    parser.add_argument("--deadline", type=float, default=10.0, help="the RUN's, s")
    parser.add_argument("--wait", type=float, default=1.0, help="wait limit, seconds")
    args = parser.parse_args()
    out = {"cores": args.cores, "x280": args.x280, "seeds": args.seeds}
    out["hops"] = args.hops
    with km.Session(log=log) as session:
        grid = session.grid
        zeros = np.zeros((grid[0] * grid[1], RANK["kResWords"]), np.uint32)
        results = session.tensor(zeros, "uint32", shard_grid=grid)
        made = program(args, results)
        run = session.load(made, deadline=args.deadline, wait=args.wait)
        log(f"loaded: {args.cores} core ranks, {args.x280} X280 ranks")
        group = (0, args.cores + COUNTERS)
        started = time.perf_counter()
        handle = run.submit("grid", group, [args.seeds, args.hops], exclusive=True)
        event = run.wait(handle, timeout=args.deadline + 5.0)
        out["host_seconds"] = round(time.perf_counter() - started, 4)
        out["event"] = event.kind
        log(f"RUN: {event.kind} after {out['host_seconds']} s")
        thread_results = run.thread_results(timeout=10.0)
        thread_result = thread_results[CLUSTER]
        out["thread"] = [thread_result.state, thread_result.code]
        stopped = run.stop()
        out["dispatcher_error"] = stopped["error_name"]
        rows = results.read().astype(np.int64) & M32
        totals = np.frombuffer(run.threads[CLUSTER].read("totals"), np.uint32)
        totals = totals.reshape(-1, RANK["kX280Words"]).astype(np.int64)
    expected = replay(args.cores, args.x280, args.seeds, args.hops)
    got = [
        [
            int(row[RANK[f"kRes{name}"]])
            for name in ("Consumed", "SumLo", "SumHi", "Xor")
        ]
        for row in rows[: args.cores]
    ] + [
        [
            int(row[RANK[f"kX280{name}"]])
            for name in ("Consumed", "SumLo", "SumHi", "Xor")
        ]
        for row in totals[: args.x280]
    ]
    got = [[consumed, lo | high << 32, mixed] for consumed, lo, high, mixed in got]
    wrong = [index for index, (a, b) in enumerate(zip(got, expected)) if a != b]
    to_x280 = int(rows[: args.cores, RANK["kResToX280"]].sum())
    visits = int(rows[: args.cores, RANK["kResVisits"]].sum())
    takes = int(totals[: args.x280, RANK["kX280Takes"]].sum())
    sent = int(totals[: args.x280, RANK["kX280Sent"]].sum())
    out["messages"] = {
        "consumed_cores": sum(end[0] for end in got[: args.cores]),
        "consumed_x280": sum(end[0] for end in got[args.cores :]),
        "to_x280": to_x280,
        "x280_takes": takes,
        "x280_sent": sent,
        "core_visits": visits,
    }
    out["wrong_consumers"] = wrong
    out["gave"] = [int(g) for g in totals[: args.x280, RANK["kX280Gave"]]]
    out["rejected"] = int(rows[: args.cores, RANK["kResRejected"]].sum())
    out["ok"] = bool(
        out["event"] == "done"
        and out["thread"] == ["done", 0]
        and stopped["error"] == 0
        and not wrong
        and to_x280 == takes
        and sent == visits
        and not any(out["gave"])
        and out["rejected"] == 0
        and sum(end[0] for end in got) == args.cores * args.seeds
    )
    print(json.dumps(power.clocked(out)), flush=True)
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
