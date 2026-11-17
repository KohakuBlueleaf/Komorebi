"""Example 24 on two cards: island 0 holds the X280 ranks; the cores of both islands
send to them and they to the cores of both (deliveries across the link); every consumer
and every core's visits against the numpy replay. One JSON line. See README.md."""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.bench import power

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("thread_rank", HERE / "thread_rank.py")
one = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(one)
RANK, CLUSTER, COUNTERS = one.RANK, one.CLUSTER, one.COUNTERS
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument(
        "--first", type=int, default=0, help="the group's first rank (island 0's)"
    )
    parser.add_argument(
        "--ranks", type=int, default=13, help="the group's ranks, counter tiles too"
    )
    parser.add_argument("--x280", type=int, default=4, help="X280 ranks, 1 to 4")
    parser.add_argument("--seeds", type=int, default=64, help="messages a core seeds")
    parser.add_argument("--hops", type=int, default=100, help="visits a message makes")
    parser.add_argument(
        "--to-cores", action="store_true", help="each seed to a core first"
    )
    parser.add_argument("--even", action="store_true", help="odd ids no hop more")
    parser.add_argument("--live", type=int, default=1 << 13, help="max_live")
    parser.add_argument("--deadline", type=float, default=10.0, help="the RUN's, s")
    parser.add_argument("--wait", type=float, default=1.0, help="wait limit, seconds")
    return parser.parse_args()


def parts(session, group):
    """Per island: (its first core of the group, the group's cores there)."""
    out = []
    for island in range(len(session.sessions)):
        first = session.graph.first(island)
        low = max(group[0], first)
        high = min(group[0] + group[1], first + session.graph.count(island))
        out.append((low - first, max(high - low, 0)))
    return out


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    flags = (RANK["kSeedToCores"] if args.to_cores else 0) | (
        RANK["kSeedEven"] if args.even else 0
    )
    group = (args.first, args.ranks)
    out = {"cards": cards, "group": list(group), "x280": args.x280}
    out |= {"seeds": args.seeds, "hops": args.hops, "flags": flags}
    with km.Session(cards=cards, log=log) as session:
        areas = [
            session.island(i).tensor(
                np.zeros((np.prod(session.island(i).grid), RANK["kResWords"])),
                "uint32",
                shard_grid=session.island(i).grid,
            )
            for i in range(len(session.sessions))
        ]
        if len({area.address for area in areas}) != 1:
            raise km.KomorebiError("the results areas are not at one address")
        made = one.program(args, areas[0])
        run = session.load(made, deadline=args.deadline, wait=args.wait)
        split = parts(session, group)
        out["parts"] = split
        cores = sum(max(n - COUNTERS, 0) for _, n in split)  # the message ranks
        log(f"loaded: {cores} core ranks over {split}, {args.x280} X280 ranks")
        started = time.perf_counter()
        ticket = run.submit(
            "grid", group, [args.seeds, args.hops, flags], exclusive=True
        )
        try:
            run.drain(timeout=args.deadline + 5.0)
        except km.CommandTimeout as error:
            out["timeout"] = str(error)[:300]
        out["host_seconds"] = round(time.perf_counter() - started, 4)
        out["events"] = [event.kind if event else None for event in run.event(ticket)]
        log(f"RUN: {out['events']} after {out['host_seconds']} s")
        holder = run.island(0)
        try:  # a give-up's recovery may have closed the runtime: said, not raised
            thread = holder.thread_results(timeout=10.0)[CLUSTER]
            out["thread"] = [thread.state, thread.code]
        except (km.KomorebiError, RuntimeError) as error:
            out["thread"] = str(error)[:200]
        try:
            out["kernel_errors"] = [part["error_name"] for part in run.stop()]
        except km.KomorebiError as error:
            out["stop_error"] = str(error)[:300]
            run.abort()
        rows = []  # the message ranks' rows in rank order: island 0's part first
        for area, (low, n) in zip(areas, split):
            words = area.read().astype(np.int64).reshape(-1, RANK["kResWords"])
            rows.append(words[low : low + max(n - COUNTERS, 0)])
        rows = np.concatenate(rows)
        try:
            region = holder.threads[CLUSTER].read("totals")
        except (km.KomorebiError, RuntimeError) as error:
            out["totals_error"] = str(error)[:200]
            region = bytes(4 * RANK["kX280Words"] * args.x280)
        totals = np.frombuffer(region, np.uint32)
        totals = totals.reshape(-1, RANK["kX280Words"]).astype(np.int64)
    visits = [0] * (cores + args.x280)
    expected = one.replay(cores, args.x280, args.seeds, args.hops, flags, visits)
    got = [
        [
            int(row[RANK[f"kRes{name}"]])
            for name in ("Consumed", "SumLo", "SumHi", "Xor")
        ]
        for row in rows
    ] + [
        [
            int(row[RANK[f"kX280{name}"]])
            for name in ("Consumed", "SumLo", "SumHi", "Xor")
        ]
        for row in totals[: args.x280]
    ]
    got = [[consumed, lo | high << 32, mixed] for consumed, lo, high, mixed in got]
    out["wrong_consumers"] = [
        i for i, (a, b) in enumerate(zip(got, expected)) if a != b
    ]
    seen = [int(v) for v in rows[:, RANK["kResVisits"]]]
    out["wrong_visits"] = [r for r in range(cores) if seen[r] != visits[r]]
    out["gave"] = [int(g) for g in totals[: args.x280, RANK["kX280Gave"]]]
    out["messages"] = {
        "consumed_cores": sum(end[0] for end in got[:cores]),
        "consumed_x280": sum(end[0] for end in got[cores:]),
        "to_x280": int(rows[:, RANK["kResToX280"]].sum()),
        "x280_takes": int(totals[: args.x280, RANK["kX280Takes"]].sum()),
        "x280_sent": int(totals[: args.x280, RANK["kX280Sent"]].sum()),
        "core_visits": sum(seen),
        "visits_other_islands": sum(seen[max(split[0][1] - COUNTERS, 0) :]),
    }
    out["rejected"] = int(rows[:, RANK["kResRejected"]].sum())
    out["ok"] = bool(
        out["events"] == ["done"] * sum(1 for _, n in split if n)
        and out["thread"] == ["done", 0]
        and out.get("kernel_errors") == ["none"] * len(split)
        and not out["wrong_consumers"]
        and not out["wrong_visits"]
        and not any(out["gave"])
        and out["rejected"] == 0
        and sum(end[0] for end in got) == cores * args.seeds
    )
    out["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(power.clocked(out)), flush=True)
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
