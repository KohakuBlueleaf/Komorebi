"""Tokens hopping over declared pairs of ranks: by a direct message, L1 to L1, or by the
queues in DRAM; every token replayed on the host (README.md has the cases)."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi import abi  # noqa: E402
from komorebi.bench import power  # noqa: E402
from komorebi.program import messages_header  # noqa: E402

sys.path.insert(0, str(HERE))
from ring_class import DirectRing  # noqa: E402

RING = abi.constants(HERE / "kernels" / "ring.h")
RUNTIME = ROOT / "src" / "komorebi_native" / "kernels"
STARTED = time.monotonic()
M32 = 0xFFFFFFFF


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def ring_next(state):
    state ^= (state << 13) & M32
    state ^= state >> 17
    state ^= (state << 5) & M32
    return state


def ring_seed(token):
    state = (token + 1) * 0x9E3779B1 & M32
    return state or 1


def ring_pair(rank, which, ranks):
    return (5 * rank + 1) % ranks if which == 0 else (rank * rank + 3) % ranks


def ring_to(rank, state, pattern, ranks):
    if pattern == RING["kPatternRing"]:
        return (rank + 1) % ranks
    return ring_pair(rank, state >> 7 & 1, ranks)


def pairs_of(pattern, ranks):
    """The (sender, receiver) pairs a pattern uses over `ranks` ranks."""
    if pattern == RING["kPatternRing"]:
        return [(rank, (rank + 1) % ranks) for rank in range(ranks)]
    return sorted({(r, ring_pair(r, w, ranks)) for r in range(ranks) for w in (0, 1)})


def replay(tokens, hops, pattern, ranks, cores):
    """Each token's path as the kernel takes it: per core its visits, and the
    records (id + 1, state, sum) of the tokens that end there."""
    visits = np.zeros(cores, np.int64)
    records = [set() for _ in range(cores)]
    for rank_seeded in range(ranks):
        for index in range(tokens):
            token = rank_seeded * tokens + index
            rank, state, total = rank_seeded, ring_seed(token), 0
            for left in range(hops, -1, -1):
                visits[rank] += 1
                total = (total + rank) & M32
                state = ring_next(state)
                if left == 0:
                    records[rank].add((token + 1, state, total))
                else:
                    rank = ring_to(rank, state, pattern, ranks)
    return visits, records


def stamp_report(run, rows, count):
    """The stamps of a run with --stamps: each stamp word summed over the group's
    cores, and pack's cycles in next() and from a packet to done()."""
    constants = messages_header()[0]
    names = {
        key[len("kStamp") :]: value
        for key, value in constants.items()
        if key.startswith("kStamp") and key != "kStampWords"
    }
    words = run.backend.stamps.read().astype(np.int64)[:count]
    report = {name: int(words[:, word].sum()) for name, word in names.items()}
    report["pack_next"] = int(rows[:count, RING["kResNext"]].sum())
    report["pack_handle"] = int(rows[:count, RING["kResHandle"]].sum())
    return report


def load(session, direct, body_bytes, age, args):
    """The ring loaded with `direct` declared, bodies of `body_bytes` (0: none) and
    the writer's flush_age `age`: (run, results tensor, record room)."""
    grid = session.grid
    cores = grid[0] * grid[1]
    room = max(args.tokens * (cores - 2), 1)
    words = RING["kResRecords"] + RING["kRecordWords"] * room
    results = session.tensor(
        np.zeros((cores, words), np.uint32), "uint32", shard_grid=grid
    )
    seed = session.tensor(  # a token's first body
        np.zeros((cores, max(body_bytes, 64) // 4), np.uint32),
        "uint32",
        shard_grid=grid,
    )
    options = {
        "max_live": 1024,
        "queue_messages": 1024,
        "counters": 2,
        "direct": direct,
        "body_bytes": body_bytes,
        "stamps": args.stamps,
        "flush_age": age,
    }
    if args.as_class:  # ring_class.py: the same results words, written by the class
        program = DirectRing(log=log).dynamic("ring", **options)
        run = session.load(program, deadline=args.deadline, wait=args.wait)
        return run, results, room
    program = km.Program(
        "direct_ring",
        kernels={
            "reader": RUNTIME / "messages_reader.cpp",
            "compute": HERE / "kernels" / "compute.cpp",
            "writer": RUNTIME / "messages_writer.cpp",
        },
        args=lambda core: ([], [], [results.address, seed.address]),
        handlers={"ring": RING["kHandlerRing"]},
    ).dynamic("ring", **options)
    run = session.load(program, deadline=args.deadline, wait=args.wait)
    return run, results, room


def one_run(run, results, room, case, args, replays):
    """One RUN of a case (name, group cores, ranks, tokens, hops, mode, pattern) and
    its checks against its replay (computed before the session)."""
    name, count, ranks, tokens, hops, mode, pattern = case
    log(f"{name}: submit, {ranks} ranks, {tokens} token(s) a rank, {hops} hops")
    words = [tokens, hops, RING[mode], RING[pattern], ranks, room]
    if args.as_class:
        words.append(results.address)
        handle = run.submit(DirectRing.ring, (0, count), args=words, exclusive=True)
    else:
        handle = run.submit("ring", (0, count), words, exclusive=True)
    run.drain(timeout=args.deadline + 5.0)
    run.stop()
    event, sequence = run.event(handle), run.sequence(handle)
    log(f"{name}: {event.kind if event else None}; reading the results")
    rows = results.read().astype(np.int64) & M32
    log(f"{name}: results read")
    visits, records = replays[name]
    seen_records = [
        {
            tuple(int(word) for word in rows[core, at : at + 3])
            for at in range(
                RING["kResRecords"],
                RING["kResRecords"]
                + RING["kRecordWords"] * int(rows[core, RING["kResConsumed"]]),
                RING["kRecordWords"],
            )
        }
        for core in range(count)
    ]
    group = rows[:count]
    hops_total = ranks * tokens * hops
    went = int(group[:, RING["kResDirect"]].sum())
    busy = int(group[:, RING["kResBusy"]].sum())
    checks = {
        "done": bool(event and event.kind == "done"),
        "run": bool((group[:, RING["kResRun"]] == sequence).all()),
        "visits": bool((group[:, RING["kResVisits"]] == visits).all()),
        "records": seen_records == records,
        "bodies": not any(  # every body's checksum where it ended
            rows[core, at + 3]
            for core in range(count)
            for at in range(
                RING["kResRecords"],
                RING["kResRecords"]
                + RING["kRecordWords"] * int(rows[core, RING["kResConsumed"]]),
                RING["kRecordWords"],
            )
        ),
        "hops": mode == "kModeQueue" or went + busy == hops_total,
    }
    microseconds = event.microseconds if event else None
    stamps = stamp_report(run, rows, count) if args.stamps else None
    if stamps:
        log(f"{name}: stamps {stamps}")
    log(f"{name}: {checks}; {microseconds} us, direct {went}, busy {busy}")
    return {
        "case": name,
        "ranks": ranks,
        "tokens_per_rank": tokens,
        "hops_per_token": hops,
        "hops": hops_total,
        "microseconds": round(microseconds, 1) if microseconds else None,
        "microseconds_a_hop": round(microseconds / hops, 4) if microseconds else None,
        "direct": went,
        "busy": busy,
        "checks": checks,
        "stamps": stamps,
        "ok": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tokens", type=int, default=1, help="tokens a rank")
    parser.add_argument("--hops", type=int, default=8475, help="hops a token")
    parser.add_argument("--deadline", type=float, default=5.0)
    parser.add_argument("--wait", type=float, default=1.0)
    parser.add_argument("--json", default=None)
    parser.add_argument("--cores", type=int, default=120, help="the card's grid")
    parser.add_argument("--body", type=int, default=2048, help="the body case's bytes")
    parser.add_argument(
        "--cases", nargs="+", default=["small", "ring", "pairs", "body"]
    )
    parser.add_argument(
        "--stamps", action="store_true", help="the runtime's stamps (a build apart)"
    )
    parser.add_argument(
        "--as-class", action="store_true", help="ring_class.py (ring pattern, no body)"
    )
    parser.add_argument(
        "--flush-ages", type=int, nargs="+", default=[500], help="flush_age of each"
    )
    parser.add_argument(
        "--modes", nargs="+", default=["kModeDirect", "kModeQueue"], help="per case"
    )
    args = parser.parse_args()
    report = {"load_before": Path("/proc/loadavg").read_text().split()[:3], "runs": []}
    cores, small = args.cores, 4  # first: four ranks of a group of six cores
    ranks = cores - 2
    log("replaying every token on the host")
    replays = {
        "small": replay(1, 10, RING["kPatternRing"], small, 6),
        "ring": replay(args.tokens, args.hops, RING["kPatternRing"], ranks, cores),
        "pairs": replay(args.tokens, args.hops, RING["kPatternPairs"], ranks, cores),
    }
    replays["body"] = replays["ring"]
    log("replayed")
    with km.Session(log=log) as session:
        log("device open")
        if session.grid[0] * session.grid[1] != cores:
            raise SystemExit(f"the grid has {session.grid}, not {cores} cores")
        loads = [
            (pairs_of(RING["kPatternRing"], small), 0, ("small", 6, small, 1, 10)),
            ((1,), 0, ("ring", cores, ranks, args.tokens, args.hops)),
            (
                pairs_of(RING["kPatternPairs"], ranks),
                0,
                ("pairs", cores, ranks, args.tokens, args.hops),
            ),
            ((1,), args.body, ("body", cores, ranks, args.tokens, args.hops)),
        ]
        loads = [one for one in loads if one[2][0] in args.cases]
        if args.as_class and any(one[2][0] in ("pairs", "body") for one in loads):
            raise SystemExit("--as-class: the class has the ring pattern, no bodies")
        for direct, body_bytes, (name, count, case_ranks, tokens, hops) in loads:
            pattern = "kPatternPairs" if name == "pairs" else "kPatternRing"
            runs = [(mode, age) for mode in args.modes for age in args.flush_ages]
            for mode, age in runs:  # a load a run: read stopped
                log(
                    f"loading: {len(direct)} direct pairs, body {body_bytes}, age {age}"
                )
                run, results, room = load(session, direct, body_bytes, age, args)
                case = (name, count, case_ranks, tokens, hops, mode, pattern)
                entry = one_run(run, results, room, case, args, replays)
                entry["mode"], entry["flush_age"] = mode, age
                report["runs"].append(entry)
                if not entry["ok"]:
                    break
    report["load_after"] = Path("/proc/loadavg").read_text().split()[:3]
    report["ok"] = all(entry["ok"] for entry in report["runs"]) and len(
        report["runs"]
    ) == len(args.modes) * len(args.flush_ages) * len(loads)
    summary = {
        f"{entry['case']} {entry['mode'][5:].lower()} {entry['flush_age']}": entry[
            "microseconds_a_hop"
        ]
        for entry in report["runs"]
    }
    print(json.dumps(power.clocked({"ok": report["ok"], "us_a_hop": summary})))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    return report["ok"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
