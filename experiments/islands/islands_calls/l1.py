"""An L1 tensor of the island session (`session.tensor(..., shard_grid=grid)`, one
address on every island) behind example 14's dynamic class's L1 field, on two cards:
each message rank's totals against the replay over both islands' ranks. One JSON line;
exits 1 when a check fails. See README.md."""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parents[2] / "examples" / "14_message_handlers"
sys.path.insert(0, str(EXAMPLE))
SPEC = importlib.util.spec_from_file_location(
    "handlers_class", EXAMPLE / "as_class" / "handlers_class.py"
)
one = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(one)
import replay  # noqa: E402  (example 14's replay)

STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--threads", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=10, help="lookups per thread")
    parser.add_argument("--counters", type=int, default=2)
    parser.add_argument("--deadline", type=float, default=5.0)
    return parser.parse_args()


def checked(args, session, totals):
    """Each island's message ranks' totals rows (its counter ranks last), in flat
    order, against the replay over all of them."""
    graph, cores = session.graph, totals.shape[0] // len(session.sessions)
    rows = (totals.read().astype(np.int64) & 0xFFFFFFFF).reshape(-1, cores, 16)
    counts = [graph.count(i) - args.counters for i in range(len(session.sessions))]
    got = np.concatenate(
        [
            rows[island][list(graph.rank_cores[island])[:count]]
            for island, count in enumerate(counts)
        ]
    )
    expected = replay.replay(args.threads, args.steps, sum(counts))
    names = {"lookups": one.LOOKUPS, "values": one.VALUES, "reports": one.REPORTS}
    names |= {"own": one.OWN, "probes": one.PROBES}
    differ = [
        name for name, word in names.items() if (got[:, word] != expected[name]).any()
    ]
    total = (got[:, one.SUM_LOW] | got[:, one.SUM_HIGH] << 32).astype(np.uint64)
    if (total != expected["sum"]).any():
        differ.append("sum")
    return {
        "message_ranks": counts,
        "exact": not differ and not got[:, one.WRONG].any(),
        "differ": differ,
        "messages": int(sum(got[:, names[key]].sum() for key in list(names)[:3])),
    }


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    report = {"cards": cards, "threads": args.threads, "steps": args.steps}
    with km.Session(cards=cards, log=log) as session:
        grid = session.grid
        rows = len(session.sessions) * grid[0] * grid[1]
        totals = session.tensor(
            np.zeros((rows, one.MessageHandlers.WORDS.default), np.uint32),
            "uint32",
            shard_grid=grid,
        )
        report["totals"] = [list(totals.shape), hex(totals.address)]
        report["parts_at"] = sorted({hex(part.address) for part in totals.parts})
        program = one.MessageHandlers(totals=totals, log=log).dynamic(
            "lookup",
            "value",
            "report",
            max_live=1 << 14,
            queue_messages=1 << 14,
            counters=args.counters,
        )
        run = session.load(program, deadline=args.deadline, wait=1.0)
        group = (0, session.ranks)
        log(f"loaded; RUN lookup on {group}, totals at {report['totals']}")
        handle = run.submit(
            one.MessageHandlers.lookup,
            group,
            args=(args.threads, args.steps),
            exclusive=True,
        )
        try:
            run.drain(timeout=args.deadline + 5.0)
        except km.CommandTimeout as error:
            report["timeout"] = str(error)[:300]
        event = run.event(handle)  # None until every island's part arrived
        report["event"] = event.kind if event else None
        stopped = run.stop()
        report["kernel_errors"] = [part["error_name"] for part in stopped.islands]
        log(f"RUN {report['event']}; stopped: {report['kernel_errors']}")
        report |= checked(args, session, totals)
        log(f"totals: exact {report['exact']}, {report['messages']} messages")
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    report["ok"] = bool(
        report["exact"]
        and report["event"] == "done"
        and report["kernel_errors"] == ["none", "none"]
        and len(report["parts_at"]) == 1
    )
    print(json.dumps(report), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
