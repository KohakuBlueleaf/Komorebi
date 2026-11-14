"""The message body example over two cards (`Session(cards=[0, 1])`): the same kernels,
one dynamic run over every message rank of both islands, each thread one message whose
body crosses the link with it; every body against the numpy replay. One JSON line. See
README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(HERE),
    str(ROOT / "tests" / "hardware" / "feeder"),
    str(ROOT / "src"),
]

import commands as feeder_commands  # noqa: E402
import replay  # noqa: E402

import komorebi as km  # noqa: E402

WORDS = feeder_commands.constants(HERE / "kernels" / "chain.h")
RUNTIME = ROOT / "src" / "komorebi_native" / "kernels"
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--links", type=int, default=1, help="links between the cards")
    parser.add_argument("--threads", type=int, default=1, help="at most the ranks")
    parser.add_argument("--hops", type=int, default=1200, help="arrivals per thread")
    parser.add_argument(
        "--bodies",
        type=int,
        nargs="+",
        default=[0, 4096],
        help="body bytes of each case (0: the 64-byte message alone)",
    )
    parser.add_argument("--max-live", type=int, default=128)
    parser.add_argument("--queue-messages", type=int, default=128)
    parser.add_argument(
        "--counters", type=int, default=2, help="counter tiles an island"
    )
    parser.add_argument(
        "--kernel-bytes",
        type=int,
        default=None,
        help="kernel buffer bytes (default 163840; less: link 0's service alone)",
    )
    parser.add_argument("--deadline", type=float, default=5.0, help="the run, seconds")
    parser.add_argument("--wait", type=float, default=1.0, help="wait limit, seconds")
    return parser.parse_args()


def areas(session):
    """The body and result area on every core of every island, at one address."""
    out = [
        session.island(i).tensor(
            np.zeros((np.prod(session.island(i).grid), WORDS["kAreaWords"]), np.int32),
            "int32",
            shard_grid=session.island(i).grid,
        )
        for i in range(len(session.sessions))
    ]
    if len({tensor.address for tensor in out}) != 1 or out[0].address % 64:
        raise km.KomorebiError("the areas are not at one 64-byte aligned address")
    return out


def run_once(session, args, area, body_bytes):
    """One load with `body_bytes`, its RUN over every rank, the end and the check."""
    address = area[0].address
    program = km.Program(
        "message_body",
        kernels={
            "reader": RUNTIME / "messages_reader.cpp",
            "compute": HERE / "kernels" / "compute.cpp",
            "writer": RUNTIME / "messages_writer.cpp",
        },
        args=lambda core: ([], [], [address]),
        handlers={"chain": WORDS["kHandlerChain"]},
    ).dynamic(
        "chain",
        max_live=args.max_live,
        queue_messages=args.queue_messages,
        counters=args.counters,
        body_bytes=body_bytes,
    )
    run = session.load(program, deadline=args.deadline, wait=args.wait)
    islands = len(session.sessions)
    words = [args.threads, args.hops, body_bytes, 1]
    before = run.crossings()
    ticket = run.submit("chain", (0, session.ranks), words, exclusive=True)
    timed_out = None
    try:
        run.drain(timeout=args.deadline + 5.0)
    except km.CommandTimeout as error:
        timed_out = str(error)[:400]
        log(f"timeout: {timed_out}")
    events = run.event(ticket)
    kinds = [event.kind if event else None for event in events]
    done = kinds == ["done"] * islands
    crossed = None
    if done:
        moved = run.crossings() - before
        crossed = {
            f"{link} {source}->{target}": row["messages"]
            for (link, source, target), row in sorted(moved.rows.items())
        }
        errors = [one["error_name"] for one in run.stop()]
    else:
        run.abort()
        errors = None
    rows = [
        (tensor.read().astype(np.int64) & 0xFFFFFFFF).reshape(-1, WORDS["kAreaWords"])
        for tensor in area
    ]
    result = check(args, session, rows, body_bytes)
    microseconds = [
        round(event.microseconds, 1) if event and event.kind == "done" else None
        for event in events
    ]
    result |= {
        "events": kinds,
        "run_microseconds": microseconds,
        "hop_microseconds": (round(max(microseconds) / args.hops, 3) if done else None),
        "messages_crossed": crossed,
        "kernel_errors": errors,
        "timed_out": timed_out,
    }
    result["ok"] = bool(result["exact"] and done and errors == ["none"] * islands)
    return result


def check(args, session, rows, body_bytes):
    """Every message rank's arrivals, which thread ended where, its state word and its
    body against the replay over the message ranks of both islands, in flat order."""
    counts = [session.graph.count(i) - args.counters for i in range(len(rows))]
    ranks = sum(counts)
    threads = min(args.threads, ranks)
    body_words = body_bytes // 4
    arrivals, last, state, body = replay.replay(
        threads, args.hops, ranks, body_words, True, 1
    )
    got = np.concatenate([one[:count] for one, count in zip(rows, counts)])
    reserved = np.concatenate([one[count:] for one, count in zip(rows, counts)])
    ended = np.zeros(ranks, dtype=np.int64)
    ended[last] = np.arange(threads) + 1
    differ = []
    if not (got[:, WORDS["kResArrivals"]] == arrivals).all():
        differ.append("arrivals")
    if not (got[:, WORDS["kResThread"]] == ended).all():
        differ.append("thread")
    if not (got[last, WORDS["kResHop"]] == args.hops).all():
        differ.append("hop")
    if not (got[last, WORDS["kResState"]] == state.astype(np.int64)).all():
        differ.append("state")
    if body_words and not (got[last, :body_words] == body.astype(np.int64)).all():
        differ.append("body")
    if reserved[:, WORDS["kResArrivals"]].any():
        differ.append("counter tiles")
    return {
        "body_bytes": body_bytes,
        "message_ranks": counts,
        "threads": threads,
        "hops": args.hops,
        "exact": not differ,
        "differ": differ,
        "body_words_checked": int(threads * body_words),
    }


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    runs = []
    with km.Session(
        cards=cards, links=args.links, kernel_bytes=args.kernel_bytes, log=log
    ) as session:
        area = areas(session)
        for body_bytes in args.bodies:
            runs.append(run_once(session, args, area, body_bytes))
            one = runs[-1]
            log(
                f"body {body_bytes}: {'exact' if one['exact'] else one['differ']}, "
                f"{one['run_microseconds']} us, {one['hop_microseconds']} us a hop"
            )
    result = {
        "cards": cards,
        "links": args.links,
        "ok": all(one["ok"] for one in runs),
        "runs": runs,
        "wall_s": round(time.monotonic() - STARTED, 1),
    }
    print(json.dumps(result), flush=True)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
