"""Example 29 on two cards: each island walks the chain, its SUM a group an island;
then with TOTAL (an all-reduce over the islands) and a step reading it into w, island
0's alone. A command a step and one chain (--thread: also island 0's thread's single
command over every rank), exact against numpy. One JSON line."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chain_steps as cs

import komorebi as km
from komorebi import chain as kc
from komorebi import formats, power

WORDS = cs.WORDS
HERE = Path(__file__).resolve().parent
TICKS_PER_US = 50  # the X280 timer, 50 MHz


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--width", type=int, default=2, help="tiles a rank")
    parser.add_argument("--runs", type=int, default=50, help="timed runs each way")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--thread", action="store_true", help="island 0's thread sends the chain too"
    )
    return parser.parse_args()


def island_calls(t, ranks, island_ranks, width):
    """Example 29's calls on every rank, SUM a group an island into the copy u (each
    island's root its own copy); with TOTAL, u summed over the islands, then w = z + u
    (page 0 of u for every tile). (the four calls, the six)."""
    group, tiles = (0, ranks), island_ranks * width
    four = cs.calls(t | {"t": t["u"]}, ranks, width)[:3]
    four.append(
        kc.Call(
            "sum",
            (0, t["s"], t["u"], tiles, 0, 0, 1, island_ranks),
            group,
            island_ranks,
            1,
            "u",
            "s",
            ranges=(kc.Range(0, tiles, tiles),),  # group g: its island's tiles
            sends=True,
            name="sum",
        )
    )
    total = kc.Call(
        "total",
        (0, t["u"], t["u"], t["scratch"], 0, 0, 1, island_ranks),
        group,
        island_ranks,
        1,
        "u",
        "u",
        ranges=(kc.Range(0, 1, 1),),  # group g: its island's root's u
        name="total",
    )
    add = kc.Call(
        "add",
        (0, t["z"], t["w"], t["u"], 0, 0, width, 0, 1),
        group,
        island_ranks,
        tiles,
        "w",
        "u",
        ranges=(kc.Range(0, 1, 1),),
        reads=(kc.Read("z", ranges=(kc.Range(0, tiles, tiles),)),),
        name="add total",
    )
    return four, [*four, total, add]


def read_copy(session, tensor, island):
    (raw,) = session.sessions[island].read(tensor.parts[island])
    return formats.from_bytes(raw, tensor.shape, tensor.dtype, tensor.layout)


def card_us(events):
    return max(event.microseconds for event in events)  # the slowest island's


def by_steps(run, steps, factor):
    total = 0.0
    for index, call in enumerate(steps):
        args = list(call.args)
        args[5] |= factor if index == 0 else 0
        total += card_us(run.wait(run.submit(call.handler, call.group, args, True)))
    return total


def by_chain(run, chain, factor):
    return card_us(run.wait(run.submit(chain, factor=factor)))


def by_thread(run, chain, factor):
    """Island 0's thread sends `chain` as one command over its whole group (records
    at one address on every island): the thread's µs from submit to its event."""
    one = chain.chains[0]
    if chain.address is None:
        raise RuntimeError("the chain's records are not at one address")
    if one.pieces > 1:  # a mock run's chain (a prebuild) has 0
        raise RuntimeError("a chain past 256 records: a command a piece, not one")
    args = [chain.address, one.handler, *chain.group, one.mode, factor, one.flags, 1]
    first = run.island(0)
    event = first.wait(first.submit("chain", cluster=1, args=args, result=True), 10.0)
    if event.kind != "done" or event.result == 0xFFFFFFFF:
        raise RuntimeError(f"the thread's chain: {event.kind}, word {event.result:#x}")
    return event.result / TICKS_PER_US


def wrong(session, tensors, x, factor, islands, island_ranks, width, six):
    want = cs.expected(x, factor)
    got = session.read(*(tensors[name] for name in ("y", "z", "s")))
    bad = {
        name: int(np.sum(raw.view(np.int32).reshape(-1, WORDS) != want[name]))
        for name, raw in zip(("y", "z", "s"), got, strict=True)
    }
    tiles = island_ranks * width
    parts = [
        want["s"][i * tiles : (i + 1) * tiles].sum(axis=0, dtype=np.int32)
        for i in range(islands)
    ]
    total = np.sum(parts, axis=0, dtype=np.int32)
    for island in range(islands):
        u = read_copy(session, tensors["u"], island)[0]
        bad[f"u{island}"] = int(np.sum(u != (total if six else parts[island])))
    if six:
        w = session.read(tensors["w"])[0].view(np.int32).reshape(-1, WORDS)
        bad["w"] = int(np.sum(w != want["z"] + total))
    return bad


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    generator = np.random.default_rng(args.seed)
    result = {"cards": cards, "width": args.width}
    with km.Session(cards=cards, log=cs.log) as session:
        ranks, islands = session.ranks, len(session.sessions)
        island_ranks = session.graph.count(0)
        if any(session.graph.count(i) != island_ranks for i in range(islands)):
            raise SystemExit("islands of equal ranks")
        tiles = ranks * args.width
        x = generator.integers(-50, 51, (tiles, WORDS)).astype(np.int32)
        result |= {"ranks": ranks, "islands": islands, "tiles": tiles}
        tensors = {"x": session.tensor(x, "int32")}
        for name in ("y", "z", "s"):
            tensors[name] = session.tensor(np.zeros((tiles, WORDS), np.int32), "int32")
        zeros = np.zeros((tiles, WORDS), np.int32)  # w: island 0's ranks hold it
        tensors["w"] = session.tensor(zeros, "int32", group=(0, island_ranks))
        one = np.zeros((1, WORDS), np.int32)
        tensors["u"] = session.tensor(one, "int32", copy=True)
        tensors["scratch"] = session.tensor(
            np.zeros((2, WORDS), np.int32), "int32", copy=True
        )
        t = {name: tensor.address for name, tensor in tensors.items()}
        four, six = island_calls(t, ranks, island_ranks, args.width)
        made = cs.program(tensors["x"])
        if args.thread:
            made.thread(
                cluster=1,
                sources=[HERE / "thread" / "across.c"],
                regions={"scratch": 64},
                handlers={"chain": 1},
            )
        run = session.load(made, wait=0.5, deadline=1.0)
        ways = [("steps", 2), ("chain", 5), ("steps", 7), ("chain", 3)]
        ways += [("thread", 4), ("thread", 6)] if args.thread else []
        by = {"steps": by_steps, "chain": by_chain, "thread": by_thread}
        for name, steps in (("four", four), ("six", six)):
            chain = run.chain(steps, cs.C["kChainMode"])
            records = [part.tensor.address for part in chain.chains]  # each island's
            checks = {}
            for way, factor in ways:
                by[way](run, steps if way == "steps" else chain, factor)
                checks[f"{way} x{factor}"] = wrong(
                    session,
                    tensors,
                    x,
                    factor,
                    islands,
                    island_ranks,
                    args.width,
                    name == "six",
                )
            timed, card = {}, {}
            for way in ("steps", "chain", "thread")[: 3 if args.thread else 2]:
                start = time.perf_counter()
                times = [
                    by[way](run, steps if way == "steps" else chain, 1)
                    for _ in range(args.runs)
                ]
                timed[way] = round((time.perf_counter() - start) * 1e6 / args.runs, 2)
                card[way] = round(float(np.median(times)), 2)
            checks["after timing"] = wrong(
                session, tensors, x, 1, islands, island_ranks, args.width, name == "six"
            )
            chain.free()
            result[name] = {
                "checks": checks,
                "records": records,
                "microseconds_a_run": timed,  # the host's, submit to the last wait
                "card_microseconds_a_run": card,  # the slowest island's, median
            }
            cs.log(f"{name}: {json.dumps(result[name])}")
        result["kernel"] = [r["error_name"] for r in run.stop()]
    result["ok"] = bool(
        all(error == "none" for error in result["kernel"])
        and all(
            not any(bad.values())
            for name in ("four", "six")
            for bad in result[name]["checks"].values()
        )
    )
    result["wall_s"] = round(time.monotonic() - cs.STARTED, 1)
    print(json.dumps(power.clocked(result)), flush=True)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
