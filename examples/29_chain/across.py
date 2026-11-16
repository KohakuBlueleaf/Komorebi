"""Example 29's chain across islands: its four calls on 8 ranks over the edge of
islands 0 and 1, then three calls each on another island (on a line of three, a mark
two links away). One card, a split card or two cards; a command a step and one chain,
exact against numpy. One JSON line (README.md)."""

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
from komorebi import power

WORDS = cs.WORDS


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0")
    parser.add_argument("--islands", type=int, default=0, help="one card split in N")
    parser.add_argument("--line", action="store_true", help="the split card a line")
    parser.add_argument("--cores", type=int, default=8, help="ranks a call")
    parser.add_argument("--width", type=int, default=2, help="tiles a rank")
    parser.add_argument("--runs", type=int, default=50, help="timed runs each way")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def hops(t, groups, width):
    """SCALE x into y on group A, SCALE y into z on C, ADD z and y (earlier) into s
    on B: each rank g reads the tiles rank g of the call before wrote."""
    own = kc.Range(0, width, width)
    a, c, b = groups
    return [
        kc.Call(
            "scale",
            (0, t["x"], t["y"], 0, 0, 0, width),
            a,
            1,
            width,
            "y",
            word=("factor", 5),
            name="scale x",
        ),
        kc.Call(
            "scale",
            (0, t["y"], t["z"], 0, 0, 3, width),
            c,
            1,
            width,
            "z",
            "y",
            ranges=(own,),
            name="scale y",
        ),
        kc.Call(
            "add",
            (0, t["z"], t["s"], t["y"], 0, 0, width),
            b,
            1,
            width,
            "s",
            "z",
            ranges=(own,),
            reads=(kc.Read("y", ranges=(own,)),),
            name="add",
        ),
    ]


def card_us(done):
    """The card's microseconds: an island run's slowest part."""
    if isinstance(done, list):
        return max(event.microseconds for event in done)
    return done.microseconds


def by_steps(run, steps, factor):
    total = 0.0
    for index, call in enumerate(steps):
        args = list(call.args)
        args[5] |= factor if index == 0 else 0
        total += card_us(run.wait(run.submit(call.handler, call.group, args, True)))
    return total


def by_chain(run, chain, factor):
    return card_us(run.wait(run.submit(chain, factor=factor)))


def bad_words(session, tensors, names, want):
    got = session.read(*(tensors[name] for name in names))
    return {
        name: int(np.sum(raw.view(np.int32).reshape(-1, WORDS) != want[name]))
        for name, raw in zip(names, got, strict=True)
    }


def placed(session, cores):
    """(islands, the straddle's first rank, the three groups of `hops`): island 0's
    edge and its last island's first ranks; one card: ranges side by side."""
    islands = len(session.sessions) if hasattr(session, "sessions") else 1
    if islands == 1:
        return 1, 0, ((0, cores), (2 * cores, cores), (cores, cores))
    graph = session.graph
    last = islands - 1
    groups = ((0, cores), (graph.first(last), cores), (graph.first(1) + cores, cores))
    return islands, graph.first(1) - cores // 2, groups


def run_form(session, run, tensors, steps, names, want, runs):
    chain = run.chain(steps, cs.C["kChainMode"])
    checks = {}
    for way, factor in (("steps", 2), ("chain", 5), ("steps", 7), ("chain", 3)):
        (by_steps if way == "steps" else by_chain)(
            run, steps if way == "steps" else chain, factor
        )
        checks[f"{way} x{factor}"] = bad_words(session, tensors, names, want(factor))
        cs.log(f"{way} x{factor}: {checks[f'{way} x{factor}']}")
    timed, card = {}, {}
    for way in ("steps", "chain"):
        start = time.perf_counter()
        times = [
            by_steps(run, steps, 1) if way == "steps" else by_chain(run, chain, 1)
            for _ in range(runs)
        ]
        timed[way] = round((time.perf_counter() - start) * 1e6 / runs, 2)
        card[way] = round(float(np.median(times)), 2)
    checks["after timing"] = bad_words(session, tensors, names, want(1))
    chain.free()
    return {
        "groups": [list(call.group) for call in steps],
        "checks": checks,
        "microseconds_a_run": timed,  # the host's, submit to the last wait
        "card_microseconds_a_run": card,  # the slowest island's, median
    }


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    options = {"log": cs.log}
    if len(cards) > 1 or args.islands:
        options["cards"] = cards
    if args.islands:
        options.update(islands=args.islands, line=args.line)
    generator = np.random.default_rng(args.seed)
    result = {"cards": cards, "islands_split": args.islands, "line": args.line}
    with km.Session(**options) as session:
        islands, edge, groups = placed(session, args.cores)
        tiles = args.cores * args.width
        x = generator.integers(-50, 51, (tiles, WORDS)).astype(np.int32)
        tensors = {"x": session.tensor(x, "int32")}
        for name, rows in (("y", tiles), ("z", tiles), ("s", tiles), ("t", 1)):
            tensors[name] = session.tensor(np.zeros((rows, WORDS), np.int32), "int32")
        t = {name: tensor.address for name, tensor in tensors.items()}
        result |= {"islands": islands, "edge": edge, "tiles": tiles}
        run = session.load(cs.program(tensors["x"]), wait=0.5, deadline=1.0)
        cs.log(f"loaded: {islands} islands, the straddle from rank {edge}")
        result["straddle"] = run_form(
            session,
            run,
            tensors,
            cs.calls(t, args.cores, args.width, first=edge),
            ("y", "z", "s", "t"),
            lambda factor: cs.expected(x, factor),
            args.runs,
        )
        cs.log(f"straddle: {json.dumps(result['straddle'])}")
        result["hops"] = run_form(
            session,
            run,
            tensors,
            hops(t, groups, args.width),
            ("y", "z", "s"),
            lambda factor: {
                name: value
                for name, value in cs.expected(x, factor).items()
                if name != "t"
            },
            args.runs,
        )
        cs.log(f"hops: {json.dumps(result['hops'])}")
        stopped = run.stop()
        kernels = stopped if isinstance(stopped, list) else [run.result()]
        result["kernel"] = [one["error_name"] for one in kernels]
    result["ok"] = bool(
        all(error == "none" for error in result["kernel"])
        and all(
            not any(bad.values())
            for form in ("straddle", "hops")
            for bad in result[form]["checks"].values()
        )
    )
    result["wall_s"] = round(time.monotonic() - cs.STARTED, 1)
    print(json.dumps(power.clocked(result)), flush=True)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
