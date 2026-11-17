"""The merge tree over two cards (`Session(cards=..., links=1)`), its kernels unchanged:
MERGE on one flat group over both islands folds inside each island, then crosses once;
every command done, its result word (the total's checksum) and the total against numpy,
beside the same on one island's ranks. Prints one JSON line. See README.md."""

import argparse
import json
import sys
import time

import numpy as np
from merge import CONSTANTS, TILE, checksum, program

import komorebi as km
from komorebi.bench import power

STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--rounds", type=int, default=20, help="MERGE commands a case")
    parser.add_argument("--tiles", type=int, default=512, help="input tiles T")
    parser.add_argument("--part", type=int, default=1, help="tiles per part P (1 to 3)")
    parser.add_argument("--wait", type=float, default=0.5, help="wait limit, seconds")
    parser.add_argument(
        "--deadline", type=float, default=1.0, help="a command, seconds"
    )
    parser.add_argument("--depth", type=int, default=4)
    return parser.parse_args()


def cases(session):
    """(name, group, form, host): the whole of both islands by tree, from island 0's
    dispatcher and split by the host; a group across the boundary by owner (it fits
    the arrival buffer); island 0's ranks alone, both forms."""
    ranks, first1 = session.ranks, session.graph.first(len(session.sessions) - 1)
    span = min(CONSTANTS["kArrivePages"] // 2, first1)
    out = [
        ("all_tree", (0, ranks), 1, False),
        ("all_tree_host", (0, ranks), 1, True),
        ("across_owner", (first1 - span, 2 * span), 0, False),
        ("island0_tree", (0, session.graph.count(0)), 1, False),
        ("island0_owner", (0, session.graph.count(0)), 0, False),
    ]
    return out if len(session.sessions) > 1 else out[3:]


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    generator = np.random.default_rng(1)
    report = {"cards": cards, "tiles": args.tiles, "part": args.part}
    links = 1 if len(cards) > 1 else 0
    with km.Session(cards=cards, links=links, log=log) as session:
        report["islands"] = [
            [session.graph.first(i), session.graph.count(i)]
            for i in range(len(session.sessions))
        ]
        # exact: integers in [-2, 2], every partial sum exact in float32
        exact = generator.integers(-2, 3, (args.tiles * TILE, TILE)).astype(np.float32)
        inputs = session.tensor(exact, "float32", "tile", copy=True)
        output = session.tensor(
            np.zeros((8 * args.part * TILE, TILE), np.float32),
            "float32",
            "tile",
            copy=True,
        )

        class Args:  # what merge.program reads
            depth = args.depth

        run = session.load(
            program(Args, inputs, output), deadline=args.deadline, wait=args.wait
        )
        log(f"loaded, islands {report['islands']}")
        tiles = exact.reshape(-1, TILE, TILE).astype(np.float64)
        want = np.stack([tiles[s :: args.part].sum(0) for s in range(args.part)])
        island0 = run.island(0)  # a spanning group crosses from island 0's dispatcher
        results = {}
        for index, (name, group, form, host) in enumerate(cases(session)):
            words = [
                inputs.address,
                args.tiles,
                output.address,
                index * args.part,
                form,
                0xFFFFFFFF,
                0,
                args.part,
            ]
            times, bad = [], 0
            try:
                for _ in range(args.rounds):
                    start = time.perf_counter()
                    if host:  # one part an island; the first part's word is the result
                        ticket = run.submit("merge", group, words, result=True)
                        event = run.wait(ticket, timeout=5.0)[0]
                    else:  # a group that crosses goes from island 0's dispatcher
                        handle = island0.submit("merge", group, words, result=True)
                        event = island0.wait(handle, timeout=5.0)
                    times.append((time.perf_counter() - start) * 1e6)
                    bad += event.kind != "done" or event.result != checksum(want)
                error = None
            except km.KomorebiError as caught:
                error = str(caught)
                log(f"{name}: {error}")
            results[name] = {
                "group": list(group),
                "rounds": len(times),
                "us_a_round": round(float(np.median(times)), 2) if times else None,
                "not_done_or_result_wrong": bad,
                "error": error,
            }
            log(f"{name}: {results[name]}")
        report["cases"] = results
        try:
            report["kernel_errors"] = [r["error_name"] for r in run.stop().islands]
        except km.KomorebiError as error:
            report["stop_error"] = str(error)
        got = output.parts[0].read().reshape(-1, TILE, TILE)  # island 0's copy
        for index, name in enumerate(results):
            page = index * args.part
            mine = got[page : page + args.part]
            results[name]["exact"] = bool(np.array_equal(mine, want.astype(np.float32)))
            results[name]["max_error"] = float(np.abs(mine - want).max())
    report["ok"] = all(
        case["exact"] and not case["not_done_or_result_wrong"] and case["error"] is None
        for case in report["cases"].values()
    ) and set(report.get("kernel_errors", ["?"])) == {"none"}
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(power.clocked(report)), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
