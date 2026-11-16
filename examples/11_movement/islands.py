"""The movement example's streams over two cards as islands, the same kernels as on one
card: island 0's cores copy an input island 1 holds (every word checked), then READ it
timed; --write-across: the input island 0's, the output island 1's, COPY only. One
JSON line."""

import argparse
import json
import sys
import time

import numpy as np
from movement import pages_of, program

import komorebi as km
from komorebi.bench import power

STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--cores", type=int, default=4, help="island 0's first cores")
    parser.add_argument("--tiles", type=int, default=48, help="tiles a core reads")
    parser.add_argument("--block", type=int, default=8, help="tiles per block")
    parser.add_argument("--pattern", default="AB", help="the pattern of A and B")
    parser.add_argument("--rounds", type=int, default=3, help="timed READ rounds")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--write-across", action="store_true", help="the output on island 1: COPY only"
    )
    return parser.parse_args()


def one(session, run, handler, cores, words):
    """One command on island 0's first `cores` ranks; host microseconds."""
    started = time.perf_counter()
    handle = run.island(0).submit(handler, (0, cores), words)
    (event,) = session.wait_together([(0, handle)], timeout=30.0)
    if event.kind != "done":
        raise km.KomorebiError(f"{handler}: {event}")
    return (time.perf_counter() - started) * 1e6


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    pattern = km.Pattern(args.pattern)
    report = {"cards": cards, "cores": args.cores, "tiles": args.tiles}
    report["across"] = "write" if args.write_across else "read"
    held = (0, 1) if args.write_across else (1, 0)  # the input's island, the output's
    with km.Session(cards=cards, links=1, log=log) as session:
        islands = [
            (session.graph.first(i), session.graph.count(i))
            for i in range(len(session.sessions))
        ]
        log(f"open: islands {islands}")
        data = pages_of(np.random.default_rng(args.seed), args.cores * args.tiles)
        tensors = {
            "input": session.tensor(data, "uint16", group=islands[held[0]]),
            "output": session.tensor(
                np.zeros_like(data), "uint16", group=islands[held[1]]
            ),
        }
        report["input_parts"] = list(tensors["input"].placement.parts)
        report["output_parts"] = list(tensors["output"].placement.parts)
        log(f"input {report['input_parts']}, output {report['output_parts']}")
        run = session.load(program(tensors, args.block), deadline=5.0)
        log("loaded on every island")
        try:
            words = [
                tensors["input"].address,
                tensors["output"].address,
                args.tiles,
                args.block,
                pattern.word,
                1,
            ]
            report["copy_us"] = round(one(session, run, "copy", args.cores, words), 1)
            log(f"COPY across: {report['copy_us']} us")
            if not args.write_across:
                words[1] = 0
                spent = [
                    one(session, run, "read", args.cores, words)
                    for _ in range(args.rounds)
                ]
                report["read_us"] = round(float(np.median(spent)), 1)
                moved = args.cores * args.tiles * data.shape[1] * 2
                report["read_gb_s"] = round(moved / report["read_us"] / 1e3, 3)
                log(f"READ across: {report['read_us']} us, {report['read_gb_s']} GB/s")
        finally:
            results = run.stop()
            report["kernel_errors"] = [result["error_name"] for result in results]
            log(f"stopped: {report['kernel_errors']}")
        (got,) = session.read(tensors["output"])
        got = got.view(np.uint16).reshape(len(got), -1)
        report["pages_wrong"] = int((got != data).any(axis=1).sum())
        log(f"output read back: {report['pages_wrong']} pages differ")
    report["exact"] = report["pages_wrong"] == 0
    report["ok"] = report["exact"] and report["kernel_errors"] == ["none"] * len(cards)
    print(json.dumps(power.clocked(report)), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
