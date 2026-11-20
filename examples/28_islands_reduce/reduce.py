"""All-reduce across islands: a tensor held as a copy on every island, each copy that
island's partial; one call and every island holds the sum, added in the tensor's
format. The call from a kernel (`km::all_reduce`) and from the host
(`run.all_reduce`), on two cards and on one. Prints one JSON line. See README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi.bench import power
from komorebi.islands import line_pairs

KERNELS = Path(__file__).resolve().parent / "kernels"
CONSTANTS = km.abi.constants(KERNELS / "reduce.h", known=km.abi.headers().constants)
TILE = 32
SUM = {"int32": 0, "float32": 1, "bfloat16": 2}  # km::Sum
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument(
        "--islands", type=int, default=0, help="one card split into this many islands"
    )
    parser.add_argument("--dtype", choices=sorted(SUM), default="float32")
    parser.add_argument("--pages", type=int, default=256, help="tiles")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--wait", type=float, default=0.5, help="wait limit, seconds")
    return parser.parse_args()


def write_copy(session, tensor, island, values):
    """Island `island`'s copy of `tensor` becomes `values` (through its host channel,
    with the run loaded)."""
    raw, _ = formats.to_bytes(values, tensor.dtype, tensor.layout)
    pages = raw.reshape(tensor.pages, tensor.page_bytes)
    session.sessions[island].write(tensor.parts[island], pages)


def read_copy(session, tensor, island):
    """Island `island`'s copy of `tensor`, its values."""
    (raw,) = session.sessions[island].read(tensor.parts[island])
    return formats.from_bytes(raw, tensor.shape, tensor.dtype, tensor.layout)


def program():
    return km.Program(
        "islands_reduce",
        kernels=KERNELS,
        buffers=[
            km.Buffer(
                CONSTANTS["kCbFold"],
                "uint32",
                CONSTANTS["kFoldBytes"],
                CONSTANTS["kFoldPages"],
            )
        ],
        handlers=KERNELS / "reduce.h",
    )


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    opened = {"cards": cards}
    if args.islands:  # one card split, its islands in a line
        line = line_pairs(args.islands)
        opened = {"card": cards[0], "islands": args.islands, "graph": line}
    report = {"cards": cards, "dtype": args.dtype, "pages": args.pages}
    generator = np.random.default_rng(1)
    with km.Session(**opened, log=log) as session:
        ranks = session.ranks
        islands = len(session.sessions)
        report["islands"] = islands
        # small integers: every sum is exact in each format, in any order
        partials = [
            generator.integers(-8, 9, (args.pages * TILE, TILE)).astype(np.float32)
            for _ in range(islands)
        ]
        want = np.sum(partials, axis=0)
        if args.dtype == "int32":
            partials = [p.astype(np.int32) for p in partials]
            want = want.astype(np.int32)
        zeros = np.zeros_like(partials[0])
        tensor = session.tensor(zeros, args.dtype, "tile", copy=True)
        # the most neighbours of an island, a partial a child; one at least (one
        # island has none, and its all_reduce returns at once)
        ports = max(
            [len(session.graph.neighbours(i)) for i in range(len(session.sessions))]
            + [1]
        )
        scratch = session.tensor(
            np.concatenate([zeros] * ports), args.dtype, "tile", copy=True
        )
        report["bytes"] = tensor.pages * tensor.page_bytes
        run = session.load(program(), wait=args.wait)
        words = [tensor.address, 0, args.pages, scratch.address, SUM[args.dtype]]

        def from_a_kernel():  # REDUCE on every rank: each makes the call
            return run.wait(run.submit("reduce", (0, ranks), words))

        def from_the_host():
            return run.all_reduce(tensor, scratch)

        for name, call in (("kernel", from_a_kernel), ("host", from_the_host)):
            times, card, exact = [], [], True
            for _ in range(args.rounds):  # the partials written again each round
                for island, partial in enumerate(partials):
                    write_copy(session, tensor, island, partial)
                start = time.perf_counter()
                events = call()
                times.append((time.perf_counter() - start) * 1e6)
                # on the card, submit to done, the slowest island (one island: none)
                if events is not None:
                    card.append(max(event.microseconds for event in events))
                exact &= all(
                    np.array_equal(read_copy(session, tensor, island), want)
                    for island in range(islands)
                )
            report[f"{name}_exact"] = bool(exact)
            us = round(float(np.median(times)), 2)
            on_card = round(float(np.median(card)), 2) if card else None
            report[f"{name}_us"], report[f"{name}_card_us"] = us, on_card
            log(f"from the {name}: {us} us ({on_card} on the card), exact {exact}")
        report["kernel_errors"] = [r["error_name"] for r in run.stop().islands]
    report["ok"] = (
        report["kernel_exact"]
        and report["host_exact"]
        and set(report["kernel_errors"]) == {"none"}
    )
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(power.clocked(report)), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
