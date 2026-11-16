"""The tree collectives on a copy's pages: all_gather, all_reduce, reduce_to and
broadcast_from over the islands' trees, from a kernel (every rank makes the call) and
from the host, each copy checked against numpy. Prints one JSON line. See README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import formats, power

KERNELS = Path(__file__).resolve().parent / "kernels"
CONSTANTS = km.abi.constants(KERNELS / "pages.h", known=km.abi.headers().constants)
TILE = 32
CALLS = {
    "gather": CONSTANTS["kCallGather"],
    "reduce": CONSTANTS["kCallReduce"],
    "reduce_to": CONSTANTS["kCallReduceTo"],
    "broadcast": CONSTANTS["kCallBroadcast"],
}
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0")
    parser.add_argument("--islands", type=int, default=0, help="one card split")
    parser.add_argument("--shape", default=None, help="a split card's: line, ring")
    parser.add_argument("--pages", type=int, default=64, help="tiles")
    parser.add_argument("--calls", nargs="+", default=sorted(CALLS), choices=CALLS)
    parser.add_argument("--root", type=int, default=None, help="default: the last")
    parser.add_argument(
        "--tree", type=int, nargs="+", default=None, help="a parent an island"
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--wait", type=float, default=0.5, help="wait limit, seconds")
    return parser.parse_args()


def write_copy(session, tensor, island, values):
    """Island `island`'s copy of `tensor` becomes `values`."""
    raw, _ = formats.to_bytes(values, tensor.dtype, tensor.layout)
    session.sessions[island].write(
        tensor.parts[island], raw.reshape(tensor.pages, tensor.page_bytes)
    )


def read_copy(session, tensor, island):
    """Island `island`'s copy of `tensor`, its values."""
    (raw,) = session.sessions[island].read(tensor.parts[island])
    return formats.from_bytes(raw, tensor.shape, tensor.dtype, tensor.layout)


def program():
    return km.Program(
        "islands_tree_pages",
        kernels=KERNELS,
        buffers=[
            km.Buffer(
                CONSTANTS["kCbFold"],
                "uint32",
                CONSTANTS["kFoldBytes"],
                CONSTANTS["kFoldPages"],
            )
        ],
        handlers=KERNELS / "pages.h",
    )


def wanted(call, inputs, root):
    """What every island's copy holds after `call` (the islands' partials `inputs`)."""
    n = len(inputs)
    if call == "gather":  # island c's share partition(pages, c, n) from island c
        pages = inputs[0].shape[0] // TILE
        out, at = np.empty_like(inputs[0]), 0
        for c in range(n):
            count = pages // n + (c < pages % n)
            rows = slice(at * TILE, (at + count) * TILE)
            out[rows] = inputs[c][rows]
            at += count
        return [out] * n
    if call == "reduce":
        return [np.sum(inputs, axis=0)] * n
    if call == "broadcast":
        return [inputs[root]] * n
    return None  # reduce_to: the root's copy alone is checked


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    opened = {"cards": cards}
    if args.islands:
        opened = {"card": cards[0], "islands": args.islands, "shape": args.shape}
    report = {"cards": cards, "islands": args.islands, "shape": args.shape}
    report.update(pages=args.pages, rows=[])
    generator = np.random.default_rng(1)
    with km.Session(**opened, log=log) as session:
        ranks, islands = session.ranks, len(session.sessions)
        root = islands - 1 if args.root is None else args.root
        if args.tree is not None:  # its root: the island that is its own parent
            root = next(i for i, parent in enumerate(args.tree) if parent == i)
        report.update(root=root, tree=args.tree)
        # small integers: every sum exact in float32, in any order
        inputs = [
            generator.integers(-8, 9, (args.pages * TILE, TILE)).astype(np.float32)
            for _ in range(islands)
        ]
        zeros = np.zeros_like(inputs[0])
        tensor = session.tensor(zeros, "float32", "tile", copy=True)
        ports = 4  # a card's most neighbours
        scratch = session.tensor(
            np.concatenate([zeros] * ports), "float32", "tile", copy=True
        )
        run = session.load(program(), wait=args.wait)
        log(f"loaded, {islands} islands, {ranks} ranks")
        tree = args.tree
        hosts = {
            "gather": lambda: run.all_gather(tensor),
            "reduce": lambda: run.all_reduce(tensor, scratch, tree=tree),
            "reduce_to": lambda: run.reduce_to(tensor, scratch, root, tree=tree),
            "broadcast": lambda: run.broadcast_from(tensor, root, tree=tree),
        }
        packed = sum(parent << 4 * i for i, parent in enumerate(tree or ()))
        for call in args.calls:
            words = [tensor.address, 0, args.pages, scratch.address, 1, CALLS[call]]
            words += [root, int(tree is not None), packed & 0xFFFFFFFF, packed >> 32]

            def kernel(words=words):
                return run.wait(run.submit("tree", (0, ranks), words))

            want = wanted(call, inputs, root)
            for name, go in (("kernel", kernel), ("host", hosts[call])):
                card, exact = [], True
                for _ in range(args.rounds):
                    for island, values in enumerate(inputs):
                        write_copy(session, tensor, island, values)
                    events = go()
                    card.append(events.microseconds)  # the slowest island's part
                    if want is None:
                        got = [read_copy(session, tensor, root)]
                        good = [np.sum(inputs, axis=0)]
                    else:
                        got = [read_copy(session, tensor, i) for i in range(islands)]
                        good = want
                    exact &= all(np.array_equal(g, w) for g, w in zip(got, good))
                us = round(float(np.median(card)), 2)
                report["rows"].append(
                    {"call": call, "from": name, "card_us": us, "exact": bool(exact)}
                )
                log(f"{call} from the {name}: {us} us on the card, exact {exact}")
        report["kernel_errors"] = [r["error_name"] for r in run.stop()]
    report["ok"] = all(row["exact"] for row in report["rows"]) and set(
        report["kernel_errors"]
    ) == {"none"}
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(power.clocked(report)), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
