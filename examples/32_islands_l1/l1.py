"""An L1 tensor moved across islands: each rank's float32 tile, held in its L1 at one
address, goes to the rank `shift` after it (on its island or another) by km::write_l1,
and each rank writes the tile it got out. Two cards or one split; one JSON line (see
README.md)."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi.bench import power

KERNELS = Path(__file__).resolve().parent / "kernels"
CONSTANTS = km.abi.constants(KERNELS / "l1.h", known=km.abi.headers().constants)
TILE = 32
PORT = 1  # the port each receiver counts its tile at
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--islands", type=int, default=0, help="one card split")
    parser.add_argument(
        "--shifts", type=int, nargs="+", default=[1], help="0: half the ranks"
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--wait", type=float, default=0.5, help="wait limit, seconds")
    return parser.parse_args()


def program(tensor):
    page = CONSTANTS["kTileBytes"]
    return km.Program(
        "islands_l1",
        kernels=KERNELS,
        buffers=[
            km.Buffer(CONSTANTS["kCbHeld"], "float32", page, 1),
            km.Buffer(CONSTANTS["kCbLanding"], "float32", page, 1),
        ],
        # input and output alike: a copy of a tile a rank, one accessor for both
        compile_args={"reader": tensor.tensor_args()},
        handlers=KERNELS / "l1.h",
    )


def read_by_rank(session, tensor):
    """`tensor`'s tiles, rank r's from its own island's copy."""
    got = None
    for island, sub in enumerate(session.sessions):
        (raw,) = sub.read(tensor.parts[island])
        copy = formats.from_bytes(raw, tensor.shape, tensor.dtype, tensor.layout)
        copy = copy.reshape(-1, TILE, TILE)
        got = copy.copy() if got is None else got
        first, count = session.graph.first(island), session.graph.count(island)
        got[first : first + count] = copy[first : first + count]
    return got


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    opened = {"cards": cards}
    if args.islands:  # one card split, its islands in a line
        opened = {"card": cards[0], "islands": args.islands, "line": True}
    report = {"cards": cards, "islands": args.islands, "rows": []}
    generator = np.random.default_rng(1)
    with km.Session(**opened, log=log) as session:
        ranks = session.ranks
        report["ranks"] = ranks
        values = generator.standard_normal((ranks * TILE, TILE)).astype(np.float32)
        tiles = values.reshape(-1, TILE, TILE)
        inputs = session.tensor(values, "float32", "tile", copy=True)
        output = session.tensor(np.zeros_like(values), "float32", "tile", copy=True)
        run = session.load(program(inputs), wait=args.wait)
        for shift in args.shifts:
            shift = shift or ranks // 2
            words = [inputs.address, output.address, shift, PORT]
            card = []
            for _ in range(args.rounds):
                event = run.wait(run.submit("move", (0, ranks), words))
                card.append(event.microseconds)
            got = read_by_rank(session, output)
            # rank r got the tile of rank r - shift
            want = np.roll(tiles, shift, axis=0)
            wrong = [r for r in range(ranks) if not np.array_equal(got[r], want[r])]
            us = round(float(np.median(card)), 2)
            report["rows"].append(
                {"shift": shift, "card_us": us, "exact": not wrong, "wrong": wrong[:8]}
            )
            log(f"shift {shift}: {us} us on the card, exact {not wrong}")
        report["kernel_errors"] = [r["error_name"] for r in run.stop()]
    report["ok"] = all(row["exact"] for row in report["rows"]) and set(
        report["kernel_errors"]
    ) == {"none"}
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(power.clocked(report)), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
