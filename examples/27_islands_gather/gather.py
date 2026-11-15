"""All-gather across islands: a tensor held as a copy on every island, island c
having written only its share; one call and every island holds the whole. The call
from a kernel (`km::all_gather`) and from the host (`run.all_gather`), on two cards
and on one. Prints one JSON line. See README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import formats, power
from komorebi.layout import partition

KERNELS = Path(__file__).resolve().parent / "kernels"
TILE = 32
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
    parser.add_argument("--pages", type=int, default=256, help="float32 tiles, 4 KiB")
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


def write_shares(session, tensor, whole):
    """Island c's copy: its share of `whole` (by tiles), zeros elsewhere."""
    islands = len(session.sessions)
    for c in range(islands):
        share = partition(range(tensor.pages), c, islands)
        mine = np.zeros_like(whole)
        rows = slice(share.start * TILE, share.stop * TILE)
        mine[rows] = whole[rows]
        write_copy(session, tensor, c, mine)


def every_copy_whole(session, tensor, whole):
    """True when every island's copy is `whole`."""
    return all(
        np.array_equal(read_copy(session, tensor, c), whole)
        for c in range(len(session.sessions))
    )


def card_us(events):
    """A command's time on the card, submit to done, its slowest island (None: no
    command, one island)."""
    return None if events is None else max(event.microseconds for event in events)


def timed(rounds, call):
    """The median microseconds of `call` over `rounds`: on the host, from the call to
    its return, and on the card (card_us)."""
    host, card = [], []
    for _ in range(rounds):
        start = time.perf_counter()
        events = call()
        host.append((time.perf_counter() - start) * 1e6)
        card.append(card_us(events))
    on_card = None if None in card else round(float(np.median(card)), 2)
    return round(float(np.median(host)), 2), on_card


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    opened = {"cards": cards}
    if args.islands:  # one card split, its islands in a line
        opened = {"card": cards[0], "islands": args.islands, "line": True}
    report = {"cards": cards, "pages": args.pages, "bytes": args.pages * 4096}
    whole = np.random.default_rng(1).standard_normal((args.pages * TILE, TILE))
    whole = whole.astype(np.float32)
    with km.Session(**opened, log=log) as session:
        ranks = session.ranks
        report["islands"] = len(session.sessions)
        tensor = session.tensor(np.zeros_like(whole), "float32", "tile", copy=True)
        program = km.Program(
            "islands_gather", kernels=KERNELS, handlers=KERNELS / "gather.h"
        )
        run = session.load(program, wait=args.wait)
        words = [tensor.address, 0, args.pages]

        def from_a_kernel():  # GATHER on every rank: each makes the call
            return run.wait(run.submit("gather", (0, ranks), words))

        def from_the_host():
            return run.all_gather(tensor)

        for name, call in (("kernel", from_a_kernel), ("host", from_the_host)):
            write_shares(session, tensor, whole)
            call()
            report[f"{name}_exact"] = every_copy_whole(session, tensor, whole)
            us, on_card = timed(args.rounds, call)
            report[f"{name}_us"], report[f"{name}_card_us"] = us, on_card
            exact = report[f"{name}_exact"]
            log(f"from the {name}: {us} us ({on_card} on the card), exact {exact}")
        report["kernel_errors"] = [r["error_name"] for r in run.stop()]
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
