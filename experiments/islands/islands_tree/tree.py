"""The ports' reduce as an all-reduce, its sum cast back to every rank in one command:
each rank's P float32 tiles summed over the group, every rank's copy checked against
numpy, beside the plain reduce (rank 0 alone). Prints one JSON line. See README.md."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import formats

KERNELS = Path(__file__).resolve().parent / "kernels"
CONSTANTS = km.abi.constants(KERNELS / "tree.h", known=km.abi.headers().constants)
TILE = 32
X280_HZ = 50e6  # a command's ticks (the X280's mtime)
FORMS = {"owner": 0, "tree": 1}
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0")
    parser.add_argument("--islands", type=int, default=0, help="one card split")
    parser.add_argument("--shape", default=None, help="a split card's: line, ring")
    parser.add_argument(
        "--ranks", type=int, nargs="+", default=[16], help="group sizes (0: all)"
    )
    parser.add_argument(
        "--plain", type=int, nargs="*", default=None, help="the plain reduce's sizes"
    )
    parser.add_argument("--part", type=int, default=1, help="tiles a rank (1 to 3)")
    parser.add_argument("--form", choices=sorted(FORMS), default="tree")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--wait", type=float, default=0.5, help="wait limit, seconds")
    parser.add_argument("--deadline", type=float, default=0.05, help="a command")
    args = parser.parse_args()
    if not 1 <= args.part <= CONSTANTS["kPartMax"]:
        parser.error(f"--part: 1 to {CONSTANTS['kPartMax']}")
    return args


def program(input_tensor, output_tensor):
    page = CONSTANTS["kTileBytes"]
    return km.Program(
        "islands_tree",
        kernels=KERNELS,
        buffers=[
            km.Buffer(CONSTANTS["kCbIn"], "float32", page, CONSTANTS["kInPages"]),
            km.Buffer(
                CONSTANTS["kCbArrive"], "float32", page, CONSTANTS["kArrivePages"]
            ),
            km.Buffer(CONSTANTS["kCbOut"], "float32", page, CONSTANTS["kOutPages"]),
        ],
        compile_args={
            "reader": input_tensor.tensor_args(),
            "writer": output_tensor.tensor_args(),
        },
        compute={
            "math_fidelity": "HiFi4",
            "fp32_dest_acc_en": True,
            "unpack_to_dest_mode": "UnpackToDestFp32",
        },
        handlers=KERNELS / "tree.h",
    )


def zero_copies(session, tensor):
    """Every island's copy of `tensor` zeros (through its host channel, the run
    loaded)."""
    pages = np.zeros((tensor.pages, tensor.page_bytes), np.uint8)
    for island, sub in enumerate(session.sessions):
        sub.write(tensor.parts[island], pages)


def read_by_rank(session, tensor, part):
    """Tiles of `tensor`, rank r's P tiles from its own island's copy."""
    got = None
    for island, sub in enumerate(session.sessions):
        (raw,) = sub.read(tensor.parts[island])
        copy = formats.from_bytes(raw, tensor.shape, tensor.dtype, tensor.layout)
        copy = copy.reshape(-1, TILE, TILE)
        if got is None:
            got = copy.copy()
        first, count = session.graph.first(island), session.graph.count(island)
        got[first * part : (first + count) * part] = copy[
            first * part : (first + count) * part
        ]
    return got


def check(got, tiles, ranks, part, cast):
    """Ranks whose copy of the sum (cast: every rank's; else rank 0's) differs from
    the sum of ranks 0 .. ranks - 1's tiles."""
    want = tiles[: ranks * part].reshape(ranks, part, TILE, TILE).sum(0)
    holders = range(ranks) if cast else range(1)
    return [
        r
        for r in holders
        if not np.array_equal(got[r * part : (r + 1) * part], want.astype(np.float32))
    ]


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    generator = np.random.default_rng(1)
    opened = {"cards": cards}
    if args.islands:
        opened = {"card": cards[0], "islands": args.islands, "shape": args.shape}
    report = {"cards": cards, "islands": args.islands, "shape": args.shape}
    report.update(part=args.part, form=args.form, rows=[])
    with km.Session(**opened, log=log) as session:
        total = session.ranks
        pages = total * args.part
        values = generator.integers(-8, 9, (pages * TILE, TILE)).astype(np.float32)
        inputs = session.tensor(values, "float32", "tile", copy=True)
        output = session.tensor(
            np.zeros((pages * TILE, TILE), np.float32), "float32", "tile", copy=True
        )
        run = session.load(
            program(inputs, output), deadline=args.deadline, wait=args.wait
        )
        log(f"loaded, {total} ranks")
        tiles = values.reshape(-1, TILE, TILE).astype(np.float64)
        error = None
        try:
            for ranks in args.ranks:
                ranks = ranks or total
                plain = args.plain is None or ranks in args.plain
                for cast in (0, 1) if plain else (1,):
                    words = [
                        inputs.address,
                        output.address,
                        FORMS[args.form],
                        args.part,
                        cast,
                    ]
                    times, bad, wrong = [], 0, []
                    for _ in range(args.rounds):
                        zero_copies(session, output)
                        handle = run.submit("all_reduce", (0, ranks), words)
                        event = run.wait(handle, timeout=5.0)
                        bad += event.kind != "done"
                        times.append(event.ticks / X280_HZ * 1e6)
                        got = read_by_rank(session, output, args.part)
                        wrong += check(got, tiles, ranks, args.part, cast)
                    row = {
                        "ranks": ranks,
                        "cast": bool(cast),
                        "us_on_card": round(float(np.median(times)), 2),
                        "us_all": [round(t, 2) for t in times],
                        "not_done": bad,
                        "wrong_ranks": sorted(set(wrong)),
                    }
                    report["rows"].append(row)
                    log(json.dumps(row))
        except km.KomorebiError as caught:
            error = str(caught)
            log(f"error: {error}")
        report["error"] = error
        try:
            report["kernel_errors"] = [r["error_name"] for r in run.stop()]
        except km.KomorebiError as caught:
            report["stop_error"] = str(caught)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
