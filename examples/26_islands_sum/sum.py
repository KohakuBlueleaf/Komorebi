"""A sum written for islands: each island folds its part (`km::local`), each island's
leader (`group.leaders()`) sends it to the first leader (`km::to_island(i).put`). The
same script and kernels on two cards and on one; the total against numpy.
Prints one JSON line. See README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

KERNELS = Path(__file__).resolve().parent / "kernels"
CONSTANTS = km.abi.constants(KERNELS / "sum.h", known=km.abi.headers().constants)
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
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--tiles", type=int, default=512, help="input tiles T")
    parser.add_argument("--part", type=int, default=1, help="tiles per part P (1 to 3)")
    parser.add_argument("--wait", type=float, default=0.5, help="wait limit, seconds")
    parser.add_argument(
        "--deadline", type=float, default=1.0, help="a command, seconds"
    )
    return parser.parse_args()


def program(input_tensor, output_tensor):
    page = CONSTANTS["kTileBytes"]
    return km.Program(
        "islands_sum",
        kernels=KERNELS,
        buffers=[
            km.Buffer(CONSTANTS["kCbIn"], "float32", page, CONSTANTS["kInPages"]),
            km.Buffer(
                CONSTANTS["kCbArrive"], "float32", page, CONSTANTS["kArrivePages"]
            ),
            km.Buffer(CONSTANTS["kCbOut"], "float32", page, CONSTANTS["kOutPages"]),
        ],
        compile_args={
            "reader": input_tensor.accessor_args(),
            "writer": output_tensor.accessor_args(),
        },
        compute={
            "math_fidelity": "HiFi4",
            "fp32_dest_acc_en": True,
            "unpack_to_dest_mode": "UnpackToDestFp32",
        },
        handlers=KERNELS / "sum.h",
    )


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    generator = np.random.default_rng(1)
    report = {"cards": cards, "tiles": args.tiles, "part": args.part}
    links = 1 if len(cards) > 1 or args.islands else 0
    # the link cores' services make the kernels 83.5 KB: past the default
    opened = {"cards": cards} if not args.islands else {"card": cards[0]}
    with km.Session(
        **opened,
        islands=args.islands or None,
        links=links,
        kernel_bytes=82 * 1024,
        log=log,
    ) as session:
        ranks = session.ranks
        report["islands"] = [
            [session.graph.first(i), session.graph.count(i)]
            for i in range(len(session.sessions))
        ]
        exact = generator.integers(-2, 3, (args.tiles * TILE, TILE)).astype(np.float32)
        inputs = session.tensor(exact, "float32", "tile", copy=True)
        output = session.tensor(
            np.zeros((args.part * TILE, TILE), np.float32), "float32", "tile", copy=True
        )
        run = session.load(
            program(inputs, output), deadline=args.deadline, wait=args.wait
        )
        log(f"loaded, islands {report['islands']}")
        tiles = exact.reshape(-1, TILE, TILE).astype(np.float64)
        want = np.stack([tiles[s :: args.part].sum(0) for s in range(args.part)])
        words = [inputs.address, args.tiles, output.address, 0, 1, 0, 0, args.part]
        island0 = run.island(0)  # a group over every rank, from island 0's dispatcher
        times, bad, error = [], 0, None
        try:
            for _ in range(args.rounds):
                start = time.perf_counter()
                handle = island0.submit("sum", (0, ranks), words)
                event = island0.wait(handle, timeout=5.0)
                times.append((time.perf_counter() - start) * 1e6)
                bad += event.kind != "done"
        except km.KomorebiError as caught:
            error = str(caught)
            log(f"error: {error}")
        report.update(
            group=[0, ranks],
            rounds=len(times),
            us_a_round=round(float(np.median(times)), 2) if times else None,
            not_done=bad,
            error=error,
        )
        try:
            report["kernel_errors"] = [r["error_name"] for r in run.stop()]
        except km.KomorebiError as caught:
            report["stop_error"] = str(caught)
        got = output.parts[0].read().reshape(-1, TILE, TILE)  # island 0's copy
        report["exact"] = bool(
            np.array_equal(got[: args.part], want.astype(np.float32))
        )
        report["max_error"] = float(np.abs(got[: args.part] - want).max())
    report["ok"] = (
        report["exact"]
        and not report["not_done"]
        and report["error"] is None
        and set(report.get("kernel_errors", ["?"])) == {"none"}
    )
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
