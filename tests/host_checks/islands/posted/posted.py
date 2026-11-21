"""A posted write into another island's L1 (km::write_l1 with km::posted) beside the
confirmed one: rows sent across a round at a time, each checked where it lands, the
order of two posted writes and the fence of a confirmed call after them. On the mock
(kmr prebuild) it builds; on cards it prints one JSON line."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.device import mock

KERNELS = Path(__file__).resolve().parent / "kernels"
C = km.abi.constants(KERNELS / "posted.h", known=km.abi.headers().constants)
PAGE = C["kPage"]
LAST = {0: 0, 1: 0xC000}  # the landing's last row: the last round's, the fence's
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1", help="the chips, as named (visible)")
    parser.add_argument("--islands", type=int, default=0, help="one card split")
    parser.add_argument("--links", type=int, default=0, help="joined islands' links")
    parser.add_argument("--bytes", type=int, nargs="+", default=[10240, 20480])
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--wait", type=float, default=5.0, help="wait limit, seconds")
    return parser.parse_args()


def program(output):
    def buffer(index, pages):
        return km.Buffer(index, "float32", PAGE, pages)

    return km.Program(
        "posted_l1",
        kernels=KERNELS,
        buffers=[
            buffer(C["kCbRows"], C["kRowsHeld"] * C["kRowBytes"] // PAGE),
            buffer(C["kCbLanding"], 2 * C["kRowBytes"] // PAGE),
            buffer(C["kCbWords"], 1),
            buffer(C["kCbResults"], C["kResultPages"]),
        ],
        compile_args={"reader": output.tensor_args()},
        handlers=KERNELS / "posted.h",
    )


def pattern(salt, words):
    """The kernel's row for `salt` (reader.cpp's pattern)."""
    i = np.arange(words, dtype=np.uint64)
    x = (salt * 0x9E3779B1 + i * 0x85EBCA6B + 0x7F4A7C15) & 0xFFFFFFFF
    return (x ^ (x >> 15)).astype(np.uint32)


def words_of(session, output, island):
    (raw,) = session.sessions[island].read(output.parts[island])
    return np.frombuffer(bytes(raw), dtype=np.uint32)


def summary(values, mhz):
    us = np.asarray(values, dtype=np.float64) / mhz
    return {
        "median_us": round(float(np.median(us)), 2),
        "min_us": round(float(us.min()), 2),
        "max_us": round(float(us.max()), 2),
    }


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    opened = {"cards": cards}
    if args.islands:
        opened = {"card": cards[0], "islands": 2, "graph": [(0, 1)]}
    if args.links:
        opened["links"] = args.links
    report = {"cards": cards, "rounds": args.rounds, "rows": []}
    with km.Session(**opened, log=log) as session:
        zeros = np.zeros((C["kOutPages"] * 32, 32), np.float32)
        output = session.tensor(zeros, "float32", "tile", copy=True)
        run = session.load(program(output), wait=args.wait)
        mhz = run.tick_mhz
        sender = session.graph.first(0) + session.graph.count(0) - 1
        report["group"], report["mhz"] = [sender, 2], mhz
        for size in args.bytes:
            for posted in (0, 1):
                words = [output.address, posted, size, args.rounds]
                event = run.wait(run.submit("rounds", (sender, 2), words))
                sent = words_of(session, output, 0)
                got = words_of(session, output, 1)
                rounds = sent[: args.rounds * 4].reshape(-1, 4)
                base = C["kOutWords"] * PAGE // 4
                here = got[base + 12 : base + 16]  # kSent: bad, order, fence, rounds
                last = got[C["kOutRow"] * PAGE // 4 :][: size // 4]
                row = {
                    "bytes": size,
                    "form": "posted" if posted else "confirmed",
                    "command_us": round(event.microseconds, 1),
                    "bad_words": int(here[0]) + int(rounds[:, 3].sum()),
                    "last_row_exact": bool(
                        np.array_equal(
                            last, pattern(LAST[posted] or args.rounds - 1, size // 4)
                        )
                    ),
                    "call": summary(rounds[:, 0], mhz),
                    "landed": summary(rounds[:, 1].view(np.int32), mhz),
                    "round_trip": summary(rounds[:, 2], mhz),
                }
                if posted:
                    fence = sent[base + 12 : base + 14]
                    row["order_bad_words"] = int(here[1])
                    row["fence_bad_words"] = int(here[2])
                    row["two_posted_calls_us"] = round(float(fence[0]) / mhz, 2)
                    row["fence_call_us"] = round(float(fence[1]) / mhz, 2)
                row["exact"] = (
                    row["bad_words"] == 0
                    and row["last_row_exact"]
                    and row.get("order_bad_words", 0) == 0
                    and row.get("fence_bad_words", 0) == 0
                    and int(here[3]) == args.rounds
                )
                report["rows"].append(row)
                log(json.dumps(row))
        report["kernel_errors"] = [r["error_name"] for r in run.stop().islands]
    exact = mock.active() or all(row["exact"] for row in report["rows"])  # zeros
    report["ok"] = exact and set(report["kernel_errors"]) == {"none"}
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
