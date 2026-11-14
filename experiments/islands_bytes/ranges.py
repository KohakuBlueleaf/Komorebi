"""Ranges of a page across islands: island 0's ranks read ranges of 4 to 2048 bytes at
many pages and offsets of a tensor island 1 holds (and, beside it, one island 0
holds), one in flight and many; every word checked. Prints one JSON line. See
README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

KERNELS = Path(__file__).resolve().parent / "kernels"
PAGE = 4096
SLOT = 64 + 64 + 2048 + 64  # kRangesSlot
IN_FLIGHT = 32  # kRangesInFlight
MAGIC = 0x52414E47  # kRangesMagic
NONE = 0xFFFFFFFF
COPY = 64 // 4  # kRangesCopy, in words
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--links", type=int, default=4)
    parser.add_argument("--pages", type=int, default=256, help="pages of the source")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 64, 128, 2048])
    parser.add_argument("--flights", type=int, nargs="+", default=[1, IN_FLIGHT])
    parser.add_argument("--ranges", type=int, default=64, help="ranges a rank")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def pattern(pages):
    """The source's words: word w of page p is (p << 16 | w) ^ 0x9E3779B9."""
    p = np.arange(pages, dtype=np.uint32)[:, None] << np.uint32(16)
    w = np.arange(PAGE // 4, dtype=np.uint32)[None, :]
    return (p | w) ^ np.uint32(0x9E3779B9)


def aiclk(card):
    """The card's AICLK in MHz as the driver reports it (None: not readable)."""
    try:
        path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
        return int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def check(rows, data, size, ranges):
    """Each rank's report: its ranges all read, no word wrong, no guard changed, its
    last range's bytes those of the host's data."""
    reports = []
    for row in rows:
        page, offset = int(row[8]), int(row[9])
        copy = row[COPY : COPY + size // 4]
        host = data[page, offset // 4 : offset // 4 + size // 4]
        reports.append(
            {
                "magic": int(row[0]) == MAGIC,
                "done": int(row[2]),
                "wrong": int(row[3]),
                "guard": int(row[4]),
                "first": None if int(row[5]) == NONE else int(row[5]),
                "cycles": int(row[6]),
                "failed": int(row[7]),
                "last_exact": bool((copy == host).all()),
            }
        )
    bad = [
        rank
        for rank, one in enumerate(reports)
        if not one["magic"]
        or one["done"] != ranges
        or one["wrong"]
        or one["guard"]
        or one["failed"]
        or not one["last_exact"]
    ]
    return reports, bad


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    data = pattern(args.pages)
    configs = [
        (place, size, flight)
        for place in ("across", "here")
        for size in args.sizes
        for flight in args.flights
    ]
    report = {"cards": cards, "links": args.links, "page": PAGE, "rows": []}
    with km.Session(cards=cards, links=args.links, log=log) as session:
        graph = session.graph
        own = [(graph.first(i), graph.count(i)) for i in range(2)]
        ranks = own[0][1]
        sources = {
            "across": session.tensor(data, "uint32", group=own[1]),
            "here": session.tensor(data, "uint32", group=own[0]),
        }
        out = session.empty((len(configs) * ranks, PAGE // 4), "uint32", group=own[0])
        for name, tensor in sources.items():
            log(f"source {name}: {tensor.placement.parts}")
        program = km.Program(
            "ranges",
            kernels=KERNELS,
            handlers=KERNELS / "ranges.h",
            buffers=[  # READ's slots, then its report page
                km.Buffer(
                    index=0,
                    format="bfloat16",
                    page_bytes=IN_FLIGHT * SLOT + PAGE,
                    pages=1,
                )
            ],
            compile_args={"reader": sources["across"].accessor_args()},
        )
        run = session.load(program, deadline=2.0)
        log(f"loaded on both islands, {ranks} ranks of island 0 read")
        handler = program.handler("read")
        mine = run.island(0)
        events = []
        for number, (place, size, flight) in enumerate(configs):
            words = [
                sources[place].address,
                out.address,
                args.pages,
                size,
                args.ranges,
                flight,
                args.seed + number,
                number * ranks,
            ]
            clock = aiclk(cards[0])
            before = run.crossings()
            event = mine.wait(mine.submit(handler, own[0], words), timeout=30.0)
            moved = run.crossings() - before
            events.append((event.ticks / 50e6, clock, aiclk(cards[0]), moved))
            ms = event.ticks / 50e3  # the X280's mtime: 50 MHz
            log(f"READ {place} {size} B, {flight} in flight: {ms:.3f} ms")
        try:
            results = run.stop()
            report["kernel_errors"] = [result["error_name"] for result in results]
        except km.KomorebiError as error:
            report["kernel_errors"] = str(error)[:300]
        log(f"stopped: {report['kernel_errors']}")
        for number, (place, size, flight) in enumerate(configs):
            (got,) = session.read(out, first=number * ranks, pages=ranks)
            rows = got.view(np.uint32).reshape(ranks, -1)
            reports, bad = check(rows, data, size, args.ranges)
            seconds, clock0, clock1, moved = events[number]
            mhz = clock1 or 1350  # sampled after the command, the run still loaded
            cycles = np.array([one["cycles"] for one in reports], dtype=np.float64)
            per = cycles / max(args.ranges, 1) / mhz
            row = {
                "place": place,
                "bytes": size,
                "in_flight": flight,
                "ranks": ranks,
                "ranges": ranks * args.ranges,
                "command_ms": round(seconds * 1e3, 3),
                "us_a_range_mean": round(float(per.mean()), 3),
                "us_a_range_max": round(float(per.max()), 3),
                "aiclk_mhz": [clock0, clock1],
                "crossed": {
                    f"{a}->{b}": moved.total(a, b) for a, b in ((0, 1), (1, 0))
                },
                "words_wrong": sum(one["wrong"] for one in reports),
                "guards_changed": sum(one["guard"] for one in reports),
                "bad_ranks": bad[:8],
            }
            log(
                f"{place} {size} B x{flight}: {row['us_a_range_mean']} us a range "
                f"(max {row['us_a_range_max']}), wrong {row['words_wrong']}, "
                f"guards {row['guards_changed']}, bad ranks {len(bad)}"
            )
            report["rows"].append(row)
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    report["ok"] = bool(
        report["kernel_errors"] == ["none", "none"]
        and all(not row["bad_ranks"] for row in report["rows"])
    )
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
