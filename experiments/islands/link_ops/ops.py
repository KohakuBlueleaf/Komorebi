"""Writes into another rank's L1 and adds to its word (km::write_l1, km::add_word):
for each case S:T, island S's first rank makes --calls calls to island T's last rank,
then the target's ring, counter and port count are checked. One JSON line; README.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import islands

KERNELS = Path(__file__).resolve().parent / "kernels"
CLEAR, OPS, REPORT = 1, 2, 3  # kHandler* (ops.h)
KINDS = {"write": 1, "add": 2, "both": 3, "posted": 6, "all": 7}  # kKind*
NO_PORT = 0xFFFFFFFF
AREA_RING = 64  # kAreaRing: bytes before the ring
PAGE = 4096  # kAreaBytes: the area, one report page
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--card", type=int, help="one card split into --islands")
    parser.add_argument("--islands", type=int, default=0)
    parser.add_argument(
        "--graph",
        type=islands.parse_pairs,
        default=None,
        help="a split card's island pairs a-b,c-d (default: a line)",
    )
    parser.add_argument("--cases", default="0:0,0:1", help="S:T island pairs")
    parser.add_argument("--calls", type=int, default=1000)
    parser.add_argument("--kind", choices=tuple(KINDS), default="both")
    parser.add_argument("--bytes", type=int, default=16, help="bytes a write")
    parser.add_argument("--slots", type=int, default=128, help="the ring's slots")
    parser.add_argument("--port", type=int, default=3, help="-1: no count")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--refused",
        type=int,
        default=0,
        help="cases this many hops away or more: refused",
    )
    return parser.parse_args()


def stride(size):
    """A ring slot's bytes: `size` in whole 16s (the kernel's)."""
    return (size + 15) // 16 * 16


def expected_ring(calls, slots, size, rank):
    """The ring's words after `calls` calls: slot s (16-aligned) holds the last call
    s + k slots."""
    words = size // 4
    ring = np.zeros((slots, stride(size) // 4), np.uint32)
    for slot in range(min(slots, calls)):
        call = slot + (calls - 1 - slot) // slots * slots
        for word in range(words):
            ring[slot, word] = (
                call if word == 0 else (call ^ word << 24 ^ rank << 12) & 0xFFFFFFFF
            )
    return ring.reshape(-1)


def check(row, args, target):
    """What the target's report shows against what the calls should have left."""
    kind = KINDS[args.kind]
    counter, port = int(row[0]), int(row[1])
    out = {"counter": counter, "port": port}
    good = True
    if kind & 2:
        good &= counter == args.calls
    if kind & 1:
        if args.port >= 0:
            good &= port == args.calls
        ring = row[
            AREA_RING // 4 : AREA_RING // 4 + args.slots * stride(args.bytes) // 4
        ]
        wrong = np.flatnonzero(
            ring != expected_ring(args.calls, args.slots, args.bytes, target)
        )
        out["ring_wrong"] = int(wrong.size)
        if wrong.size:
            out["first_wrong"] = [int(wrong[0]), int(ring[wrong[0]])]
        good &= wrong.size == 0
    out["ok"] = bool(good)
    return out


def page_of(session, tensor, page):
    """Page `page` of a report tensor as words, read while the run is loaded."""
    (got,) = session.read(tensor, first=page, pages=1)
    return got.view(np.uint32).reshape(-1)


def main():
    args = parse()
    if AREA_RING + args.slots * stride(args.bytes) > PAGE or args.bytes % 4:
        sys.exit("the ring does not fit the area, or --bytes is not whole 4s")
    if args.card is not None:
        graph = args.graph or islands.line_pairs(args.islands)
        opened = {"card": args.card, "islands": args.islands, "graph": graph}
    else:
        opened = {"cards": [int(card) for card in args.cards.split(",")]}
    cases = [tuple(int(x) for x in case.split(":")) for case in args.cases.split(",")]
    report = {
        "opened": opened,
        "calls": args.calls,
        "kind": args.kind,
        "bytes": args.bytes,
        "slots": args.slots,
        "port": args.port,
        "cases": [],
    }
    ok = True
    with km.Session(**opened, log=log) as session:
        graph = session.graph
        count = len(graph.islands)
        own = [(graph.first(i), graph.count(i)) for i in range(count)]
        report["islands"] = count
        reports = [
            session.empty((2, PAGE // 4), "uint32", group=own[i]) for i in range(count)
        ]
        program = km.Program(
            "link_ops",
            kernels=KERNELS,
            handlers=KERNELS / "ops.h",
            buffers=[
                km.Buffer(index=0, format="bfloat16", page_bytes=PAGE, pages=1),
                km.Buffer(index=1, format="bfloat16", page_bytes=PAGE, pages=1),
            ],
            compile_args={"reader": reports[0].accessor_args()},
        )
        run = session.load(program, deadline=args.timeout)

        def one(island, handler, rank, words):
            part = run.island(island)
            return part.wait(
                part.submit(handler, (rank, 1), words), timeout=args.timeout
            )

        try:
            ok = cases_run(args, cases, own, graph, reports, one, session, report)
        finally:
            try:
                report["kernel_errors"] = [r["error_name"] for r in run.stop().islands]
            except km.KomorebiError as error:
                report["kernel_errors"] = str(error)[:300]
                ok = False
    report["ok"] = bool(ok) and set(report["kernel_errors"]) == {"none"}
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


def cases_run(args, cases, own, graph, reports, one, session, report):
    """Each case S:T in turn, its record into `report`; False at the first wrong."""
    port = NO_PORT if args.port < 0 else args.port
    clear_port = max(args.port, 0)
    ok = True
    for s, t in cases:
        sender = own[s][0]
        target = own[t][0] + own[t][1] - 1
        record = {
            "case": f"{s}:{t}",
            "sender": sender,
            "target": target,
            "hops": graph.hops(s, t),
        }
        try:
            one(t, CLEAR, target, [clear_port])
            one(s, CLEAR, sender, [clear_port])
            words = [target, args.calls, port, args.slots, args.bytes, KINDS[args.kind]]
            event = one(s, OPS, sender, words)
            record["us"] = round(event.ticks / 50.0, 2)  # mtime: 50 MHz
            record["us_a_call"] = round(event.ticks / 50.0 / args.calls, 3)
            one(t, REPORT, target, [reports[t].address, 0, clear_port])
            one(s, REPORT, sender, [reports[s].address, 1, clear_port])
            target_row = page_of(session, reports[t], 0)
            sender_row = page_of(session, reports[s], 1)
            record["done"] = int(sender_row[2])
            record.update(check(target_row, args, target))
            record["ok"] = record["ok"] and record["done"] == args.calls
            if args.refused and record["hops"] >= args.refused:  # nothing done
                record["refused"] = record["done"] == 0 and record["counter"] == 0
                record["ok"] = record["refused"]
        except km.KomorebiError as error:
            record["error"] = f"{type(error).__name__}: {error}"[:800]
            record["ok"] = False
        log(json.dumps(record))
        report["cases"].append(record)
        ok &= record["ok"]
        if "error" in record:
            break
    return ok


if __name__ == "__main__":
    main()
