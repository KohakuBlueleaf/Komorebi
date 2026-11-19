"""Commands across islands from island 0's own run that do not finish there: a HANG
whose hung rank is on island 1 is given up naming island 1 and the run goes on; a
command onto island 1's cores holding another program gets island 1's rejection, not a
hang. Two cards. One JSON line (README.md)."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

ROOT = Path(__file__).resolve().parents[3]
KERNELS = ROOT / "examples" / "01_mapcollect" / "kernels"
TWO = ROOT / "examples" / "16_two_programs" / "kernels"  # sum, scale: runtime's form
TILE = 32
EVENT_ISLAND = 0x80  # KM_EVENT_ISLAND
HANG_PORT = 1  # mapcollect.h's kHangPort: the reader waits for an arrival at port 0
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument(
        "--dispatcher", help="the islands' dispatcher (default: as chosen)"
    )
    parser.add_argument("--deadline", type=float, default=1.0, help="seconds a command")
    return parser.parse_args()


def program(source, output):
    """Example 01's kernels (its HANG among their handlers)."""
    return km.Program(
        "mapcollect",
        kernels=KERNELS,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
        ],
        compile_args={
            "reader": source.accessor_args(),
            "writer": output.accessor_args(),
        },
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        handlers=KERNELS / "mapcollect.h",
    )


def two(source, output):
    """Example 16's sum and scale, each with EMPTY (handler 2: nothing)."""
    return [
        km.Program(
            name,
            kernels=TWO / name,
            buffers=[
                km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
                km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
            ],
            compile_args={
                "reader": source.tensor_args(),
                "writer": output.tensor_args(),
            },
            compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
            handlers={name: 1, "empty": 2},
        )
        for name in ("sum", "scale")
    ]


def done(part, group):
    """A NOP on `group` from `part` (an island's own run): whether it is done."""
    return part.wait(part.submit(0, group), timeout=10.0).kind == "done"


def given_up(first, own):
    """A HANG on island 0's last 10 ranks and island 1's first 10 from island 0's run,
    island 1's rank 5 hung: TIMEOUT naming island 1; then a NOP across past the hung
    rank and a NOP on island 0's ranks are done."""
    group = (own[1][0] - 10, 20)
    try:
        hang = first.submit("hang", group, [15, HANG_PORT])
        event = first.wait(hang, timeout=20.0)
        named, given = event.kind, False
    except km.CommandTimeout as error:
        named, given = error.slot & 0xFF, bool(error.given_up)
    log(f"HANG on {group}, island 1's rank 5 hung: given up {given}, named {named}")
    across = (own[0][0], own[0][1] + 5)  # island 0 and island 1's ranks before it
    after = {"across": done(first, across), "local": done(first, own[0])}
    log(f"after it: a NOP across {across} and on {own[0]}: {after}")
    return {
        "hang_given_up": given,
        "hang_named": named,
        "hang_named_island_1": given and named == EVENT_ISLAND + 1,
        "after_hang": after,
    }


def rejected(run, first, own, programs):
    """Island 1's first 5 ranks switched to program b, then a's EMPTY on island 0's last
    5 ranks and those from island 0's run, unchecked by the host: island 1's dispatcher
    rejects it (its cores hold another program) and island 0's run gets the rejection;
    after a's load back there, the same EMPTY is done."""
    a, b = programs
    part = (own[1][0], 5)
    group = (own[1][0] - 5, 10)
    run.wait(run.load(b, group=part), 10.0)
    first.check_programs = False  # the far dispatcher's own refusal
    try:
        kind = first.wait(first.submit(a.handler("empty"), group), 10.0).kind
        reason = None
    except km.CommandRejected as error:
        kind, reason = "rejected", str(error)
    except km.CommandTimeout as error:
        kind, reason = "timeout", str(error)
    first.check_programs = True
    log(f"a's EMPTY across onto b's cores: {kind} ({reason})")
    run.wait(run.load(a, group=part), 10.0)
    again = first.wait(first.submit(a.handler("empty"), group), 10.0).kind
    log(f"after a's load back: {again}")
    return {
        "across_onto_other_program": kind,
        "rejection": reason,
        "rejected_program": kind == "rejected" and "program" in (reason or ""),
        "after_rejection": again == "done",
    }


def stopped(run):
    """The run's stop: "clean", or its error's first line."""
    try:
        run.stop()
        return "clean"
    except km.KomorebiError as error:
        return str(error).split("\n")[0][:200]


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    report = {"cards": cards}
    with km.Session(cards=cards, dispatcher=args.dispatcher, log=log) as session:
        ranks, graph = session.ranks, session.graph
        own = [(graph.first(i), graph.count(i)) for i in range(len(cards))]
        report |= {"dispatcher": session.dispatcher, "ranks": ranks, "own": own}
        source = session.tensor(
            np.zeros((2 * ranks * TILE, TILE), np.float32), "bfloat16", "tile"
        )
        output = session.tensor(
            np.zeros((ranks * TILE, TILE), np.float32), "bfloat16", "tile"
        )
        run = session.load(program(source, output), deadline=args.deadline)
        first = run.island(0)
        report["across_before"] = done(first, (0, ranks))
        report |= given_up(first, own)
        report["stop"] = [stopped(run)]
        made = two(source, output)
        run = session.load(
            {made[0]: (0, ranks - 10), made[1]: (ranks - 10, 10)},
            deadline=args.deadline,
        )
        report |= rejected(run, run.island(0), own, made)
        report["stop"].append(stopped(run))
    report["ok"] = bool(
        report["across_before"]
        and report["hang_named_island_1"]
        and all(report["after_hang"].values())
        and report["rejected_program"]
        and report["after_rejection"]
        and report["stop"] == ["clean", "clean"]
    )
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
