"""Thread split: every core pulls ranges of work from its X280 core; each answer's word
names the overlay to run on the range (three, fast on different units). Passes: each
overlay alone, the host's split, the thread's choice; the pull alone with and without a
word. --far: island 0 of two cards runs, island 1 holds the pages. See README.md."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.session import aiclk_mhz

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
HEADER = HERE / "split.h"
C = km.abi.constants(HEADER, known=km.abi.headers().constants)
CLUSTER = 1
TICKS_PER_MICROSECOND = 50  # kmx_time() counts the 50 MHz timer
PASSES = ["plain", "sparse", "dense", "host", "thread", "pick", "word"]
OVERLAYS = ["plain", "sparse", "dense"]
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--units", type=int, default=4096, help="tiles of work")
    parser.add_argument("--per", type=int, default=1, help="units per range")
    parser.add_argument(
        "--run", type=int, default=16, help="units of one kind in a row"
    )
    parser.add_argument("--penalty", type=int, default=20000, help="cycles off kind")
    parser.add_argument("--pull-units", type=int, default=0, help="0: 64 a core")
    parser.add_argument("--slots", type=int, default=2, help="the reader's slots")
    parser.add_argument(
        "--far",
        action="store_true",
        help="cards 0 and 1: island 0's cores write tiles and lines island 1 holds",
    )
    parser.add_argument(
        "--addon",
        default="",
        help="overlays not loaded (a comma list, e.g. sparse): run.addon adds each to "
        "the live run, then the host starts the thread's passes",
    )
    args = parser.parse_args()
    if not 1 <= args.per <= 255 or args.run < 1:
        parser.error("--per: 1 to 255 (a tell counts kinds in bytes); --run: 1 or more")
    return args


def kinds(units, run):
    """Each unit's kind (TS_KIND in numpy)."""
    x = (np.arange(units, dtype=np.uint64) // run * 0x9E3779B1) & 0xFFFFFFFF
    mixed = ((x ^ (x >> 15)) * 0x85EBCA77) & 0xFFFFFFFF
    return ((mixed >> 16) % 3).astype(np.int64)


def stripes(units, per, cores):
    """Each core's units [first, end): its stripe of the ranges, as the thread deals."""
    ranges = -(-units // per)
    edges = [core * ranges // cores * per for core in range(cores + 1)]
    return [(min(edges[c], units), min(edges[c + 1], units)) for c in range(cores)]


def host_split(unit_kinds, bounds):
    """The host's fixed split: each core the overlay of its stripe's commonest kind."""
    return [
        1 + int(np.bincount(unit_kinds[first:end], minlength=3).argmax())
        for first, end in bounds
    ]


def split_words(choice):
    """The split as the thread reads it: core c's overlay in 2 bits of word c // 16."""
    words = [0] * (C["TS_WORDS"] - C["TS_WORD_SPLIT"])
    for core, overlay in enumerate(choice):
        words[core // 16] |= overlay << 2 * (core % 16)
    return words


def results(units):
    """Every unit's result tile: value j of unit u is 2 * (1 + (u + j) % 7), bf16."""
    small = 1 + (np.arange(units)[:, None] + np.arange(C["TS_TILE_VALUES"])) % 7
    return (np.asarray(2.0 * small, np.float32).view(np.uint32) >> 16).astype(np.uint16)


# --8<-- [start:program]
def program(args, words, tiles, lines, thread=True):
    """Three kernel-form kernels, three overlays, the thread, every core's port (none
    of the two with `thread` False); the reader and its overlays write `tiles` and
    `lines`, the dense overlay's writer part `tiles`."""
    tile = dict(format="bfloat16", page_bytes=C["TS_TILE_BYTES"], pages=2)
    made = km.Program(
        "thread_split",
        kernels=KERNELS,
        buffers=[
            km.Buffer(C["TS_CB_IN"], **tile),
            km.Buffer(C["TS_CB_OUT"], **tile),
            km.Buffer(C["TS_CB_TILE"], "uint32", C["TS_TILE_BYTES"], 1),
            km.Buffer(C["TS_CB_JOB_COMPUTE"], "uint32", 64, 2),
            km.Buffer(C["TS_CB_JOB_WRITER"], "uint32", 64, 2),
            km.Buffer(C["TS_CB_LINE"], "uint32", 64, 1),
        ],
        compile_args={
            "reader": tiles.tensor_args() + lines.tensor_args(),
            "writer": tiles.tensor_args(),
        },
    )
    later = args.addon.split(",")  # added to the live run instead (--addon)
    for name in ("plain", "sparse", "dense"):
        if name not in later:
            made.overlay(name, KERNELS / name, number=C[f"TS_OVERLAY_{name.upper()}"])
    made.slots(count={"reader": args.slots})  # fewer than three: loads happen
    if not thread:
        return made
    made.thread(
        cluster=CLUSTER,
        sources=[HERE / "thread" / "split.c"],
        includes=[HERE],
        regions={"ticks": 4096},
        words=words,
        deadline=60.0,
        # islands, or --addon: the passes wait for the host's command
        **(
            {"handlers": {"passes": C["TS_THREAD_PASSES"]}}
            | {"defines": {"TS_STARTED_BY_HOST": "1"}}
            if args.far or args.addon
            else {}
        ),
    )
    made.ports()  # every core, dealt over the thread's X280 cores
    return made


# --8<-- [end:program]


def check(args, cores, lines, tiles, bounds, choice):
    """The wrong counts: tiles of the work passes, and lines of every pass."""
    want = results(args.units)
    wrong = {"tiles": 0, "lines": 0}
    for work in range(C["TS_WORK_PASSES"]):
        got = tiles[work * args.units : (work + 1) * args.units]
        wrong["tiles"] += int((got != want).any(axis=1).sum())
    per_core = np.array([end - first for first, end in bounds])
    for number, name in enumerate(PASSES):
        line = lines[number * cores : (number + 1) * cores].astype(np.int64)
        bad = line[:, C["TS_LINE_PASS"]] != number + 1
        bad |= line[:, C["TS_LINE_BAD"]] != 0
        if number < C["TS_WORK_PASSES"]:
            bad |= line[:, C["TS_LINE_UNITS"]] != per_core
            used = line[:, C["TS_LINE_PLAIN"] : C["TS_LINE_DENSE"] + 1]
            bad |= used.sum(axis=1) != per_core
            if number <= C["TS_PASS_DENSE"]:
                bad |= used[:, number] != per_core
            if name == "host":
                bad |= used[np.arange(cores), np.array(choice) - 1] != per_core
        else:
            pulls = line[:, C["TS_LINE_PULLS"]]
            bad |= pulls != [
                end - first for first, end in stripes(args.pull_units, 1, cores)
            ]
            bad |= line[:, C["TS_LINE_WORDS"]] != (pulls if name == "word" else 0)
        wrong["lines"] += int(bad.sum())
    return wrong


def table(args, cores, lines, ticks, megahertz, unit_kinds, choice):
    """Per pass: the thread's time, the slowest core, loads, units per overlay; the
    pull's cost with and without a word."""
    rows = {}
    for number, name in enumerate(PASSES):
        line = lines[number * cores : (number + 1) * cores].astype(np.int64)
        pulls = int(line[:, C["TS_LINE_PULLS"]].sum())
        rows[name] = {
            "pass_us": round(ticks[number] / TICKS_PER_MICROSECOND, 1),
            "slowest_core_us": round(
                float(line[:, C["TS_LINE_SPAN"]].max()) / megahertz, 1
            ),
            "loads": int(line[:, C["TS_LINE_LOADS"]].sum()),
            "loads_max_core": int(line[:, C["TS_LINE_LOADS"]].max()),
            "units": dict(
                zip(
                    OVERLAYS,
                    line[:, C["TS_LINE_PLAIN"] : C["TS_LINE_DENSE"] + 1]
                    .sum(0)
                    .tolist(),
                )
            ),
            "pulls": pulls,
            "next_cycles_mean": round(
                float(line[:, C["TS_LINE_WAITED"]].sum()) / max(pulls + cores, 1), 1
            ),
        }
    penalized = {
        name: int((unit_kinds != number).sum()) for number, name in enumerate(OVERLAYS)
    }
    bounds = stripes(args.units, args.per, cores)
    penalized["host"] = sum(
        int((unit_kinds[first:end] != choice[core] - 1).sum())
        for core, (first, end) in enumerate(bounds)
    )
    return rows, penalized


class Passes:
    """The thread's passes as the host's command ran them: their event."""

    def __init__(self, event):
        done = event.kind == "done"
        self.state = event.kind
        self.code = event.result if done else None
        self.error = "none" if done else event.kind
        self.words = []


def started_passes(part):
    """The thread runs the passes on its command, waited for (--far, --addon)."""
    handle = part.submit("passes", cluster=CLUSTER, args=(), result=True)
    return Passes(part.wait(handle, timeout=120.0))


def opened(args):
    """The session: one card, or (--far) cards 0 and 1 with a link."""
    if args.far:
        return km.Session(cards=[0, 1], links=1, log=log)
    return km.Session(log=log)


def main():
    args = parse()
    with opened(args) as session:
        held = {}
        if args.far:  # island 0's ranks and thread run; island 1's ranks hold the pages
            graph = session.graph
            cores, card = graph.count(0), session.sessions[0].card
            held = {"group": (graph.first(1), graph.count(1))}
        else:
            grid = session.grid
            cores, card = grid[0] * grid[1], session.card
        args.pull_units = args.pull_units or 64 * cores
        unit_kinds = kinds(args.units, args.run)
        bounds = stripes(args.units, args.per, cores)
        choice = host_split(unit_kinds, bounds)
        tiles = session.empty(
            (C["TS_WORK_PASSES"] * args.units, C["TS_TILE_VALUES"]), "uint16", **held
        )
        lines = session.empty((C["TS_PASSES"] * cores, C["TS_LINE"]), "uint32", **held)
        words = [
            args.units, args.per, args.run, args.penalty, tiles.address,
            lines.address, args.pull_units, *split_words(choice),
        ]  # fmt: skip
        made = program(args, words, tiles, lines)
        if args.far:  # island 1 without the thread: its cores get no command
            made = [made, program(args, words, tiles, lines, thread=False)]
        run = session.load(made)
        megahertz = aiclk_mhz(card)
        log(f"loaded, {cores} cores, aiclk {megahertz} MHz, far {args.far}")
        if args.far:
            result = started_passes(run.island(0))
        elif args.addon:  # a pull's answer names each as any overlay
            for name in args.addon.split(","):
                number = C[f"TS_OVERLAY_{name.upper()}"]
                run.addon(name, KERNELS / name, number=number)
            result = started_passes(run)
        else:
            try:
                thread = run.thread_results(timeout=180.0)
            except km.ThreadFailed as failure:
                thread = failure.results
            result = thread[CLUSTER]
        log(f"thread: {result.state}, code {result.code}, error {result.error}")
        run.stop()
        if args.far:
            raw = run.island(0).threads[CLUSTER].read("ticks").view(np.uint32)
            result.words = raw[: C["TS_PASSES"]].tolist()
        elif args.addon:
            raw = run.thread.read("ticks").view(np.uint32)
            result.words = raw[: C["TS_PASSES"]].tolist()
        (raw_lines,) = session.read(lines)
        (raw_tiles,) = session.read(tiles)
    got_lines = raw_lines.view(np.uint32).reshape(C["TS_PASSES"] * cores, -1)
    got_tiles = raw_tiles.view(np.uint16).reshape(-1, C["TS_TILE_VALUES"])
    ticks = list(result.words) + [0] * C["TS_PASSES"]
    wrong = check(args, cores, got_lines, got_tiles, bounds, choice)
    rows, penalized = table(
        args, cores, got_lines, ticks, megahertz, unit_kinds, choice
    )
    for name, row in rows.items():
        print(
            f"{name:6}: pass {row['pass_us']:9.1f} us, slowest core "
            f"{row['slowest_core_us']:9.1f} us, loads {row['loads']:5} "
            f"(max {row['loads_max_core']} a core), units {row['units']}, "
            f"next() {row['next_cycles_mean']:7.1f} cycles"
        )
    out = {
        "cores": cores,
        "far": args.far,
        "addon": args.addon,
        "units": args.units,
        "per": args.per,
        "run": args.run,
        "penalty": args.penalty,
        "pull_units": args.pull_units,
        "slots": args.slots,
        "aiclk_mhz": megahertz,
        "thread": {"state": result.state, "code": result.code, "error": result.error},
        "host_split": np.bincount(choice, minlength=4)[1:].tolist(),
        "penalized_units": penalized,
        "passes": rows,
        "wrong": wrong,
    }
    out["ok"] = bool(
        result.code == 0 and result.error == "none" and not any(wrong.values())
    )
    print(json.dumps(out), flush=True)
    if not out["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
