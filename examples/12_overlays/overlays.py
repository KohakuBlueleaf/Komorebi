"""Two overlays loaded and called in turn on every core: mix (a reader part) and scale
(reader, compute and writer parts). Every result is checked; the cycles of load() and
call() are printed, loading and held. --far: two cards, island 0's cores, the lines and
tiles held by island 1. One JSON line. Example: python overlays.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import power
from komorebi.session import aiclk_mhz

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
HEADER = KERNELS / "overlays.h"
CONSTANTS = km.abi.constants(HEADER, known=km.abi.headers().constants)
TILE_BYTES = 2048
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=0, help="0: the whole grid")
    parser.add_argument("--rounds", type=int, default=10, help="commands per phase")
    parser.add_argument("--tiles", type=int, default=1, help="scale's tiles a core")
    parser.add_argument("--steps", type=int, default=64, help="mix's loop steps")
    parser.add_argument("--slots", type=int, default=1, help="slots of each kernel")
    parser.add_argument(
        "--far",
        action="store_true",
        help="cards 0 and 1: island 0's cores write lines and tiles island 1 holds",
    )
    return parser.parse_args()


# --8<-- [start:program]
def program(slots, lines, tiles):
    """Three kernels of the kernel form; two overlays, `slots` slots each kernel; the
    reader writes `lines`, scale's writer part `tiles` (their tensor arguments)."""
    tile = dict(format="bfloat16", page_bytes=TILE_BYTES, pages=2)
    made = km.Program(
        "overlays",
        kernels=KERNELS,
        handlers=HEADER,  # kHandlerRun; kOverlayMix = 1, kOverlayScale = 2
        buffers=[
            km.Buffer(CONSTANTS["kCbIn"], **tile),
            km.Buffer(CONSTANTS["kCbOut"], **tile),
            km.Buffer(CONSTANTS["kCbLine"], "uint32", 64, 1),
        ],
        compile_args={"reader": lines.tensor_args(), "writer": tiles.tensor_args()},
    )
    made.overlay("mix", KERNELS / "mix")  # mix/reader.cpp
    made.overlay("scale", KERNELS / "scale")  # scale/: reader, compute, writer.cpp
    return made.slots(count=slots)


# --8<-- [end:program]


def mix(value, steps):
    """The mix overlay's loop on the host."""
    for step in range(steps):
        value = (value * 2654435761 + step) & 0xFFFFFFFF
        value ^= value >> 13
    return value


def bf16(values):
    """bfloat16 bits of float32 values that bfloat16 holds exactly."""
    return (np.asarray(values, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def scaled(cores, tiles, scalar):
    """Scale's output pages: value j of tile t on core c is (c + t + j) % 7 + 1 times
    the scalar, one page a tile."""
    core = np.arange(cores)[:, None, None]
    tile = np.arange(tiles)[None, :, None]
    index = np.arange(CONSTANTS["kTileValues"])[None, None, :]
    values = ((core + tile + index) % CONSTANTS["kSmallValues"] + 1) * scalar
    return bf16(values).reshape(cores * tiles, -1)


class Rounds:
    """Submits RUN for one overlay to every core and checks every result."""

    def __init__(self, session, cores, args, held):
        self.session, self.run, self.cores, self.args = session, None, cores, args
        self.lines = session.empty((cores, 16), "uint32", **held)
        self.tiles = session.empty(
            (cores * args.tiles, TILE_BYTES // 2), "uint16", **held
        )
        self.times = {}  # (overlay, loading or held) -> [(load, call)]
        self.wrong = 0
        self.commands = 0

    def once(self, name, seed):
        args, number = self.args, CONSTANTS[f"kOverlay{name.capitalize()}"]
        scalar = float(2 + seed % 3)
        words = [number, seed, self.lines.address, args.tiles]
        words += [
            int(np.float32(scalar).view(np.uint32)),
            args.steps,
            self.tiles.address,
        ]
        event = self.run.wait(self.run.submit("run", (0, self.cores), words), 30.0)
        if event.kind != "done":
            raise km.KomorebiError(f"{name}: {event}")
        (pages,) = self.session.read(self.lines)
        lines = pages.reshape(self.cores, -1).view(np.uint32).astype(np.int64)
        if name == "mix":
            want = [mix(seed ^ core, args.steps) for core in range(self.cores)]
        else:
            want = [args.tiles] * self.cores
            (raw,) = self.session.read(self.tiles)
            got = raw.view(np.uint16).reshape(self.cores * args.tiles, -1)
            self.wrong += int((got != scaled(self.cores, args.tiles, scalar)).sum())
        self.wrong += int((lines[:, 1] != want).sum() + (lines[:, 5] != number).sum())
        self.commands += 1
        for load, call, held in lines[:, 2:5]:
            key = (name, "held" if held else "loading")
            self.times.setdefault(key, []).append((int(load), int(call)))


def opened(args):
    """The session: one card, or (--far) cards 0 and 1 with a link."""
    if args.far:
        return km.Session(cards=[0, 1], links=1, log=log)
    return km.Session(log=log)


def main():
    args = parse()
    with opened(args) as session:
        held, card = {}, None
        if args.far:  # island 0's ranks run; island 1's ranks hold the pages
            graph = session.graph
            cores = args.cores or graph.count(0)
            held = {"group": (graph.first(1), graph.count(1))}
            card = session.sessions[0].card
        else:
            grid = session.grid
            cores = args.cores or grid[0] * grid[1]
            card = session.card
        log(
            f"device open; {cores} cores, {args.slots} slot(s) a kernel, far {args.far}"
        )
        rounds = Rounds(session, cores, args, held)
        run = session.load(program(args.slots, rounds.lines, rounds.tiles))
        rounds.run = run.island(0) if args.far else run
        built = rounds.run.program.overlays_built
        sizes = {f"{n}-{p}": len(i) for (n, p), i in sorted(built.images.items())}
        mhz = [aiclk_mhz(card)]
        try:
            # --8<-- [start:turns]
            for seed in range(args.rounds):  # in turn: each load replaces the other
                rounds.once("mix", seed)
                rounds.once("scale", seed)
            for name in ("mix", "scale"):  # again and again: held after the first
                for seed in range(args.rounds):
                    rounds.once(name, seed)
            # --8<-- [end:turns]
            log(f"{rounds.commands} commands, {rounds.wrong} wrong")
            mhz.append(aiclk_mhz(card))
        finally:
            run.stop()
    table = {}
    for (name, how), pairs in sorted(rounds.times.items()):
        load, call = np.median(np.asarray(pairs), axis=0)
        table[f"{name} {how}"] = {"load_cycles": load, "call_cycles": call}
        print(
            f"{name:5} {how:7}: load {load:7.0f} cycles ({load / mhz[0]:5.2f} us), "
            f"call {call:7.0f} cycles ({call / mhz[0]:5.2f} us), {len(pairs)} runs"
        )
    print(f"every result exact: {rounds.wrong == 0} ({rounds.commands} commands)")
    out = {
        "cores": cores,
        "far": args.far,
        "slots": args.slots,
        "tiles": args.tiles,
        "steps": args.steps,
        "commands": rounds.commands,
        "wrong": rounds.wrong,
        "exact": rounds.wrong == 0,
        "aiclk_mhz": mhz,
        "image_bytes": sizes,
        "slots_bytes": {part: list(value) for part, value in built.slots.items()},
        "medians": table,
    }
    print(json.dumps(power.clocked(out), default=float), flush=True)
    if rounds.wrong:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
