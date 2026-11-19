"""Code added to a live run (run.addon): an overlay (mix) named by the loaded RUN's
word, then a handler (SCALE) submitted as any handler, from the host and an X280
thread; commands on every core before, between and after, every word against numpy,
the times printed and one JSON line. See README.md."""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.bench import power
from komorebi.bench.power import aiclk_mhz

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
HEADER = KERNELS / "add.h"
CONSTANTS = km.abi.constants(HEADER, known=km.abi.headers().constants)
TILE_BYTES = 2048
THREAD_HANDLERS = {"relay": 1}  # kHandlerRelay in thread/relay.c
STARTED = time.monotonic()
M32 = 0xFFFFFFFF


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=0, help="0: the session's ranks")
    parser.add_argument("--rounds", type=int, default=8, help="commands per phase")
    parser.add_argument("--tiles", type=int, default=2, help="SCALE's tiles a core")
    parser.add_argument("--steps", type=int, default=64, help="mix's loop steps")
    parser.add_argument(
        "--reset-core", type=int, default=7, help="reset after the adds (-1: none)"
    )
    parser.add_argument("--json", help="also write the JSON line here")
    return parser.parse_args()


# --8<-- [start:program]
def program(lines, tiles):
    """The load: three kernels with one handler, RUN; slots for code added later (the
    reader's three as large as the service overlay's part, a writer slot and a compute
    slot of the bytes named); an X280 thread that relays commands."""
    tile = dict(format="bfloat16", page_bytes=TILE_BYTES, pages=2)
    made = km.Program(
        "add_overlay",
        kernels=KERNELS,
        handlers=HEADER,  # kHandlerRun = 1, the only one
        buffers=[
            km.Buffer(CONSTANTS["kCbIn"], **tile),
            km.Buffer(CONSTANTS["kCbOut"], **tile),
            km.Buffer(CONSTANTS["kCbLine"], "uint32", 64, 1),
        ],
        compile_args={"reader": lines.tensor_args(), "writer": tiles.tensor_args()},
    )
    made.slots(
        count={"reader": 4}, bytes={"reader": 2048, "writer": 2048, "compute": 4096}
    )
    made.thread(
        cluster=1,
        sources=[HERE / "thread" / "relay.c"],
        regions={"unused": 64},  # a thread has a region at least
        handlers=THREAD_HANDLERS,
    )
    return made


# --8<-- [end:program]


def mix(value, steps):
    """The mix overlay's loop on the host."""
    for step in range(steps):
        value = (value * 2654435761 + step) & M32
        value ^= value >> 13
    return value


def bf16(values):
    """bfloat16 bits of float32 values that bfloat16 holds exactly."""
    return (np.asarray(values, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def scaled(cores, tiles, scalar):
    """SCALE's output pages: value j of tile t on core c is (c + t + j) % 7 + 1 times
    the scalar, one page a tile."""
    core = np.arange(cores)[:, None, None]
    tile = np.arange(tiles)[None, :, None]
    index = np.arange(CONSTANTS["kTileValues"])[None, None, :]
    values = ((core + tile + index) % CONSTANTS["kSmallValues"] + 1) * scalar
    return bf16(values).reshape(cores * tiles, -1)


class Phases:
    """Submits the commands and checks every word; keeps each command's time."""

    def __init__(self, session, cores, args):
        self.session, self.cores, self.args, self.run = session, cores, args, None
        self.lines = session.empty((cores, 16), "uint32")
        self.tiles = session.empty((cores * args.tiles, TILE_BYTES // 2), "uint16")
        self.wrong, self.commands = 0, 0
        self.times = {}  # what -> [event microseconds] (or the thread's ticks / 50)
        self.cycles = {}  # "mix loading" / "mix held" -> [(load, call)] of the cores

    def note(self, what, microseconds):
        self.times.setdefault(what, []).append(float(microseconds))
        self.commands += 1

    def run_words(self, number, seed):
        return [number, seed, self.lines.address, self.args.steps]

    def check_lines(self, number, seed):
        (pages,) = self.session.read(self.lines)
        lines = pages.reshape(self.cores, -1).view(np.uint32).astype(np.int64)
        start = [(seed ^ core) & M32 for core in range(self.cores)]
        want = [mix(v, self.args.steps) for v in start] if number else start
        self.wrong += int((lines[:, 1] != want).sum() + (lines[:, 5] != number).sum())
        if number:
            for load, call, held in lines[:, 2:5]:
                key = f"mix {'held' if held else 'loading'}"
                self.cycles.setdefault(key, []).append((int(load), int(call)))

    def run_command(self, what, number, seed, thread=False):
        """RUN naming overlay `number` (0: none), from the host or the thread."""
        words = self.run_words(number, seed)
        if thread:
            ticks = self.relay(CONSTANTS["kHandlerRun"], words)
            self.note(what, ticks / 50.0)
        else:
            event = self.run.wait(self.run.submit("run", (0, self.cores), words), 30.0)
            if event.kind != "done":
                raise km.KomorebiError(f"{what}: {event}")
            self.note(what, event.microseconds)
        self.check_lines(number, seed)

    def scale_words(self, scalar):
        bits = int(np.float32(scalar).view(np.uint32))
        return [0, 0, 0, self.args.tiles, bits, 0, self.tiles.address]

    def check_tiles(self, scalar):
        (raw,) = self.session.read(self.tiles)
        got = raw.view(np.uint16).reshape(self.cores * self.args.tiles, -1)
        self.wrong += int((got != scaled(self.cores, self.args.tiles, scalar)).sum())

    def scale_command(self, what, handler, seed, thread=False):
        """The added handler SCALE on every core, from the host or the thread."""
        scalar = float(2 + seed % 3)
        words = self.scale_words(scalar)
        if thread:
            self.note(what, self.relay(handler, words) / 50.0)
        else:
            event = self.run.wait(
                self.run.submit(handler, (0, self.cores), words), 30.0
            )
            if event.kind != "done":
                raise km.KomorebiError(f"{what}: {event}")
            self.note(what, event.microseconds)
        self.check_tiles(scalar)

    def relay(self, handler, words):
        """The thread's RELAY of one command to cores [0, cores): its ticks."""
        relayed = [handler, self.cores, *words, *[0] * (8 - len(words))]
        event = self.run.wait(
            self.run.submit("relay", cluster=1, args=relayed, result=True), 30.0
        )
        if event.kind != "done" or event.result == M32:
            raise km.KomorebiError(f"relay of {handler}: {event}")
        return event.result


def median(values):
    return round(float(np.median(values)), 2) if values else None


def main():
    args = parse()
    with km.Session(log=log) as session:
        cores = args.cores or session.ranks  # the grid less a Tensix dispatcher's
        log(f"device open; {cores} cores")
        phases = Phases(session, cores, args)
        run = phases.run = session.load(program(phases.lines, phases.tiles))
        mhz = [aiclk_mhz(session.card)]
        try:
            # --8<-- [start:add]
            for seed in range(args.rounds):  # before: the load's code only
                phases.run_command("run before", 0, seed)
            mix_number = run.addon("mix", KERNELS / "mix")  # an overlay, a reader part
            for seed in range(args.rounds):  # between: RUN names it by its number
                phases.run_command("run with mix", mix_number, seed)
            scale = run.addon("scale", KERNELS / "scale", handler=True)  # three parts
            for seed in range(args.rounds):  # after: submitted as any handler
                phases.scale_command("scale", scale, seed)
            # --8<-- [end:add]
            again = run.addon("mix again", KERNELS / "mix")  # the same code: cached
            reset = None
            if args.reset_core >= 0:  # its reset list writes the area with the adds
                reset = run.reset_core(args.reset_core)
                log(f"core {args.reset_core} reset by the {reset['by']}")
            for seed in range(args.rounds):
                phases.run_command("run with mix again", again, seed)
                phases.run_command("run after", 0, seed)
                phases.run_command(
                    "run with mix from the thread", mix_number, seed, True
                )
                phases.scale_command("scale from the thread", scale, seed, True)
            log(f"{phases.commands} commands, {phases.wrong} wrong")
            mhz.append(aiclk_mhz(session.card))
            added = {name: dataclasses.asdict(each) for name, each in run.added.items()}
        finally:
            run.stop()
    report(args, cores, phases, added, mhz, reset)


def report(args, cores, phases, added, mhz, reset=None):
    """Prints each command's first and later times, the adds' times and the JSON
    line (also to --json); exits 1 when a word was wrong."""
    first = {what: values[0] for what, values in phases.times.items()}
    later = {what: median(values[1:]) for what, values in phases.times.items()}
    for what in phases.times:
        print(
            f"{what:32}: first {first[what]:8.2f} us, later (median) "
            f"{later[what] or 0:8.2f} us, {len(phases.times[what])} commands"
        )
    cycles = {
        key: dict(zip(("load", "call"), np.median(np.asarray(pairs), axis=0).tolist()))
        for key, pairs in sorted(phases.cycles.items())
    }
    for name, each in added.items():
        print(
            f"add {name!r}: number {each['number']}, build {each['build_s'] * 1e3:.1f} "
            f"ms{' (cached)' if each['cached'] else ''}, write "
            f"{each['write_s'] * 1e3:.1f} ms, entries {each['entry_s'] * 1e3:.1f} ms"
        )
    print(f"every word exact: {phases.wrong == 0} ({phases.commands} commands)")
    out = {
        "cores": cores,
        "rounds": args.rounds,
        "tiles": args.tiles,
        "steps": args.steps,
        "commands": phases.commands,
        "wrong": phases.wrong,
        "exact": phases.wrong == 0,
        "aiclk_mhz": mhz,
        "added": added,
        "reset": reset,
        "first_us": first,
        "later_us": later,
        "mix_cycles": cycles,
    }
    line = json.dumps(power.clocked(out), default=float)
    print(line, flush=True)
    if args.json:
        Path(args.json).write_text(line + "\n")
    if phases.wrong:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
