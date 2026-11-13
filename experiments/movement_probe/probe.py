"""Both data movers of a core on one stream: reads, writes and copies through the
movement engine (a pattern per stream) against the classic reader / writer layout.
One JSON line per case, checked against numpy before it is timed. Example:
python probe.py read --cores 1 --blocks 8 32 --layouts classic A AB"""

import argparse
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import abi

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
PROBE_H = KERNELS / "probe.h"
CONSTANTS = abi.constants(PROBE_H, known=abi.headers().constants)
PROGRAMS = {"read": 0, "write": 1, "copy": 2}
TILE_BYTES = CONSTANTS["kTileBytes"]
WORDS = TILE_BYTES // 4
TILES_PER_CORE = 256
TARGET_CYCLES = 2_700_000  # the shorter launch of a pair: about 2 ms
T0 = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - T0:7.2f}] {message}", flush=True)


class Clock:
    """AICLK (MHz) and card power (W) sampled every 50 ms while it runs."""

    def __init__(self):
        card = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
        base = Path(f"/sys/class/tenstorrent/tenstorrent!{card}")
        self.aiclk = base / "tt_aiclk"
        hwmon = next(iter(sorted((base / "device" / "hwmon").glob("hwmon*"))))
        self.power = hwmon / "power1_input"
        self.samples, self.stop = [], threading.Event()

    def read(self):
        mhz = int(self.aiclk.read_text().split()[0])
        return mhz, int(self.power.read_text()) / 1e6

    def run(self):
        while not self.stop.wait(0.05):
            self.samples.append(self.read())

    def __enter__(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()

    def summary(self):
        mhz = np.array([s[0] for s in self.samples] or [self.read()[0]])
        watts = np.array([s[1] for s in self.samples] or [self.read()[1]])
        return {
            "aiclk_mhz_min": int(mhz.min()),
            "aiclk_mhz_mean": round(float(mhz.mean()), 1),
            "power_w_mean": round(float(watts.mean()), 1),
            "samples": len(self.samples),
        }


def pattern_word(pattern):
    """The pattern's word: its length, then a bit per position owned by mover B."""
    if not 1 <= len(pattern) <= 24 or set(pattern) - {"A", "B"}:
        raise ValueError(f"pattern {pattern!r}: 1 to 24 letters A and B")
    owners = sum(1 << i for i, letter in enumerate(pattern) if letter == "B")
    return len(pattern) | owners << 8


def owners_of(pattern, blocks):
    """The mover (0: A, 1: B) owning each of `blocks` blocks."""
    return np.array([pattern[k % len(pattern)] == "B" for k in range(blocks)], int)


def buffers_of(program):
    """The data buffers a program's load declares: a buffer keeps its filling and
    draining CPUs for the whole load, so copies have halves of their own."""
    if program == "copy":
        return ["kBufferCopyA", "kBufferCopyB", "kBufferClassicCopy"]
    return ["kBufferA", "kBufferB"]


def ring_pages(block, free_bytes, buffers):
    """Pages of each buffer: four blocks, two when four do not fit beside the rest."""
    for count in (4, 2):
        if buffers * count * block * TILE_BYTES + (64 << 10) <= free_bytes:
            return count * block
    raise ValueError(f"blocks of {block} tiles do not fit {free_bytes} bytes of L1")


def program_of(arguments, ring, program):
    buffers = [
        km.Buffer(CONSTANTS[name], "uint32", TILE_BYTES, ring)
        for name in buffers_of(program)
    ]
    buffers.append(
        km.Buffer(CONSTANTS["kBufferProfile"], "uint32", CONSTANTS["kProfileBytes"], 1)
    )
    movement = KERNELS / "movement.cpp"
    return km.Program(
        "movement_probe",
        kernels={
            "reader": movement,
            "writer": movement,
            "compute": KERNELS / "compute.cpp",
        },
        buffers=buffers,
        compile_args={"reader": arguments, "writer": arguments},
        handlers=PROBE_H,
    )


def expected_stamps(pattern, block, ring, salt):
    """The words the write program stores at a core's pages, [tiles, words]."""
    owners = owners_of(pattern, TILES_PER_CORE // block)
    position = np.arange(TILES_PER_CORE)
    owner = owners[position // block]
    earlier = np.array([(owners[:k] == owners[k]).sum() for k in range(len(owners))])
    slot = (earlier[position // block] * block + position % block) % ring
    head = (salt << 20) | (owner << 19) | (slot << 9)
    return (head[:, None] | np.arange(WORDS)[None, :]).astype(np.uint32)


def sums(pages):
    """The compute kernel's two sums over [tiles, words] pages (mod 2^32)."""
    tile_sums = pages.astype(np.uint64).sum(axis=1) & 0xFFFFFFFF
    weights = np.arange(1, len(tile_sums) + 1, dtype=np.uint64)
    return int(tile_sums.sum() & 0xFFFFFFFF), int(
        (tile_sums * weights).sum() & 0xFFFFFFFF
    )


def describe(page, core, position, data):
    """What a wrong page holds: an input page (which), a stamp (its fields), or
    neither."""
    found = np.flatnonzero((data[:, 0] == page[0]) & (data[:, 1] == page[1]))
    out = {"core": core, "position": int(position)}
    if len(found):
        return out | {"input_page": int(found[0])}
    head = int(page[0])
    return out | {
        "word0": head,
        "stamp": [head >> 20, (head >> 19) & 1, (head >> 9) & 1023, head & 511],
    }


class Probe:
    """One load of the probe program (one block size) on the session's grid."""

    def __init__(self, session, tensors, block, ring, program):
        self.session, self.t, self.block, self.ring = session, tensors, block, ring
        self.run = session.load(
            program_of(tensors["input"].accessor_args(), ring, program),
            deadline=2.0,
            wait=1.0,
        )

    def command(
        self, program, layout, pattern, cores, repetitions, salt=0, input_tensor="input"
    ):
        words = [
            self.t[input_tensor].address,
            self.t["output"].address,
            TILES_PER_CORE,
            self.block,
            pattern_word(pattern),
            repetitions,
            PROGRAMS[program],
            CONSTANTS["kLayoutClassic" if layout == "classic" else "kLayoutEngine"],
            salt,
        ]
        handle = self.run.submit("run", (0, cores), words)
        event = self.run.wait(handle, timeout=30.0)
        if event.kind != "done":
            raise km.KomorebiError(f"RUN {program} {layout}: {event}")
        return event.microseconds

    def profile(self, cores):
        """[cores, words] of the profile pages after the last RUN."""
        handle = self.run.submit("profile", (0, cores), [self.t["profile"].address])
        self.run.wait(handle, timeout=10.0)
        (pages,) = self.session.read(self.t["profile"], pages=cores)
        words = pages.reshape(cores, -1).view(np.uint32)
        return words[:, : CONSTANTS["kProfileWords"]].astype(np.int64)

    def span(self, words, program):
        """Cycles from the first mover's start to the last end; busy shares by mover."""
        ref = words[0, CONSTANTS["kProfileMoverA"]]
        starts, ends, out = [], [], {}
        for name in ("A", "B"):
            at = CONSTANTS[f"kProfileMover{name}"]
            start = (words[:, at] - ref + 2**31) % 2**32 - 2**31
            end = (words[:, at + 1] - ref + 2**31) % 2**32 - 2**31
            starts.append(start)
            ends.append(end)
            out[f"tiles_{name}"] = int(words[:, at + 2].sum())
            out[f"busy_cycles_{name}"] = words[:, at + 3]
        span = int(max(e.max() for e in ends) - min(s.min() for s in starts))
        for name in ("A", "B"):
            busy = out.pop(f"busy_cycles_{name}")
            out[f"busy_{name}"] = round(float(busy.max()) / span, 3)
        return span, out

    def read_core(self, tensor, core):
        (raw,) = self.session.read(
            tensor, first=core * TILES_PER_CORE, pages=TILES_PER_CORE
        )
        return raw.reshape(TILES_PER_CORE, -1).view(np.uint32)

    def check(self, case, cores, data, salt):
        """Runs the case once (one repetition) and compares it with numpy."""
        program, layout, pattern = case["program"], case["layout"], case["pattern"]
        if program == "copy":  # a decoy first, so a page the copy missed shows
            self.command("copy", "classic", "A", cores, 1, input_tensor="decoy")
        self.command(program, layout, pattern, cores, 1, salt)
        if program == "read":
            words = self.profile(cores)
            bad = 0
            for core in range(cores):
                pages = data[core * TILES_PER_CORE : (core + 1) * TILES_PER_CORE]
                got = (
                    int(words[core, CONSTANTS["kProfileSum"]]),
                    int(words[core, CONSTANTS["kProfileWeighted"]]),
                )
                bad += got != sums(pages)
            return {"cores_checked": cores, "cores_wrong": bad}
        sample = sorted({0, cores // 2, cores - 1})
        bad, seen = 0, []
        for core in sample:
            got = self.read_core(self.t["output"], core)
            if program == "copy":
                want = data[core * TILES_PER_CORE : (core + 1) * TILES_PER_CORE]
            else:
                want = expected_stamps(pattern, self.block, self.ring, salt)
            wrong = np.flatnonzero((got != want).any(axis=1))
            bad += len(wrong)
            seen.extend(
                describe(got[position], core, position, data)
                for position in wrong[: 8 - len(seen)]
            )
            log(f"  core {core} read back")
        return {"cores_checked": len(sample), "pages_wrong": bad, "first_wrong": seen}

    def time(self, case, cores, mhz):
        """Rate from the difference of R and 4R repetitions, best of three."""
        program, layout, pattern = case["program"], case["layout"], case["pattern"]
        self.command(program, layout, pattern, cores, 2)
        per_repetition, _ = self.span(self.profile(cores), program)
        repetitions = max(2, math.ceil(TARGET_CYCLES / (per_repetition / 2)))
        trials = []
        for _ in range(3):
            row = {}
            for label, count in (("short", repetitions), ("long", 4 * repetitions)):
                row[f"host_microseconds_{label}"] = round(
                    self.command(program, layout, pattern, cores, count), 1
                )
                row[f"cycles_{label}"], shares = self.span(self.profile(cores), program)
            row |= shares
            trials.append(row)
        bytes_per_repetition = cores * TILES_PER_CORE * TILE_BYTES
        best = min(t["cycles_long"] - t["cycles_short"] for t in trials)
        seconds = best / (mhz * 1e6)
        return {
            "repetitions": [repetitions, 4 * repetitions],
            "rate_gbs": round(
                bytes_per_repetition * 3 * repetitions / seconds / 1e9, 2
            ),
            "trials": trials,
        }

    def stop(self):
        self.run.stop()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("program", choices=list(PROGRAMS))
    p.add_argument("--cores", type=int, nargs="+", default=[1])
    p.add_argument("--blocks", type=int, nargs="+", default=[32])
    p.add_argument("--layouts", nargs="+", default=["classic", "A", "AB"])
    p.add_argument("--out", type=Path, help="append the JSON lines to this file")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    with km.Session() as session:
        grid = session.grid
        total = grid[0] * grid[1]
        cores_list = [total if c == 0 else c for c in args.cores]
        pages = max(cores_list) * TILES_PER_CORE
        log(f"device open, grid {grid}; building {pages} pages of input")
        rng = np.random.default_rng(args.seed)
        data = rng.integers(0, 2**32, size=(pages, WORDS), dtype=np.uint32)
        tensors = {
            "input": session.tensor(data, "uint32"),
            "output": session.tensor(np.zeros_like(data), "uint32"),
            "decoy": session.tensor(
                data[:1] if args.program != "copy" else data[::-1].copy(), "uint32"
            ),
            "profile": session.tensor(np.zeros((total, 256), np.uint32), "uint32"),
        }
        log("tensors written")
        _, free = session.l1()
        nocs = session.nocs()
        row = [nocs.noc0[(x, 0)] for x in range(min(8, grid[0]))]
        salt = 1
        for block in args.blocks:
            ring = ring_pages(block, free, len(buffers_of(args.program)))
            log(f"load: blocks of {block}, {ring} pages per half")
            probe = Probe(session, tensors, block, ring, args.program)
            try:
                with Clock() as clock:
                    mhz = clock.read()[0]
                    for cores in cores_list:
                        for layout in args.layouts:
                            pattern = layout
                            if layout == "classic":
                                pattern = "B" if args.program == "write" else "A"
                            case = {
                                "program": args.program,
                                "layout": (
                                    "classic" if layout == "classic" else "engine"
                                ),
                                "pattern": pattern,
                                "block": block,
                                "cores": cores,
                            }
                            log(f"case {case}")
                            salt = salt % 4095 + 1
                            checked = probe.check(case, cores, data, salt)
                            exact = not (
                                checked.get("cores_wrong") or checked.get("pages_wrong")
                            )
                            line = case | {"exact": exact, "check": checked}
                            if exact:
                                line |= probe.time(case, cores, mhz)
                            line |= {
                                "mhz_used": mhz,
                                "mhz_watts_now": list(clock.read()),
                                "ring_pages": ring,
                                "row_noc0": row,
                                "loadavg": Path("/proc/loadavg")
                                .read_text()
                                .split()[:3],
                            }
                            print(json.dumps(line), flush=True)
                            if args.out:
                                with args.out.open("a") as f:
                                    f.write(json.dumps(line) + "\n")
                    log(f"clock {clock.summary()}")
            finally:
                probe.stop()
                log("run stopped")


if __name__ == "__main__":
    main()
