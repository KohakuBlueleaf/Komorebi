"""W read in DRAM pages of k tiles: each core reads gate's share (336 tiles) from a
uint8 row tensor whose page is k tiles' bytes, one NoC read a page, on mover A or on
both movers by a pattern; compute frees the tiles. One JSON line a case.
python experiments/language/bounds/page_tiles/probe.py --pages 1 2 4 8"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import abi, formats
from komorebi.device import mock

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
PAGES_H = KERNELS / "pages.h"
CONSTANTS = abi.constants(PAGES_H, known=abi.headers().constants)
FORMATS = {"bfp8_b": 0, "bfp4_b": 1}
HALVES = {
    "bfp8_b": ("kBufferBfp8A", "kBufferBfp8B"),
    "bfp4_b": ("kBufferBfp4A", "kBufferBfp4B"),
}
TILES = 336  # a core's share of gate's W at 120 cores (42 rows x 8 columns)
PAGE_CYCLES, BYTE_CYCLES = 156.0, 0.22  # the mover's measured line
T0 = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - T0:7.2f}] {message}", flush=True)


def aiclk():
    """The card's AICLK in MHz."""
    card = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
    path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
    return int(path.read_text().split()[0])


def program_of(block):
    """The probe's load: each half four blocks of `block` tiles."""
    buffers = [
        km.Buffer(CONSTANTS[name], "uint32", formats.tile_bytes(dtype), 4 * block)
        for dtype, names in HALVES.items()
        for name in names
    ]
    buffers.append(
        km.Buffer(
            CONSTANTS["kBufferProfile"], "uint32", 2 * CONSTANTS["kProfileBytes"], 1
        )
    )
    movement = KERNELS / "movement.cpp"
    return km.Program(
        "page_tiles",
        kernels={
            "reader": movement,
            "writer": movement,
            "compute": KERNELS / "compute.cpp",
        },
        buffers=buffers,
        handlers=PAGES_H,
    )


def line(tile, k, cores, mhz):
    """The line's cycles a tile and the card's GB/s over `cores`."""
    cycles = PAGE_CYCLES / k + BYTE_CYCLES * tile
    return cycles, cores * tile * mhz * 1e6 / cycles / 1e9


def pattern_word(pattern):
    """The command's pattern: bit i set, block i (mod its length) is mover B's."""
    if not 1 <= len(pattern) <= 8 or set(pattern) - set("AB"):
        raise SystemExit(f"pattern {pattern!r}: 1 to 8 letters A or B")
    bits = sum(1 << i for i, letter in enumerate(pattern) if letter == "B")
    return len(pattern) << 8 | bits


def case(session, run, profile, cores, dtype, k, block, pattern, repetitions, mhz):
    """Three READs of one case; the best span's rate and the median core's cycles."""
    tile = formats.tile_bytes(dtype)
    pages = TILES // k
    tensor = session.empty((cores * pages, k * tile), "uint8")
    words = [
        tensor.address,
        k * tile,
        pages,
        block // k,
        FORMATS[dtype],
        repetitions,
        profile.address,
        block,
        pattern_word(pattern),
    ]
    trials = []
    for _ in range(3):
        event = run.wait(run.submit("read", (0, cores), words), timeout=30.0)
        if mock.active():  # a prebuild: the load is built, nothing to time
            tensor.free()
            return {"dtype": dtype, "k": k, "pattern": pattern, "mock": True}
        if event.kind != "done":
            raise km.KomorebiError(f"READ {dtype} k={k}: {event}")
        (raw,) = session.read(profile, pages=2 * cores)
        rows = raw.view(np.uint32).reshape(cores, 2, -1).astype(np.int64)
        read = rows[:, :, CONSTANTS["kProfilePages"]]
        if not (read.sum(axis=1) == pages * repetitions).all():
            raise km.KomorebiError(f"READ {dtype} k={k}: pages {read.tolist()}")
        movers = read > 0  # a mover with no block of the pattern is not timed
        start = rows[:, :, CONSTANTS["kProfileStart"]]
        end = rows[:, :, CONSTANTS["kProfileEnd"]]
        ref = start[0, 0]
        early = np.where(movers, (start - ref + 2**31) % 2**32 - 2**31, 2**62)
        late = np.where(movers, (end - ref + 2**31) % 2**32 - 2**31, -(2**62))
        spans = late.max(axis=1) - early.min(axis=1)
        trials.append(
            {
                "span": int(late.max() - early.min()),
                "b_share": round(float(read[:, 1].sum() / read.sum()), 3),
                "core_median": int(np.median(spans)),
                "core_max": int(spans.max()),
                "us": round(event.microseconds, 1),
            }
        )
    tensor.free()
    best = min(trials, key=lambda t: t["span"])
    total = cores * TILES * tile * repetitions
    tiles = TILES * repetitions
    cycles, line_gbs = line(tile, k, cores, mhz)
    return {
        "dtype": dtype,
        "k": k,
        "block": block,
        "pattern": pattern,
        "page_bytes": k * tile,
        "cores": cores,
        "repetitions": repetitions,
        "gbs": round(total / (best["span"] / (mhz * 1e6)) / 1e9, 1),
        "line_gbs": round(line_gbs, 1),
        "cycles_a_tile_median": round(best["core_median"] / tiles, 1),
        "cycles_a_tile_slowest": round(best["core_max"] / tiles, 1),
        "line_cycles_a_tile": round(cycles, 1),
        "mhz": mhz,
        "trials": trials,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pages", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--formats", nargs="+", default=list(FORMATS))
    p.add_argument("--block", type=int, default=16, help="tiles a block")
    p.add_argument("--patterns", nargs="+", default=["A"], help="e.g. A AAAB")
    p.add_argument("--cores", type=int, default=120)
    p.add_argument("--repetitions", type=int, default=8)
    args = p.parse_args()
    with km.Session() as session:
        grid = session.grid
        cores = min(args.cores, grid[0] * grid[1])
        profile = session.tensor(np.zeros((2 * cores, 16), np.uint32), "uint32")
        run = session.load(program_of(args.block), deadline=2.0, wait=1.0)
        log(f"loaded on {grid}, {cores} cores")
        try:
            for pattern in args.patterns:
                for dtype in args.formats:
                    for k in args.pages:
                        if args.block % k or TILES % args.block:
                            raise SystemExit(
                                f"k={k}, block {args.block}: k must divide the "
                                f"block and the block {TILES}"
                            )
                        row = case(
                            session,
                            run,
                            profile,
                            cores,
                            dtype,
                            k,
                            args.block,
                            pattern,
                            args.repetitions,
                            aiclk(),
                        )
                        print(json.dumps(row), flush=True)
        finally:
            result = run.stop()
            log(f"stopped: {result['state_name']}, error {result['error_name']}")


if __name__ == "__main__":
    main()
