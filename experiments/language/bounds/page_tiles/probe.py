"""W read in DRAM pages of k tiles: each core reads gate's share (336 tiles of a
bfp8_b or bfp4_b size) from a uint8 row tensor whose page is k tiles' bytes, one NoC
read a page, the tiles pushed to compute, which frees them. One JSON line a case.
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
BUFFERS = {"bfp8_b": "kBufferBfp8", "bfp4_b": "kBufferBfp4"}
TILES = 336  # a core's share of gate's W at 120 cores (42 rows x 8 columns)
BLOCK = 16  # tiles a block, the decode class's CHUNK x W8
RING = 4 * BLOCK
PAGE_CYCLES, BYTE_CYCLES = 156.0, 0.22  # the mover's measured line
T0 = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - T0:7.2f}] {message}", flush=True)


def aiclk():
    """The card's AICLK in MHz."""
    card = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
    path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
    return int(path.read_text().split()[0])


def program_of():
    buffers = [
        km.Buffer(CONSTANTS[name], "uint32", formats.tile_bytes(dtype), RING)
        for dtype, name in BUFFERS.items()
    ]
    buffers.append(
        km.Buffer(CONSTANTS["kBufferProfile"], "uint32", CONSTANTS["kProfileBytes"], 1)
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


def case(session, run, profile, cores, dtype, k, repetitions, mhz):
    """Three READs of one case; the best span's rate and the median core's cycles."""
    tile = formats.tile_bytes(dtype)
    pages = TILES // k
    tensor = session.empty((cores * pages, k * tile), "uint8")
    words = [
        tensor.address,
        k * tile,
        pages,
        BLOCK // k,
        CONSTANTS[BUFFERS[dtype]],
        repetitions,
        profile.address,
        BLOCK,
    ]
    trials = []
    for _ in range(3):
        event = run.wait(run.submit("read", (0, cores), words), timeout=30.0)
        if mock.active():  # a prebuild: the load is built, nothing to time
            tensor.free()
            return {"dtype": dtype, "k": k, "mock": True}
        if event.kind != "done":
            raise km.KomorebiError(f"READ {dtype} k={k}: {event}")
        (raw,) = session.read(profile, pages=cores)
        rows = raw.view(np.uint32).reshape(cores, -1).astype(np.int64)
        start, end = (
            rows[:, CONSTANTS["kProfileStart"]],
            rows[:, CONSTANTS["kProfileEnd"]],
        )
        spans = (end - start) % 2**32
        late = (end - start[0] + 2**31) % 2**32 - 2**31
        early = (start - start[0] + 2**31) % 2**32 - 2**31
        if not (rows[:, CONSTANTS["kProfilePages"]] == pages * repetitions).all():
            raise km.KomorebiError(f"READ {dtype} k={k}: pages {rows[:, 2].tolist()}")
        trials.append(
            {
                "span": int(late.max() - early.min()),
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
    p.add_argument("--formats", nargs="+", default=list(BUFFERS))
    p.add_argument("--cores", type=int, default=120)
    p.add_argument("--repetitions", type=int, default=8)
    args = p.parse_args()
    with km.Session() as session:
        grid = session.grid
        cores = min(args.cores, grid[0] * grid[1])
        profile = session.tensor(np.zeros((cores, 16), np.uint32), "uint32")
        run = session.load(program_of(), deadline=2.0, wait=1.0)
        log(f"loaded on {grid}, {cores} cores")
        try:
            for dtype in args.formats:
                for k in args.pages:
                    if BLOCK % k or TILES % k:
                        raise SystemExit(f"k={k}: not a divisor of {BLOCK} and {TILES}")
                    row = case(
                        session,
                        run,
                        profile,
                        cores,
                        dtype,
                        k,
                        args.repetitions,
                        aiclk(),
                    )
                    print(json.dumps(row), flush=True)
        finally:
            result = run.stop()
            log(f"stopped: {result['state_name']}, error {result['error_name']}")


if __name__ == "__main__":
    main()
