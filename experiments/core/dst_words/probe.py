"""Dst words through km::dst, bit for bit: random words and special patterns into Dst
and back in each 32-bit format, the vector unit's reordering against the window's
order, and a tile's spans each way; a JSON line a format (README.md).
Example: python probe.py; python probe.py --formats float32 --save out.npz"""

import argparse
import json
import secrets
import sys
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.device import chips, mock
from komorebi.program import several
from komorebi.session import kernels as session_kernels
from komorebi.tools.feeder import commands as rec

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
CB_IN, CB_OUT, CB_INFO = 0, 16, 24  # layout.hpp
NONCE, TILES, WRITE, READ = 0, 1, 16, 64
TILE_WORDS = 1024
RANDOM_TILES = 4


def special_tile():
    """1,024 words: NaN payloads, infinities, subnormals, both zeros, all ones, the
    extremes of each exponent, then every exponent with random signs and mantissas."""
    rng = np.random.default_rng(7)
    named = [
        0x7F800001,
        0x7FBFFFFF,
        0x7FC00000,
        0x7FC00001,
        0x7FFFFFFF,
        0x7FA5A5A5,
        0xFF800001,
        0xFFBFFFFF,
        0xFFC00000,
        0xFFFFFFFF,
        0xFFD2B4C6,
        0x7F800000,
        0xFF800000,
        0x00000001,
        0x00000002,
        0x00400000,
        0x007FFFFF,
        0x00012345,
        0x80000001,
        0x80400000,
        0x807FFFFF,
        0x00000000,
        0x80000000,
        0x00800000,
        0x80800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x3F800000,
        0xBF800000,
        0x0000FFFF,
        0xFFFF0000,
        0x55555555,
        0xAAAAAAAA,
        0x0F0F0F0F,
        0xF0F0F0F0,
    ]
    words = list(named)
    for exponent in range(256):  # each exponent: mantissa 0, all ones, two random
        for sign in (0, 1):
            head = sign << 31 | exponent << 23
            words += [head, head | 0x7FFFFF]
    while len(words) < TILE_WORDS:
        words.append(int(rng.integers(0, 1 << 32, dtype=np.uint64)))
    return np.array(words[:TILE_WORDS], np.uint32)


def inputs(seed):
    rng = np.random.default_rng(seed)
    random = rng.integers(0, 1 << 32, (RANDOM_TILES, TILE_WORDS), dtype=np.uint64)
    return np.concatenate([random.astype(np.uint32), special_tile()[None]], 0)


def reordered(tile):
    """Tile 1 as the vector unit leaves it in the window's order: Dst word w is row
    w // 16, column w % 16; vector v holds rows 4 (v // 2) + 0..3, the even (v even)
    or odd columns; vector v of tile 0 goes to vector 5 v + 3 (mod 32) of tile 1."""
    rows = tile.reshape(64, 16)
    out = np.zeros_like(rows)
    for v in range(32):
        target = (5 * v + 3) % 32
        source_rows = slice(4 * (v // 2), 4 * (v // 2) + 4)
        target_rows = slice(4 * (target // 2), 4 * (target // 2) + 4)
        out[target_rows, target % 2 :: 2] = rows[source_rows, v % 2 :: 2]
    return out.reshape(-1)


def run(session, fmt, data):
    """One program on core (0, 0): (out pages [2 tiles, 1,024], info words)."""
    nonce = secrets.randbits(32) | 1
    tiles = len(data)
    page = 4 * TILE_WORDS
    t_in = session.tensor(data.view(np.int32), "int32")
    t_out = session.tensor(np.zeros((2 * tiles, TILE_WORDS), np.int32), "int32")
    t_info = session.tensor(np.zeros((1, TILE_WORDS), np.uint32), "uint32")
    core = [(0, 0, 0, 0)]
    launch = session.launch()
    launch.cb(core, CB_IN, fmt, page, 2 * page)
    launch.cb(core, CB_OUT, fmt, page, 4 * page)
    launch.cb(core, CB_INFO, "uint32", page, page)
    launch.kernel(
        str(KERNELS / "reader.cpp"),
        core,
        1,
        t_in.accessor_args(),
        [],
        [],
        [(0, 0, [t_in.address, tiles])],
    )
    launch.kernel(
        str(KERNELS / "writer.cpp"),
        core,
        0,
        t_out.accessor_args() + t_info.accessor_args(),
        [],
        [],
        [(0, 0, [t_out.address, t_info.address, 2 * tiles])],
    )
    options = {
        "fp32_dest_acc_en": True,
        "unpack_to_dest_mode": {CB_IN: "UnpackToDestFp32"},
    }
    launch.kernel(
        str(KERNELS / "compute.cpp"),
        core,
        2,
        [],
        session_kernels.compute_defines(session.arch),
        [str(rec.INCLUDE)],
        [(0, 0, [nonce, tiles])],
        **several.compute_config(options),
    )
    launch.run()
    session.device.finish()
    out = t_out.read().view(np.uint32).reshape(tiles, 2, TILE_WORDS).copy()
    info = t_info.read().view(np.uint32).reshape(-1).copy()
    return nonce, out, info


def wrong(got, want):
    """Words that differ, and the first few as (index, wanted, got) in hex."""
    bad = np.flatnonzero(got != want)
    first = [(int(i), f"{int(want[i]):08x}", f"{int(got[i]):08x}") for i in bad[:6]]
    return int(bad.size), first


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--formats", nargs="+", default=["float32", "int32", "uint32"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save", help="the output words, by format, into this .npz")
    args = p.parse_args()
    data = inputs(args.seed)
    saved, ok = {}, True
    with km.Session() as session:
        chip = chips.chip(session.arch)
        ticks_per_us = chip.busy_mhz * chip.tick_ratio
        for fmt in args.formats:
            nonce, out, info = run(session, fmt, data)
            if mock.active():  # a prebuild: its reads give zeros
                continue
            back, back_first = wrong(out[:, 0].reshape(-1), data.reshape(-1))
            want = np.stack([reordered(t) for t in data])
            moved, moved_first = wrong(out[:, 1].reshape(-1), want.reshape(-1))
            special, _ = wrong(out[-1, 0], data[-1])
            tiles = len(data)
            record = {
                "arch": session.arch,
                "form": "window" if chip.dst_window else "unpacker and packer",
                "format": fmt,
                "nonce_ok": int(info[NONCE]) == nonce,
                "words": int(data.size),
                "wrong_back": back,
                "wrong_back_special_tile": special,
                "wrong_reordered": moved,
                "first_wrong_back": back_first,
                "first_wrong_reordered": moved_first,
                "write_us": [
                    round(int(t) / ticks_per_us, 3) for t in info[WRITE : WRITE + tiles]
                ],
                "read_us": [
                    round(int(t) / ticks_per_us, 3) for t in info[READ : READ + tiles]
                ],
            }
            ok = ok and record["nonce_ok"] and back == 0 and moved == 0
            saved[fmt] = out
            print(json.dumps(record), flush=True)
    if args.save and saved:
        np.savez(args.save, **saved)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
