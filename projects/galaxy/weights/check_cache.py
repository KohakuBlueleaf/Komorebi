"""Checks the converted cache: every weight's CRC-32 against its island's log, and no
block-float negative zero code in a bfp arena (formats.bfp_negative_zeros).

python check_cache.py PLACEMENT_JSON CACHE_DIR [--jobs 31]
"""

import argparse
import json
import os
import sys
import zlib
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import convert  # noqa: E402

formats = convert.formats


def check_island(args):
    placement_path, cache, island = args
    with open(placement_path) as f:
        p = json.load(f)
    done = convert.read_done(cache, island)
    bad_crc, negative, weights = [], 0, 0
    files = {}
    try:
        for e in p["entries"]:
            if e["island"] != island:
                continue
            dtype, tile = e["dtype"], p["tile_bytes"][e["dtype"]]
            if dtype not in files:
                files[dtype] = os.open(
                    convert.arena_path(cache, island, dtype), os.O_RDONLY
                )
            raw = os.pread(files[dtype], e["pages"] * tile, e["first"] * tile)
            weights += 1
            if e["name"] not in done or zlib.crc32(raw) != done[e["name"]][0]:
                bad_crc.append(e["name"])
            if dtype in formats.BLOCK:
                found = formats.bfp_negative_zeros(
                    memoryview(raw), formats.BLOCK[dtype]
                )
                negative += int(found.any(axis=1).sum())
    finally:
        for fd in files.values():
            os.close(fd)
    return island, weights, bad_crc, negative


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("placement")
    ap.add_argument("cache")
    ap.add_argument("--jobs", type=int, default=31)
    a = ap.parse_args()
    with open(a.placement) as f:
        count = len(json.load(f)["cards"])
    jobs = [(a.placement, a.cache, i) for i in range(count)]
    total = bad = neg = 0
    with Pool(min(a.jobs, count)) as pool:
        for island, weights, bad_crc, negative in pool.imap_unordered(
            check_island, jobs
        ):
            total, bad, neg = total + weights, bad + len(bad_crc), neg + negative
            print(
                f"island {island}: {weights} weights, {len(bad_crc)} CRC differ "
                f"{bad_crc[:3]}, {negative} tiles with a negative zero code",
                flush=True,
            )
    print(f"all: {total} weights, {bad} CRC differ, {neg} tiles with a negative zero")


if __name__ == "__main__":
    main()
