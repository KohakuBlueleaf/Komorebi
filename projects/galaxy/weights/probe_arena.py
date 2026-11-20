"""Probe on one card: the largest arena an island needs (a bfp4_b tile tensor of
PAGES one-tile pages) is allocated whole, a span of it written and read back, with
the write and read rates. python probe_arena.py PAGES [WRITE_MIB]"""

import sys
import time

import numpy as np

import komorebi as km

pages = int(sys.argv[1])
mib = int(sys.argv[2]) if len(sys.argv) > 2 else 256
TILE = 576
with km.Session(cards=[0]) as s:
    t0 = time.time()
    arena = s.empty((pages * 32, 32), "bfp4_b", "tile")
    print(
        f"arena {pages:,} pages {pages * TILE:,} B: made in {time.time() - t0:.3f} s,"
        f" address {arena.address:#x}, page_bytes {arena.page_bytes}",
        flush=True,
    )
    n = mib * 2**20 // TILE
    rng = np.random.default_rng(1)
    data = rng.integers(0, 256, (n, TILE), dtype=np.uint8)
    for first in (0, pages - n):
        t0 = time.time()
        s.write(arena, data, first=first)
        tw = time.time() - t0
        t0 = time.time()
        (back,) = s.read(arena, first=first, pages=n)
        tr = time.time() - t0
        print(
            f"first {first:,}: wrote {n * TILE / 2**20:.0f} MiB in {tw:.3f} s "
            f"({n * TILE / tw / 1e9:.2f} GB/s)",
            flush=True,
        )
        print(
            f"  read back in {tr:.3f} s ({n * TILE / tr / 1e9:.2f} GB/s), "
            f"equal {np.array_equal(back.reshape(n, -1), data)}",
            flush=True,
        )
