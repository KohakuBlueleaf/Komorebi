"""Development load: the weights already in the cache's logs of the given placement
islands, onto a session of CARDS (one island a card, placement island k onto session
island k mod islands). python try_load.py CARDS ISLANDS [MAX_WEIGHTS] [threads]"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import convert  # noqa: E402
import loader  # noqa: E402

import komorebi as km  # noqa: E402

cards = [int(x) for x in sys.argv[1].split(",")]
src = [int(x) for x in sys.argv[2].split(",")]
most = int(sys.argv[3]) if len(sys.argv) > 3 else 50
threads = int(sys.argv[4]) if len(sys.argv) > 4 else None
names = []
for island in src:
    names += list(convert.read_done(loader.CACHE, island))[:most]
t0 = time.time()
with km.Session(cards=cards) as s:
    print(f"open {time.time() - t0:.1f} s", flush=True)
    n = len(cards)
    W = loader.load(
        s, names, islands={k: i % n for i, k in enumerate(src)}, threads=threads
    )
    print(W.report)
    h = W[names[0]]
    print(h, "word", h.word)
    t1 = time.time()
    W.free()
    W = loader.load(
        s,
        names,
        islands={k: i % n for i, k in enumerate(src)},
        resume=True,
        verify="full",
        threads=threads,
    )
    print("reload with resume:", W.report, f"{time.time() - t1:.1f} s")
