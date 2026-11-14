"""Fetch and store over islands: island 0's work cluster 1 thread fetches a tensor
island 0 holds and the same bytes held by island 1 into its region (kmx_fetch), then
stores the region into a tensor of each island (kmx_store); every word checked. One
JSON line. Example: python islands.py --sizes 1048576 67108864"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
KERNELS = HERE.parents[1] / "examples" / "08_parallel_thread" / "kernels"  # NOPs
PAGE = 4096  # bytes of a page (one fabric packet carries one)
TICKS = 50  # kmx_time() ticks a microsecond
T0 = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - T0:7.2f}] {message}", flush=True)


def hashed(data):
    """islands.c's hash: the sum of 64-bit word j times 2 j + 1, mod 2**64."""
    words = data.view(np.uint64)
    return int(np.sum(words * (2 * np.arange(words.size, dtype=np.uint64) + 1)))


def wrong_pages(got, want):
    """The pages of `got` that differ from `want`: how many, the first 8, and for
    each of those: all zero, or the page of `want` its first 16 bytes match."""
    pages = got.size // PAGE
    got = got[: pages * PAGE].reshape(pages, PAGE)
    want = want[: pages * PAGE].reshape(pages, PAGE)
    bad = np.flatnonzero((got != want).any(axis=1))
    heads = {bytes(want[j, :16]): j for j in range(pages)}
    first = []
    for j in bad[:8]:
        zero = not got[j].any()
        first.append([int(j), "zero" if zero else heads.get(bytes(got[j, :16]), -1)])
    return {"count": int(bad.size), "first": first}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[1 << 20, 64 << 20])
    parser.add_argument("--cores", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--links", type=int, default=4)
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    largest = max(args.sizes)
    rng = np.random.default_rng(args.seed)
    data = rng.integers(0, 1 << 32, (largest // PAGE, PAGE // 4), dtype=np.uint64)
    data = data.astype(np.uint32)
    want = data.reshape(-1).view(np.uint8)
    cards = [int(card) for card in args.cards.split(",")]
    report = {"sizes": args.sizes, "cores": args.cores, "links": args.links}
    with km.Session(cards=cards, links=args.links, log=log) as session:
        own = [(session.graph.first(i), session.graph.count(i)) for i in range(2)]
        local = session.tensor(data, "uint32", group=own[0])
        far = session.tensor(data, "uint32", group=own[1])
        backs = []
        for size in args.sizes:
            for island in range(2):
                shape = (size // PAGE, PAGE // 4)
                backs.append(session.empty(shape, "uint32", group=own[island]))
        log(
            f"tensors: island 0 {local.placement.parts}, island 1 {far.placement.parts}"
        )
        words = [local.address, far.address, PAGE, args.cores, args.repeats]
        words += [len(args.sizes), *args.sizes, *[0] * (2 - len(args.sizes))]
        words += [back.address for back in backs]
        words += [0] * (12 - len(words))
        with_thread = km.Program("bank_fetch", kernels=KERNELS)
        with_thread.thread(
            cluster=1,
            sources=[HERE / "thread" / "islands.c"],
            regions={"dst": largest},
            words=words,
            deadline=60.0,
        )
        run = session.load([with_thread, km.Program("bank_fetch", kernels=KERNELS)])
        log("loaded: the thread on island 0")
        mine = run.island(0)
        result = mine.thread_result(timeout=90.0)
        log(f"thread: {result.state}, code {result.code}, error {result.error}")
        log(f"thread words: {list(result.words)}")
        run.stop()
        log("stopped")
        region = mine.thread.read("dst")
        log("region read")
        stored = session.read(*backs)
        held = session.read(local, far)
        log("stored and fetched tensors read")
    out = list(result.words) + [0] * 20
    report["thread"] = [result.state, result.code, result.error]
    got = np.asarray(region).view(np.uint8)[: args.sizes[-1]]
    report["region_exact"] = bool((got == want[: args.sizes[-1]]).all())
    report["region_wrong"] = wrong_pages(got, want)
    report["held_wrong"] = [
        wrong_pages(np.asarray(t).view(np.uint8).reshape(-1), want) for t in held
    ]
    rows = []
    for i, size in enumerate(args.sizes):
        o = out[10 * i : 10 * i + 10]
        want_hash = hashed(want[:size])
        row = {"bytes": size, "status": o[0]}
        for name, at in (("local", 1), ("far", 4)):
            ticks, low, high = o[at : at + 3]
            row[f"fetch_{name}_ms"] = round(ticks / TICKS / 1e3, 3)
            row[f"fetch_{name}_gb_s"] = round(size * TICKS / max(ticks, 1) / 1e3, 3)
            row[f"fetch_{name}_exact"] = (low | high << 32) == want_hash
        for name, at, back in (("local", 7, 2 * i), ("far", 8, 2 * i + 1)):
            row[f"store_{name}_ms"] = round(o[at] / TICKS / 1e3, 3)
            got = np.asarray(stored[back]).view(np.uint8).reshape(-1)[:size]
            row[f"store_{name}_exact"] = bool((got == want[:size]).all())
            # against what the region held (what the store wrote, right or not)
            held_region = np.asarray(region).view(np.uint8)[:size]
            row[f"store_{name}_as_region"] = bool((got == held_region).all())
        rows.append(row)
    report["rows"] = rows
    report["ok"] = bool(
        result.ran
        and result.code == 0
        and report["region_exact"]
        and all(
            row["status"] == 0
            and all(v for k, v in row.items() if k.endswith(("exact", "as_region")))
            for row in rows
        )
    )
    print(json.dumps(report), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
