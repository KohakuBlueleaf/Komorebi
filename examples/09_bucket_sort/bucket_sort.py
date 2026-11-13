"""Bucket sort, first pass: COUNT on the Tensix group, PLAN on 1, 4 or 12 X280 cores,
PLACE on the Tensix group, conducted by an X280 thread. The host loads, starts and
checks against numpy (reference.py). Prints one JSON line; see README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import reference

import komorebi as km
from komorebi.thread import BANK_OF_CLUSTER, TIMER_HZ

HERE = Path(__file__).resolve().parent
T0 = time.monotonic()
# bucket_sort.h: the words' order, the kernels' sizes
WORDS = ("keys_n", "log2_b", "cores", "keys", "out", "x280_cores", "repeats", "banks")
TIMES = ("count", "plan", "place", "whole")
CHUNK, BUCKETS_MAX, KEYS_PER_PAGE, SLOTS = 16, 16384, 1024, 64
WAY = 128 << 10  # an X280 cluster's L3: one cache way of 128 KiB
CODES = {0: "ok", 1: "bad words or regions", 2: "a kmx_par call failed"}
CODES.update({3: "a push failed", 4: "the ring failed", 5: "a command did not finish"})


def log(msg):
    """A progress line with seconds since start, flushed (the card job's watcher
    reads them)."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def keys_tensor(session, values):
    """Step 0 (host): a DRAM tensor of 4 KiB pages, 1,024 keys a page, interleaved
    over the banks; the last page padded with zeros the kernels never read."""
    rows = np.zeros((-(-values.size // KEYS_PER_PAGE), KEYS_PER_PAGE), np.uint32)
    rows.reshape(-1)[: values.size] = values
    return session.tensor(rows, "uint32")


def program(keys, out, words, x280_cores):
    """The Tensix side (kernels/, its two circular buffers) and the X280 thread
    (thread/plan.c) on cluster 1, or on clusters 1 to 3 for twelve cores."""
    prog = km.Program(
        "bucket_sort",
        kernels=HERE / "kernels",
        buffers=[
            km.Buffer(index=0, format="uint32", page_bytes=4 * KEYS_PER_PAGE, pages=2),
            km.Buffer(
                index=16,
                format="uint32",
                page_bytes=4 * BUCKETS_MAX + 16 * SLOTS,
                pages=1,
            ),
        ],
        compile_args={
            "reader": keys.accessor_args(),
            "writer": out.accessor_args(),
        },
    )
    table = words[2] * 4 << words[1]  # cores x B counters of 4 bytes
    # The counts region is whole cache ways plus half a way, so that the plan, after
    # it, never sits a whole number of ways from the counts the cores read beside it.
    padded = -(-table // WAY) * WAY + WAY // 2
    # B totals, room for two rows: the thread starts them half a row off (thread/plan.c)
    regions = {"counts": padded, "plan": table, "totals": 8 << words[1], "peers": 256}
    prog.thread(
        clusters=(1, 2, 3) if x280_cores == 12 else (1,),
        sources=[HERE / "thread" / "plan.c"],
        includes=[HERE],
        regions=regions,
        words=words,
        deadline=20.0,
    )
    return prog


def run_card(args, values, log2_b):
    """Loads the program, lets the thread run it, stops, reads the output back. Returns
    {cluster: ThreadResult}, the cores used and the output keys."""
    with km.Session(log=log) as session:
        gx, gy = session.grid
        cores = args.cores or gx * gy
        log(f"device open: {cores} cores; uploading {values.size} keys")
        keys = keys_tensor(session, values)
        out = keys_tensor(session, np.zeros(values.size, np.uint32))
        log("keys and output tensors in DRAM")
        banks = sum(BANK_OF_CLUSTER[c] << 8 * (c - 1) for c in (1, 2, 3))
        words = [values.size, log2_b, cores, keys.address]
        words += [out.address, args.x280_cores, args.repeats, banks]
        run = session.load(program(keys, out, words, args.x280_cores), deadline=2.0)
        log(
            "loaded: the grid, the dispatcher, the thread"
            f" ({args.x280_cores} X280 cores)"
        )
        try:
            results = run.thread_results(timeout=30.0)
        except km.ThreadFailed as e:
            results = e.results
        for c, r in results.items():
            log(f"cluster {c}: {r.state}, code {r.code}, error {r.error}, at {r.where}")
        run.stop()
        log("stopped; reading the output back")
        got = []
        for first, pages in session.read_chunks(out):
            got.append(pages.view(np.uint32).reshape(-1))
            log(f"output pages {first} to {first + pages.shape[0] - 1} read")
        return results, cores, np.concatenate(got)[: values.size]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--keys", type=int, default=1 << 22, help="N, the keys to sort")
    p.add_argument("--buckets", type=int, default=4096, help="B, a power of 2, 16+")
    p.add_argument("--x280-cores", type=int, choices=(1, 4, 12), default=12)
    p.add_argument("--cores", type=int, default=0, help="Tensix cores (0: all)")
    p.add_argument("--repeats", type=int, default=1, help="rounds; the best is kept")
    p.add_argument("--one-bucket", action="store_true", help="every key in bucket 0")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    log2_b = args.buckets.bit_length() - 1
    if args.buckets != 1 << log2_b or not CHUNK <= args.buckets <= BUCKETS_MAX:
        raise SystemExit(f"--buckets {args.buckets}: a power of 2, 16 to 16384")
    rng = np.random.default_rng(args.seed)
    top = 1 << (32 - log2_b) if args.one_bucket else 1 << 32
    values = rng.integers(0, top, args.keys, dtype=np.uint64).astype(np.uint32)
    results, cores, got = run_card(args, values, log2_b)
    lead = results[1]
    w = list(lead.words) + [0] * len(TIMES) * 2
    out = {"keys": args.keys, "buckets": args.buckets, "x280_cores": args.x280_cores}
    out["cores"], out["one_bucket"] = cores, args.one_bucket
    out["codes"] = {c: CODES.get(r.code, r.code) for c, r in results.items()}
    out["ms"] = {k: round(w[i] * 1e3 / TIMER_HZ, 3) for i, k in enumerate(TIMES)}
    out["placed"], out["rounds"] = w[4], w[5]
    out["thread_ms"] = round(lead.us / 1e3, 2)
    want = reference.output(values, log2_b)
    out["wrong"] = int((got != want).sum())
    out["ok"] = (
        all(r.ran and r.code == 0 and r.error == "none" for r in results.values())
        and out["placed"] == args.keys
        and out["rounds"] == args.repeats
        and out["wrong"] == 0
    )
    out["wall_s"] = round(time.monotonic() - T0, 1)
    print(json.dumps(out), flush=True)
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
