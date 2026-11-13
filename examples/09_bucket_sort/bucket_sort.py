"""Bucket sort, first pass: COUNT on the Tensix group, PLAN on 1, 4 or 12 X280 cores,
PLACE on the Tensix group, driven by an X280 thread. The host loads, starts and
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
STARTED = time.monotonic()
# bucket_sort.h: the words' order, the kernels' sizes
WORDS = (
    "key_count", "log2_buckets", "cores", "keys_address", "output_address",
    "x280_cores", "repeats", "banks",
)  # fmt: skip
TIMES = ("count", "plan", "place", "whole")
CHUNK, BUCKETS_MAX, KEYS_PER_PAGE, SLOTS = 16, 16384, 1024, 64
WAY = 128 << 10  # an X280 cluster's L3: one cache way of 128 KiB
CODES = {0: "ok", 1: "bad words or regions", 2: "a kmx_parallel call failed"}
CODES.update({3: "a push failed", 4: "the ring failed", 5: "a command did not finish"})


def log(message):
    """A progress line with seconds since start, flushed (the card job's watcher
    reads them)."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def keys_tensor(session, values):
    """Step 0 (host): a DRAM tensor of 4 KiB pages, 1,024 keys a page, interleaved
    over the banks; the last page padded with zeros the kernels never read."""
    rows = np.zeros((-(-values.size // KEYS_PER_PAGE), KEYS_PER_PAGE), np.uint32)
    rows.reshape(-1)[: values.size] = values
    return session.tensor(rows, "uint32")


def program(keys, output, words, x280_cores):
    """The Tensix side (kernels/, its two circular buffers) and the X280 thread
    (thread/plan.c) on cluster 1, or on clusters 1 to 3 for twelve cores."""
    sort_program = km.Program(
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
            "writer": output.accessor_args(),
        },
    )
    table = words[2] * 4 << words[1]  # cores x buckets counters of 4 bytes
    # The counts region is whole cache ways plus half a way, so that the plan, after
    # it, never sits a whole number of ways from the counts the cores read beside it.
    padded = -(-table // WAY) * WAY + WAY // 2
    # A total per bucket, room for two rows: the thread starts them half a row off
    # (thread/plan.c).
    regions = {"counts": padded, "plan": table, "totals": 8 << words[1], "peers": 256}
    sort_program.thread(
        clusters=(1, 2, 3) if x280_cores == 12 else (1,),
        sources=[HERE / "thread" / "plan.c"],
        includes=[HERE],
        regions=regions,
        words=words,
        deadline=20.0,
    )
    return sort_program


def run_card(args, values, log2_buckets):
    """Loads the program, lets the thread run it, stops, reads the output back. Returns
    {cluster: ThreadResult}, the cores used and the output keys."""
    with km.Session(log=log) as session:
        grid_x, grid_y = session.grid
        cores = args.cores or grid_x * grid_y
        log(f"device open: {cores} cores; uploading {values.size} keys")
        keys = keys_tensor(session, values)
        output = keys_tensor(session, np.zeros(values.size, np.uint32))
        log("keys and output tensors in DRAM")
        banks = sum(
            BANK_OF_CLUSTER[cluster] << 8 * (cluster - 1) for cluster in (1, 2, 3)
        )
        words = [values.size, log2_buckets, cores, keys.address]
        words += [output.address, args.x280_cores, args.repeats, banks]
        run = session.load(program(keys, output, words, args.x280_cores), deadline=2.0)
        log(
            "loaded: the grid, the dispatcher, the thread"
            f" ({args.x280_cores} X280 cores)"
        )
        try:
            results = run.thread_results(timeout=30.0)
        except km.ThreadFailed as failure:
            results = failure.results
        for cluster, result in results.items():
            log(
                f"cluster {cluster}: {result.state}, code {result.code}, "
                f"error {result.error}, at {result.where}"
            )
        run.stop()
        log("stopped; reading the output back")
        got = []
        for first, pages in session.read_chunks(output):
            got.append(pages.view(np.uint32).reshape(-1))
            log(f"output pages {first} to {first + pages.shape[0] - 1} read")
        return results, cores, np.concatenate(got)[: values.size]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--keys", type=int, default=1 << 22, help="the keys to sort")
    parser.add_argument(
        "--buckets", type=int, default=4096, help="a power of 2, 16 to 16384"
    )
    parser.add_argument("--x280-cores", type=int, choices=(1, 4, 12), default=12)
    parser.add_argument("--cores", type=int, default=0, help="Tensix cores (0: all)")
    parser.add_argument(
        "--repeats", type=int, default=1, help="rounds; the best is kept"
    )
    parser.add_argument(
        "--one-bucket", action="store_true", help="every key in bucket 0"
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    log2_buckets = args.buckets.bit_length() - 1
    if args.buckets != 1 << log2_buckets or not CHUNK <= args.buckets <= BUCKETS_MAX:
        raise SystemExit(f"--buckets {args.buckets}: a power of 2, 16 to 16384")
    generator = np.random.default_rng(args.seed)
    top = 1 << (32 - log2_buckets) if args.one_bucket else 1 << 32
    values = generator.integers(0, top, args.keys, dtype=np.uint64).astype(np.uint32)
    results, cores, got = run_card(args, values, log2_buckets)
    lead = results[1]
    result_words = list(lead.words) + [0] * len(TIMES) * 2
    report = {
        "keys": args.keys,
        "buckets": args.buckets,
        "x280_cores": args.x280_cores,
    }
    report["cores"], report["one_bucket"] = cores, args.one_bucket
    report["codes"] = {
        cluster: CODES.get(result.code, result.code)
        for cluster, result in results.items()
    }
    report["ms"] = {
        name: round(result_words[index] * 1e3 / TIMER_HZ, 3)
        for index, name in enumerate(TIMES)
    }
    report["placed"], report["rounds"] = result_words[4], result_words[5]
    report["thread_ms"] = round(lead.microseconds / 1e3, 2)
    want = reference.output(values, log2_buckets)
    report["wrong"] = int((got != want).sum())
    report["ok"] = (
        all(
            result.ran and result.code == 0 and result.error == "none"
            for result in results.values()
        )
        and report["placed"] == args.keys
        and report["rounds"] == args.repeats
        and report["wrong"] == 0
    )
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
