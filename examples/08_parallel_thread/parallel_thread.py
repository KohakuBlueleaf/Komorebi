"""Parallel thread: a map, a reduction and a two-pass prefix sum on the work cluster's
four cores with kmx_parallel (thread/parallel.c), each run with one partition and with
four, or (--twelve) the map and prefix sum on 1, 4, 8 and 12 cores of the three clusters
(thread/clusters.c), checked against numpy and timed. One JSON line; see README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
STARTED = time.monotonic()
TICKS_PER_MICROSECOND = 50  # kmx_time() counts the 50 MHz timer
REGIONS = (
    "in", "map_1", "map_4", "map_quarters", "map_aliased", "vmap_1", "vmap_4",
    "heavy_1", "heavy_4", "sequence", "prefix_1", "prefix_4", "counters",
)  # fmt: skip
TIMES = (
    "map_1", "map_4", "map_quarters", "map_aliased", "vmap_1", "vmap_4", "heavy_1",
    "heavy_4", "sum_1", "sum_4", "sum_4_packed", "sum_4_padded", "prefix_sequence",
    "prefix_1", "prefix_4", "invalidate_1", "invalidate_4", "flush_1", "flush_4",
    "parallel_1000",
)  # fmt: skip
SUMS = ("sum_1", "sum_4", "sum_4_packed", "sum_4_padded")
WAY = 128 << 10  # the cluster's L3: one way of 128 KiB (README.md, "The rules")
CODES = {
    0: "ok",
    1: "bad words or regions",
    2: "a kmx_parallel call was late or stopped",
    3: "the fetch of the input failed",
}
FETCH_BANK = 7  # --fetch: the input's bank, which cluster 1 (bank 6) cannot read
ROW = 16384  # --fetch: a page of the source; its bank's pages lie one after another
TWELVE_REGIONS = (
    "in", "ref_map", "ref_heavy", "ref_pre", "map", "heavy", "pre", "sums",
)  # fmt: skip
PATTERNS = ("map", "heavy", "prefix")
X280_CORES = (1, 4, 8, 12)
SUMS_BYTES = 4096 * 64  # the chunk sums: a line for each of CHUNKS_MAX chunks
TWELVE_CODES = {0: "ok", 1: "bad words or regions", 2: "late or stopped", 3: "a push"}


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def mapped(x, rounds=1):
    """thread/parallel.c's mix() in numpy, applied `rounds` times."""
    x = x.astype(np.uint64)
    for _ in range(rounds):
        x = ((x ^ (x >> 15)) * 2246822519 + 1) & 0xFFFFFFFF
    return x.astype(np.uint32)


def exclusive_prefix(x):
    """The exclusive prefix sum mod 2**32."""
    result = np.zeros_like(x)
    np.cumsum(x[:-1], dtype=np.uint32, out=result[1:])
    return result


def program(data, repeats, rounds, deadline, fetch=None):
    """The grid (every command a no-op) and the thread with its regions: each array
    rounded up to whole 128 KiB, the input 64 KiB more, so that every output starts
    half a way from the input; map_aliased is written half a way in, on its way.
    `fetch` (bank, address): the input starts zero and the thread fetches it there."""
    thread_program = km.Program("parallel_thread", kernels=HERE / "kernels")
    size = -(-data.nbytes // WAY) * WAY
    padded = np.zeros((size + WAY // 2) // 4, np.uint32)
    if fetch is None:
        padded[: data.size] = data
    regions = {name: size for name in REGIONS}
    regions.update({"in": padded, "map_aliased": size + WAY, "counters": 4096})
    thread_program.thread(
        cluster=1,
        sources=[HERE / "thread" / "parallel.c", HERE / "thread" / "vmap.c"],
        includes=[HERE / "thread"],
        regions=regions,
        words=[data.size, repeats, rounds, *(fetch or ())],
        deadline=deadline,
    )
    return thread_program


def fetch_source(session, data):
    """The input in DRAM bank FETCH_BANK: a tensor of ROW-byte pages whose pages in that
    bank hold it; returns the tensor (held while the thread runs) and (bank, address).
    """
    pages = -(-data.nbytes // ROW)
    rows = np.zeros((8 * pages, ROW // 4), np.uint32)
    mine = np.zeros(pages * ROW // 4, np.uint32)
    mine[: data.size] = data
    rows[FETCH_BANK::8] = mine.reshape(pages, -1)
    tensor = session.tensor(rows, "uint32")
    return tensor, (FETCH_BANK, tensor.address)


def program_twelve(data, repeats, rounds, deadline):
    """The same thread on clusters 1 to 3, its regions placed as program()'s: the
    input half a way longer, every other array whole ways."""
    thread_program = km.Program("parallel_clusters", kernels=HERE / "kernels")
    size = -(-data.nbytes // WAY) * WAY
    padded = np.zeros((size + WAY // 2) // 4, np.uint32)
    padded[: data.size] = data
    regions = {name: size for name in TWELVE_REGIONS}
    regions.update({"in": padded, "sums": SUMS_BYTES})
    thread_program.thread(
        clusters=(1, 2, 3),
        sources=[HERE / "thread" / "clusters.c"],
        regions=regions,
        words=[data.size, repeats, rounds],
        deadline=deadline,
    )
    return thread_program


def times_microseconds(words, first_word):
    """{cores: microseconds} of four result words from `first_word`."""
    return {
        cores: round(words[first_word + index] / TICKS_PER_MICROSECOND, 1)
        for index, cores in enumerate(X280_CORES)
    }


def run_twelve(args, data):
    """The --twelve mode: per pattern and cores the best time, the input push's time,
    and whether each output is exact (against numpy for 1 and 12 cores, against the
    1-core output in the thread for 4 and 8)."""
    with km.Session(log=log) as session:
        thread_program = program_twelve(data, args.repeats, args.rounds, deadline=30.0)
        run = session.load(thread_program)
        log("loaded")
        try:
            results = run.thread_results(timeout=40.0)
        except km.ThreadFailed as failure:
            results = failure.results
        for cluster, result in results.items():
            log(
                f"cluster {cluster}: {result.state}, code {result.code}, "
                f"error {result.error}, at {result.where}"
            )
        run.stop()
        lead = run.threads[1]
        got = {
            name: lead.read(name).view(np.uint32)[: data.size]
            for name in TWELVE_REGIONS[1:7]
        }
        log("regions read")
    result_words = list(results[1].words) + [0] * 28
    want = {"map": mapped(data), "heavy": mapped(data, args.rounds)}
    want["prefix"] = exclusive_prefix(data)
    report = {"words": args.words, "rounds": args.rounds}
    report["codes"] = {
        cluster: TWELVE_CODES.get(result.code, result.code)
        for cluster, result in results.items()
    }
    report["microseconds"], report["exact"] = {}, {}
    for pattern_index, pattern in enumerate(PATTERNS):
        region = "pre" if pattern == "prefix" else pattern
        report["microseconds"][pattern] = times_microseconds(
            result_words, 4 * pattern_index
        )
        # words that differ from the 1-core output
        differing = result_words[12 + 4 * pattern_index : 16 + 4 * pattern_index]
        differ = dict(zip(X280_CORES, differing))
        report["exact"][f"{pattern}_1"] = bool(
            (got[f"ref_{region}"] == want[pattern]).all()
        )
        report["exact"][f"{pattern}_4"] = differ[4] == 0
        report["exact"][f"{pattern}_8"] = differ[8] == 0
        report["exact"][f"{pattern}_12"] = (
            bool((got[region] == want[pattern]).all()) and not differ[12]
        )
    report["microseconds"]["push_in"] = times_microseconds(result_words, 24)
    report["ok"] = all(
        result.ran and result.code == 0 and result.error == "none"
        for result in results.values()
    ) and all(report["exact"].values())
    return report


def check(data, rounds, got, words):
    """{pattern: exact against numpy} from the regions read and the result words."""
    light, heavy, pre = mapped(data), mapped(data, rounds), exclusive_prefix(data)
    want = {"map_1": light, "map_4": light, "map_quarters": light, "vmap_1": light}
    want["map_aliased"] = light
    want.update({"vmap_4": light, "heavy_1": heavy, "heavy_4": heavy})
    want.update({"sequence": pre, "prefix_1": pre, "prefix_4": pre})
    exact = {name: bool((got[name] == array).all()) for name, array in want.items()}
    total, first_sum_word = int(data.astype(np.uint64).sum()), len(TIMES)
    for index, name in enumerate(SUMS):
        low = words[first_sum_word + 2 * index]
        high = words[first_sum_word + 2 * index + 1]
        exact[name] = (low | high << 32) == total
    return exact


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--words", type=int, default=1 << 18, help="32-bit words per array"
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="runs per timing (the best)"
    )
    parser.add_argument(
        "--rounds", type=int, default=16, help="mix() per word, heavy map"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--twelve", action="store_true", help="1 to 12 X280 cores, 3 clusters"
    )
    parser.add_argument(
        "--fetch", action="store_true", help="the input fetched from bank 7 first"
    )
    args = parser.parse_args()
    generator = np.random.default_rng(args.seed)
    data = generator.integers(0, 1 << 32, args.words, dtype=np.uint64).astype(np.uint32)
    if args.twelve:
        report = run_twelve(args, data)
        report["wall_s"] = round(time.monotonic() - STARTED, 1)
        print(json.dumps(report), flush=True)
        sys.exit(0 if report["ok"] else 1)
    with km.Session(log=log) as session:
        source, fetch = fetch_source(session, data) if args.fetch else (None, None)
        thread_program = program(data, args.repeats, args.rounds, 30.0, fetch)
        run = session.load(thread_program)
        log("loaded")
        result = run.thread_result(timeout=40.0)
        log(
            f"thread: {result.state}, code {result.code}, error {result.error}, "
            f"at {result.where}"
        )
        run.stop()
        del source  # the fetched input's tensor: needed until the thread returned
        got = {}
        for name in REGIONS[1:-1]:
            skip = WAY // 8 if name == "map_aliased" else 0  # written half a way in
            got[name] = run.thread.read(name).view(np.uint32)[skip : skip + data.size]
        log("regions read")
    words = list(result.words) + [0] * (len(TIMES) + 2 * len(SUMS) + 1)
    report = {
        "words": args.words,
        "rounds": args.rounds,
        "code": CODES.get(result.code, result.code),
    }
    report["error"] = result.error
    report["thread_ms"] = round(result.microseconds / 1e3, 1)
    report["microseconds"] = {
        name: round(words[index] / TICKS_PER_MICROSECOND, 1)
        for index, name in enumerate(TIMES)
    }
    if args.fetch:
        fetched = words[len(TIMES) + 2 * len(SUMS)]
        report["microseconds"]["fetch"] = round(fetched / TICKS_PER_MICROSECOND, 1)
    report["exact"] = check(data, args.rounds, got, words)
    microseconds = report["microseconds"]
    pairs = {"map": ("map_1", "map_4"), "vmap": ("vmap_1", "vmap_4")}
    pairs.update({"heavy": ("heavy_1", "heavy_4"), "sum": ("sum_1", "sum_4")})
    pairs.update({"prefix": ("prefix_sequence", "prefix_4")})
    report["four_cores_x"] = {
        name: (
            round(microseconds[slow] / microseconds[fast], 2)
            if microseconds[fast]
            else None
        )
        for name, (slow, fast) in pairs.items()
    }
    report["ok"] = (
        result.ran
        and result.code == 0
        and result.error == "none"
        and all(report["exact"].values())
    )
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
