"""Parallel thread: a map, a reduction and a two-pass prefix sum on the work cluster's
four harts with kmx_par (thread/parallel.c), each run with one share and with four,
or (--twelve) the map and prefix sum on 1, 4, 8 and 12 harts of the three clusters
(thread/clusters.c), checked against numpy and timed. One JSON line; see README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
T0 = time.monotonic()
TICKS_PER_US = 50  # kmx_time() counts the 50 MHz timer
REGIONS = (
    "in", "map_1", "map_4", "map_quarters", "map_aliased", "vmap_1", "vmap_4",
    "heavy_1", "heavy_4", "seq", "prefix_1", "prefix_4", "counters",
)  # fmt: skip
TIMES = (
    "map_1", "map_4", "map_quarters", "map_aliased", "vmap_1", "vmap_4", "heavy_1",
    "heavy_4", "sum_1", "sum_4", "sum_4_packed", "sum_4_padded", "prefix_seq",
    "prefix_1", "prefix_4", "take_1", "take_4", "give_1", "give_4", "par_1000",
)  # fmt: skip
SUMS = ("sum_1", "sum_4", "sum_4_packed", "sum_4_padded")
WAY = 128 << 10  # the cluster's L3: one way of 128 KiB (README.md, "The rules")
CODES = {0: "ok", 1: "bad words or regions", 2: "a kmx_par call was late or stopped"}
TWELVE_REGIONS = (
    "in", "ref_map", "ref_heavy", "ref_pre", "map", "heavy", "pre", "sums",
)  # fmt: skip
PATTERNS = ("map", "heavy", "prefix")
HARTS = (1, 4, 8, 12)
SUMS_BYTES = 4096 * 64  # the chunk sums: a line for each of CHUNKS_MAX chunks
TWELVE_CODES = {0: "ok", 1: "bad words or regions", 2: "late or stopped", 3: "a push"}


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def mapped(x, rounds=1):
    """thread/parallel.c's f() in numpy, applied `rounds` times."""
    x = x.astype(np.uint64)
    for _ in range(rounds):
        x = ((x ^ (x >> 15)) * 2246822519 + 1) & 0xFFFFFFFF
    return x.astype(np.uint32)


def exclusive_prefix(x):
    """The exclusive prefix sum mod 2**32."""
    out = np.zeros_like(x)
    np.cumsum(x[:-1], dtype=np.uint32, out=out[1:])
    return out


def program(data, repeats, rounds, deadline):
    """The grid (every record a no-op) and the thread with its regions: each array
    rounded up to whole 128 KiB, the input 64 KiB more, so that every output starts
    half a way from the input; map_aliased is written half a way in, on its way."""
    prog = km.Program("parallel_thread", kernels=HERE / "kernels")
    size = -(-data.nbytes // WAY) * WAY
    padded = np.zeros((size + WAY // 2) // 4, np.uint32)
    padded[: data.size] = data
    regions = {name: size for name in REGIONS}
    regions.update({"in": padded, "map_aliased": size + WAY, "counters": 4096})
    prog.thread(
        cluster=1,
        sources=[HERE / "thread" / "parallel.c", HERE / "thread" / "vmap.c"],
        includes=[HERE / "thread"],
        regions=regions,
        words=[data.size, repeats, rounds],
        deadline=deadline,
    )
    return prog


def program_twelve(data, repeats, rounds, deadline):
    """The same thread on clusters 1 to 3, its regions placed as program()'s: the
    input half a way longer, every other array whole ways."""
    prog = km.Program("parallel_clusters", kernels=HERE / "kernels")
    size = -(-data.nbytes // WAY) * WAY
    padded = np.zeros((size + WAY // 2) // 4, np.uint32)
    padded[: data.size] = data
    regions = {name: size for name in TWELVE_REGIONS}
    regions.update({"in": padded, "sums": SUMS_BYTES})
    prog.thread(
        clusters=(1, 2, 3),
        sources=[HERE / "thread" / "clusters.c"],
        regions=regions,
        words=[data.size, repeats, rounds],
        deadline=deadline,
    )
    return prog


def times_us(words, at):
    """{harts: microseconds} of four result words from `at`."""
    return {h: round(words[at + i] / TICKS_PER_US, 1) for i, h in enumerate(HARTS)}


def run_twelve(args, data):
    """The --twelve mode: per pattern and harts the best time, the input push's time,
    and whether each output is exact (against numpy for 1 and 12 harts, against the
    1-hart output in the thread for 4 and 8)."""
    with km.Session(log=log) as session:
        prog = program_twelve(data, args.repeats, args.rounds, deadline=30.0)
        run = session.load(prog)
        log("loaded")
        try:
            results = run.thread_results(timeout=40.0)
        except km.ThreadFailed as e:
            results = e.results
        for c, r in results.items():
            log(f"cluster {c}: {r.state}, code {r.code}, error {r.error}, at {r.where}")
        run.stop()
        lead = run.threads[1]
        got = {
            n: lead.read(n).view(np.uint32)[: data.size] for n in TWELVE_REGIONS[1:7]
        }
        log("regions read")
    w = list(results[1].words) + [0] * 28
    want = {"map": mapped(data), "heavy": mapped(data, args.rounds)}
    want["prefix"] = exclusive_prefix(data)
    out = {"words": args.words, "rounds": args.rounds}
    out["codes"] = {c: TWELVE_CODES.get(r.code, r.code) for c, r in results.items()}
    out["us"], out["exact"] = {}, {}
    for p, k in enumerate(PATTERNS):
        region = "pre" if k == "prefix" else k
        out["us"][k] = times_us(w, 4 * p)
        differ = dict(zip(HARTS, w[12 + 4 * p : 16 + 4 * p]))  # words that differ
        out["exact"][f"{k}_1"] = bool((got[f"ref_{region}"] == want[k]).all())
        out["exact"][f"{k}_4"] = differ[4] == 0
        out["exact"][f"{k}_8"] = differ[8] == 0
        out["exact"][f"{k}_12"] = (
            bool((got[region] == want[k]).all()) and not differ[12]
        )
    out["us"]["push_in"] = times_us(w, 24)
    out["ok"] = all(
        r.ran and r.code == 0 and r.error == "none" for r in results.values()
    ) and all(out["exact"].values())
    return out


def check(data, rounds, got, words):
    """{pattern: exact against numpy} from the regions read and the result words."""
    light, heavy, pre = mapped(data), mapped(data, rounds), exclusive_prefix(data)
    want = {"map_1": light, "map_4": light, "map_quarters": light, "vmap_1": light}
    want["map_aliased"] = light
    want.update({"vmap_4": light, "heavy_1": heavy, "heavy_4": heavy})
    want.update({"seq": pre, "prefix_1": pre, "prefix_4": pre})
    exact = {k: bool((got[k] == v).all()) for k, v in want.items()}
    total, at = int(data.astype(np.uint64).sum()), len(TIMES)
    for i, name in enumerate(SUMS):
        exact[name] = (words[at + 2 * i] | words[at + 2 * i + 1] << 32) == total
    return exact


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--words", type=int, default=1 << 18, help="32-bit words per array")
    p.add_argument("--repeats", type=int, default=3, help="runs per timing (the best)")
    p.add_argument("--rounds", type=int, default=16, help="f() per word, heavy map")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--twelve", action="store_true", help="1 to 12 harts, 3 clusters")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    data = rng.integers(0, 1 << 32, args.words, dtype=np.uint64).astype(np.uint32)
    if args.twelve:
        out = run_twelve(args, data)
        out["wall_s"] = round(time.monotonic() - T0, 1)
        print(json.dumps(out), flush=True)
        sys.exit(0 if out["ok"] else 1)
    with km.Session(log=log) as session:
        prog = program(data, args.repeats, args.rounds, deadline=30.0)
        run = session.load(prog)
        log("loaded")
        r = run.thread_result(timeout=40.0)
        log(f"thread: {r.state}, code {r.code}, error {r.error}, at {r.where}")
        run.stop()
        got = {}
        for name in REGIONS[1:-1]:
            skip = WAY // 8 if name == "map_aliased" else 0  # written half a way in
            got[name] = run.thread.read(name).view(np.uint32)[skip : skip + data.size]
        log("regions read")
    words = list(r.words) + [0] * (len(TIMES) + 2 * len(SUMS))
    out = {
        "words": args.words,
        "rounds": args.rounds,
        "code": CODES.get(r.code, r.code),
    }
    out["error"] = r.error
    out["thread_ms"] = round(r.us / 1e3, 1)
    out["us"] = {k: round(words[i] / TICKS_PER_US, 1) for i, k in enumerate(TIMES)}
    out["exact"] = check(data, args.rounds, got, words)
    us = out["us"]
    pairs = {"map": ("map_1", "map_4"), "vmap": ("vmap_1", "vmap_4")}
    pairs.update({"heavy": ("heavy_1", "heavy_4"), "sum": ("sum_1", "sum_4")})
    pairs.update({"prefix": ("prefix_seq", "prefix_4")})
    out["four_harts_x"] = {
        k: round(us[a] / us[b], 2) if us[b] else None for k, (a, b) in pairs.items()
    }
    out["ok"] = (
        r.ran and r.code == 0 and r.error == "none" and all(out["exact"].values())
    )
    out["wall_s"] = round(time.monotonic() - T0, 1)
    print(json.dumps(out), flush=True)
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
