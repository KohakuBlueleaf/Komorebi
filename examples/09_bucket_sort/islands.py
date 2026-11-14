"""Bucket sort over two cards as the islands of one session: island 0's ranks and
thread sort, the output held by island 0 (each key's word a NoC write), then by island
1 (each key's word a byte put across). Checks both against numpy; one JSON line."""

import argparse
import json
import sys
import time

import numpy as np
from bucket_sort import keys_tensor, program

import komorebi as km
import reference
from komorebi.thread import BANK_OF_CLUSTER, TIMER_HZ

STARTED = time.monotonic()
CLUSTER = 1  # the thread's work cluster, on island 0
TIMES = ("count", "plan", "place", "whole")
RESULT_WORDS = 6  # bucket_sort.h: BS_RESULT_WORDS


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def sort_into(run, output, row):
    """The thread's rounds into `output`, waited for, their result words to row `row`
    of its region: its state and code, and what crossed each way meanwhile."""
    part = run.island(0)
    before = run.crossings()
    handle = part.submit(
        "sort", cluster=CLUSTER, args=[output.address, row], result=True
    )
    event = part.wait(handle, timeout=60.0)
    moved = run.crossings() - before
    out = {"state": event.kind, "code": event.result if event.kind == "done" else None}
    out["crossed"] = {f"{a}->{b}": moved.total(a, b) for a, b in ((0, 1), (1, 0))}
    return out


def result_words(run, row):
    """Row `row` of the thread's result words (read once the grid stopped): the best ms
    of each step, the keys placed, the rounds."""
    words = run.island(0).threads[CLUSTER].read("results").view(np.uint32)
    words = words[row * RESULT_WORDS : (row + 1) * RESULT_WORDS]
    out = {
        "ms": {
            name: round(int(words[i]) * 1e3 / TIMER_HZ, 3)
            for i, name in enumerate(TIMES)
        }
    }
    out["placed"], out["rounds"] = int(words[4]), int(words[5])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--keys", type=int, default=1 << 16, help="the keys to sort")
    parser.add_argument("--buckets", type=int, default=4096, help="a power of 2")
    parser.add_argument("--x280-cores", type=int, choices=(1, 4), default=4)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    log2_buckets = args.buckets.bit_length() - 1
    if args.buckets != 1 << log2_buckets or not 16 <= args.buckets <= 16384:
        raise SystemExit(f"--buckets {args.buckets}: a power of 2, 16 to 16384")
    generator = np.random.default_rng(args.seed)
    values = generator.integers(0, 1 << 32, args.keys, dtype=np.uint64)
    values = values.astype(np.uint32)
    cards = [int(card) for card in args.cards.split(",")]
    report = {"cards": cards, "keys": args.keys, "buckets": args.buckets}
    report["x280_cores"] = args.x280_cores
    outputs, results = {}, {}
    with km.Session(cards=cards, log=log) as session:
        graph = session.graph
        islands = [(graph.first(i), graph.count(i)) for i in range(2)]
        cores = islands[0][1]  # island 0's ranks: ranks 0 to cores - 1
        report["cores"] = cores
        log(f"open: islands {islands}; island 0's {cores} ranks sort")
        keys = keys_tensor(session, values, group=islands[0])
        empty = np.zeros(values.size, np.uint32)
        for name, island in (("here", 0), ("across", 1)):
            outputs[name] = keys_tensor(session, empty, group=islands[island])
        log(f"keys and outputs in DRAM: {[o.address for o in outputs.values()]}")
        banks = sum(BANK_OF_CLUSTER[c] << 8 * (c - 1) for c in (1, 2, 3))
        words = [values.size, log2_buckets, cores, keys.address]
        words += [outputs["here"].address, args.x280_cores, 1, banks]
        made = [
            program(keys, outputs["here"], words, args.x280_cores, thread, by_host=True)
            for thread in (True, False)  # island 1 without the thread
        ]
        run = session.load(made, deadline=2.0)
        log("loaded on both islands")
        for row, (name, output) in enumerate(outputs.items()):
            results[name] = sort_into(run, output, row)
            log(f"sort into the output {name}: {results[name]}")
        run.stop()
        for row, name in enumerate(outputs):
            results[name].update(result_words(run, row))
        log(f"stopped: {results}; reading the outputs back")
        got = {}
        for name, output in outputs.items():
            (raw,) = session.read(output)
            got[name] = raw.view(np.uint32).reshape(-1)[: values.size]
            log(f"output {name} read")
    want = reference.output(values, log2_buckets)
    per_core = -(-args.keys // cores)  # the most keys a core places
    for name, result in results.items():
        result["wrong"] = int((got[name] != want).sum())
        result["place_us_a_key"] = round(result["ms"]["place"] * 1e3 / per_core, 3)
    report.update(results)
    crossed = results["across"]["crossed"]
    report["crossings_agree"] = (
        crossed["0->1"]["puts"] == args.keys
        and crossed["1->0"]["puts"] == args.keys
        and not any(results["here"]["crossed"][way]["puts"] for way in crossed)
    )
    report["ok"] = report["crossings_agree"] and all(
        result["state"] == "done"
        and result["code"] == 0
        and result["placed"] == args.keys
        and result["rounds"] == 1
        and result["wrong"] == 0
        for result in results.values()
    )
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
