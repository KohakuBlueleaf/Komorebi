"""Example 21's load from the thread, written as classes: Adder's thread, in a load of
Adder (cores 0 to 3) and Doubler (4 to 7), loads Doubler on Adder's cores, doubles x on
all eight, loads Adder back and adds x to the doubled; each step filled by Adder's
header, which has Doubler's parts at this load. Every word against numpy."""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi.bench import power

WORDS = 1024  # float32 words a page: one tile's
WIDTH = 2  # tiles a core of eight
TILES = 8 * WIDTH
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function
def plus(a, b):
    return a + b


class Doubler(km.Program):
    """d = x + x, `width` tiles a core from rank x width."""

    @km.handler
    def double(self, a: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for tile in km.Stream(a.pages(first, width)):
            results.push(plus(tile, tile))


class Adder(km.Program):
    """s = x + d, `width` tiles a core; its thread runs the load's steps."""

    notes = km.region(4096)  # a thread has one region at least; unused here
    thread = km.x280_thread("thread/adder.c", cluster=1, deadline=10.0)

    @km.handler
    def add(self, a: km.Tensor, b: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for x, y in zip(
            km.Stream(a.pages(first, width)), km.Stream(b.pages(first, width))
        ):
            results.push(plus(x, y))


def differences(got, want):
    """Wrong words: their count and the first eight (page, word)."""
    wrong = np.argwhere(got != want)
    return {"wrong": int(len(wrong)), "first": [[int(p), int(w)] for p, w in wrong[:8]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dump", default=None, help="an .npz of the outputs")
    args = parser.parse_args()
    x = np.random.default_rng(args.seed).integers(-50, 51, (TILES, WORDS))
    x = x.astype(np.float32)
    result = {"tiles": TILES}
    with km.Session(log=log) as session:
        tensors = {"x": session.tensor(x, "float32")}
        for name in "ds":
            tensors[name] = session.tensor(np.zeros_like(x), "float32")
        x_, d, s = (tensors[name] for name in "xds")
        doubler = Doubler(a=x_, out=d, form="runtime", session=session, log=log)
        adder = Adder(a=x_, b=d, out=s, thread_words=[WIDTH], form="runtime",
                    session=session, log=log)  # fmt: skip
        run = session.load({adder: (0, 4), doubler: (4, 4)}, deadline=10.0)
        thread = run.thread_results(timeout=15.0)[1]
        result["thread"] = {
            "state": thread.state,
            "error": thread.error,
            "code": thread.code,
            "microseconds": round(thread.microseconds, 2),
        }
        got = {
            name: raw.view(np.float32).reshape(-1, WORDS)
            for name, raw in zip("ds", session.read(d, s), strict=True)
        }
        run.stop()
        result["kernel"] = run.result()["error_name"]
    if args.dump:
        np.savez_compressed(args.dump, x=x, **got)
    result["checks"] = {
        "d": differences(got["d"], x + x),
        "s": differences(got["s"], x + (x + x)),
    }
    result["ok"] = bool(
        result["kernel"] == "none"
        and result["thread"]["code"] == 0
        and all(check["wrong"] == 0 for check in result["checks"].values())
    )
    result["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(power.clocked(result)), flush=True)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
