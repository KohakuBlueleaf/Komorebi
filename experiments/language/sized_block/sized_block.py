"""A stream's block given at each submit (the internal hook front._sized(8), no
public name yet): one kernel run at blocks 8, 4, 2 and 1 over 20 tiles a core (8
leaves a short last block), each word against numpy, beside block 8 built in."""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi import power
from komorebi.dsl.front import _sized

WORDS = 1024  # float32 words a page: one tile's
CORES, TILES = 8, 20  # the cores and each core's tiles
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function
def twice(a):
    return a + a


class Sized(km.Program):
    """out = a + a, `tiles` tiles a core from rank x tiles, in blocks of `block`."""

    @km.handler
    def scale(self, a: km.Tensor, out: km.Tensor, tiles: int,
              block: int = _sized(8)):  # fmt: skip
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=block)
        for x in km.Stream(a.pages(first, tiles), block=block):
            results.push(twice(x))


class Fixed(km.Program):
    """The same with block 8 built in."""

    @km.handler
    def scale(self, a: km.Tensor, out: km.Tensor, tiles: int):
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=8)
        for x in km.Stream(a.pages(first, tiles), block=8):
            results.push(twice(x))


def differences(got, want):
    """Wrong words: their count and the first eight (page, word)."""
    wrong = np.argwhere(got != want)
    return {"wrong": int(len(wrong)), "first": [[int(p), int(w)] for p, w in wrong[:8]]}


def timed(run, handler, args, runs):
    """The dispatcher's median microseconds of `runs` commands."""
    group = (0, CORES)
    times = [run.wait(run.submit(handler, group, args=args)).microseconds
             for _ in range(runs)]  # fmt: skip
    return round(float(np.median(times)), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--dump", default=None, help="an .npz of the outputs")
    args = parser.parse_args()
    x = np.random.default_rng(1).integers(-50, 51, (CORES * TILES, WORDS))
    x = x.astype(np.float32)
    result, dumped = {"cores": CORES, "tiles": TILES, "blocks": {}}, {"x": x}
    with km.Session(log=log) as session:
        a = session.tensor(x, "float32")
        out = session.tensor(np.zeros_like(x), "float32")
        for cls in (Sized, Fixed):
            run = session.load(cls(a=a, out=out, session=session, log=log))
            for block in (8, 4, 2, 1) if cls is Sized else (8,):
                session.write(out, np.zeros_like(x).view(np.uint8).reshape(-1, 4096))
                words = (a, out, TILES, block) if cls is Sized else (a, out, TILES)
                run.wait(run.submit(cls.scale, (0, CORES), args=words))
                got = session.read(out)[0].view(np.float32).reshape(-1, WORDS)
                dumped[f"{cls.__name__}_{block}"] = got
                result["blocks"][f"{cls.__name__} {block}"] = {
                    **differences(got, x + x),
                    "us": timed(run, cls.scale, words, args.runs),
                }
            run.stop()
            result[f"kernel {cls.__name__}"] = run.result()["error_name"]
    if args.dump:
        np.savez_compressed(args.dump, **dumped)
    result["ok"] = bool(
        all(v == "none" for k, v in result.items() if k.startswith("kernel"))
        and all(one["wrong"] == 0 for one in result["blocks"].values())
    )
    result["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(power.clocked(result)), flush=True)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
