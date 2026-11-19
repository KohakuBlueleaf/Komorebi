"""Both data movers of each core on one stream: every core reads its tiles through a
shared stream under a pattern, checked word by word through a copy, then timed beside
pattern A (mover A alone). One JSON line. Example: python movement.py --pattern AAAB
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.bench import power

KERNELS = Path(__file__).resolve().parent / "kernels"
HEADER = KERNELS / "movement.h"
CONSTANTS = km.abi.constants(HEADER, known=km.abi.headers().constants)
PAGE_BYTES = 2048  # one bfloat16 tile
TARGET_MICROSECONDS = 2000.0  # the shorter run of a timed pair
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--pattern", nargs="+", default=["AB"], help="one or more patterns of A and B"
    )
    parser.add_argument("--cores", type=int, default=0, help="0: the session's ranks")
    parser.add_argument("--tiles", type=int, default=256, help="tiles a core reads")
    parser.add_argument("--block", type=int, default=16, help="tiles per block")
    parser.add_argument(
        "--check-only", action="store_true", help="check the copies, time nothing"
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    args.pattern = [km.Pattern(letters) for letters in args.pattern]
    return args


def program(tensors, block):
    """Input halves of four blocks, output halves of two; the input's and the
    output's tensor arguments."""
    return km.Program(
        "movement",
        kernels=KERNELS,  # movement.cpp for both movers, compute.cpp
        buffers=[
            km.BufferPair(
                CONSTANTS["kInputMoverA"],
                CONSTANTS["kInputMoverB"],
                "bfloat16",
                PAGE_BYTES,
                4 * block,
            ),
            km.BufferPair(
                CONSTANTS["kOutputMoverA"],
                CONSTANTS["kOutputMoverB"],
                "bfloat16",
                PAGE_BYTES,
                2 * block,
            ),
        ],
        compile_args={
            "movement": tensors["input"].tensor_args() + tensors["output"].tensor_args()
        },
        handlers=HEADER,
    )


def pages_of(rng, count):
    """`count` pages of bfloat16 bit patterns of finite normal numbers."""
    values = rng.standard_normal((count, PAGE_BYTES // 2)).astype(np.float32)
    return (values.view(np.uint32) >> 16).astype(np.uint16)


def submit(run, handler, cores, words):
    event = run.wait(run.submit(handler, (0, cores), words), timeout=30.0)
    if event.kind != "done":
        raise km.KomorebiError(f"{handler}: {event}")
    return event.microseconds


def check(session, run, tensors, data, cores, tiles, block, pattern):
    """COPY once and read the output back: the pages that differ from the input."""
    words = [
        tensors["input"].address,
        tensors["output"].address,
        tiles,
        block,
        pattern.word,
        1,
    ]
    submit(run, "copy", cores, words)
    wrong = 0
    for first, pages in session.read_chunks(tensors["output"]):
        got = pages.view(np.uint16).reshape(len(pages), -1)
        last = min(first + len(pages), cores * tiles)
        if first < last:
            wrong += int((got[: last - first] != data[first:last]).any(axis=1).sum())
        log(f"  {pattern}: output pages from {first} read back")
    return wrong


def rate(run, tensors, cores, tiles, block, pattern):
    """GB/s read from R and 4R repetitions, best of three; card microseconds."""
    words = [tensors["input"].address, 0, tiles, block, pattern.word]
    once = submit(run, "read", cores, words + [1])
    repetitions = max(2, int(TARGET_MICROSECONDS / max(once, 1.0)))
    best = None
    for _ in range(3):
        shorter = submit(run, "read", cores, words + [repetitions])
        longer = submit(run, "read", cores, words + [4 * repetitions])
        best = longer - shorter if best is None else min(best, longer - shorter)
    moved = cores * tiles * PAGE_BYTES * 3 * repetitions
    return round(moved / (best * 1e-6) / 1e9, 2), repetitions


def main():
    args = parse()
    patterns = [km.Pattern("A")] + [p for p in args.pattern if p != km.Pattern("A")]
    check_tiles = args.block + args.block // 2  # one whole block and a ragged one
    with km.Session() as session:
        grid = session.grid
        cores = args.cores or session.ranks  # the grid less a Tensix dispatcher's
        rng = np.random.default_rng(args.seed)
        log(f"device open, grid {grid}; {cores} cores, {args.tiles} tiles each")
        data = pages_of(rng, cores * args.tiles)
        tensors = {
            "input": session.tensor(data, "uint16"),
            "output": session.tensor(
                np.zeros((cores * check_tiles, data.shape[1]), np.uint16), "uint16"
            ),
        }
        log("tensors written")
        run = session.load(program(tensors, args.block))
        results = []
        try:
            for pattern in patterns:
                wrong = check(
                    session,
                    run,
                    tensors,
                    data[: cores * check_tiles],
                    cores,
                    check_tiles,
                    args.block,
                    pattern,
                )
                result = {"pattern": str(pattern), "pages_wrong": wrong}
                if not wrong and not args.check_only:
                    result["rate_gbs"], result["repetitions"] = rate(
                        run, tensors, cores, args.tiles, args.block, pattern
                    )
                log(f"pattern {pattern}: {result}")
                results.append(result)
        finally:
            run.stop()
            log("run stopped")
    for result in results:
        if "rate_gbs" in result:
            beside = result["rate_gbs"] / results[0]["rate_gbs"]
            print(
                f"pattern {result['pattern']:>8}: {result['rate_gbs']:7.2f} GB/s, "
                f"{beside:.2f} times pattern A"
            )
    exact = all(result["pages_wrong"] == 0 for result in results)
    print("every word of every copy is the input" if exact else "copies differ")
    line = {"cores": cores, "tiles": args.tiles, "block": args.block, "exact": exact}
    print(json.dumps(power.clocked(line | {"patterns": results})))
    raise SystemExit(0 if exact else 1)


if __name__ == "__main__":
    main()
