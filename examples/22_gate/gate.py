"""The gate of a SwiGLU layer, out = silu(a) * b, over two tensors in DRAM on a group
of cores, checked against numpy in float64. One JSON line.
Example: python gate.py --cores 12
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.bench import power

KERNELS = Path(__file__).resolve().parent / "kernels"
HEADER = KERNELS / "gate.h"
CONSTANTS = km.abi.constants(HEADER, known=km.abi.headers().constants)
PAGE_BYTES = 2048  # one bfloat16 tile
SMALL = 2.0**-4  # values below this are judged by absolute error only
BOUND = 2.0**-6  # relative error allowed: bfloat16 holds 8 bits, the vector unit's silu
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pattern", default="AAAB", help="a pattern of A and B")
    parser.add_argument("--cores", type=int, default=0, help="0: the session's ranks")
    parser.add_argument("--tiles", type=int, default=64, help="tiles a core takes")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    args.pattern = km.Pattern(args.pattern)
    if args.tiles % CONSTANTS["kBlock"]:
        parser.error(f"--tiles must be a multiple of {CONSTANTS['kBlock']}")
    return args


def program(tensors):
    """Three pairs of buffers, a, b and out: four blocks of tiles in each half; the
    tensor arguments of a, b and out."""
    pages = 4 * CONSTANTS["kBlock"]
    return km.Program(
        "gate",
        kernels=KERNELS,  # movement.cpp for both movers, compute.cpp
        buffers=[
            km.BufferPair(
                CONSTANTS["kAMoverA"],
                CONSTANTS["kAMoverB"],
                "bfloat16",
                PAGE_BYTES,
                pages,
            ),
            km.BufferPair(
                CONSTANTS["kBMoverA"],
                CONSTANTS["kBMoverB"],
                "bfloat16",
                PAGE_BYTES,
                pages,
            ),
            km.BufferPair(
                CONSTANTS["kOutMoverA"],
                CONSTANTS["kOutMoverB"],
                "bfloat16",
                PAGE_BYTES,
                pages,
            ),
        ],
        compile_args={
            "movement": [word for tensor in tensors for word in tensor.tensor_args()]
        },
        handlers=HEADER,
    )


def pages_of(rng, count):
    """`count` pages of bfloat16 bit patterns of normal numbers."""
    values = rng.standard_normal((count, PAGE_BYTES // 2)).astype(np.float32)
    return (values.view(np.uint32) >> 16).astype(np.uint16)


def floats(bits):
    """The values of bfloat16 bit patterns, in float64."""
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def reference(a, b):
    """silu(a) * b in float64 from the bfloat16 inputs."""
    return a / (1.0 + np.exp(-a)) * b


def main():
    args = parse()
    with km.Session() as session:
        grid = session.grid
        cores = args.cores or session.ranks  # the grid less a Tensix dispatcher's
        pages = cores * args.tiles
        rng = np.random.default_rng(args.seed)
        log(f"device open, grid {grid}; {cores} cores, {args.tiles} tiles each")
        a_bits, b_bits = pages_of(rng, pages), pages_of(rng, pages)
        a, b = session.tensor(a_bits, "uint16"), session.tensor(b_bits, "uint16")
        out = session.tensor(np.zeros_like(a_bits), "uint16")
        log("tensors written")
        run = session.load(program((a, b, out)))
        try:
            words = [a.address, b.address, out.address, args.tiles, args.pattern.word]
            event = run.wait(run.submit("gate", (0, cores), words), timeout=30.0)
            log(f"command: {event}")
            if event.kind != "done":
                raise km.KomorebiError(f"gate: {event}")
            wanted = reference(floats(a_bits), floats(b_bits))
            error = scale = relative = 0.0
            wrong = 0
            for first, got in session.read_chunks(out):
                got = floats(got.view(np.uint16).reshape(len(got), -1))
                want = wanted[first : first + len(got)]
                difference = np.abs(got - want)
                error = max(error, float(difference.max()))
                scale = max(scale, float(np.abs(want).max()))
                large = np.abs(want) >= SMALL  # relative error where it means something
                relative = max(
                    relative, float((difference / np.abs(want))[large].max())
                )
                wrong += int((difference > BOUND * np.abs(want) + 2.0**-10).sum())
                log(f"  output pages from {first} read back")
        finally:
            run.stop()
            log("run stopped")
    ok = wrong == 0
    print(f"largest error {error:.3e} of values up to {scale:.2f}")
    print(f"largest relative error {relative:.3e}, where the value is 2^-4 or more")
    print(f"{wrong} values outside {BOUND:.3e} of their size plus 2^-10")
    line = {"cores": cores, "tiles": args.tiles, "pattern": str(args.pattern)}
    line |= {
        "max_error": error,
        "max_relative": relative,
        "wrong": wrong,
        "microseconds": event.microseconds,
    }
    print(json.dumps(power.clocked(line)))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
