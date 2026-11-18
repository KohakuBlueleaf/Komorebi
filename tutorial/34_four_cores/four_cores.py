"""Four cores: an X280 thread adds up numbers on one core, then on all four."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent

N = 4_000_000

program = km.Program("four_cores", kernels=HERE / "kernels")
program.thread(
    cluster=1,
    sources=[HERE / "thread" / "four.c"],
    regions={"unused": 64},
    words=[N],
)

with km.Session() as session:
    run = session.load(program)
    result = run.thread_result(timeout=10.0)
    run.stop()

if not result.words:  # a run without a card: its reads give zeros
    raise SystemExit("four_cores.py: no result words (reads were zeros)")

# --8<-- [start:check]
numbers = np.arange(N, dtype=np.uint32)
numbers ^= numbers >> 15
numbers *= np.uint32(2246822519)
numbers ^= numbers >> 13
expected = int(numbers.sum(dtype=np.uint64)) % 2**32
alone, together, one_us, four_us = result.words
print(f"scrambled numbers 0 to {N - 1}, modulo 2**32: {expected}")
print(f"one core:   {alone} in {one_us} microseconds")
print(f"four cores: {together} in {four_us} microseconds")
print(f"four cores took {one_us / four_us:.1f} times less time")
# --8<-- [end:check]

assert result.code == 0 and alone == together == expected
