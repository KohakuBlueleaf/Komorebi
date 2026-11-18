"""Twelve cores: one thread on three clusters adds up numbers on 12 X280 cores."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent

N = 12_000_000

program = km.Program("twelve_cores", kernels=HERE / "kernels")
# --8<-- [start:thread]
program.thread(
    clusters=(1, 2, 3),
    sources=[HERE / "thread" / "twelve.c"],
    regions={"pieces": 256},
    words=[N],
)
# --8<-- [end:thread]

with km.Session() as session:
    run = session.load(program)
    results = run.thread_results(timeout=20.0)
    run.stop()
    # --8<-- [start:read]
    pieces = []
    for thread in run.threads.values():
        lines = thread.read("pieces").view(np.uint32).reshape(4, 16)
        pieces += lines[:, 0].tolist()
    # --8<-- [end:read]

if not results[1].words:  # a run without a card: its reads give zeros
    raise SystemExit("twelve_cores.py: no result words (reads were zeros)")

total = sum(pieces) % 2**32
numbers = np.arange(N, dtype=np.uint32)
numbers ^= numbers >> 15
numbers *= np.uint32(2246822519)
numbers ^= numbers >> 13
expected = int(numbers.sum(dtype=np.uint64)) % 2**32
print(f"twelve pieces: {pieces}")
print(f"their sum, modulo 2**32: {total}")
print(f"the scrambled numbers 0 to {N - 1}:    {expected}")
first, second = results[1].words
print(f"the first call took {first} microseconds, the second {second}")

assert all(result.code == 0 for result in results.values())
assert total == expected
