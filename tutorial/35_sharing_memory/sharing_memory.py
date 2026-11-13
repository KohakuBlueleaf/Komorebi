"""Sharing memory: Python fills one region, the thread another; Python reads both."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent

numbers = np.arange(1024, dtype=np.uint32)

program = km.Program("sharing_memory", kernels=HERE / "kernels")
# --8<-- [start:regions]
program.thread(
    cluster=1,
    sources=[HERE / "thread" / "triple.c"],
    regions={"numbers": numbers, "tripled": 4096},
)
# --8<-- [end:regions]

with km.Session() as session:
    run = session.load(program)
    result = run.thread_result(timeout=10.0)
    run.stop()
    # --8<-- [start:read]
    tripled = run.thread.read("tripled").view(np.uint32)
    # --8<-- [end:read]

print(f"the thread went through {result.words[0]} numbers")
print(f"first five tripled: {tripled[:5].tolist()}")
print(f"last five tripled:  {tripled[-5:].tolist()}")

assert result.code == 0
assert (tripled == 3 * numbers).all()
