"""Buffers and data: one core sums a DRAM tensor with plain loads from L1."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"

# --8<-- [start:host]
with km.Session() as session:
    x = np.random.default_rng(1).integers(0, 1 << 30, (8, 256), dtype=np.uint32)
    src = session.tensor(x, "uint32", "row")  # a page a row: 8 pages of 1,024 bytes
    program = km.Program(
        "sum",
        kernels=KERNELS,
        buffers=[km.Buffer(index=0, format="uint32", page_bytes=1024, pages=4)],
        compile_args={"reader": src.accessor_args()},
        handlers=KERNELS / "handlers.h",
    )
    run = session.load(program)
    handle = run.submit("sum", group=(0, 1), args=[src.address, src.pages], result=True)
    event = run.wait(handle, timeout=10.0)
    run.stop()
    want = int(x.sum(dtype=np.uint64) % (1 << 32))  # the card adds modulo 2^32
    print(f"sum of {x.size} words: card {event.result}, numpy {want}")
    assert event.result == want
# --8<-- [end:host]
