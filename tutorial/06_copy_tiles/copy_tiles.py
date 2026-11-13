"""Buffers and data: one core copies tiles from one DRAM tensor to another."""

from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import formats

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"

# --8<-- [start:host]
with km.Session() as session:
    x = np.random.default_rng(1).uniform(-1, 1, (64, 96)).astype(np.float32)
    x = formats.bf16_values(formats.bf16_bits(x))  # what bfloat16 holds of x
    src = session.tensor(x, "bfloat16", "tile")  # 2 x 3 tiles, in DRAM
    dst = session.empty(x.shape, "bfloat16", "tile")
    program = km.Program(
        "copy",
        kernels=KERNELS,
        buffers=[km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=4)],
        compile_args={"reader": src.accessor_args(), "writer": dst.accessor_args()},
        segments=KERNELS / "handlers.h",
    )
    run = session.load(program)
    args = [src.address, dst.address, src.pages]
    run.wait(run.issue("copy", team=(0, 1), args=args), timeout=10.0)
    run.stop()
    same = np.array_equal(dst.read(), x)
    print(f"{src.pages} tiles copied, identical: {same}")
    assert same
# --8<-- [end:host]
