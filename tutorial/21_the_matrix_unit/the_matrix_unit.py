"""The matrix unit: the compute kernel multiplies two tiles as matrices."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent

with km.Session() as session:
    # --8<-- [start:tensors]
    numbers = np.random.default_rng(1).integers(0, 3, (4 * 32, 32))
    source = session.tensor(numbers, "bfloat16", "tile")
    destination = session.tensor(np.zeros((32, 32)), "bfloat16", "tile")
    # --8<-- [end:tensors]

    # --8<-- [start:program]
    tile = dict(format="bfloat16", page_bytes=source.page_bytes, pages=1)
    program = km.Program(
        "the_matrix_unit",
        kernels=HERE / "kernels",
        handlers=HERE / "kernels" / "handlers.h",
        buffers=[
            km.Buffer(index=0, **tile),
            km.Buffer(index=1, **tile),
            km.Buffer(index=16, **tile),
        ],
        compute={"math_fidelity": "HiFi4"},
        compile_args={
            "reader": source.tensor_args(),
            "writer": destination.tensor_args(),
        },
    )
    # --8<-- [end:program]
    run = session.load(program)

    # --8<-- [start:add]
    arguments = [source.address, 1, 2, destination.address, 0]
    run.wait(run.submit("multiply", group=(0, 1), args=arguments), timeout=10.0)
    run.stop()
    # --8<-- [end:add]

    # --8<-- [start:check]
    total = destination.read()
    expected = numbers[32:64] @ numbers[64:96]
    print("tile 1, first row:      ", numbers[32, :8].tolist(), "...")
    print("tile 2, first column:   ", numbers[64:72, 0].tolist(), "...")
    print("product, first number:  ", int(total[0, 0]))
    assert (total == expected).all()
    print("every number of the product is the matrix product of tile 1 and tile 2")
    # --8<-- [end:check]
