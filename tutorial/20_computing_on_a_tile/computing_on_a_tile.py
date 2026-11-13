"""Computing on a tile: the compute kernel adds two tiles."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent

with km.Session() as session:
    # --8<-- [start:tensors]
    numbers = np.random.default_rng(1).integers(-8, 9, (4 * 32, 32))
    source = session.tensor(numbers, "bfloat16", "tile")
    destination = session.tensor(np.zeros((32, 32)), "bfloat16", "tile")
    # --8<-- [end:tensors]

    # --8<-- [start:program]
    tile = dict(format="bfloat16", page_bytes=source.page_bytes, pages=1)
    program = km.Program(
        "computing_on_a_tile",
        kernels=HERE / "kernels",
        handlers=HERE / "kernels" / "handlers.h",
        buffers=[
            km.Buffer(index=0, **tile),
            km.Buffer(index=1, **tile),
            km.Buffer(index=16, **tile),
        ],
        compile_args={
            "reader": source.accessor_args(),
            "writer": destination.accessor_args(),
        },
    )
    # --8<-- [end:program]
    run = session.load(program)

    # --8<-- [start:add]
    arguments = [source.address, 1, 2, destination.address, 0]
    run.wait(run.submit("add", group=(0, 1), args=arguments), timeout=10.0)
    run.stop()
    # --8<-- [end:add]

    # --8<-- [start:check]
    total = destination.read()
    expected = numbers[32:64] + numbers[64:96]
    print("tile 1, first row:      ", numbers[32, :8].tolist(), "...")
    print("tile 2, first row:      ", numbers[64, :8].tolist(), "...")
    print("destination, first row: ", total[0, :8].astype(int).tolist(), "...")
    assert (total == expected).all()
    print("every number of the destination is the sum of tile 1 and tile 2")
    # --8<-- [end:check]
