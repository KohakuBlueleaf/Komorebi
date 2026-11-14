"""Reduce: eight cores each hold a tile; the total arrives at core 0, in two ways."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
CORES = 8

with km.Session() as session:
    numbers = np.random.default_rng(1).integers(-8, 9, (CORES * 32, 32))
    source = session.tensor(numbers, "bfloat16", "tile")
    destination = session.tensor(np.zeros((2 * 32, 32)), "bfloat16", "tile")

    tile = dict(format="bfloat16", page_bytes=source.page_bytes)
    program = km.Program(
        "reduce",
        kernels=HERE / "kernels",
        handlers=HERE / "kernels" / "handlers.h",
        buffers=[
            km.Buffer(index=0, pages=1, **tile),
            km.Buffer(index=1, pages=8, **tile),
            km.Buffer(index=16, pages=1, **tile),
        ],
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        compile_args={
            "reader": source.tensor_args(),
            "writer": destination.tensor_args(),
        },
    )
    run = session.load(program)

    # --8<-- [start:reduce]
    for form in (0, 1):
        arguments = [source.address, 0, destination.address, form, form]
        handle = run.submit("reduce", group=(0, CORES), args=arguments, exclusive=True)
        microseconds = run.wait(handle, timeout=10.0).microseconds
        print(
            f"form {form}: {CORES} tiles added into one in "
            f"{microseconds:5.1f} microseconds"
        )
    # --8<-- [end:reduce]
    run.stop()

    total = destination.read()
    expected = numbers.reshape(CORES, 32, 32).sum(axis=0)
    assert (total[:32] == expected).all() and (total[32:] == expected).all()
    print("both forms: the destination tile is the sum of the 8 tiles")
