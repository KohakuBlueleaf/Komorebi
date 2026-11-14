"""Your own kernel: the sum of the squares of 48 tiles, checked in numpy."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
TILES = 48

with km.Session() as session:
    # --8<-- [start:tensors]
    numbers = np.random.default_rng(1).integers(-2, 3, (TILES * 32, 32))
    source = session.tensor(numbers, "bfloat16", "tile")
    destinations = [
        session.tensor(np.zeros((32, 32)), "bfloat16", "tile") for _ in range(2)
    ]
    # --8<-- [end:tensors]

    # --8<-- [start:program]
    tile = dict(format="bfloat16", page_bytes=source.page_bytes)
    program = km.Program(
        "your_own_kernel",
        kernels=HERE / "kernels",
        handlers=HERE / "kernels" / "handlers.h",
        buffers=[
            km.Buffer(index=0, pages=2, **tile),
            km.Buffer(index=1, pages=8, **tile),
            km.Buffer(index=16, pages=1, **tile),
        ],
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        compile_args={
            "reader": source.tensor_args(),
            "writer": destinations[0].tensor_args(),
        },
    )
    # --8<-- [end:program]
    run = session.load(program)

    # --8<-- [start:sum]
    times = []
    for cores, destination in zip((1, 8), destinations):
        arguments = [source.address, 0, destination.address, 0, 0, TILES]
        handle = run.submit(
            "sum_squares", group=(0, cores), args=arguments, exclusive=True
        )
        times.append(run.wait(handle, timeout=10.0).microseconds)
    run.stop()

    expected = (numbers**2).reshape(TILES, 32, 32).sum(axis=0)
    for cores, microseconds, destination in zip((1, 8), times, destinations):
        print(
            f"{cores} cores: squares of {TILES} tiles added in {microseconds:5.1f} us"
        )
        total = destination.read()
        wrong = total != expected
        if wrong.any():
            print(
                f"{wrong.sum()} of {wrong.size} numbers differ from numpy; the first:"
            )
            print(f"  the card made {total[wrong][0]}, numpy made {expected[wrong][0]}")
            raise SystemExit(1)
    # --8<-- [end:sum]
    print("both sums are the numpy sum of the squares of the 48 tiles")
