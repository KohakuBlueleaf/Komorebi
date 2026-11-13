"""A whole map: a group of cores adds two tensors, each core its share of the tiles."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
TILES = 60

with km.Session() as session:
    # --8<-- [start:tensors]
    rng = np.random.default_rng(1)
    first = rng.integers(-8, 9, (TILES * 32, 32))
    second = rng.integers(-8, 9, (TILES * 32, 32))
    source = session.tensor(np.concatenate([first, second]), "bfloat16", "tile")
    destination = session.tensor(np.zeros((3 * TILES * 32, 32)), "bfloat16", "tile")
    # --8<-- [end:tensors]

    tile = dict(format="bfloat16", page_bytes=source.page_bytes, pages=2)
    program = km.Program(
        "a_whole_map",
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
    run = session.load(program)

    # --8<-- [start:map]
    for round_number, count in enumerate((1, 4, 12)):
        arguments = [source.address, TILES, destination.address, round_number * TILES]
        handle = run.submit("map", group=(0, count), args=arguments)
        microseconds = run.wait(handle, timeout=10.0).microseconds
        print(f"group of {count:2}: {TILES} tiles added in {microseconds:5.1f} us")
    # --8<-- [end:map]
    run.stop()

    total = destination.read()
    for round_number in range(3):
        rows = slice(round_number * TILES * 32, (round_number + 1) * TILES * 32)
        assert (total[rows] == first + second).all()
    print("all three results are the sum of the two tensors")
