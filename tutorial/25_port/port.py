"""A port: core 1 sends a tile from DRAM to core 0, which reports a number of it."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent

with km.Session() as session:
    numbers = np.arange(4 * 32 * 32, dtype=np.uint32).reshape(4 * 32, 32) + 1000
    source = session.tensor(numbers, "uint32", "tile")

    program = km.Program(
        "port",
        kernels=HERE / "kernels",
        handlers=HERE / "kernels" / "handlers.h",
        buffers=[
            km.Buffer(index=0, format="uint32", page_bytes=source.page_bytes, pages=1),
            km.Buffer(index=1, format="uint32", page_bytes=source.page_bytes, pages=1),
        ],
        compile_args={"reader": source.accessor_args()},
    )
    run = session.load(program)

    # --8<-- [start:send]
    for tile, position in [(1, 0), (3, 5)]:
        arguments = [source.address, tile, position]
        handle = run.submit(
            "send", group=(0, 2), args=arguments, exclusive=True, result=True
        )
        number = run.wait(handle, timeout=10.0).result
        print(
            f"core 1 sent tile {tile}; core 0 holds number {position} of it: {number}"
        )
        assert number == 1000 + tile * 1024 + position
    # --8<-- [end:send]
    run.stop()
