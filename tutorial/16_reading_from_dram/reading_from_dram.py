"""Reading from DRAM: a core copies a tile into its L1; the writer reads a number."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent

with km.Session() as session:
    # --8<-- [start:source]
    tile_numbers = np.arange(4, dtype=np.uint32).reshape(4, 1, 1)
    positions = np.arange(32 * 32, dtype=np.uint32).reshape(1, 32, 32)
    source = session.tensor(
        (tile_numbers * 1000 + positions).reshape(128, 32), "uint32", "tile"
    )
    # --8<-- [end:source]

    # --8<-- [start:program]
    program = km.Program(
        "reading_from_dram",
        kernels=HERE / "kernels",
        handlers=HERE / "kernels" / "handlers.h",
        buffers=[
            km.Buffer(index=0, format="uint32", page_bytes=source.page_bytes, pages=1)
        ],
        compile_args={"reader": source.tensor_args()},
    )
    # --8<-- [end:program]
    run = session.load(program)

    # --8<-- [start:fetch]
    for tile, position in [(2, 0), (2, 5), (3, 0)]:
        handle = run.submit(
            "fetch", group=(0, 1), args=[source.address, tile, position], result=True
        )
        number = run.wait(handle, timeout=10.0).result
        print(f"tile {tile}, number {position}: {number}")
        assert number == tile * 1000 + position
    # --8<-- [end:fetch]
    run.stop()
