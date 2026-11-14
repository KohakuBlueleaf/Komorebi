"""Three kernels: the reader and the writer each loop over the tiles of one command."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
TILES = 8

with km.Session() as session:
    numbers = np.arange(TILES * 32 * 32, dtype=np.uint32).reshape(TILES * 32, 32)
    source = session.tensor(numbers, "uint32", "tile")
    destination = session.tensor(np.zeros_like(numbers), "uint32", "tile")

    program = km.Program(
        "three_kernels",
        kernels=HERE / "kernels",
        handlers=HERE / "kernels" / "handlers.h",
        buffers=[
            km.Buffer(index=0, format="uint32", page_bytes=source.page_bytes, pages=1)
        ],
        compile_args={
            "reader": source.tensor_args(),
            "writer": destination.tensor_args(),
        },
    )
    run = session.load(program)

    def copy(first_tile, tile_count):
        """Copies tile_count tiles in one command; returns its microseconds."""
        handle = run.submit(
            "copy",
            group=(0, 1),
            args=[
                source.address,
                first_tile,
                destination.address,
                first_tile,
                tile_count,
            ],
        )
        return run.wait(handle, timeout=10.0).microseconds

    # --8<-- [start:copy]
    one_by_one = sum(copy(tile, 1) for tile in range(TILES))
    all_at_once = copy(0, TILES)
    print(f"{TILES} commands of 1 tile: {one_by_one:6.1f} microseconds in all")
    print(f"1 command of {TILES} tiles:  {all_at_once:6.1f} microseconds")
    # --8<-- [end:copy]
    run.stop()

    (source_pages,) = session.read(source)
    (destination_pages,) = session.read(destination)
    assert (destination_pages == source_pages).all()
    print(f"all {TILES} tiles of the destination are the tiles of the source")
