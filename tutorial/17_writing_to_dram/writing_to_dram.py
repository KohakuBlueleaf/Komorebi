"""Writing to DRAM: a core copies tiles from one tensor to another."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent

with km.Session() as session:
    numbers = np.arange(4 * 32 * 32, dtype=np.uint32).reshape(4 * 32, 32) + 1000
    source = session.tensor(numbers, "uint32", "tile")
    # --8<-- [start:destination]
    destination = session.tensor(np.zeros((64, 32), np.uint32), "uint32", "tile")
    # --8<-- [end:destination]

    program = km.Program(
        "writing_to_dram",
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

    # --8<-- [start:copy]
    for source_tile, destination_tile in [(2, 0), (3, 1)]:
        handle = run.submit(
            "copy",
            group=(0, 1),
            args=[source.address, source_tile, destination.address, destination_tile],
        )
        run.wait(handle, timeout=10.0)
    # --8<-- [end:copy]
    run.stop()

    # --8<-- [start:check]
    (source_pages,) = session.read(source)
    (destination_pages,) = session.read(destination)
    first = destination_pages.view(np.uint32)[:, 0].tolist()
    print(f"the first number of each destination tile: {first}")
    assert (destination_pages[0] == source_pages[2]).all()
    assert (destination_pages[1] == source_pages[3]).all()
    print("tiles 2 and 3 of the source are tiles 0 and 1 of the destination")
    # --8<-- [end:check]
