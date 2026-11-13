"""Both movers: one core copies 256 tiles with one data-movement CPU, then with both."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
TILES = 256

with km.Session() as session:
    numbers = np.arange(TILES * 32 * 32, dtype=np.uint32).reshape(TILES * 32, 32)
    source = session.tensor(numbers, "uint32", "tile")
    destination = session.tensor(
        np.zeros((2 * TILES * 32, 32), np.uint32), "uint32", "tile"
    )

    # --8<-- [start:program]
    program = km.Program(
        "both_movers",
        kernels=HERE / "kernels",  # movement.cpp and compute.cpp
        handlers=HERE / "kernels" / "handlers.h",
        buffers=[km.BufferPair(0, 1, "uint32", page_bytes=source.page_bytes, pages=16)],
        compile_args={"movement": source.accessor_args()},
    )
    # --8<-- [end:program]
    run = session.load(program)

    # --8<-- [start:copy]
    for round_number, letters in enumerate(("A", "AB")):
        pattern = km.Pattern(letters)
        first = round_number * TILES
        arguments = [source.address, 0, destination.address, first, TILES, pattern.word]
        handle = run.submit("copy", group=(0, 1), args=arguments)
        microseconds = run.wait(handle, timeout=10.0).microseconds
        print(
            f"pattern {letters:2}: {TILES} tiles copied in "
            f"{microseconds:6.1f} microseconds"
        )
    # --8<-- [end:copy]
    run.stop()

    (source_pages,) = session.read(source)
    (destination_pages,) = session.read(destination)
    for round_number in range(2):
        copied = destination_pages[round_number * TILES : (round_number + 1) * TILES]
        assert (copied == source_pages).all()
    print("both copies hold every tile of the source")
