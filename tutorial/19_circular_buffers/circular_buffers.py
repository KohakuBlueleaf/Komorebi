"""Circular buffers: the same copy with room for 1, 2 and 8 tiles in the buffer."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
TILES = 64

with km.Session() as session:
    numbers = np.arange(TILES * 32 * 32, dtype=np.uint32).reshape(TILES * 32, 32)
    source = session.tensor(numbers, "uint32", "tile")
    destination = session.tensor(np.zeros_like(numbers), "uint32", "tile")

    for pages in (1, 2, 8):
        destination.write(np.zeros_like(numbers))
        # --8<-- [start:buffer]
        program = km.Program(
            "circular_buffers",
            kernels=HERE / "kernels",
            handlers=HERE / "kernels" / "handlers.h",
            buffers=[
                km.Buffer(
                    index=0, format="uint32", page_bytes=source.page_bytes, pages=pages
                )
            ],
            compile_args={
                "reader": source.accessor_args(),
                "writer": destination.accessor_args(),
            },
        )
        # --8<-- [end:buffer]
        run = session.load(program)
        arguments = [source.address, 0, destination.address, 0, TILES]
        handle = run.submit("copy", group=(0, 1), args=arguments)
        microseconds = run.wait(handle, timeout=10.0).microseconds
        run.stop()
        print(f"buffer pages {pages}: {TILES} tiles copied in {microseconds:5.1f} us")

        (source_pages,) = session.read(source)
        (destination_pages,) = session.read(destination)
        assert (destination_pages == source_pages).all()
