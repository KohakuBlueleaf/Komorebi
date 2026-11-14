"""Finding a bug: two copies that go wrong, and what the card says about each."""

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
        "finding_a_bug",
        kernels=HERE / "kernels",
        handlers=HERE / "kernels" / "handlers.h",
        buffers=[
            km.Buffer(index=0, format="uint32", page_bytes=source.page_bytes, pages=2)
        ],
        compile_args={
            "reader": source.tensor_args(),
            "writer": destination.tensor_args(),
        },
    )
    for bug in (1, 2, 0):
        run = session.load(program, deadline=1.0)
        arguments = [source.address, 0, destination.address, 0, TILES, bug]
        handle = run.submit("copy", group=(0, 1), args=arguments)
        # --8<-- [start:wait]
        try:
            event = run.wait(handle, timeout=10.0)
        except km.CommandTimeout as error:
            print(f"with bug {bug}: {error}")
            run.stop()  # the card gave the command up; this run is no longer needed
            continue
        # --8<-- [end:wait]
        run.stop()
        print(f"with bug {bug}: the command is {event.kind}")
        (source_pages,) = session.read(source)
        (destination_pages,) = session.read(destination)
        assert (destination_pages == source_pages).all()
