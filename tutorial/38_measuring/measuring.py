"""Measuring: how long a command takes, by the card's clock and by the host's."""

import statistics
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
TILES = 64
REPEATS = 20

with km.Session() as session:
    numbers = np.arange(TILES * 32 * 32, dtype=np.uint32).reshape(TILES * 32, 32)
    source = session.tensor(numbers, "uint32", "tile")
    destination = session.tensor(np.zeros_like(numbers), "uint32", "tile")
    program = km.Program(
        "measuring",
        kernels=HERE / "kernels",
        handlers=HERE / "kernels" / "handlers.h",
        buffers=[
            km.Buffer(index=0, format="uint32", page_bytes=source.page_bytes, pages=8)
        ],
        compile_args={
            "reader": source.tensor_args(),
            "writer": destination.tensor_args(),
        },
    )
    run = session.load(program)
    arguments = [source.address, 0, destination.address, 0, TILES]

    # --8<-- [start:loop]
    card = []
    host = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        handle = run.submit("copy", group=(0, 1), args=arguments)
        event = run.wait(handle, timeout=10.0)
        host.append((time.perf_counter() - started) * 1e6)
        card.append(event.microseconds)
    # --8<-- [end:loop]
    run.stop()

    (source_pages,) = session.read(source)
    (destination_pages,) = session.read(destination)
    assert (destination_pages == source_pages).all()

# --8<-- [start:report]
print(f"{REPEATS} copies of {TILES} tiles")
print(
    f"card clock: first {card[0]:.1f}, middle {statistics.median(card):.1f}, "
    f"smallest {min(card):.1f}, largest {max(card):.1f} microseconds"
)
print(
    f"host clock: first {host[0]:.1f}, middle {statistics.median(host):.1f}, "
    f"smallest {min(host):.1f}, largest {max(host):.1f} microseconds"
)
# --8<-- [end:report]
