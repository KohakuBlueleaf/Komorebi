"""Partitions: ten items shared by four cores, one contiguous piece each."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

program = km.Program(
    "partitions",
    kernels=HERE / "kernels",
    handlers=HERE / "kernels" / "handlers.h",
)

ITEMS = 10
CORES = 4

with km.Session() as session:
    run = session.load(program)
    covered = []
    # --8<-- [start:shares]
    for rank in range(CORES):
        handle = run.submit("share", group=(0, CORES), args=[ITEMS, rank], result=True)
        word = run.wait(handle, timeout=10.0).result
        first_item, item_count = word >> 16, word & 0xFFFF
        covered += range(first_item, first_item + item_count)
        print(f"rank {rank} takes {item_count} items, from item {first_item}")
    # --8<-- [end:shares]
    run.stop()

assert covered == list(range(ITEMS))
print("together they cover items 0 to 9, each exactly once")
