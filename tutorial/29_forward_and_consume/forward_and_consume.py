"""Forward and consume: messages hop from core to core, each hop chosen by its value."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
RUNTIME = HERE.parents[1] / "src" / "komorebi_native" / "kernels"
GROUP = 8  # 6 cores take messages; the last 2 cores of a dynamic group keep counters
RANKS = GROUP - 2
MESSAGES = 6
HOPS = 4

with km.Session() as session:
    columns, rows = session.grid
    notes = session.tensor(
        np.zeros((columns * rows, 4), np.int32), "int32", shard_grid=session.grid
    )
    program = km.Program(
        "forward_and_consume",
        kernels={
            "reader": RUNTIME / "messages/reader.cpp",
            "compute": KERNELS / "compute.cpp",
            "writer": RUNTIME / "messages/writer.cpp",
        },
        handlers=KERNELS / "handlers.h",
    ).dynamic("wander", message_bytes=64, max_live=4096, queue_messages=4096)
    run = session.load(program)

    # --8<-- [start:run]
    arguments = [MESSAGES, HOPS, notes.address]
    handle = run.submit("wander", group=(0, GROUP), args=arguments, exclusive=True)
    run.wait(handle, timeout=10.0)
    run.stop()
    # --8<-- [end:run]

    kept = notes.read()[:RANKS, :2]
    for rank, (visits, ended_here) in enumerate(kept.tolist()):
        print(f"rank {rank}: {visits} visits, {ended_here} messages ended here")

# --8<-- [start:replay]
visits = [0] * RANKS
ended_here = [0] * RANKS
for identifier in range(MESSAGES):
    rank, value, hops_left = identifier % RANKS, identifier + 1, HOPS
    while True:
        visits[rank] += 1
        value = (value * 5 + 1) % 97
        if hops_left == 0:
            ended_here[rank] += 1
            break
        hops_left -= 1
        rank = value % RANKS
# --8<-- [end:replay]
assert kept.tolist() == [[v, e] for v, e in zip(visits, ended_here)]
print(
    f"{sum(visits)} visits in all, {sum(ended_here)} messages ended, as the replay says"
)
