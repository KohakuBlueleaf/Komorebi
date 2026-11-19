"""Handlers as overlays: each handler's body is loaded where its messages land."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
GROUP = 8  # 6 cores take messages; the last 2 cores of a dynamic group keep counters
RANKS = GROUP - 2
MESSAGES = 6
HOPS = 3

with km.Session() as session:
    columns, rows = session.grid
    notes = session.tensor(
        np.zeros((columns * rows, 4), np.int32), "int32", shard_grid=session.grid
    )
    # --8<-- [start:program]
    program = km.Program(
        "handlers_as_overlays",
        kernels=None,  # Komorebi's reader, compute and writer
        args=lambda core: ([], [], [notes.address]),  # compute argument 0: the notes
        handlers=KERNELS / "handlers.h",
    ).dynamic(
        "climb",
        "land",
        parts=KERNELS,  # climb.cpp and land.cpp
        message_bytes=64,
        max_live=4096,
        queue_messages=4096,
    )
    # --8<-- [end:program]
    run = session.load(program)

    handle = run.submit(
        "climb", group=(0, GROUP), args=[MESSAGES, HOPS], exclusive=True
    )
    run.wait(handle, timeout=10.0)
    run.stop()

    kept = notes.read()[:RANKS, :3]
    for rank, (climbs, landed, total) in enumerate(kept.tolist()):
        print(f"rank {rank}: {climbs} climbs, {landed} landed, values add to {total}")

climbs = [0] * RANKS
landed = [0] * RANKS
total = [0] * RANKS
for identifier in range(MESSAGES):
    rank, value, hops_left = identifier % RANKS, identifier + 1, HOPS
    while True:
        climbs[rank] += 1
        value += rank
        if hops_left == 0:
            landed[rank] += 1  # the same core, as the handler land
            total[rank] += value
            break
        hops_left -= 1
        rank = (rank + 1) % RANKS
assert kept.tolist() == [list(row) for row in zip(climbs, landed, total)]
print(f"{sum(climbs)} climbs and {sum(landed)} landings, as the replay says")
