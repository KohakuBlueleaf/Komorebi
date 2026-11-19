"""Message handlers: a message climbs as one handler and lands as another."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
RUNTIME = HERE.parents[1] / "src" / "komorebi_native" / "kernels"
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
        "message_handlers",
        kernels={
            "reader": RUNTIME / "messages/reader.cpp",
            "compute": KERNELS / "compute.cpp",
            "writer": RUNTIME / "messages/writer.cpp",
        },
        handlers=KERNELS / "handlers.h",
    ).dynamic("climb", "land", message_bytes=64, max_live=4096, queue_messages=4096)
    # --8<-- [end:program]
    run = session.load(program)

    arguments = [MESSAGES, HOPS, notes.address]
    handle = run.submit("climb", group=(0, GROUP), args=arguments, exclusive=True)
    run.wait(handle, timeout=10.0)
    run.stop()

    kept = notes.read()[:RANKS, :3]
    for rank, (climbs, landed, total) in enumerate(kept.tolist()):
        print(f"rank {rank}: {climbs} climbs, {landed} landed, values add to {total}")

# --8<-- [start:replay]
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
# --8<-- [end:replay]
assert kept.tolist() == [list(row) for row in zip(climbs, landed, total)]
print(f"{sum(climbs)} climbs and {sum(landed)} landings, as the replay says")
