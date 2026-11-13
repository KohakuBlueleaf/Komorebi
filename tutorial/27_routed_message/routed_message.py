"""A routed message: rank 0 sends twelve messages; each goes where its value says."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
RUNTIME = HERE.parents[1] / "src" / "komorebi_native" / "kernels"
GROUP = 8  # 6 cores take messages; the last 2 cores of a dynamic group keep counters
MESSAGES = 12

with km.Session() as session:
    notes = session.tensor(
        np.zeros((120, 4), np.int32), "int32", shard_grid=session.grid
    )

    # --8<-- [start:program]
    program = km.Program(
        "routed_message",
        kernels={
            "reader": RUNTIME / "messages_reader.cpp",
            "compute": KERNELS / "compute.cpp",
            "writer": RUNTIME / "messages_writer.cpp",
        },
        handlers=KERNELS / "handlers.h",
    ).dynamic("route", message_bytes=64, max_live=4096, queue_messages=4096)
    # --8<-- [end:program]
    run = session.load(program)

    # --8<-- [start:route]
    handle = run.submit(
        "route", group=(0, GROUP), args=[MESSAGES, notes.address], exclusive=True
    )
    run.wait(handle, timeout=10.0)
    run.stop()
    # --8<-- [end:route]

    kept = notes.read()[: GROUP - 2, :2]
    for rank, (received, total) in enumerate(kept.tolist()):
        print(f"rank {rank}: received {received} messages, values add up to {total}")
    values = np.arange(MESSAGES)
    destinations = values * values % (GROUP - 2)
    for rank in range(GROUP - 2):
        assert kept[rank].tolist() == [
            int((destinations == rank).sum()),
            int(values[destinations == rank].sum()),
        ]
