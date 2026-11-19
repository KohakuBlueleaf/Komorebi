"""Full buffers: a core may have only so many messages alive; send says no past that."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
RUNTIME = HERE.parents[1] / "src" / "komorebi_native" / "kernels"
GROUP = 8  # 6 cores take messages; the last 2 cores of a dynamic group keep counters
RANKS = GROUP - 2
QUOTA = 20  # sends each core may have alive
MAX_LIVE = RANKS * QUOTA  # the run's ranks share it; at least the load's ranks

with km.Session() as session:
    columns, rows = session.grid
    notes = session.tensor(
        np.zeros((columns * rows, 4), np.int32), "int32", shard_grid=session.grid
    )
    # --8<-- [start:program]
    program = km.Program(
        "full_buffers",
        kernels={
            "reader": RUNTIME / "messages/reader.cpp",
            "compute": KERNELS / "compute.cpp",
            "writer": RUNTIME / "messages/writer.cpp",
        },
        handlers=KERNELS / "handlers.h",
    ).dynamic("flood", message_bytes=64, max_live=MAX_LIVE, queue_messages=2048)
    # --8<-- [end:program]
    run = session.load(program)
    handle = run.submit("flood", group=(0, GROUP), args=[notes.address], exclusive=True)
    run.wait(handle, timeout=10.0)
    run.stop()

    kept = notes.read()[:RANKS, :3]
    for rank, (accepted, received, quota_left) in enumerate(kept.tolist()):
        print(
            f"rank {rank}: accepted {accepted} sends, received {received}, "
            f"quota left {quota_left}"
        )
    assert kept[:, 0].tolist() == [QUOTA] + [0] * (RANKS - 1)
    assert kept[:, 1].sum() == QUOTA
    assert (kept[:, 2] == QUOTA - kept[:, 0] + kept[:, 1]).all()
    print(f"rank 0 asked {2 * QUOTA} times and {QUOTA} sends were accepted")
