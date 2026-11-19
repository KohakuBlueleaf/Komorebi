"""Why cores talk: each core keeps a number in its own L1; only Python reads them."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent

with km.Session() as session:
    # --8<-- [start:notes]
    columns, rows = session.grid
    notes = session.tensor(
        np.zeros((columns * rows, 4), np.int32), "int32", shard_grid=session.grid
    )
    # --8<-- [end:notes]
    program = km.Program(
        "why_cores_talk",
        kernels=HERE / "kernels",
        handlers=HERE / "kernels" / "handlers.h",
    )
    run = session.load(program)

    # --8<-- [start:remember]
    handle = run.submit("remember", group=(0, 6), args=[notes.address])
    run.wait(handle, timeout=10.0)
    run.stop()
    # --8<-- [end:remember]

    # --8<-- [start:read]
    kept = notes.read()[:, 0]
    print("what cores 0 to 7 hold in their L1:", kept[:8].tolist())
    assert kept[:6].tolist() == [0, 1, 4, 9, 16, 25]
    assert kept[6:].tolist() == [0] * (len(kept) - 6)
    # --8<-- [end:read]
