"""Casting a word: core 0 reads eight values and, for each, tells its row what to do."""

from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import layout

HERE = Path(__file__).resolve().parent
CORES = 6  # the left panel of row 0
VALUES = [4, 7, 10, 3, 8, 8, 1, 6]

with km.Session() as session:
    values = session.tensor(np.array([VALUES], dtype=np.uint32), "uint32")
    totals = session.tensor(np.zeros((CORES, 8), dtype=np.uint32), "uint32")

    # --8<-- [start:plan]
    nocs = session.nocs()
    group = layout.Group(0, CORES)
    words = layout.broadcast_args(
        nocs.grid, nocs.panels, group, (0, 0, CORES, 1), "row", 0,
        nocs.noc0, nocs.noc_xy, (0, 0, 0, 0),
    )  # fmt: skip
    fields = km.abi.struct_fields(km.abi.headers().device[-1], "BroadcastArgs")

    def arguments_of(core):
        """The program's arguments of a core: the reader's, the writer's, compute's."""
        if core not in words:
            return [0] * len(fields), [], []
        return [words[core][field] for field in fields], [], []

    # --8<-- [end:plan]

    program = km.Program(
        "casting_a_word",
        kernels=HERE / "kernels",
        handlers=HERE / "kernels" / "handlers.h",
        buffers=[
            km.Buffer(index=0, format="uint32", page_bytes=values.page_bytes, pages=1),
            km.Buffer(index=1, format="uint32", page_bytes=totals.page_bytes, pages=1),
        ],
        args=arguments_of,
        compile_args={"reader": values.tensor_args() + totals.tensor_args()},
    )
    run = session.load(program)

    # --8<-- [start:follow]
    arguments = [values.address, len(VALUES), totals.address]
    handle = run.submit("follow", group=(0, CORES), args=arguments, exclusive=True)
    run.wait(handle, timeout=10.0)
    # --8<-- [end:follow]
    run.stop()
    got = totals.read()[:, 0]

total = 1
for value in VALUES:
    total = total * 2 if value % 2 == 0 else total + 3
print(f"the leader read {VALUES} and cast {len(VALUES)} choices")
print(f"every core's total: {got.tolist()}, expected {total}")
assert (got == total).all()
