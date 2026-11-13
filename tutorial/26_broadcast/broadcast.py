"""Broadcast: core 0 reads four tiles and sends each to the other cores of its row."""

from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import layout

HERE = Path(__file__).resolve().parent
CORES = 6  # the left panel of row 0
BLOCKS = 4
WINDOW = 2

with km.Session() as session:
    numbers = np.random.default_rng(1).integers(-8, 9, (BLOCKS * 32, 32))
    source = session.tensor(numbers, "bfloat16", "tile")
    destination = session.tensor(np.zeros((CORES * 32, 32)), "bfloat16", "tile")

    # --8<-- [start:plan]
    nocs = session.nocs()
    group = layout.Group(0, CORES)
    words = layout.broadcast_args(
        nocs.grid, nocs.panels, group, (0, 0, CORES, 1), "row", 0,
        nocs.noc0, nocs.noc_xy, (0, 0, WINDOW, 1),
    )  # fmt: skip
    fields = km.abi.struct_fields(km.abi.headers().device[-1], "BroadcastArgs")

    def arguments_of(core):
        """The program's arguments of a core: the reader's, the writer's, compute's."""
        if core not in words:
            return [0] * len(fields), [], [WINDOW]
        return [words[core][field] for field in fields], [], [WINDOW]

    # --8<-- [end:plan]

    tile = dict(format="bfloat16", page_bytes=source.page_bytes)
    program = km.Program(
        "broadcast",
        kernels=HERE / "kernels",
        handlers=HERE / "kernels" / "handlers.h",
        buffers=[
            km.Buffer(index=0, pages=WINDOW, **tile),
            km.Buffer(index=16, pages=1, **tile),
        ],
        args=arguments_of,
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        compile_args={
            "reader": source.accessor_args(),
            "writer": destination.accessor_args(),
        },
    )
    run = session.load(program)

    # --8<-- [start:cast]
    arguments = [source.address, BLOCKS, destination.address]
    handle = run.submit("broadcast", group=(0, CORES), args=arguments, exclusive=True)
    event = run.wait(handle, timeout=10.0)
    print(f"{BLOCKS} blocks to {CORES} cores in {event.microseconds:.1f} microseconds")
    # --8<-- [end:cast]
    run.stop()

    total = destination.read().reshape(CORES, 32, 32)
    expected = numbers.reshape(BLOCKS, 32, 32).sum(axis=0)
    assert all((total[core] == expected).all() for core in range(CORES))
    print(f"each of the {CORES} cores holds the sum of the {BLOCKS} blocks")
