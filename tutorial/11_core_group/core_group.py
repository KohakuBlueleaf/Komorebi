"""A core group: one command, many cores; the core of rank r works r times as long."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

program = km.Program(
    "core_group",
    kernels=HERE / "kernels",
    handlers=HERE / "kernels" / "handlers.h",
)

with km.Session() as session:
    run = session.load(program)
    took = {}
    # --8<-- [start:groups]
    for count in (1, 4, 12):
        handle = run.submit("work", group=(0, count), args=[1_000_000])
        event = run.wait(handle, timeout=10.0)
        took[count] = event.microseconds / 1000
        print(f"group of {count:2}: the command took {took[count]:5.1f} milliseconds")
    # --8<-- [end:groups]
    run.stop()

assert took[1] < took[4] < took[12]
