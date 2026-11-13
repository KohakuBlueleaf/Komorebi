"""Handlers: one kernel, two pieces of work; each command names the one it wants."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

# --8<-- [start:program]
program = km.Program(
    "handlers",
    kernels=HERE / "kernels",
    handlers=HERE / "kernels" / "handlers.h",
)
# --8<-- [end:program]

with km.Session() as session:
    run = session.load(program)
    # --8<-- [start:submit]
    handle = run.submit("add", group=(0, 1), args=[6, 7], result=True)
    # --8<-- [end:submit]
    print(f"6 add 7      = {run.wait(handle, timeout=10.0).result}")
    handle = run.submit("multiply", group=(0, 1), args=[6, 7], result=True)
    print(f"6 multiply 7 = {run.wait(handle, timeout=10.0).result}")
    run.stop()
