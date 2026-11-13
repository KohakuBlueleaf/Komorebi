"""Hello, core: the smallest program. One command to one core, and its answer."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

# --8<-- [start:program]
program = km.Program("hello", kernels=HERE / "kernels")
# --8<-- [end:program]

# --8<-- [start:session]
with km.Session() as session:
    # --8<-- [end:session]
    # --8<-- [start:load]
    run = session.load(program)
    # --8<-- [end:load]
    # --8<-- [start:submit]
    handle = run.submit(0, group=(0, 1))
    # --8<-- [end:submit]
    # --8<-- [start:wait]
    event = run.wait(handle, timeout=10.0)
    # --8<-- [end:wait]
    # --8<-- [start:stop]
    run.stop()
    # --8<-- [end:stop]
    # --8<-- [start:print]
    print(f"the command is {event.kind}")
    # --8<-- [end:print]
