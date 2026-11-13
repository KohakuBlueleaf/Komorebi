"""A result: the core sends one number back with its report."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

program = km.Program("a_result", kernels=HERE / "kernels")

with km.Session() as session:
    run = session.load(program)
    # --8<-- [start:submit]
    handle = run.submit(0, group=(0, 1), result=True)
    # --8<-- [end:submit]
    event = run.wait(handle, timeout=10.0)
    run.stop()
    # --8<-- [start:print]
    print(f"the core answered {event.result}")
    # --8<-- [end:print]
    assert event.result == 42
