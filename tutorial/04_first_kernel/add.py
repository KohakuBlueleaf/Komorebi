"""A first kernel: one core adds two numbers and the host reads the sum."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

program = km.Program(
    "add",
    kernels=HERE / "kernels",
    segments=HERE / "kernels" / "handlers.h",
)

# --8<-- [start:run]
with km.Session() as session:
    run = session.load(program)
    for a, b in [(40, 2), (7, 35), (1000, 2345)]:
        handle = run.issue("add", team=(5, 1), args=[a, b], result=True)
        event = run.wait(handle, timeout=10.0)
        print(f"{a} + {b} = {event.result}")
        assert event.result == a + b
    run.stop()
# --8<-- [end:run]
