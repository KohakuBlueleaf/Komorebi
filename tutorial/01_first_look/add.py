"""The first look: a core adds two numbers that Python sends it."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

program = km.Program("first_look", kernels=HERE / "kernels")

with km.Session() as session:
    run = session.load(program)
    for left, right in [(40, 2), (7, 35), (1000, 2345)]:
        handle = run.submit(0, group=(0, 1), args=[left, right], result=True)
        event = run.wait(handle, timeout=10.0)
        print(f"{left} + {right} = {event.result}")
        assert event.result == left + right
    run.stop()
