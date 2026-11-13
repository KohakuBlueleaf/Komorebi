"""Many commands: submit them all, then wait for all of them."""

import time
from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

program = km.Program(
    "many_commands",
    kernels=HERE / "kernels",
    handlers=HERE / "kernels" / "handlers.h",
)

with km.Session() as session:
    run = session.load(program)
    # --8<-- [start:batch]
    handles = [
        run.submit("multiply", group=(0, 1), args=[number, number], result=True)
        for number in range(1, 9)
    ]
    events = run.wait(handles, timeout=10.0)
    print("squares:", [event.result for event in events])
    # --8<-- [end:batch]

    # --8<-- [start:one-by-one]
    started = time.perf_counter()
    for number in range(1000):
        handle = run.submit("add", group=(0, 1), args=[number, 1], result=True)
        assert run.wait(handle, timeout=10.0).result == number + 1
    one_by_one = time.perf_counter() - started
    # --8<-- [end:one-by-one]

    started = time.perf_counter()
    handles = [
        run.submit("add", group=(0, 1), args=[number, 1], result=True)
        for number in range(1000)
    ]
    events = run.wait(handles, timeout=10.0)
    together = time.perf_counter() - started
    assert [event.result for event in events] == [number + 1 for number in range(1000)]
    print(f"1000 commands, waiting for each: {one_by_one * 1000:.0f} milliseconds")
    print(f"1000 commands, waiting for all:  {together * 1000:.0f} milliseconds")
    run.stop()
