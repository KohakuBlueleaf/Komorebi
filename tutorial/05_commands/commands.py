"""Commands: the kernels are loaded once; each command only costs microseconds."""

import time
from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

program = km.Program("commands", kernels=HERE / "kernels")

with km.Session() as session:
    # --8<-- [start:load]
    started = time.perf_counter()
    run = session.load(program)
    print(f"loading the kernels took {time.perf_counter() - started:.2f} seconds")
    # --8<-- [end:load]
    # --8<-- [start:commands]
    for number in (1, 2, 3):
        handle = run.submit(0, group=(0, 1))
        event = run.wait(handle, timeout=10.0)
        micros = event.microseconds
        print(f"command {number}: {event.kind} after {micros:.1f} microseconds")
    # --8<-- [end:commands]
    run.stop()
