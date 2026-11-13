"""The leader: of a group of four cores, only the first core's result reaches Python."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

program = km.Program(
    "the_leader",
    kernels=HERE / "kernels",
    handlers=HERE / "kernels" / "handlers.h",
)

with km.Session() as session:
    run = session.load(program)
    # --8<-- [start:everyone]
    handle = run.submit("everyone", group=(5, 4), result=True)
    event = run.wait(handle, timeout=10.0)
    print(f"cores 5 to 8 all answer; Python receives {event.result}")
    # --8<-- [end:everyone]
    # --8<-- [start:only-one]
    received = []
    for rank in range(4):
        handle = run.submit("only_one", group=(5, 4), args=[rank], result=True)
        event = run.wait(handle, timeout=10.0)
        received.append(event.result)
        print(f"only rank {rank} answers; Python receives {event.result}")
    # --8<-- [end:only-one]
    run.stop()

assert received == [5, 0, 0, 0]
