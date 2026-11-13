"""When it goes wrong: the host stops waiting, and the card stops a late command."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

program = km.Program(
    "when_it_goes_wrong",
    kernels=HERE / "kernels",
    handlers=HERE / "kernels" / "handlers.h",
)

with km.Session() as session:
    run = session.load(program, deadline=1.0)

    # --8<-- [start:host-gives-up]
    handle = run.submit("count", group=(0, 1), args=[50_000_000], result=True)
    try:
        run.wait(handle, timeout=0.05)
    except km.WaitTimeout:
        print("the host stopped waiting after 0.05 seconds; the command goes on")
    event = run.wait(handle, timeout=10.0)
    print(f"a second wait: {event.kind}, the core counted to {event.result}")
    # --8<-- [end:host-gives-up]

    # --8<-- [start:card-gives-up]
    handle = run.submit("count", group=(0, 1), args=[400_000_000], result=True)
    try:
        run.wait(handle, timeout=10.0)
    except km.CommandTimeout as error:
        print(f"the card stopped the command: {error}")
    # --8<-- [end:card-gives-up]
