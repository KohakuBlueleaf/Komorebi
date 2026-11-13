"""Handlers and commands: three kinds of command in flight together, then a batch."""

from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent

program = km.Program(
    "commands",
    kernels=HERE / "kernels",
    handlers=HERE / "kernels" / "handlers.h",
)

# --8<-- [start:submit]
with km.Session() as session:
    run = session.load(program)
    handles = [
        run.submit("add", group=(0, 1), args=[40, 2], result=True),
        run.submit("dot", group=(1, 1), args=[1, 2, 3, 10, 20, 30], result=True),
        run.submit("who", group=(119, 1), args=[1000], result=True),
    ]
    events = run.wait(handles, timeout=10.0)
    print([event.result for event in events])
    assert [event.result for event in events] == [42, 140, 1119]
    # --8<-- [end:submit]

    # --8<-- [start:batch]
    rng = np.random.default_rng(1)
    pairs = [(int(a), int(b)) for a, b in rng.integers(0, 1 << 20, (100, 2))]
    handles = [
        run.submit("add", group=(i % 120, 1), args=[a, b], result=True)
        for i, (a, b) in enumerate(pairs)
    ]
    events = run.wait(handles, timeout=10.0)
    wrong = sum(e.result != a + b for e, (a, b) in zip(events, pairs))
    print(f"{len(events)} commands, {wrong} wrong")
    run.stop()
# --8<-- [end:batch]
