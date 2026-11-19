"""Two programs in one run: add and multiply, each a handler 1 of its own, loaded on
every core together; then each on its own group of cores, and part of a group
switches."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

# --8<-- [start:programs]
add = km.Program("add", kernels=HERE / "add", handlers=HERE / "handlers.h")
multiply = km.Program(
    "multiply", kernels=HERE / "multiply", handlers=HERE / "handlers.h"
)
# --8<-- [end:programs]


def ask(run, program, group):
    """APPLY of `program` to 6 and 7 on `group`; its leader's answer."""
    handle = run.submit(program.handler("apply"), group, args=[6, 7], result=True)
    return run.wait(handle, timeout=10.0).result


with km.Session() as session:
    # --8<-- [start:combined]
    run = session.load([add, multiply])  # every core holds both programs
    for program in (add, multiply, add):
        answer = ask(run, program, (0, 1))
        print(f"combined: {program.name:8} 6 and 7 = {answer}")
    run.stop()
    # --8<-- [end:combined]
    # --8<-- [start:groups]
    half = session.ranks // 2  # the cores a program runs on, less a Tensix dispatcher
    first, second = (0, half), (half, session.ranks - half)
    run = session.load({add: first, multiply: second})  # a group each
    print(f"groups:   add      6 and 7 = {ask(run, add, first)}")
    print(f"groups:   multiply 6 and 7 = {ask(run, multiply, second)}")
    try:
        ask(run, multiply, first)
    except ValueError as error:
        print(f"refused:  {error}")
    # --8<-- [end:groups]
    # --8<-- [start:switch]
    part, kept = (0, half // 2), (half // 2, half - half // 2)  # add's group halved
    run.wait(run.load(multiply, group=part), timeout=10.0)  # part switches
    print(f"switched: multiply 6 and 7 = {ask(run, multiply, part)}")
    print(f"kept:     add      6 and 7 = {ask(run, add, kept)}")
    run.stop()
    # --8<-- [end:switch]
