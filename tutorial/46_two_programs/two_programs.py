"""Two programs in one run: add and multiply, each a handler 1 of its own, loaded on
every core together; then each on its own group of cores."""

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
    run = session.load({add: (0, 60), multiply: (60, 60)})  # a group each
    print(f"groups:   add      6 and 7 = {ask(run, add, (0, 60))}")
    print(f"groups:   multiply 6 and 7 = {ask(run, multiply, (60, 60))}")
    try:
        ask(run, multiply, (0, 60))
    except ValueError as error:
        print(f"refused:  {error}")
    run.stop()
    # --8<-- [end:groups]
