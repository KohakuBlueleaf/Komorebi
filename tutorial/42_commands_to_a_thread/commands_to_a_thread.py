"""Commands to a thread: Python asks the X280 thread for a number, then hands the
answer to a Tensix core in the next command."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

program = km.Program(
    "commands_to_a_thread",
    kernels=HERE / "kernels",
    handlers=HERE / "kernels" / "handlers.h",
)
# --8<-- [start:thread]
program.thread(
    cluster=1,
    sources=[HERE / "thread" / "serve.c"],
    regions={"unused": 64},
    handlers={"triangle": 1, "square": 2},
)
# --8<-- [end:thread]

with km.Session() as session:
    run = session.load(program)
    # --8<-- [start:commands]
    triangle = run.wait(run.submit("triangle", cluster=1, args=(10,), result=True))
    square = run.wait(run.submit("square", cluster=1, args=(5,), result=True))
    total = run.wait(
        run.submit("add", (0, 1), args=(triangle.result, square.result), result=True)
    )
    # --8<-- [end:commands]
    print(f"the thread answered triangle(10) = {triangle.result}")
    print(f"the thread answered square(5) = {square.result}")
    print(f"core 0 added them: {total.result}")
    run.stop()

assert (triangle.result, square.result, total.result) == (55, 25, 80)
