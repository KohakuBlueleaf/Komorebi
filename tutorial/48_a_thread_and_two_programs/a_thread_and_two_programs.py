"""A thread and two programs: the add and multiply programs of the two programs
lesson, loaded together, and an X280 thread that asks each of them and adds the
answers."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent
PROGRAMS = HERE.parent / "46_two_programs"

# --8<-- [start:programs]
add = km.Program("add", kernels=PROGRAMS / "add", handlers=PROGRAMS / "handlers.h")
multiply = km.Program(
    "multiply", kernels=PROGRAMS / "multiply", handlers=PROGRAMS / "handlers.h"
)
multiply.thread(  # any one of the run's programs may declare the thread
    cluster=1,
    sources=[HERE / "thread" / "both.c"],
    regions={"unused": 64},
    handlers={"both": 1},
)
# --8<-- [end:programs]

with km.Session() as session:
    # --8<-- [start:run]
    run = session.load([add, multiply])  # every core holds both
    handle = run.submit("both", cluster=1, args=[6, 7], result=True)
    answer = run.wait(handle, timeout=10.0).result
    run.stop()
    # --8<-- [end:run]

print("the thread asked add and multiply about 6 and 7")
print(f"and added their answers: {answer}")

assert answer == (6 + 7) + 6 * 7
