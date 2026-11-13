"""Hello, thread: a C function on an X280 core adds two numbers and returns the sum."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

program = km.Program("hello_thread", kernels=HERE / "kernels")
# --8<-- [start:thread]
program.thread(
    cluster=1,
    sources=[HERE / "thread" / "hello.c"],
    regions={"unused": 64},
    words=[20, 22],
)
# --8<-- [end:thread]

with km.Session() as session:
    run = session.load(program)
    # --8<-- [start:result]
    result = run.thread_result(timeout=10.0)
    print(f"the thread ended with state {result.state}, code {result.code}")
    print(f"it returned {result.words[0]} = 20 + 22")
    print(f"the thread ran for {result.microseconds:.0f} microseconds")
    # --8<-- [end:result]
    run.stop()

assert (result.code, result.words[0]) == (0, 42)
