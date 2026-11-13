"""A thread that submits: the X280 sends a command to core 0 and returns its answer."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

program = km.Program("thread_submits", kernels=HERE / "kernels")
program.thread(
    cluster=1,
    sources=[HERE / "thread" / "submit.c"],
    regions={"unused": 64},
    words=[6, 7],
)

with km.Session() as session:
    run = session.load(program)
    # --8<-- [start:result]
    result = run.thread_result(timeout=10.0)
    print(f"the thread ended with code {result.code}")
    print(f"core 0 answered the thread with {result.words[0]} = 6 * 7")
    # --8<-- [end:result]
    run.stop()

assert (result.code, result.words[0]) == (0, 42)
