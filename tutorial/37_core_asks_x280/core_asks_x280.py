"""A core asks the X280: core 0 of the grid asks a thread for a square."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

NUMBER = 12

program = km.Program("core_asks_x280", kernels=HERE / "kernels")
program.thread(
    cluster=1,
    sources=[HERE / "thread" / "serve.c"],
    regions={"unused": 64},
    words=[NUMBER],
)
# --8<-- [start:ports]
program.ports({0: (1, 0)})
# --8<-- [end:ports]

with km.Session() as session:
    run = session.load(program)
    result = run.thread_result(timeout=10.0)
    run.stop()

if not result.words:  # a run without a card: its reads give zeros
    raise SystemExit("core_asks_x280.py: no result words (reads were zeros)")

print(f"core 0 asked for the square of {NUMBER}")
print(f"it told the X280 {result.words[0]}, which is the answer plus one")

assert result.code == 0 and result.words[0] == NUMBER * NUMBER + 1
