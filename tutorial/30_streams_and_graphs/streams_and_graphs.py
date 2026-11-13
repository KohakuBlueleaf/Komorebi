"""Streams and graphs: commands that depend on each other, and a loop."""

from pathlib import Path

import komorebi as km
from komorebi.engines.graph import Graph

HERE = Path(__file__).resolve().parent
LIMIT = 100

program = km.Program(
    "streams_and_graphs",
    kernels=HERE / "kernels",
    handlers=HERE / "kernels" / "handlers.h",
)

# --8<-- [start:build]
graph = Graph(max_runs=20)
value = graph.bindings("value")  # a register that Python fills in at the start
limit = graph.const(LIMIT)  # a register that holds 100
sum_register, doubled = graph.reg(), graph.reg()
add = graph.command(
    program.handler("add"), group=(0, 1), args=[value, 3], result=sum_register
)
multiply = graph.command(
    program.handler("multiply"),
    group=(1, 1),
    args=[sum_register, 2],
    after=[add],
    result=doubled,
)
step = graph.cpu(after=[multiply])
step.mov(value, doubled)  # the next round starts from the doubled sum
step.lt(graph.again, value, limit)  # and goes round again while value < 100
# --8<-- [end:build]

with km.Session() as session:
    run = session.load(program)
    # --8<-- [start:run]
    result = run.run_hostfed(graph, {"value": 1})
    print(f"the graph went round {result.runs} times, ended with {result.regs[0]}")
    # --8<-- [end:run]
    run.stop()

# --8<-- [start:check]
expected, rounds = 1, 0
while True:
    expected, rounds = (expected + 3) * 2, rounds + 1
    if expected >= LIMIT:
        break
assert (result.runs, result.regs[0]) == (rounds, expected)
print(f"the same rounds in Python: {rounds} rounds, value {expected}")
# --8<-- [end:check]
