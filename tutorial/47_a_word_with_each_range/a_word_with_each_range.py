"""A word with each range: four cores pull ranges of numbers from an X280 core, and
the word with each range says whether to add the numbers or their squares."""

from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent

program = km.Program("a_word_with_each_range", kernels=HERE / "kernels")
program.thread(
    cluster=1,
    sources=[HERE / "thread" / "pick.c"],
    regions={"unused": 64},
)
# --8<-- [start:ports]
program.ports({core: (1, 0) for core in range(4)})
# --8<-- [end:ports]

with km.Session() as session:
    run = session.load(program)
    result = run.thread_result(timeout=10.0)
    run.stop()

# --8<-- [start:check]
want = [0, 0, 0, 0]
for number in range(64):
    index = number // 4  # the range the number is in
    core = index % 4  # range r goes to core r % 4
    want[core] += number if index % 2 == 0 else number * number
# --8<-- [end:check]
for core in range(4):
    print(f"core {core} added up {result.words[core]}")
print(f"the same as Python: {list(result.words[:4]) == want}")

assert result.code == 0 and list(result.words[:4]) == want
