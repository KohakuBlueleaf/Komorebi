---
title: "Streams and graphs"
part: 4
summary: "a chain of commands described once, with dependencies and a loop that ends on a value from the cores"
---

# Streams and graphs

A graph describes several commands once, with which one waits for which, and a loop that
repeats them until a number that the cores computed says to stop; Komorebi then runs it
without Python in between. This lesson relies on [[handlers]], [[many-commands]] and
[[a-result]].

## Why a graph

Until now Python sent each command and decided what came next. Look at the cost: every
decision sends a result back to your computer, Python thinks, and the next command goes out
again. A program that repeats a step many times, each time using what the last step
computed, spends most of its time in that round trip.

A graph moves the decisions to where the results are. It is a list of commands with the
order they depend on, `after=[...]`: a command starts when everything it is after is done,
and two commands that do not depend on each other and use different cores can be in flight
at the same time. A small processor step between commands can do arithmetic on
the results and decide whether to go round again. The graph is built once and then run
with one call.

A **stream** is the same idea for a straight line of commands without branches; its
page is [Micro-op streams](../streams.md). This lesson builds a graph.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow", "steps": [{"t": "round starts: value = 1 (from Python),\nlimit = 100", "gem": "amber"}, {"t": "add on core 0: value + 3 = sum, the\nresult goes to a register"}, {"t": "multiply on core 1, after add: sum x 2\n= doubled"}, {"t": "cpu step, after multiply: value =\ndoubled; again if value < 100"}], "back": {"from": 3, "to": 0, "t": "again"}}
</script>
<figcaption>The graph of this program: one round is three nodes, and the last one decides whether there is another round.</figcaption>
</figure>

## The general form

```python
from komorebi.engines.graph import Graph

graph = Graph(max_runs=ROUNDS)
name = graph.bindings("name")
constant = graph.const(NUMBER)
register = graph.reg()
first = graph.command(HANDLER, group=(FIRST, COUNT), args=[...], result=register)
second = graph.command(HANDLER, group=(...), args=[register], after=[first])
step = graph.cpu(after=[second])
step.mov(name, register)
step.lt(graph.again, name, constant)

result = run.run_hostfed(graph, {"name": START})
```

- `Graph(max_runs=ROUNDS)` makes an empty graph, and `ROUNDS` is the most rounds it may go;
  more is an error.
- A **register** is a 32-bit number that the graph keeps between its nodes. `graph.reg()`
  makes one that starts at 0; `graph.bindings("name")` makes one that Python fills in when
  the graph starts; `graph.const(NUMBER)` makes one that holds `NUMBER`.
- `graph.command(...)` adds a command node. `args` may hold numbers and registers, and
  the registers are read when the command is sent. `result=register` stores the command's
  result word ([[a-result]]) in a register. `after=[...]` lists the nodes it waits for.
- `graph.cpu(after=...)` adds a step that runs a few instructions on the registers:
  `mov`, `addi`, `lt` and others. `step.lt(graph.again, a, b)` sets `again` to 1 if `a` is
  less than `b`, and when `again` is not 0 at the end of a round, another round starts.
- `run.run_hostfed(graph, bindings)` runs the graph: the **executor**, the part that sends a
  command when its predecessors are done, runs on your computer and uses the loaded
  program. The same graph can be uploaded to run on the X280 core, with the same result.
- `result.runs` is the number of rounds, and `result.regs` the registers at the end.

## One line at a time

```{ .python }
--8<-- "tutorial/30_streams_and_graphs/streams_and_graphs.py:build"
```

Three registers are made, and two nodes. `add` is the handler of [[handlers]], on core 0,
with the argument `value` and 3, and its result goes into `sum_register`. `multiply`
runs on core 1 and waits for `add`; its arguments are that register and 2. A `step`
after it copies the doubled number into `value` for the next round and sets `again` while
the value is below 100. The value goes `1 -> 8 -> 22 -> 50 -> 106`, and stops at 106.

```{ .python }
--8<-- "tutorial/30_streams_and_graphs/streams_and_graphs.py:run"
```

One call runs all the rounds. The result has the number of rounds and the registers; register
0 is `value`, the first one made.

```{ .python }
--8<-- "tutorial/30_streams_and_graphs/streams_and_graphs.py:check"
```

The same rounds in a few lines of Python, as the check.

## The whole program

=== "streams_and_graphs.py"

    ``` { .python }
    --8<-- "tutorial/30_streams_and_graphs/streams_and_graphs.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/30_streams_and_graphs/kernels/handlers.h"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/30_streams_and_graphs/kernels/writer.cpp"
    ```

The reader and compute files are the empty ones, and the kernels are those of
[[handlers]].

```{ .text .km-output }
--8<-- "tutorial/30_streams_and_graphs/output.txt"
```

Four rounds, ending at 106; the script's check, the same sum and product in a loop, gives
the same.

## Common mistakes

- A graph that never stops. If `again` is still set after `max_runs` rounds, the graph
  ends with an error.
- A node that names itself or a later node in `after`. A node may only wait for nodes
  made before it.
- Two nodes that can run at the same time, one writing a register that the other reads. The
  result then depends on timing. Put one `after` the other.
- More than four predecessors for a node. Add a `graph.join(...)` node to gather them.

That is the end of Part 4. Cores can now hand tiles to each other, add them up, copy one
to many, send messages that the data routes, keep them from overrunning anyone, and
describe whole chains of commands at once. Part 5 puts a program on the other kind of core.

Next: [[hello-thread]].
