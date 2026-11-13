---
title: "The leader"
part: 2
summary: "the core that reports for the whole group, and the one result a group gives"
---

# The leader

A group of cores gives one report and one result, and both come from its **leader**, the
first core of the group. This lesson relies on [[core-group]], [[a-result]] and
[[commands]].

## Why a leader

When `wait` returns, it returns one event for one command, however many cores ran it. The
cores have to agree on who tells the dispatcher ([[commands]]). The rule is simple: every
other core of the group reports to the first one, and the first one, once all have
reported, reports to the dispatcher. That core is the leader. This is why a command is
done only when every core of its group is.

The result word of a command ([[a-result]]) travels with the leader's report. So the
result a Python program reads is whatever the leader's writer set, and nothing a
different core of the group set arrives.

<figure class="km-diagram">
<script type="application/json">
{"kind": "boxes", "boxes": [{"id": "c6", "x": 0, "y": 0, "w": 110, "h": 36, "t": "core 6, rank 1", "gem": "amber"}, {"id": "c7", "x": 0, "y": 48, "w": 110, "h": 36, "t": "core 7, rank 2", "gem": "amber"}, {"id": "c8", "x": 0, "y": 96, "w": 110, "h": 36, "t": "core 8, rank 3", "gem": "amber"}, {"id": "c5", "x": 190, "y": 48, "w": 130, "h": 36, "t": "core 5, rank 0\nthe leader", "gem": "amber", "here": true}, {"id": "d", "x": 400, "y": 48, "w": 130, "h": 36, "t": "the dispatcher", "gem": "sapphire"}], "arrows": [{"from": "c6", "to": "c5"}, {"from": "c7", "to": "c5"}, {"from": "c8", "to": "c5"}, {"from": "c5", "to": "d", "t": "one report", "dx": -40, "dy": -12}]}
</script>
<figcaption>The group <code>group=(5, 4)</code>: three cores report to the leader, core 5, and the leader reports to the dispatcher with its own result.</figcaption>
</figure>

## The general form

There is nothing new to write: the leader is the core with rank 0, and the code is that
of [[a-result]]:

```cpp
writer.result(VALUE);
```

```python
handle = run.submit(KIND, group=(FIRST, COUNT), result=True)
event = run.wait(handle, timeout=SECONDS)
```

- `VALUE` is a number of 32 bits. Every core of the group may call `writer.result`; only
  the call of the core `FIRST`, the one with rank 0, reaches Python as `event.result`.
- A leader that does not call `writer.result` gives `event.result == 0`.
- The group's leader is always the core `FIRST` of `group=(FIRST, COUNT)`.

## One line at a time

```{ .cpp }
--8<-- "tutorial/13_the_leader/kernels/writer.cpp:run"
```

The writer works out its rank. For the handler `Everyone`, every core of the group sets
its own core number as the result. For the handler `OnlyOne`, only the core whose rank is
the first argument does.

```{ .python }
--8<-- "tutorial/13_the_leader/the_leader.py:everyone"
```

The group is cores 5 to 8. All four set a result, 5, 6, 7 and 8, and Python receives 5:
the leader's.

```{ .python }
--8<-- "tutorial/13_the_leader/the_leader.py:only-one"
```

Four more commands to the same group, in which one rank at a time sets its result. When the
rank is 0 the result arrives, as 5. When it is 1, 2 or 3, the core sets its result, but it
is not the leader's, and Python receives 0, as if nobody had answered.

## The whole program

=== "the_leader.py"

    ``` { .python }
    --8<-- "tutorial/13_the_leader/the_leader.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/13_the_leader/kernels/handlers.h"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/13_the_leader/kernels/writer.cpp"
    ```

```{ .text .km-output }
--8<-- "tutorial/13_the_leader/output.txt"
```

The script checks the four numbers of the second part against `[5, 0, 0, 0]`.

## Common mistakes

- Expecting a result from another core. A result of 0 can mean that the leader set 0, or
  that nothing was set, or that a core other than the leader set it. The three look the
  same from Python.
- Putting the work that sets the result in the last core, "because it finishes last". The
  group's result is the first core's.
- Thinking the leader is slower or does extra work. It only waits for the reports of the
  others, and that wait is inside the command's time.

That is the group. What a program does with many cores that each hold a different piece of
data comes in Part 3.

Next: [[placing-groups]] chooses where on the card a group goes.
