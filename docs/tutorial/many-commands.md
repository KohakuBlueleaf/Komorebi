---
title: "Many commands"
part: 1
summary: "submitting several commands and waiting for all of them"
---

# Many commands

`submit` does not wait, so you can submit as many commands as you like and then wait for
all of them with one call. This lesson relies on [[commands]] and [[handlers]].

## Why not wait every time

Until now every command was followed by its own `wait`: send, wait, send, wait. A `wait`
is a trip to the card and back, and while it is on that trip Python sends nothing and the
card has nothing to do. If the commands do not depend on each other, send them all first
and wait once. Then the card always has the next command ready.

## The general form

```python
handles = [run.submit(...) for ... in ...]
events = run.wait(handles, timeout=SECONDS)
```

- `handles` is a list of the handles that `submit` returned, one per command.
- `run.wait(handles, ...)` with a list waits until every command in it is done, and
  returns a list of events. The events are in the order of the handles you passed, not
  in the order the commands finished.
- `SECONDS` is the time for the whole wait, not for each command.

Commands for the same core run one after another, the oldest first, so a program can
rely on the order in which it submitted them.

## One line at a time

```{ .python }
--8<-- "tutorial/09_many_commands/many_commands.py:batch"
```

Eight commands, one for each number from 1 to 8, each asking the core to multiply the
number by itself. The list comprehension submits all eight before the first `wait`, then
`wait` returns eight events, and the line after prints their results.

```{ .python }
--8<-- "tutorial/09_many_commands/many_commands.py:one-by-one"
```

For comparison: a thousand commands, each followed by its own `wait`, and the time they
took.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow",
 "steps": [
  {"t": "one at a time:\nsubmit, wait, submit, wait, ...", "gem": "coral"},
  {"t": "all first:\nsubmit, submit, submit, ..., wait", "gem": "sage"}]}
</script>
<figcaption>Two ways to send the same thousand commands.</figcaption>
</figure>

## The whole program

=== "many_commands.py"

    ``` { .python }
    --8<-- "tutorial/09_many_commands/many_commands.py"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/09_many_commands/kernels/writer.cpp"
    ```

The kernels and the handlers are those of lesson 8.

```{ .text .km-output }
--8<-- "tutorial/09_many_commands/output.txt"
```

The squares come back in the order of the handles. The two times are from one run and
differ on every run; the proportion is what to take away. Waiting for each command took
many times as long as waiting once for all.

## Common mistakes

- Waiting inside the loop when nothing needs the answer yet. It is correct and slow.
- Expecting the events in the order the commands finished. They come in the order of the
  handles in your list.
- Submitting commands and never waiting for them. The cores still run them, but you never
  see their results.

Next: [[when-it-goes-wrong]] is about a command that does not finish.
