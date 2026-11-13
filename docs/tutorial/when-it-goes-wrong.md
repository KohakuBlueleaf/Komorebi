---
title: "When it goes wrong"
part: 1
summary: "a command that does not finish: the two time limits and what the card reports"
---

# When it goes wrong

A command that does not finish is stopped by one of two time limits, and they are not the
same limit: the host's `timeout` ends only your waiting, and the card's `deadline` ends
the command. This lesson relies on [[handlers]] and [[many-commands]].

## Why two limits

A program that waits for a core can wait for ever if the kernel has a bug. Komorebi has
two answers, at two places.

- The **timeout** is in Python: the `timeout=` of `wait`. When it runs out, `wait` raises
  `km.WaitTimeout` and the program decides what to do. The command is not touched. It may
  be slow, not broken, and you can call `wait` again.
- The **deadline** is on the card: how long a command may take from `submit` until every
  core of its group has reported. The card measures it itself. When it runs out, the card stops the cores and the next `wait`
  raises `km.CommandTimeout`. The deadline is a setting of `load`, and its default is one
  second.

## The general form

```python
run = session.load(program, deadline=SECONDS)

try:
    event = run.wait(handle, timeout=SECONDS)
except km.WaitTimeout:
    ...  # the host gave up waiting; the command is still on the card
except km.CommandTimeout as error:
    ...  # the card gave up on the command; the run is over
```

- `deadline=` is the card's limit for every command of this run.
- `timeout=` is the host's limit for this one wait.
- After `WaitTimeout` the run goes on: wait again, or send something else.
- After `CommandTimeout` the run has been stopped by the card, and so you do not call
  `run.stop()`. To go on, `load` again.

## A kernel that takes a while

To have something that is slow, the kernel of this lesson has a handler that counts up to
the number it is given, which takes about six milliseconds per million:

```{ .cpp }
--8<-- "tutorial/10_when_it_goes_wrong/kernels/writer.cpp:run"
```

`volatile` makes the compiler count for real instead of replacing the loop by its
result.

## One limit at a time

```{ .python }
--8<-- "tutorial/10_when_it_goes_wrong/when_it_goes_wrong.py:host-gives-up"
```

The command counts to 50 million, about 0.3 seconds. The host waits 0.05 seconds, and
`wait` raises `WaitTimeout`. The `except` prints a line, and the command goes on: the
second `wait` waits up to ten seconds and gets the event.

```{ .python }
--8<-- "tutorial/10_when_it_goes_wrong/when_it_goes_wrong.py:card-gives-up"
```

Now 400 million, more than two seconds, with a host that would wait ten. The card's
deadline of one second passes first. The card stops the cores, and `wait` raises
`CommandTimeout`, whose message is what the card reports.

<figure class="km-diagram">
<script type="application/json">
{"kind": "boxes", "boxes": [{"id": "start", "x": 0, "y": 20, "w": 90, "h": 44, "t": "submit"}, {"id": "host", "x": 130, "y": 0, "w": 150, "h": 36, "t": "host: wait(timeout)", "gem": "sapphire"}, {"id": "card", "x": 130, "y": 48, "w": 150, "h": 36, "t": "card: deadline", "gem": "coral"}, {"id": "host_end", "x": 330, "y": 0, "w": 220, "h": 36, "t": "WaitTimeout: wait again"}, {"id": "card_end", "x": 330, "y": 48, "w": 220, "h": 36, "t": "CommandTimeout: run is over", "gem": "coral"}], "arrows": [{"from": "start", "to": "host"}, {"from": "start", "to": "card"}, {"from": "host", "to": "host_end"}, {"from": "card", "to": "card_end"}], "narrow": {"boxes": [{"id": "start", "x": 120, "y": 0, "w": 90, "h": 40, "t": "submit"}, {"id": "host", "x": 0, "y": 80, "w": 150, "h": 52, "t": "host:\nwait(timeout)", "gem": "sapphire"}, {"id": "card", "x": 180, "y": 80, "w": 150, "h": 52, "t": "card:\ndeadline", "gem": "coral"}, {"id": "host_end", "x": 0, "y": 172, "w": 150, "h": 52, "t": "WaitTimeout:\nwait again"}, {"id": "card_end", "x": 180, "y": 172, "w": 150, "h": 52, "t": "CommandTimeout:\nrun is over", "gem": "coral"}], "arrows": [{"from": "start", "to": "host"}, {"from": "start", "to": "card"}, {"from": "host", "to": "host_end"}, {"from": "card", "to": "card_end"}]}}
</script>
<figcaption>Two clocks start at the same moment. The first to run out decides what you see.</figcaption>
</figure>

## The whole program

=== "when_it_goes_wrong.py"

    ``` { .python }
    --8<-- "tutorial/10_when_it_goes_wrong/when_it_goes_wrong.py"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/10_when_it_goes_wrong/kernels/handlers.h"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/10_when_it_goes_wrong/kernels/writer.cpp"
    ```

```{ .text .km-output }
--8<-- "tutorial/10_when_it_goes_wrong/output.txt"
```

Reading the last line, `CommandTimeout`'s message:

- `command 2`: the second command of this run, the one with handle 2.
- `group first 0 count 1`: the cores it was sent to.
- `not reported: core 0`: the core of the group that had not reported when the deadline
  passed.
- what follows says where `reader.cpp` and writer of that core were when the card told them
  to stop. They are Komorebi's own step names, and you do not need them yet;
  [[finding-a-bug]] uses them.

The card is fine afterwards. The counting kernel did end by itself, and a new program can
open the card at once.

## Common mistakes

- Reading `WaitTimeout` as "the command failed". It says that the host stopped waiting.
- A `timeout` that is longer than the deadline. The card's deadline passes first and the
  error is `CommandTimeout`, not `WaitTimeout`.
- A deadline shorter than the honest running time of a command. The card then stops a
  command that was working. Set the deadline for the longest command of the program.
- A writer that does not return `true` ([[hello-core]]). The core never reports, and the
  card stops the command at its deadline with `CommandTimeout`.

That is the end of Part 1. You can now send work to a core, with numbers, get a number
back, choose among several pieces of work, send many commands, and read what the card says
when one does not finish. Part 2 does the same on many cores at once.

Next: [[core-group]].
