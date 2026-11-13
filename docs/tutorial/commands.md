---
title: "Commands"
part: 1
summary: "what a command is, who sends it, and that the kernels stay loaded between commands"
---

# Commands

A command is one message from Python that makes cores run their kernels once. The
kernels stay loaded on the cores between commands, so sending a command costs
microseconds, not the seconds that loading costs. This lesson relies on [[hello-core]].

## Why commands exist

A program that starts a kernel for every piece of work pays for the start every time:
compile, copy to the cores, start. On this card that is seconds, and a piece of work on a
core may take microseconds.

Komorebi pays once. `load` puts the three kernels on every core, and each core's `run`
loop (lesson 4) waits for commands. After that, each piece of work is a command: a short
message to cores that are already running and waiting.

## Who sends it and how it travels

You call `submit` in Python. Python does not talk to the 120 cores one by one. It hands
the command to a program that Komorebi keeps running on one of the card's X280 cores,
the **dispatcher**. The dispatcher passes the command to the Tensix cores. When the cores
of the group are done, the dispatcher collects their reports into one, and `wait` returns
it as an event.

<figure class="km-diagram">
<script type="application/json">
{"kind": "boxes", "boxes": [{"id": "host", "x": 0, "y": 0, "w": 130, "h": 64, "t": "Python\non your computer"}, {"id": "dispatcher", "x": 190, "y": 0, "w": 140, "h": 64, "t": "the dispatcher\non an X280 core", "gem": "sapphire"}, {"id": "core", "x": 390, "y": 0, "w": 150, "h": 64, "t": "the cores\nkernels waiting", "gem": "amber"}], "arrows": [{"from": "host", "to": "dispatcher", "kind": "command", "t": "command", "dx": -26, "dy": -12}, {"from": "dispatcher", "to": "core", "kind": "command", "t": "command", "dx": -26, "dy": -12}], "narrow": {"boxes": [{"id": "host", "x": 0, "y": 0, "w": 200, "h": 52, "t": "Python\non your computer"}, {"id": "dispatcher", "x": 0, "y": 96, "w": 200, "h": 52, "t": "the dispatcher\non an X280 core", "gem": "sapphire"}, {"id": "core", "x": 0, "y": 192, "w": 200, "h": 52, "t": "the cores\nkernels waiting", "gem": "amber"}], "arrows": [{"from": "host", "to": "dispatcher", "kind": "command", "t": "command", "dx": 8, "dy": 4}, {"from": "dispatcher", "to": "core", "kind": "command", "t": "command", "dx": 8, "dy": 4}]}}
</script>
<figcaption>A command goes from Python through the dispatcher to the cores. The event comes back the same way.</figcaption>
</figure>

You do not write the dispatcher and you do not start it. `load` does that. What you need to
know is that it is there, and that it makes a command cheap.

## What a command holds

A command is a small message. It holds a few things, and the next lessons add them one
at a time. [[hello-core]] used two of them:

- the **kind** of work, a number: [[handlers]],
- the **group** of cores that run it: [[core-group]].

Three more come later: whether the cores send something back ([[a-result]]), up to ten
numbers for the kernel to work with ([[arguments]]) and a few settings that Part 2 and
later parts use.

## See it cost microseconds

The program below loads the kernels once and measures it, then sends the same command
three times. It uses the kernels of lesson 4 unchanged; only the Python file is new.

```{ .python }
--8<-- "tutorial/05_commands/commands.py:load"
```

`time.perf_counter()` is Python's clock. These lines measure how long `load` takes.

```{ .python }
--8<-- "tutorial/05_commands/commands.py:commands"
```

Three commands, one after the other: send, wait for the event, print. `event.microseconds` is how
long the command took, in microseconds, from the moment it was submitted until the event
was ready, measured on the card.

=== "commands.py"

    ``` { .python }
    --8<-- "tutorial/05_commands/commands.py"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/05_commands/kernels/writer.cpp"
    ```

The other two kernel files are the same empty ones as in lesson 4.

```{ .text .km-output }
--8<-- "tutorial/05_commands/output.txt"
```

The times are from one run and differ on every run. What does not differ is the
proportion: loading takes a second, about 1,000,000 microseconds, and a command takes
about two. The first run after you change a kernel file takes longer, because the C++ has
to be compiled.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow", "steps": [{"t": "load: compile, copy to the cores,\nstart\nonce, about a second", "gem": "amber"}, {"t": "submit, wait: a command, about two\nmicroseconds"}, {"t": "submit, wait: a command, about two\nmicroseconds"}, {"t": "stop: the kernels end"}], "back": {"from": 3, "to": 2, "t": "as often\nas you like"}}
</script>
<figcaption>The kernels are loaded once; each command only wakes them.</figcaption>
</figure>

## Common mistakes

- Loading again for every piece of work. Put `load` before the loop and `submit` inside it.
- Sending a command after `stop`. It is refused with `KomorebiError: the run is stopped`.
  A run ends at `stop`; to run again, `load` again.
- Expecting `submit` to wait. It returns at once. The command may not have started when
  `submit` returns; `wait` is what waits for it.

Next: [[a-result]] makes the core send a number back.
