---
title: "Commands to a thread"
part: 5
summary: "a command goes to an X280 thread instead of a core group, the thread's kmx_program answers it with a word, and that word feeds the next command"
---

# Commands to a thread

A command can go to an X280 thread instead of a group of Tensix cores. The thread answers
it with a word, and the word comes back in the command's event, just as a leader's result
word does. This lesson relies on [[thread-submits]] and [[a-result]].

## Why command a thread

In [[thread-submits]] the thread sent commands to the cores. The other way round is just as
useful: a step that a general-purpose core does well, such as a table lookup or a choice
with many branches, can be asked of the thread as one command among the others. Python
sends it with `run.submit`, and so can a stream, a graph or another thread
([[streams-and-graphs]]). A graph can then put a thread's step between two grid commands
and never go back to the host.

## The general form

In Python, the thread names its handlers, as a program does for its kernels
([[handlers]]):

```python
program.thread(cluster=CLUSTER, sources=[...], regions={...}, handlers={"NAME": NUMBER})

handle = run.submit("NAME", cluster=CLUSTER, args=(A0, A1), result=True)
event = run.wait(handle)
```

In the thread's C, instead of `kmx_main`:

```c
uint32_t kmx_program(const km_command_t* command) {
    switch (kmx_handler(command)) {
        case kHandlerName:
            return WORD;
    }
    return 0;
}
```

- `handlers={"NAME": NUMBER}` gives the thread's handlers. A header such as `handlers.h`
  of [[handlers]] works too. They are the thread's own: the cores' handlers are another
  list.
- In the C, each name becomes a constant: `"triangle"` is `kHandlerTriangle`. Python
  defines it when it builds the thread, so the C needs no header for it.
- `cluster=CLUSTER` sends the command to that work cluster's thread. A command to a thread
  takes no `group`, and it is never exclusive or queued.
- `kmx_program` is called once for each command, on core 0 of the cluster. Its return value
  is the command's result word, `event.result` in Python with `result=True`.
- `kmx_handler(command)` is the command's handler; `command->a0` to `command->a9` are its
  arguments.
- A thread with `kmx_program` does not end by itself. It serves until the run stops, and
  each command has its own deadline, as a command to cores has.

## A thread that serves

The thread has two handlers. It computes both from the command's first argument:

```{ .c }
--8<-- "tutorial/42_commands_to_a_thread/thread/serve.c:program"
```

Python declares them with the thread:

```{ .python }
--8<-- "tutorial/42_commands_to_a_thread/commands_to_a_thread.py:thread"
```

Then it sends two commands to the thread and hands both answers to core 0:

```{ .python }
--8<-- "tutorial/42_commands_to_a_thread/commands_to_a_thread.py:commands"
```

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow",
 "steps": [
  {"t": "Python: triangle(10)\nto the thread on cluster 1"},
  {"t": "thread: kmx_program\nreturns 55", "gem": "sapphire"},
  {"t": "Python: square(5)\nto the thread"},
  {"t": "thread: kmx_program\nreturns 25", "gem": "sapphire"},
  {"t": "Python: add(55, 25)\nto core 0"},
  {"t": "core 0: writer.cpp\nanswers 80", "gem": "amber"}]}
</script>
<figcaption>Two commands to the thread and one to a core, each answered with a word.</figcaption>
</figure>

## The whole program

=== "commands_to_a_thread.py"

    ``` { .python }
    --8<-- "tutorial/42_commands_to_a_thread/commands_to_a_thread.py"
    ```

=== "thread/serve.c"

    ``` { .c }
    --8<-- "tutorial/42_commands_to_a_thread/thread/serve.c"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/42_commands_to_a_thread/kernels/writer.cpp"
    ```

=== "kernels/handlers.h"

    ``` { .cpp }
    --8<-- "tutorial/42_commands_to_a_thread/kernels/handlers.h"
    ```

`reader.cpp` and `compute.cpp` are the empty ones of [[hello-core]]. Run it from the folder
`tutorial/42_commands_to_a_thread`:

```bash
python commands_to_a_thread.py
```

```{ .text .km-output }
--8<-- "tutorial/42_commands_to_a_thread/output.txt"
```

Here Python carried the thread's words to core 0. In a graph the same chain runs on the
card: `graph.command(number, cluster=1, result=register)`, with the thread's handler
number, stores the word in a register, and the next node takes the register as an
argument
([`examples/13_thread_commands`](../../examples/13_thread_commands/)).

## Common mistakes

- Giving a group as well. `run.submit(..., (0, 1), cluster=1)` stops with
  `ValueError: a command to a thread takes no group, ...`.
- A cores' handler name for the thread. `run.submit("add", cluster=1)` stops with a
  `KeyError`: the thread knows only its own handlers.
- More than 8 commands to one thread at once from Python. The thread holds 8 unanswered;
  a ninth that finds all 8 waiting is rejected (`CommandRejected`, reason `thread`). Wait
  for some first.
- A `kmx_program` that never returns. Its command ends at its deadline with a
  `"timeout"` event, and the thread answers nothing else while it is stuck.

Next: [[measuring]] starts Part 6 with how long things take.
