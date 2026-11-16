---
title: "A core asks the X280"
part: 5
summary: "an X280 port: a Tensix core asks its X280 core a question, the thread answers, and the core tells the answer back"
---

# A core asks the X280

An X280 port joins one Tensix core and one X280 core, and lets each send the other short
messages while a command runs. This lesson relies on [[thread-submits]] and [[the-leader]].

## Why a port

Until now a thread and the Tensix cores only met at the edges: a command from the thread, an
event back, a region at the end. Some work needs a talk in the middle. A Tensix core that
meets a decision it cannot make cheaply, such as which piece of work to take next, can ask a
bigger core on the card and wait for the answer. An **X280 port** is that line. It is a
different thing from the port of [[port]], which is a counter on a Tensix core: this one
connects a Tensix core to an X280 core. It has no dispatcher in its path, and it is fast: a
message is a number that identifies it and three 32-bit words.

There are two kinds of message, and a program can use both, in both directions:

- A **tell** is a message that the receiver looks at when it likes. If two tells arrive before
  the receiver looks, only the newest is seen.
- An **ask** is a message that expects an **answer**. The asker waits for the answer to that
  ask, and the other side answers it by the number that came with it.

## The general form

In Python, next to `program.thread(...)`:

```python
program.ports({CORE: (CLUSTER, CPU)})
```

On the Tensix core, in `reader.cpp`, with `#include "komorebi/device/core/km_x280.hpp"`:

```cpp
km::X280Port port(reader);
...
km::X280Msg reply;
bool got = port.call(A, B, C, reply);
port.tell(A, B, C);
```

In the thread, the function that serves, and the call that starts a command and serves it:

```c
static void serve(int cluster, int cpu, uint64_t argument) {
    while (kmx_port_open()) {
        kmx_message message;
        if (kmx_port_asked(CORE, &message) == KMX_OK) {
            kmx_port_answer(CORE, message.sequence, A, B, C);
        }
        if (kmx_port_told(CORE, &message) == KMX_OK) {
            ...
        }
    }
}

kmx_event event;
int status = kmx_port_run(&command, serve, ARGUMENT, &event);
```

- `ports({CORE: (CLUSTER, CPU)})` says which X280 core serves which Tensix core:
  Tensix core number `CORE` is served by core `CPU`, 0 to 3, of cluster `CLUSTER`. Every
  served core has exactly one. The thread must run on the card, and on that cluster.
- `km::X280Port port(reader)` makes the core's end of the port. `reader.cpp` owns it; the
  other kernels do not use it.
- `port.call(A, B, C, reply)` asks, with three numbers, and waits for the answer. It returns
  `true` when the answer came. The answer's three numbers are `reply.w[0]`, `reply.w[1]`,
  `reply.w[2]`.
- `port.tell(A, B, C)` tells the X280 core three numbers and does not wait.
- `kmx_port_owner(CORE)` is the X280 core that serves Tensix core `CORE`; the function
  `serve` runs on every X280 core, and the one that is not the owner has nothing to do.
- `kmx_port_asked(CORE, &message)` looks for an ask from that core that is not answered
  yet. It returns `KMX_OK` and the message, or `KMX_NONE`, and never waits. The ask's
  numbers are `message.w[0]` to `message.w[2]`, and `message.sequence` is the number of the ask.
- `kmx_port_answer(CORE, message.sequence, A, B, C)` answers that ask.
- `kmx_port_told(CORE, &message)` looks for a tell that is newer than the last one it saw.
  It returns `KMX_OK` and the message, or `KMX_NONE`, and never waits.
- `kmx_port_open()` is 1 while the command is running and 0 once its event has come, or
  the run was stopped.
- `kmx_port_run(&command, serve, ARGUMENT, &event)` submits the command like `kmx_submit`
  does ([[thread-submits]]), runs `serve` on every X280 core of every cluster that runs the
  thread until the event comes, and returns it in `event`.

## One line at a time

```{ .python }
--8<-- "tutorial/37_core_asks_x280/core_asks_x280.py:ports"
```

Tensix core 0 is served by core 0 of cluster 1. That is the one X280 core that the program
uses for it.

```{ .cpp }
--8<-- "tutorial/37_core_asks_x280/kernels/reader.cpp:run"
```

`reader.cpp` of core 0 asks for the number `a0` of its command, waits for the answer, and
tells the X280 core the answer plus one. If the ask gets no answer it gives up and tells
nothing. The tell is the proof, for the thread, that the answer arrived.

```{ .c }
--8<-- "tutorial/37_core_asks_x280/thread/serve.c:look"
```

One look at the port of core 0: answer an ask with the square of its number, and keep what a
tell says.

```{ .c }
--8<-- "tutorial/37_core_asks_x280/thread/serve.c:serve"
```

The serving function. Only the X280 core that owns Tensix core 0 does anything. It looks
again and again while the command runs, and once more after it ended, because the tell can
come just before the end.

```{ .c }
--8<-- "tutorial/37_core_asks_x280/thread/serve.c:run"
```

The thread sends the command and serves it until its event comes. The thread then returns
the tell that it saw as its result.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow",
 "steps": [
  {"t": "thread: kmx_port_run sends the\ncommand with a0 = 12 and serves", "gem": "sapphire"},
  {"t": "core 0, reader.cpp: port.call(12, ...)\nasks and waits", "gem": "amber"},
  {"t": "thread: kmx_port_asked sees 12,\nkmx_port_answer(144)", "gem": "sapphire"},
  {"t": "core 0, reader.cpp: the answer is 144,\nport.tell(145)", "gem": "amber"},
  {"t": "the command ends. The thread keeps 145\nand returns it to Python"}]}
</script>
<figcaption>The question goes to the X280 core, the answer comes back, and a proof goes with it.</figcaption>
</figure>

## The whole program

=== "core_asks_x280.py"

    ``` { .python }
    --8<-- "tutorial/37_core_asks_x280/core_asks_x280.py"
    ```

=== "thread/serve.c"

    ``` { .c }
    --8<-- "tutorial/37_core_asks_x280/thread/serve.c"
    ```

=== "kernels/reader.cpp"

    ``` { .cpp }
    --8<-- "tutorial/37_core_asks_x280/kernels/reader.cpp"
    ```

`compute.cpp` and `writer.cpp` are the empty ones of [[hello-core]]. Run it from the folder
`tutorial/37_core_asks_x280`:

```bash
python core_asks_x280.py
```

```{ .text .km-output }
--8<-- "tutorial/37_core_asks_x280/output.txt"
```

The 145 is the square of 12, plus the one that the Tensix core added. It could only reach
Python by going from the thread to the core in an answer, and back from the core to the
thread in a tell.

## Common mistakes

- Declaring the ports twice. A second `program.ports(...)` stops with
  `ValueError: ... already declares its X280 ports`.
- Looking only while `kmx_port_open()` is true. A message that comes just before the
  command's end is missed. Look once more after the loop, as `serve` does.
- Expecting every tell to arrive. A tell is the newest message: two tells in a row, with no
  look between, leave only the second.
- An ask that nobody answers. `port.call` waits, with a limit, and returns `false`.

Next: [[commands-to-a-thread]] sends commands to the thread itself.
