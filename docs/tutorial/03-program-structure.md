---
title: Program structure
summary: The folder of a Komorebi program, what Program describes, what Session and Run do, and the smallest program that runs.
---

# 3. Program structure

A Komorebi program is a folder. It holds the three kernels that run on every Tensix
core, a header that gives the handlers their numbers, and a Python file. The Python
file describes the program (`Program`), opens the card (`Session`), loads the program
onto the card (`session.load`) and then submits commands to it (`Run`). This page walks
through the smallest program that does all of that and does nothing else: it submits
one command to every core, and every core reports it done.

## The folder

```text
03_program_structure/
├── hello.py            the host: Program, Session, Run
└── kernels/
    ├── handlers.h      the handlers: a number and a name each
    ├── reader.cpp      the reader kernel, on RISC-V core 0 of every Tensix core
    ├── compute.cpp     the compute kernel, on the three compute cores
    └── writer.cpp      the writer kernel, on RISC-V core 1
```

A program that uses an X280 thread has one more file or folder, the thread's C code
(`thread/plan.c` in the bucket sort of chapter 23). Chapter 16 adds it. The kernels'
names are fixed: `reader.cpp`, `compute.cpp` and `writer.cpp`. Everything else in
the folder is named as you like.

## What each file does

=== "hello.py"

    ``` { .python title="hello.py" callouts="program = km.Program|kernels=HERE|handlers=|with km.Session|session.load|run.submit|run.wait|run.stop" }
    --8<-- "tutorial/03_program_structure/hello.py"
    ```

    1.  `Program` only describes. Nothing is opened or compiled here.
    2.  The folder that holds `reader.cpp`, `compute.cpp` and `writer.cpp`.
    3.  The header that names the handlers. Here it names one, `hello`, taken from
        `kHandlerHello` in `handlers.h`.
    4.  Opens the card. The `with` block closes it again, also when the code inside fails.
    5.  Compiles and loads the kernels on all 120 cores and starts the dispatcher. It
        returns when every core is ready.
    6.  Queues one command for handler `hello`. No `group=`, so the core group is every
        core. The call returns a handle.
    7.  Blocks until the whole group has reported, at most ten seconds, and returns the
        event of the command.
    8.  Sends the stop command to every core and closes the dispatcher.

=== "kernels/handlers.h"

    ```cpp
    --8<-- "tutorial/03_program_structure/kernels/handlers.h"
    ```

=== "kernels/reader.cpp"

    ```cpp
    --8<-- "tutorial/03_program_structure/kernels/reader.cpp"
    ```

=== "kernels/compute.cpp"

    ```cpp
    --8<-- "tutorial/03_program_structure/kernels/compute.cpp"
    ```

=== "kernels/writer.cpp"

    ```cpp title="kernels/writer.cpp" callouts="km::Writer wr|wr.run|return true"
    --8<-- "tutorial/03_program_structure/kernels/writer.cpp"
    ```

    1.  The runtime's object for the writer kernel. It knows the core's number, its
        buffers and where to report.
    2.  The runtime's loop. For every command whose core group holds this core it calls
        the lambda with the command (`r`).
    3.  `true` means "done here": the runtime reports it to the group's leader.

Each kernel has the same shape. `kernel_main` creates the runtime's object for its
kernel (`km::Reader`, `km::Compute` or `km::Writer`) and calls `run` on it with the
**body**, a lambda. The runtime owns the loop: it waits for the next command that
includes this core and calls the body with it. Here the three bodies are empty. The
writer's body returns `true`, which means "this command is done on this core, report
it". A body that has nothing to say about a command still returns `true`; only a writer
that gave up waiting returns `false` (chapter 21).

## What `Program` describes

`km.Program` is a plain description. Creating one does not touch the card.

| Field | What it is | Used from |
|---|---|---|
| `name` | a label, for logs and for the compile cache | this page |
| `kernels` | the folder with `reader.cpp`, `compute.cpp`, `writer.cpp` | this page |
| `handlers` | the handlers: a dict `{name: number}` or a header whose `constexpr uint32_t kHandlerName = n;` lines give them. `kHandlerBigStep` becomes `"big_step"`. Numbers are 1 to 239 | chapter 4 |
| `buffers` | `km.Buffer(index, format, page_bytes, pages)`: the circular buffers, made on every core | chapter 6 |
| `compile_args` | numbers given to a kernel at compile time, such as the layout of a DRAM tensor | chapter 6 |
| `compute` | options of the compute kernel: `math_fidelity`, `fp32_dest_acc_en` and more | chapter 9 |
| `depth` | the ring slots: how many commands can be in flight at once, 1 to 16, default 4 | chapter 7 |
| `args`, `defines`, `opt` | per-core arguments, `#define`s for the kernels, an optimisation level per kernel | reference |
| `thread(...)` | declares an X280 thread | chapter 16 |

The full list is in the [host API reference](../reference/host.md#program).

## What `Session` and `Run` do

| Call | What happens |
|---|---|
| `km.Session(card=None)` | Opens the card in slow dispatch mode, which is the mode Komorebi needs. Use it in a `with` block: leaving the block stops the program and closes the card cleanly. |
| `session.load(program)` | Compiles the three kernels (a few seconds the first time, cached after), checks that they fit L1 and that they use no instruction the cores lack, loads them on every Tensix core, starts the dispatcher on X280 cluster 0 and waits until every core says it is ready. Returns a `Run`. |
| `run.submit(handler, group=..., args=...)` | Queues one command and returns its **handle** (a handle number). `handler` is the name or the number. The core group is `group=(first, count)`; the default is every core. |
| `run.wait(handle, timeout=...)` | Sends what is queued, waits until the group has reported, returns the event. |
| `run.stop()` | Tells every core to leave its kernels, stops the dispatcher and closes the X280 runtime. |
| `session.grid` | The Tensix grid as `(x, y)`: `(12, 10)` here. |

!!! note "Call names in the tutorial"
    The calls keep the names the code has today. `run.submit`, `group=` and `handlers=` name
    what this tutorial calls submit, core group and handler.

## Run it

```bash
cd tutorial/03_program_structure
TT_LOGGER_LEVEL=error python hello.py
```

The first run compiles the kernels and prints build messages from tt-metal before the
program's own line, which is the last one. A second run is quieter. This output is
from a second run on a P150A, with the build messages left out:

```{ .text .km-output }
cores: 120, event: done
```

`event.kind` is `done` when the group reported. The other kinds are in
chapter 21.

## Notes

- A program loads once. To change a kernel, stop the run, leave the `with` block and
  start again: kernels compile on `load`.
- One process holds the card. A second process that opens the card raises
  `km.CardBusy` and names the first.
- A compile error in a kernel raises `km.ProgramError` from `session.load` with the
  compiler's message, before anything runs on the card.
- Reference: [host API](../reference/host.md), [Tensix runtime](../reference/runtime.md).
