---
title: "Hello, thread"
part: 5
summary: "C code on an X280 core, started from Python: add two numbers and return the sum"
---

# Hello, thread

A **thread** is a C program that runs on one X280 cluster while your Tensix kernels run on
the grid; Python loads it with the program and collects what it returns. This lesson
writes the smallest thread, one that adds two numbers. It relies on [[the-card]] and
[[arguments]].

## Why a thread

The Tensix cores are many and small. Some work does not suit them: a decision that needs
every core's numbers, work whose size you only learn while it runs, a loop that sends one
command after another. That work belongs on a bigger core, and the X280 cores are the
bigger cores on the card. [[the-card]] showed them in four **clusters** of four. Cluster 0
is Komorebi's own. Clusters 1, 2 and 3 are yours, and a thread runs on one of them.

A thread is ordinary C, with no standard library, and it starts when `load` runs. It does
not compute on tiles. This lesson uses it alone, to show how it is written, started and
read; the lessons after it give it work.

## The general form

The C file:

```c
#include "kmx.h"

int kmx_main(int core) {
    const kmx_args* arguments = kmx_args_get();
    BODY
    kmx_result(CODE, WORDS, COUNT);
    return 0;
}
```

In Python, next to `km.Program(...)` and before `load`:

```python
program.thread(
    cluster=CLUSTER,
    sources=[FILES],
    regions={NAME: SIZE},
    words=[NUMBERS],
)
```

and after `load`:

```python
result = run.thread_result(timeout=SECONDS)
```

- `#include "kmx.h"` brings in the thread's library, `kmx`. Every call of it starts with
  `kmx_`.
- `kmx_main` is the function the library calls on the first of the cluster's four cores,
  the way `kernel_main` is called on a Tensix core. The name is fixed. `core` is that
  core's number, 0 here. When `kmx_main` returns, the thread has ended.
- `kmx_args_get()` returns the arguments of the thread. `arguments->words[0]` is the first
  number that Python gave in `words=`, `arguments->words[1]` the second, and so on.
- `kmx_result(CODE, WORDS, COUNT)` stores what Python will read: a `CODE`, by convention 0
  for "worked", and `COUNT` numbers from the array `WORDS`, at most 48.
- `return 0;` ends the thread. The value returned is the thread's code, unless
  `kmx_result` gave one earlier.
- `CLUSTER` is the X280 cluster to run on, 1, 2 or 3.
- `FILES` are the C files of the thread, as paths.
- `NUMBERS` are up to 64 whole numbers for the thread, 32 bits each, not negative. They
  are the thread's arguments, as `args=` is a command's ([[arguments]]).
- `regions={NAME: SIZE}` names a block of memory of `SIZE` bytes that Python and the
  thread can both reach. A block like this is a **region**. A thread must have at least
  one, and this thread does not use it; [[sharing-memory]] is about regions.
- `SECONDS` is how long `thread_result` waits for the thread to end.

The Tensix side still needs its three kernel files ([[hello-core]]). Here they are the
empty ones, and no command is sent to the grid.

## One line at a time

```{ .python }
--8<-- "tutorial/32_hello_thread/hello_thread.py:thread"
```

This is the whole declaration of the thread. The line only describes it, as `Program`
describes the kernels: nothing runs until `load`.

```{ .c }
--8<-- "tutorial/32_hello_thread/thread/hello.c"
```

The thread reads its two numbers, adds them in a constant `sum`, and stores `sum` as the
one result word. `&sum` is the address of the number: `kmx_result` takes an array, and a
single number is an array of one.

```{ .python }
--8<-- "tutorial/32_hello_thread/hello_thread.py:result"
```

`thread_result` waits until the thread has ended and returns its **result**: the `state`
the thread ended in, its `code`, the list `words` that it stored, and how long it ran, in
`microseconds`.

<figure class="km-diagram">
<script type="application/json">
{"kind": "flow",
 "steps": [
  {"t": "session.load(program)\nthe C file is built and started\non X280 cluster 1", "gem": "sapphire"},
  {"t": "kmx_main runs on core 0 of the\ncluster and adds 20 + 22", "gem": "sapphire"},
  {"t": "kmx_result(0, &sum, 1)\nthe code and the sum are stored", "gem": "sapphire"},
  {"t": "kmx_main returns:\nthe thread has ended"},
  {"t": "run.thread_result(...)\nPython receives code 0 and 42"}]}
</script>
<figcaption>The life of the thread of this program.</figcaption>
</figure>

## The whole program

=== "hello_thread.py"

    ``` { .python }
    --8<-- "tutorial/32_hello_thread/hello_thread.py"
    ```

=== "thread/hello.c"

    ``` { .c }
    --8<-- "tutorial/32_hello_thread/thread/hello.c"
    ```

The three kernel files in `kernels/` are those of [[hello-core]]. Run it from the folder
`tutorial/32_hello_thread`:

```bash
python hello_thread.py
```

```{ .text .km-output }
--8<-- "tutorial/32_hello_thread/output.txt"
```

The state is `done`, the code 0 that the thread chose, and the sum 42 came from a core
that is not a Tensix core. The thread ran for about ten microseconds.

## Common mistakes

- No region. `program.thread(...)` without `regions=` stops with
  `ValueError: 1 to 16 regions, shared ones included`.
- A cluster that is not 1, 2 or 3. `cluster=4` stops with
  `ValueError: cluster 4: work clusters are 1 to 3`.
- Too many numbers, or a negative one. 65 words stop with `ValueError: at most 64 words`,
  and `words=[-1]` with `ValueError: words are 32-bit unsigned`.
- A thread that does not end. When `timeout` runs out, `thread_result` does not raise: it
  asks the thread to stop, and returns a result whose `ran` is `False`. A thread that
  loops until `kmx_stop_requested()` is true, then returns 7, shows `ran=False`, and
  `code=7`. [[core-asks-x280]] and [[finding-a-bug]] use this.

Next: [[thread-submits]] lets the thread send a command to the Tensix cores.
