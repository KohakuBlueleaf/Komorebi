---
title: "A result"
part: 1
summary: "one number from the core back to Python"
---

# A result

A result is one 32-bit number that a core sends back to Python together with its report
that a command is done. This lesson relies on [[hello-core]] and [[commands]].

## Why a result

So far the only thing Python learned from a command was that it was done. Almost every
real program needs more: the answer. The result is the smallest answer there is: one
number, with nothing to set up. Later lessons move whole arrays; this one moves one number.

## The general form

In the writer kernel, before `return true;`:

```cpp
writer.result(VALUE);
```

In Python, in the call of `submit`, and after `wait`:

```python
handle = run.submit(0, group=(FIRST, COUNT), result=True)
event = run.wait(handle, timeout=SECONDS)
event.result
```

- `VALUE` is an unsigned 32-bit number: a constant or any C++ expression that gives a
  `uint32_t`. It is what the core hands back.
- `result=True` tells the cores that this command wants a result. Without it, the
  number is not sent, even if the kernel sets it.
- `event.result` is the number, in Python, as an ordinary `int`.

## One line at a time

```{ .cpp }
--8<-- "tutorial/06_a_result/kernels/writer.cpp:run"
```

`writer.result(42)` stores 42 as this core's answer to this command. It is `writer` that
has it, not `reader` or `compute`: the writer is the kernel that reports to the card
that the command is done, so it is the one that can attach a number to the report. The
reader and the compute function cannot.

```{ .python }
--8<-- "tutorial/06_a_result/a_result.py:submit"
```

The same `submit` as before, with `result=True`.

<figure class="km-diagram">
<script type="application/json">
{"kind": "anatomy",
 "parts": [
  {"t": "handle = run.submit(0, group=(0, 1), "},
  {"t": "result=True", "n": "ask the cores to send\ntheir number back"},
  {"t": ")"}]}
</script>
<figcaption>The one new part of the line.</figcaption>
</figure>

```{ .python }
--8<-- "tutorial/06_a_result/a_result.py:print"
```

`event.result` is the 42 that the kernel gave.

## The whole program

=== "a_result.py"

    ``` { .python }
    --8<-- "tutorial/06_a_result/a_result.py"
    ```

=== "kernels/writer.cpp"

    ``` { .cpp }
    --8<-- "tutorial/06_a_result/kernels/writer.cpp"
    ```

The reader and the compute files are the same empty ones as before.

```{ .text .km-output }
--8<-- "tutorial/06_a_result/output.txt"
```

The kernel computed nothing here: 42 is a constant. What matters is the path it took.
It was set on a core of the card, in C++, and it came out of `event.result` in Python.
The next lesson sends numbers the other way.

## Common mistakes

- Forgetting `result=True`. The command works and the event arrives, but `event.result`
  is `None`.
- Reading `event.us` of a command that asked for a result. The card sends the result in
  the place where it would send the time, so `event.us` is not a time then. For a time,
  use a command without `result=True`.
- Setting the result in the reader or the compute kernel. Only the writer has
  `result`.

Next: [[arguments]] sends numbers the other way, from Python into the kernel.
