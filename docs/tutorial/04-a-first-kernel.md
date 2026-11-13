---
title: A first kernel
summary: One core, one handler, one command and a result word that comes back to Python.
---

# 4. A first kernel

The first kernel does the least a kernel can do and still be useful to look at: one
core adds two numbers and Python reads the sum. The numbers travel in the command, the
sum travels back in the **result word** of the command's event, and nothing is read from
or written to DRAM. The structure is the one from [chapter 3](03-program-structure.md);
what is new is one handler with a body.

<figure class="km-grid">
<script type="application/json">
{"steps": [
 {"label": "Python submits add(40, 2) for a core group of one: core 5. It asks for a result word.",
  "lit": ["host", "x280:0"],
  "arrows": [{"from": "host", "to": "x280:0"}]},
 {"label": "The dispatcher sends the command to every core. Only core 5 is in the group; the other 119 cores skip it.",
  "lit": ["x280:0", "core:5"],
  "arrows": [{"from": "x280:0", "to": "core:5"}, {"from": "x280:0", "to": "core:6"}]},
 {"label": "On core 5 the writer kernel runs the handler: 40 + 2 = 42. It hands 42 to the runtime as the result word.",
  "lit": ["core:5"],
  "marks": {"core:5": "42"}},
 {"label": "Core 5 is the leader of its group of one, so it acknowledges to the dispatcher with the result word. The host's wait returns an event whose result is 42.",
  "lit": ["core:5", "x280:0", "host"],
  "marks": {"core:5": "42"},
  "arrows": [{"from": "core:5", "to": "x280:0", "kind": "reply"}, {"from": "x280:0", "to": "host", "kind": "reply"}]}
]}
</script>
</figure>

## The idea

A **handler** is one named piece of kernel code. You give it a number in a header
(`kHandlerAdd = 1`), the host names it in a command (`run.submit("add", ...)`), and in the
kernel a `switch` or an `if` on the command's handler number picks the code. A command
can carry ten 32-bit **words** (`r.a0` to `r.a9` in the kernel, the list `args=[...]`
in Python). If the command is submitted with `result=True`, the core that leads its
core group may attach one more 32-bit word to its report, and the host reads it from the
event as `event.result`.

Why the writer kernel? Each core's **writer** is the one that reports to the leader, so
it is the one place that can attach the result. The reader and the compute kernel have
no handler for `add` and do nothing. They have to exist anyway, with an empty body,
because a program always has the three kernels.

| What | Where | Meaning |
|---|---|---|
| `constexpr uint32_t kHandlerAdd = 1;` | `kernels/handlers.h` | the handler's number; the host name is `"add"` |
| `km::handler(r)` | kernel | the handler number of the command `r` |
| `r.a0` ... `r.a9` | kernel | the ten argument words of the command |
| `wr.result(value)` | writer kernel | the result word, sent with the report of a command submitted with `result=True` |
| `run.submit("add", group=(5, 1), args=[a, b], result=True)` | host | submits the command to core group `(first=5, count=1)`; returns a handle |
| `run.wait(handle, timeout=10.0)` | host | waits for the report; returns the event |
| `event.result` | host | the result word; `event.us` is the time from submit to report in microseconds, but only for a command without `result=True` (the result word is sent in place of the time) |

## The example

The handler, in the writer kernel:

```cpp title="kernels/writer.cpp" callouts="wr.run|km::handler(r) == kHandlerAdd|wr.result|return true"
--8<-- "tutorial/04_first_kernel/kernels/writer.cpp:body"
```

1.  The runtime calls this body once for every command that includes this core.
2.  Handler `add`. A command for any other handler falls through.
3.  `r.a0` and `r.a1` are the first two words of the command, `uint32_t`. Their sum is
    computed in the writer's RISC-V core, modulo 2<sup>32</sup>, and handed to the
    runtime as the result word.
4.  `true`: this core is done with the command.

The header and the other two kernels:

=== "kernels/handlers.h"

    ```cpp
    --8<-- "tutorial/04_first_kernel/kernels/handlers.h"
    ```

=== "kernels/reader.cpp"

    ```cpp
    --8<-- "tutorial/04_first_kernel/kernels/reader.cpp"
    ```

=== "kernels/compute.cpp"

    ```cpp
    --8<-- "tutorial/04_first_kernel/kernels/compute.cpp"
    ```

The host:

``` { .python title="add.py" callouts="session.load|run.submit|run.wait|assert event.result" }
--8<-- "tutorial/04_first_kernel/add.py:run"
```

1.  `session.load` is the call from chapter 3: it compiles and loads the kernels on
    all 120 cores.
2.  One command. Handler `add`, core group of one core, starting at core 5; `args` are
    the words `a0` and `a1`; `result=True` asks the leader for its result word.
3.  Wait for the report of that command. Ten seconds only guards against a hang; the
    command itself takes microseconds.
4.  The host checks the card's sum against Python's. The program stops with an
    `AssertionError` if they differ.

## Run it

```bash
cd tutorial/04_first_kernel
TT_LOGGER_LEVEL=error python add.py
```

```{ .text .km-output }
40 + 2 = 42
7 + 35 = 42
1000 + 2345 = 3345
```

The kernels compile on the first run and print build messages before these lines. The
lines themselves are the same on every card.

## Notes

- A command with no `result=True` gets no result word: `event.result` is `None`.
- Only the leader's word comes back. With a core group of one core it is always
  that core's. For a larger group the leader is the group's first core
  (chapter 7).
- The handler's arithmetic is the writer's 32-bit RISC-V integer arithmetic, so a
  sum past 2<sup>32</sup> wraps. A handler that needs floating-point numbers or tiles
  goes in the compute kernel (chapter 9).
- Handler number 0 is reserved for a command that does nothing, and 240 to 255 are
  reserved for the system; yours are 1 to 239.
- Reference: [host API: Run](../reference/host.md#run), [runtime: the writer](../reference/runtime.md#kmwriter).
