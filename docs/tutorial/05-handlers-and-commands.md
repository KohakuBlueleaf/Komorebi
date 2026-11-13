---
title: Handlers and commands
summary: Several handlers in one program, the ten words of a command, handles and events, and many commands in flight at once.
---

# 5. Handlers and commands

A program has as many **handlers** as it has kinds of work: one for the matrix product,
one for the sort, one for a check. Python submits a **command** for a handler, and a
handler can run many times with different arguments. This page shows a program with three
handlers, three commands in flight at the same time, and a hundred more in a loop.

<figure class="km-grid">
<script type="application/json">
{"steps": [
 {"label": "Python submits three commands: add on core 0, dot on core 1, who on core 119. Each has its own handler and its own core group of one core.",
  "lit": ["host", "x280:0"],
  "arrows": [{"from": "host", "to": "x280:0"}]},
 {"label": "The dispatcher sends them one after the other. Every core sees all three and picks out its own.",
  "lit": ["x280:0", "core:0", "core:1", "core:119"],
  "arrows": [{"from": "x280:0", "to": "core:5"}, {"from": "x280:0", "to": "core:6"}]},
 {"label": "The three cores work at the same time, each in its own handler. Core 0 adds 40 + 2, core 1 takes a dot product, core 119 adds its own index to 1000.",
  "lit": ["core:0", "core:1", "core:119"],
  "marks": {"core:0": "42", "core:1": "140", "core:119": "1119"}},
 {"label": "Each core reports with its result word. The host's wait returns three events, in the order of the handles.",
  "lit": ["core:0", "core:1", "core:119", "x280:0", "host"],
  "marks": {"core:0": "42", "core:1": "140", "core:119": "1119"},
  "arrows": [{"from": "core:0", "to": "x280:0", "kind": "reply"}, {"from": "core:119", "to": "x280:0", "kind": "reply"}, {"from": "x280:0", "to": "host", "kind": "reply"}]}
]}
</script>
</figure>

## The idea

A command carries a handler, a core group, up to ten 32-bit words and a few flags. The
cores of the group run the handler in all three of their kernels, so a handler is really
three pieces of code, one in each kernel. A kernel that has nothing to do for a handler
ignores it, and still reports (the writer returns `true`).

Commands run in the order they are submitted, but not one after another: up to `depth`
commands (4 unless the program says otherwise, 16 at most) are in flight at once, and
two commands whose core groups do not overlap run side by side. When the order between
two commands that share cores matters, the second is submitted `exclusive=True`, which
makes it wait until the first has reported (chapter 7).

| Call | Arguments | What it does and returns |
|---|---|---|
| `run.submit(handler, group=None, args=(), exclusive=False, result=False)` | `handler`: a name from the header or a number. `group`: `(first, count)`, default every core. `args`: up to ten integers | Queues one command, returns its **handle**. Raises `KeyError` for an unknown handler name, `ValueError` for a core group outside the grid or more than ten words, before anything is sent |
| `run.wait(handles, timeout=10.0)` | one handle or a list | Sends what is queued, waits until each command has reported, returns the event (a list of events for a list of handles) |
| `run.wait_any(handles, timeout=10.0)` | a list | Returns the first event that arrives |
| `run.drain(timeout=10.0)` | | Waits for everything submitted; returns how many events it saw |
| `run.event(handle)` | | The event of a command whose event has arrived, else `None` |
| `Event.kind`, `.result`, `.us` | | `"done"` (or `"timeout"`, chapter 21); the result word; microseconds from submit to report |

## The example

Three handlers, one body each, in the writer kernel. In `kernels/handlers.h` they are
`kHandlerAdd = 1`, `kHandlerDot = 2` and `kHandlerWho = 3`.

```cpp title="kernels/writer.cpp" callouts="switch (km::handler(r))|r.a0 * r.a3|wr.me()|default:|return true"
--8<-- "tutorial/05_handlers_and_commands/kernels/writer.cpp:body"
```

1.  One `switch` on the handler number. This is the usual shape of a Komorebi kernel
    body.
2.  `dot` reads six of the ten words: two vectors of three numbers, a0 to a2 and a3 to a5.
3.  `wr.me()` is the number of the core that runs the body, 0 to 119. `who` returns it
    plus a0.
4.  Handler 0 (NOP) and any handler this kernel has no code for. The command still
    completes.
5.  Report done.

The reader and the compute kernel are the empty bodies of chapter 4. The host submits all
three commands before it waits for any:

``` { .python title="commands.py" callouts="handles = [|args=[1, 2, 3, 10, 20, 30]|group=(119, 1)|events = run.wait|assert" }
--8<-- "tutorial/05_handlers_and_commands/commands.py:submit"
```

1.  Each `run.submit` returns at once; the command is only queued on the host.
2.  The ten words are plain integers. A command may pass fewer than ten.
3.  Three different cores, three different handlers.
4.  `wait` sends all three commands, then waits for the three reports. The events come
    back in the order of the handles, whatever order the cores finished in.
5.  The host knows what each core should have answered:
    40 + 2, 1 &times; 10 + 2 &times; 20 + 3 &times; 30, and 119 + 1000.

A hundred commands, to a hundred cores, in a loop:

``` { .python title="commands.py" }
--8<-- "tutorial/05_handlers_and_commands/commands.py:batch"
```

Only four commands are in flight at a time, so the host keeps the rest queued and sends
them as room appears. `wait` takes the whole list.

## Run it

```bash
cd tutorial/05_handlers_and_commands
TT_LOGGER_LEVEL=error python commands.py
```

```{ .text .km-output }
[42, 140, 1119]
100 commands, 0 wrong
```

Both lines are the same on every card.

## Notes

- A handler name in Python must be in the header's `kHandler...` lines, written in snake
  case: `kHandlerBigStep` is `"big_step"`. A number works too.
- Only the leader of a core group sends a result word. A command for cores 3 to 6
  with `result=True` returns the answer of core 3 (chapter 7).
- Ten words is all a command has. Bigger data goes in DRAM, and the command carries
  its address ([chapter 6](06-buffers-and-data.md)).
- `wait` has a time limit. If a core does not report in time, the run stops with an
  error that names the core (chapter 21).
- Reference: [host API: Run](../reference/host.md#run), [runtime: the command](../reference/runtime.md#the-command).
