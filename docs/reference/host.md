---
title: Host API
summary: Program, Session and Run, the Python interface that loads a program onto every Tensix core under the dispatch kernel, issues records to teams, waits for them, and reads back state and trace.
tags:
  - reference
  - host
---

# Host API

Status: implemented for host-fed runs on one Blackhole card.

```python
import komorebi as km

with km.Session(card=0) as session:
    run = session.load(program)
    tickets = [run.issue("map", team=(i * 8, 8), args=[...]) for i in range(15)]
    run.wait(tickets, timeout=5.0)
    run.stop()
    trace = run.trace()
```

`import komorebi` needs no tt-metal tree. `km.Session` imports ttnn and the package's
native module on first use.

## `Program`

`km.Program(name, kernels, buffers=[], args=None, compile_args={}, compute={},
defines={}, depth=4, segments={})` describes what the resident worker runs. It opens
nothing.

| Field | Meaning |
|---|---|
| `kernels` | a directory with `reader.cpp`, `compute.cpp` and `writer.cpp`, or a dict of the three paths |
| `buffers` | `km.Buffer(index, format, page_bytes, pages)`: circular buffers on every core; `format` is a ttnn data type name (`"bfloat16"`, `"uint32"`) |
| `args` | `args(core) -> (reader, writer, compute)`: the program's own runtime arguments of each core, after the runtime's start-up arguments |
| `compile_args` | `{"reader": [...], "writer": [...], "compute": [...]}` |
| `compute` | compute kernel options: `math_fidelity` (a `ttnn.MathFidelity` name), `fp32_dest_acc_en`, and the other `ComputeConfigDescriptor` fields |
| `defines` | `{name: value}` for all three kernels |
| `depth` | mailbox slots the cores scan and the dispatch kernel uses, 1 to 16 |
| `segments` | `{name: number}`, or a header whose `constexpr uint32_t kSegName = n;` lines give them (`kSegBigStep` becomes `"big_step"`); numbers 1 to 239 |

The runtime owns circular buffers 29 and 30 (the record's token for compute and
writer) and 31 (the trace); a program uses the others.

## `Session`

`km.Session(card=None, device=None, pll_mhz=1750, tree=None, log=None)`.

- `card` selects the card through `TT_VISIBLE_DEVICES` (an error if that variable
  already names another card); `device` uses a mesh device the caller opened and
  keeps. The session opens the card in slow dispatch.
- `tree` is the tt-metal tree for builds (default `TT_METAL_HOME`). `log(msg)` receives
  one line per step (launch, kernel started, runtime closed, read back).
- As a context manager it stops a loaded run at the exit: with `run.stop()` on a normal
  exit, through the X280 runtime (no EXIT) on an exception, then it closes the device
  it opened.

`session.load(program, depth=None, trace=0, deadline=1.0, wait=2.0, idle=120.0,
kernel_idle=0.0, start=10.0, host_depth=64, event_depth=128, batch=64, settle_us=1.0,
image=None)` returns a `Run`:

1. finds or builds the dispatch kernel image and the native module (see below);
2. opens the X280 runtime and lays out the kernel's memory;
3. launches the three kernels on every core. Kernels compile here, before any X280
   image exists, so a compile error raises `km.ProgramError` and leaves nothing
   running;
4. starts the dispatch kernel and waits (bounded by `start` seconds) for every column of
   the grid to report READY; otherwise `km.StartError`.

| Argument | Meaning |
|---|---|
| `trace` | sequence numbers traced per core (0: no trace); 16 bytes of L1 each |
| `deadline` | seconds a record may take from issue to its team's completion, waiting behind earlier records included |
| `wait`, `idle` | the cores' bound on a wait for another core, and on a wait for the next record (the longest the host may stay silent) |
| `kernel_idle` | the dispatch kernel's own idle timeout; 0 turns it off |
| `host_depth`, `event_depth` | the command ring in the X280's memory and the event ring in host memory; powers of two, at most 64 and 128 |
| `batch` | commands stored per batch: every line, then one fence |
| `settle_us` | the host's pause between a fence and its next store into the card |

## `Run`

| Call | What it does |
|---|---|
| `issue(seg, team=None, args=(), exclusive=False)` | queues one record and returns its ticket (1, 2, ...). `seg` is a name or a number (`0` is NOP); `team` is `(first, count)`, default every core; `args` up to ten 32-bit words. Teams and segments are checked here. With `exclusive`, the record is held until no team in flight overlaps it |
| `wait(tickets, timeout=10.0)` | stores queued commands up to the tickets in batches, reads events until each ticket has one; returns the `Event` (or a list) |
| `drain(timeout=10.0)` | the same for everything issued; returns how many events it waited for |
| `event(ticket)` | `Event(ticket, kind, aux, ticks)` or None; `kind` is `done`, `timeout`, `pong` or `refused`; `ticks` is issue to completion in the X280's 50 MHz timer, `us` the same in microseconds |
| `seq(ticket)`, `team(ticket)` | the sequence number the kernel gave the record (the trace's index), its team |
| `stop(timeout=10.0)` | drains, sends EXIT to every core and waits for it, sends the kernel STOP, waits for its result, closes the X280 runtime, waits for every core's kernels to return, then reads the state words and the trace. Returns the result block |
| `abort()` | the failure path: stops the kernel through the runtime, closes it, reads back |
| `result()`, `state()`, `state_words()`, `trace()` | after the stop: the kernel's result block (decoded), each core's state words as a dict, the raw words, the trace |

`run.issue_s` and `run.flush_s` add up the host's time in `issue` and in the batched
stores.

`trace()` returns a `komorebi.state.Trace`: `record(ticket)` gives each member's
wall-clock words (taken, outputs written, completion sent, ring passes);
`phases(ticket)` gives microseconds after the first member took it; `lead(a, b)` is
how long before ticket `a`'s completion a member of `b` took its record (positive: both
were in flight at once).

## Errors

All derive from `km.KomorebiError`.

| Error | When | Facts it carries |
|---|---|---|
| `ProgramError` | a kernel did not compile or the launch failed | the build message |
| `StartError` | the dispatch kernel did not report READY | `result` |
| `WaitTimeout` | a host-side wait ran out; the run is still loaded | `pending` tickets |
| `RecordRefused` | the kernel consumed the command and issued nothing; the run goes on | `ticket`, `reason` |
| `RecordTimeout` | a team missed its deadline; the kernel stopped the grid, the run is stopped and read back | `ticket`, `slot`, `seq`, `team`, `cores`: each member that did not report, with where its reader and writer gave up |
| `KernelFailed` | the dispatch kernel finished with any other error | `result` |

## Builds

The dispatch kernel image (`km-dispatch.bin`) and the native module
(`_km_host.abi3.so`) are used from beside the package if they are there. Otherwise
they are built from the tt-metal tree into a cache directory, `KMR_CACHE` or
`build/komorebi/` in a checkout, keyed by a hash of their sources and the tree:
`python -m komorebi.native` builds both and prints their paths.

## Rules the host keeps

- The host never reads or writes a core's memory while the program runs. State words
  and the trace are read after the stop.
- While the dispatch kernel runs, every host wait is bounded. The X280 runtime is
  closed before any readback, and on every failure path.
- Host stores into the card go in batches: every line, one fence, then a pause of
  `settle_us` before the next store.
- Unread events plus commands in flight never exceed the event ring.
- EXIT goes to every core, after everything in flight has completed.
