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
| `compute` | compute kernel options: `math_fidelity` (a `ttnn.MathFidelity` name), `fp32_dest_acc_en`, `unpack_to_dest_mode` (a `ttnn.UnpackToDestMode` name such as `"UnpackToDestFp32"` for every circular buffer, or `{cb: name}` with the others `Default`), and the other `ComputeConfigDescriptor` fields |
| `defines` | `{name: value}` for all three kernels |
| `depth` | mailbox slots the cores scan and the dispatch kernel uses, 1 to 16 |
| `segments` | `{name: number}`, or a header whose `constexpr uint32_t kSegName = n;` lines give them (`kSegBigStep` becomes `"big_step"`); numbers 1 to 239 |

`program.thread(cluster=1, sources, regions, words, host_fallback, deadline,
prefetch="documented")` adds an
X280 thread on a work cluster (one per program); see [X280 threads](../threads.md).

The runtime owns circular buffers 29 and 30 (the record's token for compute and
writer) and 31 (the trace); a program uses the others.

## `Session`

`km.Session(card=None, device=None, pll_mhz=1750, tree=None, log=None,
grid_timeout=10.0)`.

- `card` selects the card through `TT_VISIBLE_DEVICES` (an error if that variable
  already names another card); `device` uses a mesh device the caller opened and
  keeps. The session opens the card in slow dispatch.
- `tree` is the tt-metal tree for builds (default `TT_METAL_HOME`). `log(msg)` receives
  one line per step (launch, kernel started, runtime closed, read back).
- As a context manager it stops a loaded run at the exit: with `run.stop()` on a normal
  exit, through the X280 runtime (no EXIT) on an exception, then it closes the device
  it opened.
- `grid_timeout` bounds every wait for the grid's kernels to return (seconds). When
  it passes, `km.GridTimeout` is raised and the device is left open: closing it would
  wait for the grid without a bound. The card may then need a reset.

`session.load(program, depth=None, trace=0, deadline=1.0, wait=2.0, idle=120.0,
kernel_idle=0.0, start=10.0, host_depth=64, event_depth=128, batch=64, settle_us=1.0,
image=None, thread="card")` returns a `Run`:

1. finds or builds the dispatch kernel image and the native module (see below), and
   checks that the image is there and X280 cluster 0 is idle (else `km.StartError`);
2. opens the X280 runtime and lays out the kernel's memory;
3. launches the three kernels on every core. Kernels compile here, before any X280
   image exists, so a compile error raises `km.ProgramError` and leaves nothing
   running;
4. starts the dispatch kernel and waits (bounded by `start` seconds) for every column of
   the grid to report READY; otherwise `km.StartError`, which carries the kernel's
   result block.

From the launch on, every failure path reads the grid back. When the dispatch kernel
is not there to stop the grid (it never started, or it ended before it could reach the
cores), the host writes the stop word into every core's control block itself, through
the device API, and then waits for the grid within `grid_timeout`.

| Argument | Meaning |
|---|---|
| `trace` | sequence numbers traced per core (0: no trace); 16 bytes of L1 each |
| `deadline` | seconds a record may take from issue to its team's completion, waiting behind earlier records included |
| `wait`, `idle` | the cores' bound on a wait for another core, and on a wait for the next record (the longest the host may stay silent) |
| `kernel_idle` | the dispatch kernel's own idle timeout; 0 turns it off |
| `host_depth`, `event_depth` | the command ring in the X280's memory and the event ring in host memory; powers of two, at most 64 and 128 |
| `batch` | commands stored per batch: every line, then one fence |
| `settle_us` | the host's pause between a fence and its next store into the card |
| `thread` | for a program with an X280 thread: `"card"` builds and starts it after READY, `"host"` runs its `host_fallback` instead |

Event room. The dispatch kernel takes a command that ends in an event only while the
unread events, one per busy slot and one more fit the event ring; it counts as read
what the host last wrote into its read-count line. The host keeps a tighter rule: it
stores a command only while the events it has not read plus the commands it stored
without an event fit the event ring. It writes its read count with every batch, and
while it waits whenever the kernel could be holding a command for it, so the kernel
waits for room only while that count is on its way.

Cores' NoC coordinates. `session.nocs()` returns a `layout.Nocs` for the open device,
before any load: `grid`; by core `(x, y)` the translated NoC address `noc_xy` (the one
the dispatch kernel's core table holds) and the NoC0 coordinates `noc0`; `words`, the
translated addresses as `x | y << 16` in core order; and the grid's `panels`. These are
what `layout.lane_args` and program arguments that name other cores take.

Reading tensors. `session.read(*tensors)` returns the raw pages of interleaved DRAM
tensors (`[pages, page_bytes]` uint8 each) and `session.read_bf16(*tensors)` bfloat16
tile tensors as float32 arrays. Page p of such a tensor lies in DRAM bank p % banks, so
the native module reads each bank's pages in one go, after one DRAM barrier for all
the tensors of the call: tt-metal's own read (`ttnn.to_torch`) runs that barrier for
every page. These reads do not wait for the cores to be idle, so they work while a run
is loaded; read then only what no kernel writes any more (an output whose record's
event has arrived). Reads through ttnn wait for every core of the resident grid to
return, which it does only at the stop. The control blocks live in L1 and are read
through ttnn after the stop.

## `Run`

| Call | What it does |
|---|---|
| `issue(seg, team=None, args=(), exclusive=False, result=False, queued=False)` | queues one record and returns its ticket (1, 2, ...). `seg` is a name or a number (`0` is NOP); `team` is `(first, count)`, default every core; `args` up to ten 32-bit words. Teams and segments are checked here. With `exclusive` (`KM_F_EXCL`), the record is held until no team in flight overlaps it. With `result` (`KM_F_RESULT`), its done event carries the team leader's 32-bit result word. With `queued` (`KM_F_QUEUE`), `team` is a count: the kernel gives the record the lowest first core whose team overlaps no busy one, and waits until there is one (nothing is refused for a full grid); `team(ticket)` is then `(None, count)`, since the event does not name the first core. A queued record cannot be exclusive |
| `issue_many(seg, firsts, counts, args=None, exclusive=False)` | queues n records at once and returns their tickets as a `range`. `seg`, `firsts` and `counts` are one value or n each; `args` is an `[n, m]` integer matrix (m at most 10) or None. Checked as `issue` checks, all before any is queued |
| `run_hostfed(graph, bindings=None, timeout=10.0)` | runs a [graph](../graphs.md) host-fed: its executor on the host, its records through this run; returns `HostfedResult(status, error, regs, runs, records, events)` |
| `wait(tickets, timeout=10.0)` | stores queued commands up to the tickets in batches, reads events until each ticket has one; returns the `Event` (or a list) and forgets those tickets. The stores and the event reads run in the native module with the GIL released |
| `wait_any(tickets, timeout=10.0)` | stores what is queued and waits until one of the tickets has its event; returns that `Event` (the lowest ticket if several arrived) and forgets only that ticket. Raises as `wait` does |
| `drain(timeout=10.0)` | the same for everything issued; returns how many events it waited for |
| `event(ticket)` | `Event(ticket, kind, aux, ticks, result)` or None; `kind` is `done`, `timeout`, `pong` or `refused`; `ticks` is issue to completion in the X280's 50 MHz timer, `us` the same in microseconds; for a record issued with `result=True`, `result` is the leader's word (the kernel sends it in place of the time, so `ticks` and `us` then hold it too), else None |
| `seq(ticket)`, `team(ticket)` | the sequence number the kernel gave the record (the trace's index), its team |
| `forget(tickets)` | releases what the run keeps for tickets whose events arrived; for callers that read events with `event()` after `drain()`. A traced run keeps teams and sequences for the trace |
| `stop(timeout=10.0)` | drains, sends EXIT to every core and waits for it, sends the kernel STOP, waits for its result, closes the X280 runtime, waits for every core's kernels to return, then reads the state words and the trace. Returns the result block; raises `KernelFailed` unless the result is ready, cluster 0 restored, the state DONE and the error 0 |
| `thread_result(timeout=10.0)` | a program with an X280 thread: waits for it (or stops it), returns its `ThreadResult`; after `stop()`, `run.thread.read(name)` returns a region's bytes |
| `abort()` | the failure path: stops the kernel through the runtime, closes it, stops the grid from the host, reads back; raises `KernelFailed` if the runtime could not restore cluster 0 |
| `result()`, `state()`, `state_words()`, `trace()` | after the stop: the kernel's result block (decoded), each core's state words as a dict (with `ports`: count and wanted count of every port slot, and `lanes`: each lane's VALID word, first give-up, blocks, wait cycles and DONE-BY words; a reader or writer that gave up waiting on a port or an emit names its `port`), the raw state words, the trace |

`run.issue_s` adds up the host's time in `issue` and `issue_many`; `run.wait_s` the
time in `wait` and `drain`, of which `run.native_s` was in the native loop.

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
| `StartError` | the image is missing, cluster 0 is not idle, a base handed to the kernels (control block, X280 local memory, its upload region) is not a multiple of 16 bytes, or the dispatch kernel did not report READY | `result` (the kernel's result block, when it ran) |
| `WaitTimeout` | a host-side wait ran out; the run is still loaded | `pending` tickets |
| `RecordRefused` | the kernel consumed the command and issued nothing (a reserved segment or flag, a team outside the grid, a record after EXIT); the run goes on | `ticket`, `reason` (`KMD_REFUSE_*`), `reason_name` |
| `RecordTimeout` | a team missed its deadline; the kernel stopped the grid, the run is stopped and read back | `ticket`, `slot`, `seq`, `team`, `cores`: each member that did not report, with where its reader and writer gave up |
| `KernelFailed` | the dispatch kernel finished with any other error, did not finish cleanly, or sent an event for a command not in flight | `result` |
| `GridTimeout` | the grid's kernels did not return within `grid_timeout`; the device is left open | |

The kernel's reasons for a refusal (`RecordRefused.reason`, from `KMD_REFUSE_*` in
the kernel's header; `reason_name` is the lower-case suffix):

| `reason` | `reason_name` | The command |
|---|---|---|
| 1 | `seg` | has a segment from 240 to 251 |
| 2 | `flags` | sets the reserved flag bit 0 |
| 3 | `team` | has a team of 0 cores, or one that does not lie inside the cores |
| 4 | `exited` | came after EXIT |

`issue` and `issue_many` check segments and teams before anything is queued, the
host sets no reserved flag, and a stopped run issues nothing, so through this API a
refusal means that the host and the kernel were built from different headers.

## Builds

The dispatch kernel image (`km-dispatch.bin`) and the native module
(`_km_host.abi3.so`) are used from beside the package if they are there. Otherwise
they are built from the tt-metal tree into a cache directory, `KMR_CACHE` or
`build/komorebi/` in a checkout, keyed by a hash of their sources and the tree:
`python -m komorebi.native` builds both and prints their paths.

## Rules the host keeps

- The host never reads or writes a core's memory while the program runs, with one
  exception: the stop word, when the dispatch kernel cannot send it. State words and
  the trace are read after the stop.
- While the dispatch kernel runs, every host wait is bounded. The X280 runtime is
  closed before any readback, and on every failure path.
- Host stores into the card go in batches: every line, one fence, then a pause of
  `settle_us` before the next store.
- Unread events plus commands in flight never exceed the event ring.
- EXIT goes to every core, after everything in flight has completed.
