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
defines={}, depth=4, segments={}, opt={})` describes what the resident worker runs. It opens
nothing.

| Field | Meaning |
|---|---|
| `kernels` | a directory with `reader.cpp`, `compute.cpp` and `writer.cpp`, or a dict of the three paths |
| `buffers` | `km.Buffer(index, format, page_bytes, pages)`: circular buffers on every core; `format` is a ttnn data type name (`"bfloat16"`, `"uint32"`) |
| `args` | `args(core) -> (reader, writer, compute)`: the program's own runtime arguments of each core, after the runtime's start-up arguments |
| `compile_args` | `{"reader": [...], "writer": [...], "compute": [...]}` |
| `compute` | compute kernel options: `math_fidelity` (a `ttnn.MathFidelity` name), `fp32_dest_acc_en`, `unpack_to_dest_mode` (a `ttnn.UnpackToDestMode` name such as `"UnpackToDestFp32"` for every circular buffer, or `{cb: name}` with the others `Default`), and the other `ComputeConfigDescriptor` fields, each set as an attribute (checked against tt-metal v0.81.1: `math_approx_mode`, `dst_full_sync_en`, `bfp8_pack_precise`, `enable_trisc2_rvv`); a name the descriptor lacks raises `ProgramError` |
| `defines` | `{name: value}` for all three kernels |
| `depth` | mailbox slots the cores scan and the dispatch kernel uses, 1 to 16 |
| `segments` | `{name: number}`, or a header whose `constexpr uint32_t kSegName = n;` lines give them (`kSegBigStep` becomes `"big_step"`); numbers 1 to 239 |
| `opt` | `{"reader": "Os", "writer": "Os"}`: an optimisation level per kernel, one of the runtime's `O0`, `O1`, `O2`, `O3`, `Os`, `Ofast`, `Oz` (anything else raises `ProgramError`); a kernel without one keeps the runtime's default, `O2` for the reader and writer and `O3` for the compute kernel (its three threads together). A set level also defines `KM_OPT_LEVEL` for that kernel, so that it is cached apart |

`program.thread(cluster=1, sources, regions, words, host_fallback, deadline,
prefetch="documented")` adds an
X280 thread on a work cluster (one per program); see [X280 threads](../threads.md).

A program's three kernels share the session's kernel buffer, 70,656 bytes on Blackhole
unless the session sets `kernel_bytes`. When a program is loaded its build is laid out
against that buffer: `program.sizes` (a `km.KernelSizes`) holds the packed text per
thread, the total and the buffer, and the session log has them in one line. A program
that does not fit raises `km.KernelSizeError` before anything is loaded, naming the
total, the buffer and the largest thread (in place of the runtime's "Program size
... too large for kernel config buffer"). Give the session more with `kernel_bytes`,
or lower the reader and writer to `"Os"` first and the compute kernel only when that
is not enough, since its hot loops are what `O3` is for.

A program's own header (its segments, buffers and argument indices, shared by its
kernels and its driver) is read with `km.abi.constants(path,
known=km.abi.headers().constants)`: every `constexpr uint32_t k*`, `#define KM*_*` and
enum value, as a dict; it may use the package's constants by their C++ names
(`km::kLaneArgs`). A constant the reader cannot evaluate raises `abi.HeaderError`.

The runtime owns circular buffers 29 and 30 (the record's token for compute and
writer) and 31 (the trace); a program uses the others.

## `Session`

`km.Session(card=None, device=None, pll_mhz=1000, tree=None, log=None,
grid_timeout=10.0, kernel_bytes=None)`.

- `pll_mhz` is the X280 clock (the runtime's steps: 200, 800, 1000, 1750). One PLL
  clocks all four X280 clusters, so the work clusters' threads run at it too. 1750 is an
  opt-in: it loses loads when the Tensix grid is under full matrix load (the card lowers
  its core voltage; the dispatch kernel trapped in 9 of 18 such runs, and in 0 of 23 at
  1000 MHz); safe use is the caller's claim. A host-fed record at depth 16 costs 1.228
  microseconds at 1000 MHz and 0.726 at 1750, one after the other on one card
  (`experiments/dispatch_kernel/measure.py bench --n 100000 --depth 16 --team 1
  --batch 64 --host-depth 64 --pll 1000`, then `--pll 1750`). Before the session sets
  the clock, the runtime that opens the cluster leaves it at its safe 200 MHz; 1750
  was never the card's own choice. The session's log names the clock it set.

- `card` selects the card through `TT_VISIBLE_DEVICES` (an error if that variable
  already names another card); `device` uses a mesh device the caller opened and
  keeps. The session opens the card in slow dispatch.
- `tree` is the tt-metal tree for builds (default `TT_METAL_HOME`). `log(msg)` receives
  one line per step (launch, kernel started, runtime closed, read back).
- As a context manager it stops a loaded run at the exit: with `run.stop()` on a normal
  exit, through the X280 runtime (no EXIT) on an exception, then it closes the device
  it opened. `session.close()` does the same without `with`.
- The X280 runtime must be closed while the device is still open: a process that ends
  with it open, or closes the device first, can leave the card needing a reset. So the
  session also closes itself when the process exits without closing it (`atexit`) and
  on SIGTERM (a handler is installed if the process has none; it closes the runtime,
  then exits with status 143 through the `with` blocks). A clean-up that raises leaves
  the run loaded and the device open, so the next of these paths tries again; each try
  is bounded (5 s in the runtime). SIGKILL cannot be caught: after one, reset the card.
  With `device=`, close the session before the device.
- What an abort (an exception inside `with`, `session.close()` of a loaded run, the
  exit hooks) does: it writes the stop word into every core and waits up to 2 s for
  the grid, then asks the X280 runtime to stop the dispatch kernel, which finishes
  whatever is in flight (5 s bound); `run.result()` then holds its result block, with
  `abandoned` the records left without their `KM_EV_DONE`. The card needs no reset
  after it. An exception in the kernel needs none either: the kernel finishes on its
  own and the run raises `KernelFailed` with error `trap`, naming the X280 core,
  `mcause`, `mepc` and `mtval`. After an abort or a kernel failure the session log
  also holds the first 512 bytes of the kernel's state as DRAM holds them
  (`kernel state +0x...` lines). A reset is still needed when the kernel's core 0 has
  stopped running its loop (a `WaitTimeout` says so): the runtime cannot restore
  cluster 0, and `KernelFailed` carries the runtime's account of each hart.
- `grid_timeout` bounds every wait for the grid's kernels to return (seconds). When
  it passes, `km.GridTimeout` is raised and the session does not close the device: the
  runtime closes it at the process's exit, waits for the cores up to
  `TT_METAL_OPERATION_TIMEOUT_SECONDS`, then puts them in reset. A thread that is only
  stuck (a loop, a blocked wait) is recovered by the next open; a kernel that left a
  compute unit running is not, and when the next open fails the card needs a reset.
  The variable is read once, at the process's first device open, and bounds every
  wait the runtime makes for the cores in that process, not only the exit's; unset,
  it means no bound, and only then would the close wait without one. A session that
  opens the device therefore sets it to 30 s when it is unset (with a log line) and
  keeps a value of the user's.
- `kernel_bytes` is how much of each core's L1 holds the program's kernels (the
  runtime's kernel config buffer: the five threads' text, their arguments and the
  buffer configs), a multiple of 64 from 64 to 1,531,712; None keeps the runtime's
  70,656 bytes. Every byte added is taken from the L1 left for circular buffers and
  L1 tensors (1,461,120 bytes by default; 1,400,704 with 128 KiB of kernels): the
  session opens the device with the runtime's `worker_l1_size` that gives it, so it is
  set only when the session opens the device. `session.l1()` returns `(kernel buffer,
  L1 left)` of the open device, from the runtime's layout.

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

Exclusive records. The host stores a record with `KM_F_EXCL` like any other; the
dispatch kernel holds it at the head of its command ring while a busy record's team
shares a core with it, and issues it once that record's leader has acknowledged. Host
commands behind a held one wait with it, whatever their teams. Its deadline counts from
the kernel's issue, not from the store. So back-to-back exclusive records on one team
are the kernel's boundary apart, while one issued after idle still pays the host's store
reaching the X280 (a round trip of about 14 us). Measured on 2026-11-13, one p150a, X280
at 1,000 MHz, host load 3 to 9, before (the host held each exclusive record until it
had read the event of the overlapping one) and after, alternating in one session:

| Records back to back, exclusive | Before (us) | After (us) |
|---|---|---|
| NOP on 120 cores, period | 16.25, 16.26 | 2.27, 2.27 |
| NOP on 12 cores, period | 4.28, 16.21 | 2.28, 2.28 |
| NOP on 1 core, period | 15.95, 15.96 | 1.90, 1.91 |
| `examples/03_broadcast`, 120 cores, per job | 61.1, 60.4 | 47.9, 46.9 |

The period is the first member's take of one record to that of the next (the cores'
trace, median of 99, after 10 records of warm-up). The broadcast row is
`broadcast.py --rect 0,0,12,10 --blocks 16 --pages 4 --window 2 --jobs 40 --no-check`
(host time per job). The 4.28 is a run in which the host's stores reached the X280 fast,
a state seen now and then.

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

A part of one tensor is read the same way:

| Call | Returns |
|---|---|
| `session.read(tensor, first=0, pages=None)` | `[pages]`: pages `first` .. `first + pages - 1` as one `[pages, page_bytes]` uint8 array (`pages=None`: to the end); a range outside the tensor raises `km.KomorebiError` naming its page count |
| `session.read_chunks(tensor, chunk_bytes=2 << 20)` | a generator of `(first, pages)` over the whole tensor, each chunk at most `chunk_bytes` |
| `session.read_tiles_bf16(tensor, first, pages)` | tiles `first` .. `first + pages - 1` of a bfloat16 tile tensor (one tile a page, row-major tile order) as float32 `[pages, 32, 32]` |

Reading back is slow: the card test reads a 300-page tensor (614,400 bytes) whole in
0.29 s and in ten chunks in 0.30 s, about 2 MB per second (host load about 7). A long
read is silent, and a job watcher that kills on silence cannot tell it from a hang. So
read a large tensor with `read_chunks` and log a line per chunk, and check a large
result by sample: the pages of a few rows or tiles, from `read` or `read_tiles_bf16`.

`run.thread.read(name)` reads only the region's pages of the work cluster's bank (over
4 MB, in 2 MB chunks with a log line each): a 1 MB region in 0.48 s, 5 MB in 3.8 s
(`experiments/x280_threads/region_read.py`, host load about 4). The load still writes
the regions' whole tensor, 8 MB per MB of regions: no call the host uses writes a page
range.

## `Run`

| Call | What it does |
|---|---|
| `issue(seg, team=None, args=(), exclusive=False, result=False, queued=False)` | queues one record and returns its ticket (1, 2, ...). `seg` is a name or a number (`0` is NOP); `team` is `(first, count)`, default every core; `args` up to ten 32-bit words. Teams and segments are checked here. With `exclusive` (`KM_F_EXCL`), the dispatch kernel issues the record only once no record in flight has a team that overlaps it; the host stores it as soon as the ring has room, like any other record (below, "Exclusive records"). With `result` (`KM_F_RESULT`), its done event carries the team leader's 32-bit result word. With `queued` (`KM_F_QUEUE`), `team` is a count: the kernel gives the record the lowest first core whose team overlaps no busy one, and waits until there is one (nothing is refused for a full grid); `team(ticket)` is then `(None, count)` until its done event names the first core (8 bits: at most 256 cores; `KM_CORES_MAX` is 128). A queued record cannot be exclusive |
| `issue_many(seg, firsts, counts, args=None, exclusive=False)` | queues n records at once and returns their tickets as a `range`. `seg`, `firsts` and `counts` are one value or n each; `args` is an `[n, m]` integer matrix (m at most 10) or None. Checked as `issue` checks, all before any is queued |
| `run_hostfed(graph, bindings=None, timeout=10.0)` | runs a [graph](../graphs.md) host-fed: its executor on the host, its records through this run; returns `HostfedResult(status, error, regs, runs, records, events)` |
| `wait(tickets, timeout=10.0)` | stores queued commands up to the tickets in batches, reads events until each ticket has one; returns the `Event` (or a list) and forgets those tickets. The stores and the event reads run in the native module with the GIL released |
| `wait_any(tickets, timeout=10.0)` | stores what is queued and waits until one of the tickets has its event; returns that `Event` (the lowest ticket if several arrived) and forgets only that ticket. Raises as `wait` does |
| `drain(timeout=10.0)` | the same for everything issued; returns how many events it waited for |
| `event(ticket)` | `Event(ticket, kind, aux, ticks, result)` or None; `kind` is `done`, `timeout`, `pong` or `refused`; `aux` is the first core of the team that ran the record (done; for a queued record, the one the kernel chose), the slot (timeout) or the reason (refused): the low 8 bits of the event's aux, whose next 16 bits, the low bits of the record's sequence number, `seq(ticket)` reads (layout: the kernel README, "Sources"); `ticks` is issue to completion in the X280's 50 MHz timer, `us` the same in microseconds; for a record issued with `result=True`, `result` is the leader's word (the kernel sends it in place of the time, so `ticks` and `us` then hold it too), else None |
| `seq(ticket)`, `team(ticket)` | the sequence number the kernel gave the record (the trace's index), from its done or timeout event (None before it, and for a refused record); its team, whose first core for a queued record comes from its done event (None before it) |
| `forget(tickets)` | releases what the run keeps for tickets whose events arrived; for callers that read events with `event()` after `drain()`. A traced run keeps teams and sequences for the trace |
| `stop(timeout=10.0)` | `timeout` is one deadline for all of it: cancels every engine still running (or `unknown`) and waits for their ends, drains, sends EXIT to every core and waits for it, sends the kernel STOP, waits for its result, closes the X280 runtime, waits for every core's kernels to return, then reads the state words and the trace. Returns the result block; raises `KernelFailed` unless the result is ready, cluster 0 restored, the state DONE and the error 0. Any error on the way (an `EngineTimeout` of a cancel not answered, a `WaitTimeout`) still closes the runtime first |
| `thread_result(timeout=10.0)` | a program with an X280 thread: waits for it (or stops it), returns its `ThreadResult`; after `stop()`, `run.thread.read(name)` returns a region's bytes |
| `abort()` | the failure path: stops the grid from the host (bounded), then the kernel through the runtime, closes it, reads back; `result()` then has `abandoned`; raises `KernelFailed` if the runtime could not restore cluster 0, with the runtime's account of each hart (phase, stage, trap cause and address) in its message and the session log |
| `result()`, `state()`, `state_words()`, `trace()` | after the stop: the kernel's result block (decoded), each core's state words as a dict (with `ports`: count and wanted count of every port slot, and `lanes`: each lane's VALID word, first give-up, blocks, wait cycles and DONE-BY words; a reader or writer that gave up waiting on a port or an emit names its `port`), the raw state words, the trace |

`run.issue_s` adds up the host's time in `issue` and `issue_many`; `run.wait_s` the
time in `wait` and `drain`, of which `run.native_s` was in the native loop.

`trace()` returns a `komorebi.state.Trace`: `record(ticket)` gives each member's
wall-clock words (taken, outputs written, completion sent, ring passes);
`phases(ticket)` gives microseconds after the first member took it; `lead(a, b)` is
how long before ticket `a`'s completion a member of `b` took its record (positive: both
were in flight at once).

## Engines: uploads, `start`, events

The dispatch kernel runs [streams](../streams.md) and [graphs](../graphs.md) from an
upload in its upload region, with no host step between the start and the end event.
These `Run` calls write the upload, start it, and turn its events into values.

```python
from komorebi.engines.graph import Graph

step, left, right, norm = (program.seg(n) for n in ("step", "left", "right", "norm"))
g = Graph(deadline_us=200_000, max_runs=50, req=6, keep=True)
state, it, limit = g.bindings("state", "it", "limit")
err = g.reg()
a = g.record(step, team=(0, 60), args=[state, it])
b = g.record(left, team=(0, 30), args=[state], after=[a])
c = g.record(right, team=(30, 30), args=[state], after=[a])  # in flight with b
d = g.record(norm, team=(0, 60), args=[state, 60], after=[b, c], result=err)
e = g.cpu(after=[d])
e.addi(it, it, 1)
e.lt(g.again, err, limit)  # go round again while err < limit

up = run.upload(g)
ev = run.wait_engine(run.start(up, {"state": x, "it": 0, "limit": 10**6}), 10.0)
ev = run.wait_engine(run.restart(6, {"state": y, "it": 3, "limit": 10**7}), 10.0)
r0, r1 = ev.regs  # register 0 (its low 24 bits) and register 1 at the end
run.release(up)  # the kept graph is dropped, its lines are free again
```

This is the graph of `examples/06_graph` (the same segments; `x` and `y` are the
addresses of two state tensors): a kept graph, run, then run again with other bindings
and no new upload.

| Call | What it does |
|---|---|
| `upload(obj, kind=None)` | encodes a `Stream` or a `Graph` (or takes its encoded bytes, with `kind` `"stream"` or `"graph"`) and writes it into the upload region as whole lines, at the first free place; returns an `Upload` (`offset`, `bytes`, `kind`, `req`, `keep`, `state`). Several uploads live side by side. An upload that does not fit raises `ValueError` before anything is written. The host checks nothing else: a bad upload is the kernel's to refuse, with its error event |
| `release(upload, timeout=10.0)` | frees an upload's lines. A running one is refused; a kept graph is first dropped (`CANCEL`, then its cancelled event, both within `timeout`) |
| `start(upload, bindings=(), kept=None, timeout=10.0)` | sends `KM_SEG_RUN` and returns an `EngineTicket`. A graph takes `bindings`: a dict by name for a graph uploaded from a `Graph` (its constants are added, as `binding_values` does), or a list of up to 16 words; they go to one line of the region, freed at the end. `kept=True` or `False` rewrites the header's keep bit first; None leaves it. Another running or kept engine with the same `req` raises `ValueError`. When the kernel does not answer the RUN within `timeout` it raises `EngineTimeout` with the ticket marked `unknown`: the engine may have started, and `cancel` (or `stop`) settles it |
| `restart(req, bindings=(), timeout=10.0)` | runs the kept graph with `req` again with new bindings (`KM_SEG_RUN`, `a2 = 2`) |
| `cancel(ticket, timeout=10.0)` | sends `KM_SEG_CANCEL`; the engine then ends with its error event, reason `cancelled`. Nothing is sent for an engine that has ended. For an `unknown` ticket it also learns the RUN's answer: raises `EngineRefused` if the RUN was refused (it never ran); otherwise the engine runs or has ended as usual. `EngineTimeout` if the CANCEL is not answered |
| `wait_engine(ticket, timeout=10.0)` | waits for the engine's end and returns its `EngineEvent`; raises `EngineError` for an error event, `EngineTimeout` when `timeout` passes (the engine may still run), `KernelFailed` if the kernel finished |
| `run_hostfed(obj, bindings=None, timeout=10.0)` | runs a graph or a stream on the host instead, its records through the host ring: the same records and registers as in the kernel ([graphs](../graphs.md#host-fed)); for a stream, the simulator's interpreter, `records[i].node` the program counter of the `ISSUE` or `SUBMIT`, `runs` 1 |

`start`, `restart` and `cancel` send the command and a `PING` behind it and return
when the `PING` is answered: the kernel answers a `RUN` or `CANCEL` with an event only
when it refuses it, and that event comes first. A refusal raises `EngineRefused` (a
`RecordRefused`: `reason`, `reason_name`); the host does not retry.

An `EngineTicket` has `kind`, `req`, `upload`, `events` (every event so far) and `end`.
An `EngineEvent(ticket, kind, value, regs, reason, reason_name, pc, node)` is one of:

| `kind` | From | Fields |
|---|---|---|
| `event` | a stream's `EVENT a` | `value`: register `a`, 32 bits |
| `round` | a graph's round ended and another starts | `value`: rounds done |
| `end` | `HALT`, or a graph's last node | a stream's `value`: the halt register, 32 bits; a graph's `regs`: register 0's low 24 bits, register 1 |
| `error` | any error | `reason`, `reason_name` (the `KM_SERR_*` name in lower case, read from the headers), the stream's `pc` or the graph's `node` (None when the error has no node) |

The engines' events carry the upload's `req`, not a command number: they go to their
tickets, so `wait` and `wait_any` of the host's own records see none of them, and
`wait_engine` none of the records' events. The records an engine issues are answered
to the engine, never to the host. While it
waits, `wait_engine` also writes the host's read count into the kernel, since an
engine's posts wait for event room.

The region is the one the kernel's ready block names: what is left of X280 cluster 0's
local memory after the kernel's layout (707,840 bytes on a p150a), none for a program
with an X280 thread, whose ring lies there. The host never writes into the range of a
running stream or graph, or of a kept graph. The kernel's EXIT waits until no engine
runs, so `stop` cancels the engines still running first.

The kernel numbers the records of every source (the host, an engine, a work-cluster
thread) in one sequence. A record's done or timeout event carries the low 16 bits of its
number, and the host takes the rest from its previous record's: `seq(ticket)` and the
trace's `record(ticket)` are exact while fewer than 32,768 records of other sources are
issued between two host records. When the host sees a longer gap it raises
`KomorebiError` for that ticket instead of giving another record's number; it cannot
tell a gap of 65,536 or more. A queued record's number may fall up to 2,048 behind the
host's order (the queue lets later jobs go first); a lag beyond that is read as a gap
and raises too.

## Errors

All derive from `km.KomorebiError`.

| Error | When | Facts it carries |
|---|---|---|
| `CardBusy` | `Session.open`: another live process holds the card (below) | `pid` of the holder |
| `ProgramError` | a kernel did not compile or the launch failed | the build message |
| `KernelSizeError` | the program's kernels do not fit the session's kernel buffer; a `ProgramError`, raised before anything is loaded | `sizes`: text per thread, total, buffer |
| `KernelInstructionError` | a kernel's ELF holds an instruction its core lacks ([below](#instructions-the-cores-lack)); a `ProgramError`, raised before anything is loaded | `findings`: thread, instruction, function, source line |
| `StartError` | the image is missing, cluster 0 is not idle, a base handed to the kernels (control block, X280 local memory, its upload region) is not a multiple of 16 bytes, or the dispatch kernel did not report READY | `result` (the kernel's result block, when it ran) |
| `WaitTimeout` | a host-side wait ran out; the run is still loaded. The message ends with what the kernel was doing: whether core 0 still runs its loop (its pass stamp, read twice 20 ms apart), its state and where, and the runtime's account of each hart | `pending` tickets |
| `RecordRefused` | the kernel consumed the command and issued nothing (a reserved segment or flag, a team outside the grid, a record after EXIT); the run goes on | `ticket`, `reason` (`KMD_REFUSE_*`), `reason_name` |
| `RecordTimeout` | a team missed its deadline; the kernel stopped the grid, the run is stopped and read back | `ticket`, `slot`, `seq`, `team`, `cores`: each member that did not report, with where its reader and writer gave up |
| `KernelFailed` | the dispatch kernel finished with any other error, did not finish cleanly, or sent an event for a command not in flight | `result` |
| `EngineRefused` | the kernel refused a `RUN` or `CANCEL` (four streams or two graphs running, no such req, a command after EXIT) | as `RecordRefused` |
| `EngineError` | a stream or graph ended with an error event | `event`, `ticket`, `reason`, `reason_name`, `pc`, `node` |
| `EngineTimeout` | `wait_engine` ran out, or a RUN or CANCEL was not answered; a `WaitTimeout` | `ticket` |
| `GridTimeout` | the grid's kernels did not return within `grid_timeout`; the runtime closes the device at the process's exit | |

A card stays held by a process from its first device open until that process exits,
even after the device is closed: tt-metal's driver takes the chip's lock at the first
open and gives it back only when the process ends. A second process that opens the card
meanwhile would wait without a bound, so `Session.open` first reads the lock's owner
(from the driver's lock file, without taking the lock) and raises `km.CardBusy` naming
that process when it still maps the file: a process that died holding the lock, a
zombie or a PID reused since is not taken for the holder. A test run that opens the
card in its own process therefore cannot start a card script as a child (see
`tests/hardware/README.md`).

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

### Instructions the cores lack

The compute threads are built for a target that includes a float unit (the pack
thread's vector flags imply it), so the compiler may emit instructions the Tensix
cores do not implement. A core that meets one stops without a report and the program
ends in a timeout that names no cause. After the kernels are built and before anything
is loaded, `Session.load` therefore disassembles each thread's linked ELF with the
toolchain's own objdump and raises `km.KernelInstructionError` naming the thread, the
instruction, the function and, from the debug info, the source line. Refused are the
groups the core's instruction set documents as absent: float divide and square root
(`fdiv`, `fsqrt`), vector integer divide and remainder (`vdiv`, `vrem`), and vector
float divide, square root and their estimates (`vfdiv`, `vfrdiv`, `vfsqrt`,
`vfrsqrt7`, `vfrec7`). The result is kept with the build, so a cached build is not
checked again. The check itself runs objdump once per thread ELF the first time that
build is seen, about 11 ms each (some 0.06 s for a program's five); the runtime's
compile of a new kernel, which now runs just before it, is most of a first load
(about 0.9 s for example 01). Without the toolchain's objdump, or when the build left
no ELF, the load raises `ProgramError`; without its addr2line, a finding is still
refused, without its source line. A `/` on floats is the
usual cause: divide by an integer (`div`/`divu` are implemented), multiply by a
reciprocal computed on the host, or, on the vector unit, use a reciprocal with Newton
steps as the projects do ([`recip`](../../projects/raytrace/kernels/sfpu.hpp), exact
division and square root in [`divsqrt.hpp`](../../projects/raytrace/kernels/divsqrt.hpp)).
`KMR_INSN_CHECK=0` turns the check off, for a kernel that never executes the code
found.

## Rules the host keeps

- The host never reads or writes a core's memory while the program runs, with one
  exception: the stop word, when the dispatch kernel cannot send it. State words and
  the trace are read after the stop.
- While the dispatch kernel runs, every host wait is bounded. The X280 runtime is
  closed before any readback, on every failure path, at the process's exit and on
  SIGTERM, always before the device.
- Host stores into the card go in batches: every line, one fence, then a pause of
  `settle_us` before the next store.
- Unread events plus commands in flight never exceed the event ring.
- EXIT goes to every core, after everything in flight has completed.
