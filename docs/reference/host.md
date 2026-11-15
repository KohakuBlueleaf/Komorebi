---
title: Host API
summary: Program, Session and Run, the Python interface that loads a program onto every Tensix core under the dispatcher, submits commands to groups, waits for them, and reads back state and trace.
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
    handles = [run.submit("map", group=(job * 8, 8), args=[...]) for job in range(15)]
    run.wait(handles, timeout=5.0)
    run.stop()
    trace = run.trace()
```

`import komorebi` needs no tt-metal tree. `km.Session` loads the package's native module
on first use; neither ttnn nor torch is imported. Data goes to and from the card as
numpy arrays (see [Tensors](#tensors)).

## `Program`

`km.Program(name, kernels, buffers=[], args=None, compile_args={}, compute={},
defines={}, depth=4, handlers={}, opt={}, noc={})` describes what the persistent kernel runs. It opens
nothing.

| Field | Meaning |
|---|---|
| `kernels` | a directory with `reader.cpp`, `compute.cpp` and `writer.cpp` (or `movement.cpp` and `compute.cpp`, [movement](movement.md)), or a dict of the paths. Each source defines `kernel_main` (the kernel form) or `km_program` (the runtime's form, [below](#several-programs-in-one-run)) |
| `buffers` | `km.Buffer(index, format, page_bytes, pages)`: circular buffers on every core; `format` is a data type name (`"bfloat16"`, `"float32"`, `"int32"`, `"uint32"`, `"uint16"`, `"uint8"`, `"bfloat8_b"`, `"bfloat4_b"`) |
| `args` | `args(core) -> (reader, writer, compute)`: the program's own runtime arguments of each core, after the runtime's start-up arguments |
| `compile_args` | `{"reader": [...], "writer": [...], "compute": [...]}` |
| `compute` | compute kernel options, the fields of tt-metal's `ComputeConfigDescriptor` (v0.81.1): `math_fidelity` (`"LoFi"`, `"HiFi2"`, `"HiFi3"`, `"HiFi4"`), `unpack_to_dest_mode` (`"UnpackToDestFp32"` or `"Default"` for every circular buffer, or `{cb: name}` with the others `Default`), and the flags `fp32_dest_acc_en`, `math_approx_mode`, `dst_full_sync_en`, `bfp8_pack_precise`, `enable_trisc2_rvv`; any other name raises `ProgramError` |
| `defines` | `{name: value}` for all three kernels |
| `broadcast_wait` | `True` (the default): a broadcast without a buffer waits at `finish` until every receiver cleared its VALID, +0.44 µs a command that casts a word; `False` builds the kernels with `KM_BROADCAST_WAIT` 0, and the program then never casts on a broadcast's cores right after a command that cast a word there ([broadcast](broadcast.md#a-broadcast-without-a-buffer)). A define `KM_BROADCAST_WAIT` that says otherwise raises `ProgramError` |
| `depth` | mailbox slots the cores scan and the dispatcher uses, 1 to 64 (`KM_RING_SLOTS`; [the cost of depth](../../experiments/x280/dispatcher/README.md#ring-slots-and-depth)) |
| `handlers` | `{name: number}`, or a header whose `constexpr uint32_t kHandlerName = n;` lines give them (`kHandlerBigStep` becomes `"big_step"`); numbers 1 to 239 |
| `opt` | `{"reader": "Os", "writer": "Os"}`: an optimisation level per kernel, one of the runtime's `O0`, `O1`, `O2`, `O3`, `Os`, `Ofast`, `Oz` (anything else raises `ProgramError`); a kernel without one keeps the runtime's default, `O2` for the reader and writer and `O3` for the compute kernel (its three threads together). A set level also defines `KM_OPT_LEVEL` for that kernel, so that it is cached apart |
| `noc` | `{"reader": 1, "writer": 0}`: the NoC each data-movement kernel uses, 0 or 1; without it the reader is on NoC 0 and the writer on NoC 1. Both on one NoC, or a name other than `reader` and `writer`, raises `ProgramError`; programs that share cores or switch groups give the same `noc` ([movement](movement.md#the-noc-of-each-cpu)) |

`program.thread(cluster=1, sources, regions, words, host_fallback, deadline,
prefetch="documented", handlers={})` adds an
X280 thread on a work cluster (one per cluster; `clusters=(1, 2, 3)` declares it on
each, `program.region(name, value)` a region shared by threads on one bank); see
[X280 threads](../threads.md). `handlers` names the commands its `kmx_program` answers,
as `handlers` above does for the kernels ([commands to a
thread](threads.md#commands-to-a-thread)).

A program's three kernels share the session's kernel buffer, 70,656 bytes on Blackhole
unless the session sets `kernel_bytes`. When a program is loaded its build is laid out
against that buffer: `program.sizes` (a `km.KernelSizes`) holds the packed text per
thread, the total and the buffer, and the session log has them in one line. A program
that does not fit raises `km.KernelSizeError` before anything is loaded, naming the
total, the buffer and the largest thread (in place of the runtime's "Program size
... too large for kernel config buffer"). Give the session more with `kernel_bytes`,
or lower the reader and writer to `"Os"` first and the compute kernel only when that
is not enough, since its hot loops are what `O3` is for.

A program's own header (its handlers, buffers and argument indices, shared by its
kernels and its driver) is read with `km.abi.constants(path,
known=km.abi.headers().constants)`: every `constexpr uint32_t k*`, `#define KM*_*` and
enum value, as a dict; it may use the package's constants by their C++ names
(`km::kBroadcastArgs`). A constant the reader cannot evaluate raises `abi.HeaderError`.

The runtime owns circular buffers 29 and 30 (the command's token for compute and
writer), 31 (the trace) and 32 (a chain's words); a program uses 0 to 28.

`program.handler(name)` returns a `km.Handler`: the handler's number (an `int`) that also
names its program (`.program`). `submit`, `submit_many`, `graph.command` and a stream's
`submit` take it, and put the program's place in the run, the **program field**, into
bits 24 to 31 of the command's header word (`.field`; 0 is the run's first or only
program). A name or a plain number is a handler of the run's first program, so a run of
one program is written as before. Each program numbers its handlers 1 to 239 on its own:
two programs may both have handler 1.

### Several programs in one run

A program in the **runtime's form** has a source per kernel that defines, in place of
`kernel_main`, the function the runtime's command loop calls for a command of that
program:

```c++
#include "komorebi/device/km_program.hpp"

void km_program(km::Reader& reader, const km_command_t& command) {
    switch (km::handler(command)) {
        case kHandlerTrace: trace(reader, command); break;
    }
}
```

and the same with `km::Writer&` (it may return `bool`: false sends no completion, the way a
writer body's false does) and `km::Compute&`. The host tells the forms apart from the
sources: one that defines `km_program` (comments aside) is in the runtime's form, any
other is in the kernel form (`kernel_main`). A program whose sources mix the forms, or a
source that defines both, raises `ProgramError`. The runtime owns `kernel_main` and the loop; each program is
an [overlay](overlays.md#a-program-as-an-overlay) under them.

```python
run = session.load({tracer: (0, 88), expert: (88, 32)})  # each group its own program
run = session.load([tracer, expert])  # combined: every core holds both
run.submit(tracer.handler("trace"), group=(0, 88), args=[...])
```

| `session.load(...)` | What the cores hold |
|---|---|
| `program` | the kernel form: its kernels as before; a program in the runtime's form, the runtime's kernels with that program |
| `{program: (first, count), ...}` | each group its own program and buffers (disjoint groups; a core outside them holds no program) |
| `[program, ...]` | every core every program, in one L1 layout: buffer indices disjoint, or an index both declare declared identically (then one shared buffer); otherwise `km.L1LayoutError` with the L1 budget table |

The programs' order (the list, or the dict's) is their program field. Programs that share
cores share compute options (`compute`) and `opt` levels; per-core `args` are taken when a
group holds one program. In the runtime's form a program has no dynamic handler; its own
overlays are numbered after those of the programs before it
([overlays](overlays.md#several-programs-with-overlays)). A run holds up to 63 programs. `run.programs` lists them by program
field.

**Threads in a run of programs.** A run's X280 threads and ports belong to one program:
whichever of the run's programs declares them (any one, not only the first). A second
program that declares threads or ports is refused (`ProgramError`, "belong to one
program of a run"). The thread images are built once for the run, with each program's
field and handlers as C defines when the run holds several
([threads](threads.md#a-thread-in-a-run-of-several-programs)); `run.submit(...,
cluster=c)` names a handler of that program's thread.

The host refuses (`ValueError`, before anything is sent) a command whose group has a core
that does not hold its program, a queued one unless every core holds it, and a handler of
a program the run does not hold; a stream's or graph's upload is checked the same way for
its commands with a fixed group (a group from registers is not checked). In a run
loaded with a dict the dispatcher refuses the same commands from any source (a stream, a
graph, an X280 thread, a group from registers): `CommandRejected`, reason 13 (`program`),
and the run goes on. `run.check_programs = False` leaves the refusal to the dispatcher.

#### A group switches program

In a run loaded with a dict, a group of cores can change program while the run is live:

```python
run = session.load({tracer: (0, 88), expert: (88, 32)})
handle = run.load(expert, group=(60, 28))  # those cores now hold expert
run.submit(expert.handler("refine"), group=(60, 28), args=[...])
```

`run.load(program, group=(first, count))` is a command (reserved handler 240, from the
host or an X280 thread) like any other: it returns a handle, completes with an event and has the
run's deadline. `program` is one of the dict's: its parts are built and in DRAM, so no
code moves from the host. The cores of the group take it after their earlier commands;
the dispatcher holds it until no command in flight shares a core with it and the job
queue is empty (it is exclusive, never refused for a busy group). From then on the
host and the dispatcher refuse commands for the program those cores held, and take
commands for `program`.

- **What is lost.** The cores' buffers are set up again for `program` (its indices, sizes
  and formats; the runtime's own buffers stay): whatever the old program kept in its
  buffers is gone. A load waits until every buffer of either program is empty.
- **What must fit.** A core holds one program's buffers at a time: the session reserves,
  on every core, room for the largest buffer layout among the dict's programs, beside
  the slots for every program's parts; `km.L1BudgetError` with the L1 budget table when
  it does not fit. Buffer indices are below 32.
- **One build of the runtime's kernels.** A program's parts run on any group's kernels,
  so the groups' runtime kernels must be the same build: programs of a dict share
  compute options and `opt` levels (else `run.load` raises `ProgramError` and says
  which symbol differs). A run loaded with one program or with a list has no `load`.
- **A thread may load too.** An X280 thread of the run gives a group another program
  with `kmx_load` ([threads](threads.md#a-thread-gives-a-group-another-program)). The
  host then no longer knows which program each core holds, so in a run loaded with a
  dict that has a thread on the card the host's own check is off
  (`run.check_programs` starts `False`): a command whose cores do not hold its
  program is refused by the dispatcher (reason 13, `CommandRejected` naming
  `program`), which sees every load from every source. Set it `True` for a run whose
  thread never loads.

The session sends each group a load of its own program before `session.load` returns
(the compute cores' hardware start-up on that program's formats). What it costs: the
[example](../../examples/16_two_programs/README.md) and [overlays](overlays.md#a-group-switches-program).

## `Session`

`km.Session(card=None, device=None, pll_mhz=None, tree=None, log=None,
grid_timeout=10.0, kernel_bytes=None, channel=256 << 20, slow_code=False)`, or
`km.Session(cards=[...], ...)` over several cards (below and [Islands](islands.md)).

- `cards=[a, b, ...]`, given alone (without `card` or `device`), gives one session over
  them, each card's chip an island with flat ranks ([Islands](islands.md), which lists
  that session's own arguments); `cards=[a]` is that session over one island, `card=a`
  the card's own session. `card=a, islands=n` (2 to 4) is that session over the one
  card split into n islands ([a card split into islands](islands.md#a-card-split-into-islands)).
- `pll_mhz` is the X280 clock. The runtime's table is 15, 200, 800, 1000, 1250, 1350, 1500
  and 1750 MHz; a runtime built before 1250, 1350 and 1500 were added has 15, 200, 800,
  1000 and 1750. One PLL clocks all four X280 clusters, so the work
  clusters' threads run at it too. Not given, the session uses 1350. A runtime that does
  not offer 1350 answers "no PLL solution" before it touches the clock; the session
  then uses 1000 and logs that once, and `session.pll_mhz` names the clock in use after
  the first `load`. A clock that was given and is not offered is an error whose text
  lists the clocks the runtime offers. 1750 is an opt-in: it loses loads when the Tensix
  grid is under full matrix load (the card lowers its core voltage; the dispatcher
  trapped in 9 of 18 such runs), and it is never the default; safe use is the caller's
  claim. A host-fed command at depth 16 costs 1.228 microseconds at 1000 MHz and 0.726
  at 1750, one after the other on one card (`experiments/x280/dispatcher/measure.py bench
  --n 100000 --depth 16 --group 1 --batch 64 --host-depth 64 --pll 1000`, then `--pll
  1750`). Before the session sets the clock, the runtime that opens the cluster leaves
  it at its safe 200 MHz. The session's log names the clock it set.

- `card` selects the card through `TT_VISIBLE_DEVICES` (an error if that variable
  already names another card); `device` uses a device (the native module's `Device`)
  the caller opened and keeps. The session opens the card in slow dispatch, as a 1 x 1
  mesh through tt-metal's own API, with the program cache off.
- `channel` is the session's region of the host channel: channel offsets `[0, channel)`,
  a multiple of 64 KiB up to 1 GiB; `0` or `None` uses none. Reads and writes of DRAM
  tensors go through it ([below](#the-host-channel)).
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
  has a timeout (5 s in the runtime). SIGKILL cannot be caught: after one, reset the card.
  With `device=`, close the session before the device.
- What an abort (an exception inside `with`, `session.close()` of a loaded run, the
  exit hooks) does: it writes the stop flag into every core and waits up to 2 s for
  the grid, then asks the X280 runtime to stop the dispatcher, which finishes
  whatever is in flight (5 s timeout); `run.result()` then holds its result block, with
  `abandoned` the commands left without their `KM_EVENT_DONE`. The card needs no reset
  after it. An exception in the kernel needs none either: the kernel finishes on its
  own and the run raises `KernelFailed` with error `trap`, naming the X280 core,
  `mcause`, `mepc` and `mtval`. After an abort or a kernel failure the session log
  also holds the first 512 bytes of the kernel's state as DRAM holds them
  (`kernel state +0x...` lines). A reset is still needed when the kernel's core 0 has
  stopped running its loop (a `WaitTimeout` says so): the runtime cannot restore
  cluster 0, and `KernelFailed` carries the runtime's error message for each core.
- `grid_timeout`: the timeout, in seconds, of every wait for the grid's kernels. When
  it passes, `km.GridTimeout` is raised and the session does not close the device: the
  runtime closes it at the process's exit, waits for the cores up to
  `TT_METAL_OPERATION_TIMEOUT_SECONDS`, then puts them in reset. A thread that is only
  stuck (a loop, a blocked wait) is recovered by the next open; a kernel that left a
  compute unit running is not, and when the next open fails the card needs a reset.
  The variable is read once, at the process's first device open, and is the timeout of
  every wait the runtime makes for the cores in that process, not only the exit's; unset,
  it means no timeout, and only then would the close wait without one. A session that
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
- `slow_code` lets a load place code at or above 1 MB of L1. A core's code (the
  kernels, then the overlay slots as the run's first circular buffer) sits at the
  bottom of L1 and its circular buffers above it, because a taken branch to code at or
  above 0x100000 costs about 4 cycles more; a load whose slots would end there raises
  `km.SlowCodeError` unless `slow_code=True`.

`session.load(program, depth=None, trace=0, deadline=1.0, wait=2.0, idle=0.0,
kernel_idle=0.0, start=10.0, host_depth=64, event_depth=128, batch=64, settle_microseconds=1.0,
image=None, thread="card", island=None, link=None, tensors=0, recover=True,
reserve=262144)` returns a `Run`:

1. finds or builds the dispatcher image and the native module (see below), and
   checks that the image is there and X280 cluster 0 is idle (else `km.StartError`);
2. opens the X280 runtime and lays out the kernel's memory;
3. launches the three kernels on every core. Kernels compile here, before any X280
   image exists, so a compile error raises `km.ProgramError` and leaves nothing
   running;
4. starts the dispatcher and waits (with `start` seconds as its timeout) for every column of
   the grid to report READY; otherwise `km.StartError`, which carries the kernel's
   result block;
5. with `recover`, hands the dispatcher every core's reset list (`run.keep_resets()`,
   below).

From the launch on, every failure path reads the grid back. When the dispatcher
is not there to stop the grid (it never started, or it ended before it could reach the
cores), the host writes the stop flag into every core's control block itself, through
the device API, and then waits for the grid within `grid_timeout`.

| Argument | Meaning |
|---|---|
| `trace` | sequence numbers traced per core (0: no trace); 16 bytes of L1 each |
| `deadline` | seconds a command may take from submit to its group's completion, waiting behind earlier commands included |
| `wait`, `idle` | the cores' timeout on a wait for another core, and on a wait for the next command (the longest the host may stay silent); `idle` 0, the default, is no limit |
| `kernel_idle` | the dispatcher's own idle timeout; 0 turns it off |
| `host_depth`, `event_depth` | the command ring in the X280's memory and the event ring in host memory; powers of two, at most 64 and 128 |
| `batch` | commands stored per batch: every line, then one fence |
| `settle_microseconds` | the host's pause between a fence and its next store into the card |
| `thread` | for a program with an X280 thread: `"card"` builds and starts it after READY, `"host"` runs its `host_fallback` instead |
| `recover` | hand the dispatcher every core's reset list at load, so that a give-up resets and loads again each core that did not leave the command (default True). It costs the load about 4 ms and 68,864 bytes of the upload region for the map/collect example (the list grows with the kernels and their arguments). One list a group of cores with the same kernels (several programs: each core the list of its program); on islands each island's own, link cores left out. Not done for program switches, nor for a program with semaphores: `run.recovery_refused` says why |
| `reserve` | bytes of room for code added to the live run (`Run.add`), after the overlays' parts in their DRAM tensor (each part is stored in every bank; default 256 KiB). 0 reserves none, and a load over islands reserves none ([code added to a live run](overlays.md#code-added-to-a-live-run)) |
| `island` | `(graph, i)`: this card is island `i` of a `komorebi.islands.Graph` whose island `i` has every core a rank in core order. Every core's island area holds island `i`'s header and table, the dispatcher gets `i`'s first rank and count (`kmd_args.island`), and the run's groups are flat ranks of the whole graph (`Run.ranks`); a group's part on this card runs ([islands](runtime.md#islands)) |
| `link`, `tensors` | an island's own: its `LinkPlan` (the link core and link area its commands across ride; needs `island`) and the L1 address of the tensor table its cores hold. The island sessions pass them; a program on one card never does |

Event room. The dispatcher takes a command that ends in an event only while the
unread events, one per busy slot and one more fit the event ring; it counts as read
what the host last wrote into its read-count line. The host keeps a tighter rule: it
stores a command only while the events it has not read plus the commands it stored
without an event fit the event ring. It writes its read count with every batch, and
while it waits whenever the kernel could be holding a command for it, so the kernel
waits for room only while that count is on its way.

Exclusive commands. The host stores a command with `KM_FLAG_EXCLUSIVE` like any other; the
dispatcher holds it at the head of its command ring while a busy command's group
shares a core with it, and submits it once that command's leader has acknowledged. Host
commands behind a held one wait with it, whatever their groups. Its deadline counts from
the kernel's submit, not from the store. So back-to-back exclusive commands on one group
are the kernel's boundary apart, while one submitted after idle still pays the host's store
reaching the X280 (a round trip of about 14 us). Measured on 2026-11-13, one p150a, X280
at 1,000 MHz, host load 3 to 9, before (the host held each exclusive command until it
had read the event of the overlapping one) and after, alternating in one session:

| Commands back to back, exclusive | Before (us) | After (us) |
|---|---|---|
| NOP on 120 cores, period | 16.25, 16.26 | 2.27, 2.27 |
| NOP on 12 cores, period | 4.28, 16.21 | 2.28, 2.28 |
| NOP on 1 core, period | 15.95, 15.96 | 1.90, 1.91 |
| `examples/03_broadcast`, 120 cores, per job | 61.1, 60.4 | 47.9, 46.9 |

The period is the first core's take of one command to that of the next (the cores'
trace, median of 99, after 10 commands of warm-up). The broadcast row is
`broadcast.py --rect 0,0,12,10 --blocks 16 --pages 4 --window 2 --jobs 40 --no-check`
(host time per job). The 4.28 is a run in which the host's stores reached the X280 fast,
a state seen now and then.

Cores' NoC coordinates. `session.nocs()` returns a `layout.Nocs` for the open device,
before any load: `grid`; by core `(x, y)` the translated NoC address `noc_xy` (the one
the dispatcher's core table holds) and the NoC0 coordinates `noc0`; `words`, the
translated addresses as `x | y << 16` in core order; and the grid's `panels`. These are
what `layout.broadcast_args` and program arguments that name other cores take.

### Tensors

A program's inputs and outputs are tensors the session puts on the card:

| Call | Returns |
|---|---|
| `session.tensor(data, dtype, layout="row", shard_grid=None)` | a `Tensor` holding `data` (anything `np.asarray` takes) as `dtype` (`"bfloat16"`, `"float32"`, `"int32"`, `"uint32"`, `"uint16"`, `"uint8"`), written before it returns. `layout="row"`: a page a row (the last dimension), interleaved over the DRAM banks; `"tile"`: a page a 32 x 32 tile of four 16 x 16 faces, the last two dimensions zero-padded to multiples of 32. `shard_grid=(x, y)`: rows `[cores, words]`, one in the L1 of each core of the grid at one address |
| `session.empty(shape, dtype, layout="row", shard_grid=None)` | the same, not written |
| `session.grid` | the Tensix grid `(x, y)` of the open device |

A session over several cards (`km.Session(cards=[...])`) takes, in `tensor` and
`empty`, a hint for where a tensor's pages live, in the kernel's words: `split=d` and
`group=(first, ranks)` (the group's ranks partition dimension `d` as `km::partition`
does, each rank's part on its island; default every rank), `like=other` (as `other`
is placed) or `copy=True` (on every island); no hint is `split=0` over every rank.
`tensor.address` is then the tensor's reference word, which a kernel hands to
Komorebi's page calls; on one card it is the bank address
([islands](islands.md#tensors)). A hint whose part does not fit an island's DRAM
raises `km.KomorebiError` before anything is written, naming the island, the pages
and shape asked and the most pages of that kind its DRAM takes now. Writes and reads
go through each island's own host channel at once, one host thread an island;
`km.Session(cards=[0, 1], host_reaches=[0])` (a test) carries every part through
island 0's channel, island 1's crossing by the page path, inside a loaded run.

`session.device.mesh_capsule()` hands the device to a native module that takes it: a
capsule named `komorebi.mesh_device` holding a `std::shared_ptr<MeshDevice>`.

A `Tensor` has `address` (its address in every bank, or in every core's L1),
`shape` (on the card, padded), `dtype`, `layout`, `page_bytes`, `aligned_page_bytes`,
`pages`; `accessor_args()` (the compile-time arguments of a `TensorAccessor` over it,
for `Program(compile_args=...)`); `tensor_args()` (those: what a
kernel's `km::Tensor` reads, for the page calls of
[Islands](islands.md#a-kernels-page-reads-and-writes)); `write(data)`; `read()` (the whole tensor as a numpy
array, bfloat16 as float32) and `read_bytes()`; `free()`. Its `write` and `read` wait
for the cores of a launched program to return, which the persistent grid does only at
the stop: before `load` and after `run.stop()` they are the plain way. bfloat16 values
are rounded to nearest, ties to even; `komorebi.formats` has the conversions
(`bf16_bits`, `bf16_values`, `tilize`, `untilize`, `to_bytes`, `from_bytes`) on their
own.

Reading tensors while a run is loaded. `session.read(*tensors)` returns the raw pages
of tensors in DRAM or L1 (`[pages, page_bytes]` uint8 each; an L1 tensor's page is
one core's row) and
`session.read_bf16(*tensors)` bfloat16 tile tensors as float32 arrays. Page p of such a
tensor lies in DRAM bank p % banks, so the native module reads each bank's pages in one
go, after one DRAM barrier for all the tensors of the call. These reads do not wait for
the cores to be idle, so they work while a run is loaded; read then only what no kernel
writes any more (an output whose command's event has arrived). An L1 tensor is read
this way only through the host channel (next section); without it `session.read` reads
it whole through the device API, which waits for the cores, so not while a run is
loaded (`km.ChannelError`).

A part of one tensor is read the same way:

| Call | Returns |
|---|---|
| `session.read(tensor, first=0, pages=None, out=None)` | `[pages]`: pages `first` .. `first + pages - 1` as one `[pages, page_bytes]` uint8 array (`pages=None`: to the end), or into `out`, rows of that shape, which it returns; a range outside the tensor raises `km.KomorebiError` naming its page count |
| `session.read_chunks(tensor, chunk_bytes=2 << 20)` | a generator of `(first, pages)` over the whole tensor, each chunk at most `chunk_bytes` |
| `session.write(tensor, pages, first=0)` | writes `pages` (`[n, page_bytes]` uint8, as `read` returns them) into pages `first` .. `first + n - 1` of a tensor in DRAM or L1, also while a run is loaded; without the host channel only a whole tensor, before a load (else `km.ChannelError`) |
| `session.read_tiles_bf16(tensor, first, pages)` | tiles `first` .. `first + pages - 1` of a bfloat16 tile tensor (one tile a page, row-major tile order) as float32 `[pages, 32, 32]` |

Through the device API, reading back is slow: about 2 MB per second (a 20 MiB frame in
10.1 s). With the host channel the cores move the pages instead, at the link's speed;
the next section says when. Without it, read a large tensor with `read_chunks` and log a
line per chunk (a job watcher that kills on silence cannot tell a long read from a
hang), and check a large result by sample, from `read` or `read_tiles_bf16`.

### The host channel

When a card opens, the driver maps 1 GiB of pinned host memory that the cores reach
through the PCIe tile. The session finds the mapping in `/proc/self/maps`, keeps the
region `[0, channel)` of it and zeroes the region's last 64 KiB: one 64-byte sequence
word a core. The rest holds data. Each transfer moves pages between a tensor and the
region. For an interleaved DRAM tensor up to 16 cores take a contiguous share each and
move it through two chunks of L1 (16 KiB each, or a page each when pages are larger).
For a tensor in L1 each core that holds a page moves it between its own L1 and the
region in one write or read: no DRAM, no chunks. Every move uses tt-metal's PCIe-aware
calls (set, the writes or reads, the flush, clear); then the core writes its sequence
word (a fresh nonce, its bytes, its rank). The host copies out only after every word
arrived. A transfer larger than the region's data goes in rounds, one transfer at a
time.

| The session | A read (`read`, `read_chunks`) or a write (`write`, `tensor`) goes |
|---|---|
| a run is loaded and its cores live | through the run: a command with a reserved handler to cores that hold one program, served by the runtime's service overlay ([overlays](overlays.md#transfers-to-and-from-the-host)); its cost on the host is about 0.2 ms. An L1 tensor takes one command for each run of its cores that are consecutive in core order and hold one program (one for a tensor as wide as the grid, one a row for a narrower one); if one of its cores holds no program it goes through the device API |
| no run, 80 KiB or more | through a one-shot transfer kernel: on row 0 for a DRAM tensor, on the cores that hold the pages for an L1 tensor; its launch costs about 30 ms, as the device API's read of some 76 KiB does |
| no run and less than 80 KiB, no channel | through the device API, as before |

A page over 16 KiB is its own chunk, so it goes through the channel too, in a launch
when two pages fit a core's L1 and in a run when the run's transfer buffer holds one.
A load reserves that buffer: two chunks of the largest page among the DRAM tensors the
session has made, when those take at most half of what a core has left after its
buffers (a run with overlays, or whose groups switch program, keeps 16 KiB chunks). A
tensor with larger pages made after the load, or a page too large for the L1 or for
the region's data, is read through the device API.

What still goes through the device API, and why: a read of less than 80 KiB without
a run, because a launch costs more than the device API's read; and an L1 tensor in a
run where one of its cores holds no program, because no core would serve its page.

Every offset is checked twice before anything moves: on the host (`km.ChannelError`)
and in the kernel, against the region and the core's L1; an L1 tensor's page must also
lie above the firmware's part of L1 and apart from the core's word slot (that floor is
the kernel's check alone). A core that refuses writes nothing. Card 1, a p150a at x8 16 GT/s, host seconds of a whole tensor (median of
three, every byte compared; `experiments/host/host_push/framework.py`):

| | 20 MiB frame read | 94 MB tensor read | 20 MiB write | 94 MB write |
|---|---|---|---|---|
| device API | 10.1 | 43.3 | 0.080 | 0.207 |
| channel, no run | 0.030 | 0.050 | 0.047 | 0.041 |
| channel, in a run | 0.0067 | 0.023 | 0.0055 | 0.026 |

An L1 tensor of 128 KiB a core on all 120 cores (15 MiB), card 1, AICLK 1350 MHz, host
load 4, one measurement each, every byte compared (`experiments/host/host_push/l1_tensor.py`):
inside a run written in 2.8 ms and read in 5.1 ms (5.5 and 3.1 GB/s for the host; 1.09
and 1.05 ms on the cores, 14.4 and 15.0 GB/s); without a run read in 0.11 s, the
launch's cost. The device API reads at about 2 MB per second (above).

The session takes the channel's address from the driver's own record of the card's
host memory and checks it against the process's mappings (1 GiB, read-write, a file
named `tenstorrent` for card 0 and `device_<n>_tenstorrent` for card n). The channel
falls back (reads through the device API, logged once at the open) when the driver did
not map one (no hugepage). A session given a device with `device=` maps the channel
at its first transfer; it did not see the open, so it cannot read the driver's
warning, and a transfer whose sequence words do not arrive raises `km.ChannelError`
within its timeout. If the open logged the driver's warning that the channel's NoC address
does not match, the cores' writes would not land in this process's mapping. The
session then closes the device and raises `km.ChannelError`; `channel=0` opens without
it. The hugepage file keeps its bytes between processes, so the words are zeroed at
every open and each transfer checks them against a fresh nonce.

`run.thread.read(name)` reads only the region's pages of the thread's bank (over
4 MB, in 2 MB chunks with a log line each): a 1 MB region in 0.48 s, 5 MB in 3.8 s
(`experiments/x280/x280_threads/region_read.py`, host load about 4). The load still writes
the regions' whole tensor, 8 MB per MB of regions.

## `Run`

| Call | What it does |
|---|---|
| `submit(handler, group=None, args=(), exclusive=False, result=False, queued=False, cluster=None, block=None)` | queues one command and returns its handle (1, 2, ...). `handler` is a name or a number (`0` is NOP) of the run's first program, or a `km.Handler` of any of its programs; `group` is `(first, count)`, default every core; `args` up to ten 32-bit words. Groups and handlers are checked here. With `exclusive` (`KM_FLAG_EXCLUSIVE`), the dispatcher submits the command only once every queued command sent before it has left the kernel's job queue and no command in flight has a group that overlaps it; the host stores it as soon as the ring has room, like any other command (below, "Exclusive commands"). With `result` (`KM_FLAG_RESULT`), its done event carries the group leader's 32-bit result word. With `queued` (`KM_FLAG_QUEUE`), `group` is a count: the kernel gives the command the lowest first core whose group overlaps no busy one, and waits until there is one (nothing is rejected for a full grid); `group(handle)` is then `(None, count)` until its done event names the first core (8 bits: at most 256 cores; `KM_CORES_MAX` is 128). The kernel's job queue holds 64 commands waiting for room; one more is rejected (`CommandRejected`, reason 9), so with at most 64 queued commands unfinished none is rejected. Queued commands share the program's `depth` completion slots with every other command: at most `depth` (64 at most) run at once over the whole grid. A queued command cannot be exclusive. With `cluster` (1 to 3) the command goes to that work cluster's X280 thread instead of a group: `handler` is a name among the thread's `handlers` or a number, and `group`, `exclusive` and `queued` are refused (`ValueError`, as are a cluster outside 1 to 3 and one without a thread). Its event is a group's; a full thread rejects it (`CommandRejected`, reason 14, `thread`), and a timeout is returned as its event without ending the run ([commands to a thread](threads.md#commands-to-a-thread)). With `block` (below, "A command's block"), the command carries up to 1,024 bytes more than its arguments |
| `load(program, group)` | a run loaded with a dict: the cores of `group` hold `program` from this command on (its overlay entry and its buffers, set up again); returns the handle ([a group switches program](#a-group-switches-program)). `ProgramError` in a run that cannot switch |
| `add(name, parts, handler=False, number=None, defines=None, timeout=10.0)` | builds overlay `name` from `parts` (as `Program.overlay` takes them) against the loaded kernels, writes it into the room the load reserved and gives every core its table entry by a command; returns its number. With `handler`, it is also the body of handler `number`, submitted as any handler. `run.added` is `{name: Added}`. `km.ProgramError` (`added.AddError`) when it is refused, with nothing written ([code added to a live run](overlays.md#code-added-to-a-live-run)) |
| `submit_many(handler, firsts, counts, args=None, exclusive=False)` | queues n commands at once and returns their handles as a `range`. `handler`, `firsts` and `counts` are one value or n each; `args` is an `[n, m]` integer matrix (m at most 10) or None. Checked as `submit` checks, all before any is queued |
| `run_hostfed(graph, bindings=None, timeout=10.0)` | runs a [graph](../graphs.md) host-fed: its executor on the host, its commands through this run; returns `HostfedResult(status, error, registers, runs, commands, events)` |
| `wait(handles, timeout=10.0)` | stores queued commands up to the handles in batches, reads events until each handle has one; returns the `Event` (or a list) and forgets those handles. The stores and the event reads run in the native module with the GIL released |
| `wait_any(handles, timeout=10.0)` | stores what is queued and waits until one of the handles has its event; returns that `Event` (the lowest handle if several arrived) and forgets only that handle. Raises as `wait` does |
| `drain(timeout=10.0)` | the same for everything submitted; returns how many events it waited for |
| `event(handle)` | `Event(handle, kind, auxiliary, ticks, result)` or None; `kind` is `done`, `timeout`, `pong` or `rejected`; `auxiliary` is the first core of the group that ran the command (done; for a queued command, the one the kernel chose), the slot (timeout) or the reason (rejected): the low 8 bits of the event's auxiliary, whose next 16 bits, the low bits of the command's sequence number, `sequence(handle)` reads (layout: the kernel README, "Sources"); `ticks` is submit to completion in the X280's 50 MHz timer, `microseconds` the same in microseconds; for a command submitted with `result=True`, `result` is the leader's word (the kernel sends it in place of the time, so `ticks` and `microseconds` then hold it too), else None |
| `sequence(handle)`, `group(handle)` | the sequence number the kernel gave the command (the trace's index), from its done or timeout event (None before it, and for a rejected command); its group, whose first core for a queued command comes from its done event (None before it) |
| `inspect(group=None, timeout=10.0)` | while the run goes on: the dispatcher reads each core of `group` (default every core), one core per pass of its loop, and the call returns a list with a dict per core ([inspecting a run](#inspecting-a-run)). It is the kernel's own command: it takes no slot and no sequence number |
| `keep_resets(exclude=(), timeout=10.0)` | hands the dispatcher every core's reset list (what `load(recover=True)` does): firmware, its tables, the cores' kernels and arguments and the overlay area as the load wrote it (its slots empty again, the entries of `addon` included), held at the start of the upload region for the run. Called again, its lists replace the ones kept (`addon` does that). `exclude`: cores that get none (an island's link cores). Returns True and sets `recovery` (`offset`, `bytes`, `lists`, `cores`, `seconds`), or False and sets `recovery_refused` (semaphores, no room, cores of one list with different overlay areas) |
| `reset_core(core, timeout=10.0)` | resets core `core` and loads its kernels again while the other cores run theirs: by the dispatcher from the kept list (1.6 ms on the X280), else by the host (about 65 ms). Returns `{core, by, seconds, x280_seconds}`; `KomorebiError` if the core did not come back. Commands in the core's ring are lost: use it on a core with no command in flight, or one stuck in one |
| `recoveries`, `given_up(handle)` | every give-up whose cores the dispatcher reset, as `{handle, cores, text}`, `text` being the `CommandTimeout`'s (it ends `recovered (reset and loaded again): cores [...]`); `given_up(handle)` is the `CommandTimeout` of a given-up command whose event is still held (read with `event()` after `drain()`) |
| `stalls` | the watch's events on the dispatcher's own loop: `{stage, req, ticks}`, stage `stalled`, `back` or `lost` ([the watch](../concepts/dispatch.md#the-watch)) |
| `forget(handles)` | releases what the run keeps for handles whose events arrived; for callers that read events with `event()` after `drain()`. A traced run keeps groups and sequences for the trace |
| `stop(timeout=10.0)` | `timeout` is one deadline for all of it: cancels every engine still running (or `unknown`) and waits for their ends, drains, sends EXIT to every core and waits for it, sends the kernel STOP, waits for its result, closes the X280 runtime, waits for every core's kernels to return, then reads the state words and the trace. Returns the result block; raises `KernelFailed` unless the result is ready, cluster 0 restored, the state DONE and the error 0. Any error on the way (an `EngineTimeout` of a cancel not answered, a `WaitTimeout`) still closes the runtime first |
| `thread_result(timeout=10.0)` | a program with an X280 thread: waits for it (or stops it), returns its `ThreadResult`; after `stop()`, `run.thread.read(name)` returns a region's bytes |
| `abort()` | the failure path: stops the grid from the host (with a timeout), then the kernel through the runtime, closes it, reads back; `result()` then has `abandoned`; raises `KernelFailed` if the runtime could not restore cluster 0, with the runtime's account of each core (phase, stage, trap cause and address) in its message and the session log |
| `result()`, `state()`, `state_words()`, `trace()` | after the stop: the kernel's result block (decoded), each core's state words as a dict (with `ports`: count and wanted count of every port slot, and `broadcasts`: each broadcast's VALID word, first give-up, blocks, wait cycles and DONE-BY words; a reader or writer that gave up waiting on a port or an emit names its `port`), the raw state words, the trace |

### Inspecting a run

`run.inspect(group)` reads what each core of the group is doing while the run goes on.
The dispatcher reads each core's state words, its live words and the trace entries of
the latest four commands from the core's control block in L1. It reads nothing else.
The records go to a block in its local memory, and the host reads them back after the
`pong` event.

Each core's dict holds what `komorebi.state.core_state` gives (the gave-up words, the
counters). `live` has, per kernel (`reader`, `writer`, `unpack`, `math`, `pack`), the
runtime wait it is in (a where code's name, or None), the sequence and slot of the
command it runs, and the commands it has left; `given_up` is the core's count of
commands given up. `trace` holds the entries the trace has (empty without a trace).
`read` is True for a core that was read, and `reset` is a bool, True for a core the
dispatcher reset when a give-up ended. A core the dispatcher could not reach is
`{"core": i, "read": False, "reset": reset}`. `run.inspect_ticks` is how long the read
took, in the X280's 50 MHz timer.

```python
handle = run.submit("hang", (0, 2))
cores = run.inspect((0, 4))
print([core["live"]["reader"]["where"] for core in cores])  # ['port', None, None, None]
```

An inspect does not stop the commands in flight; the dispatcher's loop takes one core
of it per pass. Only one runs at a time, and a host inspect sent meanwhile waits for
the one before it. The inspect at a command's deadline uses the same block: an inspect
that finds the block overwritten by one raises `KomorebiError`, and the call can be
made again.

When a program's command from the host misses its deadline, the dispatcher can give
it up and keep the run ([runtime](runtime.md#giving-up-one-command)). It reads the
group, tells its cores to leave the command, and sends `timeout` once none runs it.
`wait` then raises `CommandTimeout` with `given_up` True, and its `cores` are the group
as that read found them. The run goes on: the next command on those cores runs as
usual. A core still in the command 20 ms later (a loop that polls nothing) is reset and
loaded again by the dispatcher when the run holds the reset list (`recover`), and
`CommandTimeout.recovered` names it; without the list it ends the run, as does a
command the dispatcher does not give up (a thread's own, a dynamic run's).

### A command's block

`run.submit(..., block=data)` passes a command more than its ten words: `data` (bytes,
or any numpy array, taken as its bytes) of 1 to 1,024 bytes (`KM_BLOCK_BYTES`). The
host writes it, padded to whole 64-byte lines, at one address of every DRAM bank, puts
that address in `a7` and its length in `a8`, and keeps it until the command's event
(then frees it). `args` is then at most 7 words (`a0` to `a6`); `a9` is 0. A
handler reads it with `reader.block(command)` or `writer.block(command)`
([runtime](runtime.md#the-block)). A block goes to a group, not to an X280 thread
(`cluster`), and `submit_many` takes none; a thread gives its own commands a block from
its region ([threads](threads.md#the-block-of-a-command)).

The blocks written since the last store are fenced with one DRAM barrier before the
commands are stored, so a block costs the host its write and a share of that barrier.
`examples/01_mapcollect/block.py`, 64 weights (256 bytes) a command on 2, 12 and 120
cores:

| Commands | Host per command, block | without | Submit to acknowledgement, block | without |
|---|---|---|---|---|
| 200 submitted, then one wait | 167 to 213 µs | 7.4 µs | 6.6 to 6.8 µs | 1.9 to 2.4 µs |
| one at a time, each waited for | 1,981 to 2,970 µs | 134 to 151 µs | 2.7 to 3.0 µs | 1.8 to 2.0 µs |

Most of the host's time is the DRAM write (about 150 µs for the eight banks) and the
barrier (1.0 to 1.8 ms); the cores add about 1 µs per command. A block suits
commands whose words are many, not commands sent many thousands a second.

`run.submit_s` adds up the host's time in `submit` and `submit_many`; `run.wait_s` the
time in `wait` and `drain`, of which `run.native_s` was in the native loop.

`trace()` returns a `komorebi.state.Trace`: `command(handle)` gives each core's
wall-clock words (taken, outputs written, completion sent, ring passes);
`phases(handle)` gives microseconds after the first core took it; `lead(earlier, later)` is
how long before handle `a`'s completion a core of `b` took its command (positive: both
were in flight at once).

## Engines: uploads, `start`, events

The dispatcher runs [streams](../streams.md) and [graphs](../graphs.md) from an
upload in its upload region, with no host step between the start and the end event.
These `Run` calls write the upload, start it, and turn its events into values.

```python
from komorebi.engines.graph import Graph

step_handler, left_handler, right_handler, norm_handler = (
    program.handler(name) for name in ("step", "left", "right", "norm")
)
graph = Graph(deadline_microseconds=200_000, max_runs=50, req=6, keep=True)
state, iteration, limit = graph.bindings("state", "iteration", "limit")
error = graph.register()
step = graph.command(step_handler, group=(0, 60), args=[state, iteration])
left = graph.command(left_handler, group=(0, 30), args=[state], after=[step])
right = graph.command(right_handler, group=(30, 30), args=[state], after=[step])
norm = graph.command(
    norm_handler, group=(0, 60), args=[state, 60], after=[left, right], result=error
)
cpu = graph.cpu(after=[norm])
cpu.addi(iteration, iteration, 1)
cpu.lt(graph.again, error, limit)  # go round again while error < limit

upload = run.upload(graph)
first = {"state": x, "iteration": 0, "limit": 10**6}
event = run.wait_engine(run.start(upload, first), 10.0)
second = {"state": y, "iteration": 3, "limit": 10**7}
event = run.wait_engine(run.restart(6, second), 10.0)
iteration_end, error_end = event.registers  # registers 0 (its low 24 bits) and 1
run.release(upload)  # the kept graph is dropped, its lines are free again
```

This is the graph of `examples/06_graph` (the same handlers; `x` and `y` are the
addresses of two state tensors): a kept graph, run, then run again with other bindings
and no new upload.

| Call | What it does |
|---|---|
| `upload(stream_or_graph, kind=None)` | encodes a `Stream` or a `Graph` (or takes its encoded bytes, with `kind` `"stream"` or `"graph"`) and writes it into the upload region as whole lines, at the first free place; returns an `Upload` (`offset`, `bytes`, `kind`, `req`, `keep`, `state`). Several uploads live side by side. An upload that does not fit raises `ValueError` before anything is written. The host checks nothing else: a bad upload is the kernel's to reject, with its error event |
| `release(upload, timeout=10.0)` | frees an upload's lines. A running one is rejected; a kept graph is first dropped (`CANCEL`, then its cancelled event, both within `timeout`) |
| `start(upload, bindings=(), kept=None, timeout=10.0)` | sends `KM_HANDLER_RUN` and returns an `EngineHandle`. A graph takes `bindings`: a dict by name for a graph uploaded from a `Graph` (its constants are added, as `binding_values` does), or a list of up to 16 words; they go to one line of the region, freed at the end. `kept=True` or `False` rewrites the header's keep bit first; None leaves it. Another running or kept engine with the same `req` raises `ValueError`. When the kernel does not answer the RUN within `timeout` it raises `EngineTimeout` with the handle marked `unknown`: the engine may have started, and `cancel` (or `stop`) settles it |
| `restart(req, bindings=(), timeout=10.0)` | runs the kept graph with `req` again with new bindings (`KM_HANDLER_RUN`, `a2 = 2`) |
| `cancel(handle, timeout=10.0)` | sends `KM_HANDLER_CANCEL`; the engine then ends with its error event, reason `cancelled`. Nothing is sent for an engine that has ended. For an `unknown` handle it also learns the RUN's answer: raises `EngineRejected` if the RUN was rejected (it never ran); otherwise the engine runs or has ended as usual. `EngineTimeout` if the CANCEL is not answered |
| `wait_engine(handle, timeout=10.0)` | waits for the engine's end and returns its `EngineEvent`; raises `EngineError` for an error event, `EngineTimeout` when `timeout` passes (the engine may still run), `KernelFailed` if the kernel finished |
| `run_hostfed(graph, bindings=None, timeout=10.0)` | runs a graph or a stream on the host instead, its commands through the host ring: the same commands and registers as in the kernel ([graphs](../graphs.md#host-fed)); for a stream, the simulator's interpreter, `commands[i].node` the program counter of the `SUBMIT` or `ENQUEUE`, `runs` 1 |

`start`, `restart` and `cancel` send the command and a `PING` behind it and return
when the `PING` is answered: the kernel answers a `RUN` or `CANCEL` with an event only
when it rejects it, and that event comes first. A rejection raises `EngineRejected` (a
`CommandRejected`: `reason`, `reason_name`); the host does not retry.

An `EngineHandle` has `kind`, `req`, `upload`, `events` (every event so far) and `end`.
An `EngineEvent(handle, kind, value, registers, reason, reason_name, pc, node)` is one of:

| `kind` | From | Fields |
|---|---|---|
| `event` | a stream's `EVENT a` | `value`: register `a`, 32 bits |
| `round` | a graph's round ended and another starts | `value`: rounds done |
| `end` | `HALT`, or a graph's last node | a stream's `value`: the halt register, 32 bits; a graph's `registers`: register 0's low 24 bits, register 1 |
| `error` | any error | `reason`, `reason_name` (the `KM_STREAM_ERROR_*` name in lower case, read from the headers), the stream's `pc` or the graph's `node` (None when the error has no node) |

The engines' events carry the upload's `req`, not a command number: they go to their
handles, so `wait` and `wait_any` of the host's own commands see none of them, and
`wait_engine` none of the commands' events. The commands an engine submits are answered
to the engine, never to the host. While it
waits, `wait_engine` also writes the host's read count into the kernel, since an
engine's posts wait for event room.

The region is the one the kernel's ready block names: what is left of X280 cluster 0's
local memory after the kernel's layout (707,840 bytes on a p150a), 512 KiB for a
program with an X280 thread, whose rings lie after it. The host never writes into the range of a
running stream or graph, or of a kept graph. The kernel's EXIT waits until no engine
runs, so `stop` cancels the engines still running first.

The kernel numbers the commands of every source (the host, an engine, a work-cluster
thread) in one sequence. A command's done or timeout event carries the low 16 bits of its
number, and the host takes the rest from its previous command's: `sequence(handle)` and the
trace's `command(handle)` are exact while fewer than 32,768 commands of other sources are
submitted between two host commands. When the host sees a longer gap it raises
`KomorebiError` for that handle instead of giving another command's number; it cannot
tell a gap of 65,536 or more. A queued command's number may fall up to 2,048 behind the
host's order (the queue lets later jobs go first); a lag beyond that is read as a gap
and raises too.

## Errors

All derive from `km.KomorebiError`.

| Error | When | Facts it carries |
|---|---|---|
| `CardBusy` | `Session.open`: another live process holds the card (below) | `pid` of the holder |
| `ProgramError` | a kernel did not compile or the launch failed | the build message |
| `L1BudgetError` | the buffers, control blocks and overlay slots do not fit a core's L1; a `ProgramError` | `table`: the L1 budget table of a core |
| `L1LayoutError` | programs that share cores declare one buffer index two ways; an `L1BudgetError` | `table` |
| `KernelSizeError` | the program's kernels do not fit the session's kernel buffer; a `ProgramError`, raised before anything is loaded | `sizes`: text per thread, total, buffer |
| `SlowCodeError` | the overlay slots, which sit just above the kernel buffer, would end above 1 MB of L1 (a kernel buffer near 1 MB); a `ProgramError`, raised before anything is launched. `Session(slow_code=True)` loads it anyway | `end`: the slots' end in L1 |
| `KernelInstructionError` | a kernel's ELF holds an instruction its core lacks ([below](#instructions-the-cores-lack)); a `ProgramError`, raised before anything is loaded | `findings`: thread, instruction, function, source line |
| `StartError` | the image is missing, cluster 0 is not idle, a base handed to the kernels (control block, X280 local memory, its upload region) is not a multiple of 16 bytes, or the dispatcher did not report READY | `result` (the kernel's result block, when it ran) |
| `WaitTimeout` | a host-side wait ran out; the run is still loaded. The message ends with what the kernel was doing: whether core 0 still runs its loop (its pass marker, read twice 20 ms apart), its state and where, and the runtime's account of each core | `pending` handles |
| `CommandRejected` | the kernel consumed the command and submitted nothing (a reserved handler or flag, a group outside the grid, a command after EXIT); the run goes on | `handle`, `reason` (`KMD_REJECT_*`), `reason_name` |
| `CommandTimeout` | a group missed its deadline. With `given_up` True, the dispatcher gave the command up and the run goes on. Otherwise the kernel stopped the grid, and the run is stopped and read back | `handle`, `slot`, `sequence`, `group`, `given_up`, `cores`: each core of the group that did not report, with where its reader and writer gave up; given up, the group as the inspect at the deadline read it (`live`: where each kernel waited) |
| `KernelFailed` | the dispatcher finished with any other error, did not finish cleanly, or sent an event for a command not in flight | `result` |
| `EngineRejected` | the kernel rejected a `RUN` or `CANCEL` (four streams or two graphs running, no such req, a command after EXIT) | as `CommandRejected` |
| `EngineError` | a stream or graph ended with an error event | `event`, `handle`, `reason`, `reason_name`, `pc`, `node` |
| `EngineTimeout` | `wait_engine` ran out, or a RUN or CANCEL was not answered; a `WaitTimeout` | `handle` |
| `ChannelError` | the open logged the driver's warning about the host channel; a transfer does not fit the session's region or a core's L1; a core's sequence word did not arrive in time (with the kernel's status, without a run); a page range written without the channel | |
| `GridTimeout` | the grid's kernels did not return within `grid_timeout`; the runtime closes the device at the process's exit | |

A card stays held by a process from its first device open until that process exits,
even after the device is closed: tt-metal's driver takes the chip's lock at the first
open and gives it back only when the process ends. A second process that opens the card
meanwhile would wait without a timeout, so `Session.open` first reads the lock's owner
(from the driver's lock file, without taking the lock) and raises `km.CardBusy` naming
that process when it still maps the file: a process that died holding the lock, a
zombie or a PID reused since is not taken for the holder. A test run that opens the
card in its own process therefore cannot start a card script as a child (see
`tests/hardware/README.md`).

The kernel's reasons for a rejection (`CommandRejected.reason`, from `KMD_REJECT_*` in
the kernel's header; `reason_name` is the lower-case suffix):

| `reason` | `reason_name` | The command |
|---|---|---|
| 1 | `handler` | has a handler above 239 that its source may not send: the host's own commands may carry 240 to 245 and 248 (and `EXIT`), a work cluster's thread 240 and 245 |
| 2 | `flags` | sets the reserved flag bit 0, or the dispatcher's flag (`KMD_FLAG_DISPATCHER`) on a command that is none of the dispatcher's own |
| 3 | `group` | has a group of 0 cores, or one that does not lie inside the cores |
| 4 | `exited` | came after EXIT |
| 9 | `queue` | is queued (`queued=True`) while the kernel's job queue holds 64 commands |
| 13 | `program` | names a program its group's cores do not hold (named for the dispatcher's check; the host refuses such a command first) |
| 14 | `thread` | goes from the host to a work cluster whose thread already has as many commands unanswered as it takes |

`submit` and `submit_many` check handlers and groups before anything is queued, the
host sets no reserved flag, and a stopped run submits nothing, so through this API a
rejection other than 9 (a full job queue), 13 (a program the cores do not hold, where
the host does not check) and 14 (a full thread) means that the host and the kernel
were built from different headers.

## Builds

The dispatcher image (`km-dispatch.bin`) and the native module
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
instruction, the function and, from the debug info, the source line. Rejected are the
groups the core's instruction set documents as absent: float divide and square root
(`fdiv`, `fsqrt`), vector integer divide and remainder (`vdiv`, `vrem`), and vector
float divide, square root and their estimates (`vfdiv`, `vfrdiv`, `vfsqrt`,
`vfrsqrt7`, `vfrec7`). The result is kept with the build, so a cached build is not
checked again. The check itself runs objdump once per thread ELF the first time that
build is seen, about 11 ms each (some 0.06 s for a program's five); the runtime's
compile of a new kernel, which now runs just before it, is most of a first load
(about 0.9 s for example 01). Without the toolchain's objdump, or when the build left
no ELF, the load raises `ProgramError`; without its addr2line, a finding is still
rejected, without its source line. A `/` on floats is the
usual cause: divide by an integer (`div`/`divu` are implemented), multiply by a
reciprocal computed on the host, or, on the vector unit, use a reciprocal with Newton
steps as the projects do ([`recip`](../../projects/raytrace/kernels/sfpu.hpp), exact
division and square root in [`divsqrt.hpp`](../../projects/raytrace/kernels/divsqrt.hpp)).
`KMR_INSN_CHECK=0` turns the check off, for a kernel that never executes the code
found.

## Rules the host keeps

- The host never reads or writes a core's memory while the program runs, with one
  exception: the stop flag, when the dispatcher cannot send it. State words and
  the trace are read after the stop.
- While the dispatcher runs, every host wait has a timeout. The X280 runtime is
  closed before any readback, on every failure path, at the process's exit and on
  SIGTERM, always before the device.
- Host stores into the card go in batches: every line, one fence, then a pause of
  `settle_microseconds` before the next store.
- Unread events plus commands in flight never exceed the event ring.
- EXIT goes to every core, after everything in flight has completed.
