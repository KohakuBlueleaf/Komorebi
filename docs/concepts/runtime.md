---
title: The Tensix runtime
summary: The persistent kernel on every Tensix core, the 64-byte command, groups and completion, static and dynamic breakpoints, and the trace.
tags:
  - concepts
  - runtime
  - tensix
---

# The Tensix runtime

Status: implemented and run on one Blackhole card: the persistent kernel, commands,
groups and completion, static breakpoints (ports and reduce, broadcast) and the
trace, and dynamic breakpoints (one dynamic run at a time), through a `Session` and
under the test feeder.

Each program has one persistent kernel on every Tensix core it uses. The worker starts
once and then serves work: it takes the next command, switches on the handler the command
names, runs that handler and reports completion. [Dispatch](dispatch.md) is what sends
the commands, from the host, from its own engines or from an X280 thread
([threads reference](../reference/threads.md)). The [model](model.md) is what the
handlers and breakpoints mean.

| Part | As built | Reference |
|---|---|---|
| Command | One 64-byte line: sequence number, handler, flags and ring slot, group, ten argument words. The dispatcher forwards it whole. | [runtime](../reference/runtime.md#the-command) |
| Landing | Word 0 of each of the four 16-byte chunks carries the sequence number; a core runs a command only when all four agree. | [runtime](../reference/runtime.md#the-command) |
| Worker | One persistent kernel set per program. The runtime owns each kernel's loop; the program gives the body. The reader copies each command into the tile's control block and hands a token to the compute and writer kernels, which read the command from that copy. Every wait has a timeout and observes a stop flag. | [runtime](../reference/runtime.md#the-three-kernels) |
| Groups | A contiguous range of cores in core order, named in the command. Completion is per group: cores report to their row's leader, row leaders to the first core, which acknowledges with one write. Several groups run at once. | [runtime](../reference/runtime.md#the-three-kernels) |
| Static breakpoints | A value to one core: `emit` writes into a fixed slot, waits for the write, increments the receiver's port; `arrive` is a timed counted wait. `reduce` by owner and by tree. A block to a row or a column: broadcast with a window. | [ports](../reference/ports.md), [broadcasts](../reference/broadcast.md) |
| Dynamic breakpoints | A queue per core in DRAM, positions reserved by an atomic add on the receiver; packets of up to 32 messages; bins per destination; a limited population of live messages (a send quota); sent and consumed counters in a status line when a core goes idle. Generic reader and writer kernels own these, so a program's handler code lives in the compute kernel. Through a `Session` and under the test feeder; one dynamic run at a time. | [dynamic](../reference/dynamic.md) |
| Trace | Per-command wall-clock timestamps per core, read back after the run. | [runtime](../reference/runtime.md#trace) |

## Commands and handlers

A command is the unit that travels from the dispatcher to the cores. Its first
words say which handler to run, for which group and in which ring slot; the rest are
that handler's arguments. Handlers 1 to 239 are the program's; EXIT ends the worker.

A run may hold several programs. Then the kernels are the runtime's own: a reader, a writer
and a compute kernel that serve every command of the run, and each program supplies, per
kernel, the function they call (`km_program`), built as an
[overlay](../reference/overlays.md#a-program-as-an-overlay). A command names its program in
its header's program field (0 is the run's first program). For each command the runtime's
kernels compare that field with what the core's slots hold: a program held costs a look
at the slots' records, about 130 cycles a command; one not held is loaded first (its five
parts read from DRAM, about 2,200 cycles on one core). The host loads each program on a
group of cores (`session.load({program: (first, count)})`) or every program on every core
(`session.load([...])`), and refuses a command whose cores do not hold its program
([host](../reference/host.md#several-programs-in-one-run)).

## Groups and completion

A group is a contiguous range of cores in a fixed core order, and is named in the command
that starts it. Cores report completion to the group's leader; the leader pushes one
word to the dispatcher. A core reports only after its writer's outputs have
landed and its compute body has returned, so a command with no output is not complete
while compute still runs it. Because completion is per group, several groups can be busy
at once, and a group's size is not tied to a column of the grid. Where a group is
addressed by multicast, each rectangle must be a physically contiguous, Tensix-only
span in NoC0 coordinates, so a group is cut at panel boundaries. The layout module of
the Python package computes these rectangles.

A command with `KM_FLAG_RESULT` carries one 32-bit result word back with its
acknowledgement. The leader's writer gives it with `writer.result(value)` before the command
completes, and the leader acknowledges with `{ack, result}` as one 8-byte write into
the slot's acknowledgement line, so the result is there whenever the acknowledgement
is. A core that is not the leader may call it too; its value is not sent. A flagged
command whose leader never gave a result is acknowledged with 0, and a command without
the flag is acknowledged with the ack word alone.

## Breakpoints

A static breakpoint is known before the run: who sends, who receives and how many
arrive, all from the command and the core's rank. A value to one core is a write into
a slot at a fixed address on the receiver, a wait until that write has landed, and an
increment of a port the receiver waits on; there is no header and no queue
([ports](../reference/ports.md)). A block to many cores is a broadcast: one sender
multicasts it along a row or down a column of the group, with a window of blocks in
flight, and each receiver writes a progress marker to the sender when its compute is done with a block
([broadcasts](../reference/broadcast.md)). Commands that use either carry `KM_FLAG_EXCLUSIVE`, so two
commands whose groups overlap are never in flight together.

A dynamic breakpoint depends on data. Each receiver has a queue in DRAM large enough
for every live message. A sender collects messages per destination, reserves positions in
the receiver's queue with one atomic add and writes the messages there; it never waits for
the receiver, so a full inbox cannot deadlock the grid. The receiver reads its queue in
order, as packets of up to 32 messages. The number of live messages is limited by a send
quota. When a core has been idle for a while it writes its sent and consumed counters
in a status line at the submitter, and the run is over when every line says idle with the
same counters on two successive sweeps and the sums match.
[Dynamic handlers](../reference/dynamic.md) is the page for a program's author.

## Debugging

The watcher cannot run while the X280 runtime is open, so the runtime is built to be
debugged without it: every wait has a timeout, each core has a "gave up" word, "where"
codes show where a core stopped, and the trace timestamps phases per core.

A program takes and releases pages of its circular buffers only through the runtime's
wrappers (`Reader::reserve`, `Writer::wait_front`, `Compute::wait_front`,
`Compute::reserve_back`), never through the raw `cb_*` calls. A wrapper's wait is
timed and sees the stop flag, and it checks that the item it hands out stays inside
the buffer. An item of several pages is one contiguous region: tt-metal's compute side
wraps its pointer when it reaches or passes the buffer's end, the data-movement side
only when it lands exactly on it, so a straddling item would send the data-movement
pointer out of the buffer for good. The buffer must therefore be a multiple of the
item, and every push and pop on it a multiple of the item; a wrapper that sees an item
pass the end gives up with the code `bad_item` before any access.

The one kind of wait without a timeout is the wait on the NoC itself
(`noc_async_read_barrier`, `noc_async_write_barrier`, `noc_async_writes_flushed`). It
ends once the transactions to valid addresses complete; bounding it would let a kernel
leave with a transaction still in flight. The wrappers' checks are what keep those
addresses valid.

A NoC write between two cores' L1 is only correct when its source and destination
addresses share their low four bits. Measured on a Blackhole p150a
([experiment](../../experiments/noc_alignment/README.md)): when they differ, a 4-byte
write lands at the destination address but carries the bytes found at the destination's
offset within the source's 16-byte line, and a non-posted write is acknowledged as
usual. The mismatch is silent and wrong, never an error. The runtime's small writes
therefore go through one helper, `km::small_write`, that takes slot numbers, not
addresses, on both ends: from a 16-byte slot of the sender's control block to a 16-byte
slot above a base on the receiver. The control block and every base the host hands to
the kernels must be multiples of 16.
