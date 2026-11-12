---
title: The Tensix runtime
summary: The resident worker on every Tensix core, the 64-byte record, teams and completion, static and dynamic breakpoints, and the trace.
tags:
  - concepts
  - runtime
  - tensix
---

# The Tensix runtime

Status: design, not yet implemented.

Each program has one resident worker on every Tensix core it uses. The worker starts
once and then serves work: it takes the next record, switches on the segment the record
names, runs that segment and reports completion. [Dispatch](dispatch.md) is what sends
the records; the [model](model.md) is what the segments and breakpoints mean.

| Part | Contract |
|---|---|
| Record | One 64-byte line: `seq`, `seg`, team or target, a run tag, then the segment's arguments. The dispatch kernel treats it as eight opaque words. |
| Landing | A record must be seen whole. Two ways are compared: an echo word in every 16-byte chunk, so a core can tell when the whole line has landed, or the record first and a sequence word after it. The choice is not made yet. |
| Worker | One resident kernel set per program. Take the next record, switch on `seg`, run, complete. Every wait is the runtime's own, bounded, and observes a stop word. The reader copies each record into the tile's control block and hands a token to the compute and writer kernels through two small buffers; both read the record from that copy. |
| Teams | A team is named in the record. Completion is per team: members report to the team's leader, and the leader pushes one word. Several teams run at once. |
| Static breakpoints | `emit`: write into a fixed slot, wait for landing, increment the receiver's counter. `wait(n)`: a bounded counted wait. Groups are index slices; a rectangle group uses one multicast. `reduce` by owner and by tree ([implemented](../reference/ports.md)). |
| Dynamic breakpoints | An inbox ring per core; packets of up to 32 records; an outbox per destination; free-slot credit; a bounded population of live threads; sent and consumed counters pushed when a core goes idle. Generic reader and writer kernels own these, so a program's segment code lives in the compute kernel. |
| Trace | Per-core phase stamps, read back after the run. |

## Records and segments

A record is the unit that travels, both from the dispatch kernel to a core and, for a
dynamic breakpoint, from core to core. Its first words say which segment to run and for
which team; the rest are that segment's arguments. A thread that moves carries its live
variables in the same 64 bytes.

## Teams and completion

A team is a contiguous range of cores in a fixed core order, and is named in the record
that starts it. Members report completion to the team's leader; the leader pushes one
word to the dispatch kernel. A member reports only after its writer's outputs have
landed and its compute body has returned, so a record with no output is not complete
while compute still runs it. Because completion is per team, several teams can be busy
at once, and a team's size is not tied to a column of the grid. Where a team is
addressed by multicast, each rectangle must be a physically contiguous, Tensix-only
span in NoC0 coordinates, so a team is cut at panel boundaries. The layout module of
the Python package computes these rectangles.

A record with `KM_F_RESULT` carries one 32-bit result word back with its
acknowledgement. The leader's writer gives it with `wr.result(value)` before the record
completes, and the leader acknowledges with `{ack, result}` as one 8-byte write into
the slot's acknowledgement line, so the result is there whenever the acknowledgement
is. A member that is not the leader may call it too; its value is not sent. A flagged
record whose leader never gave a result is acknowledged with 0, and a record without
the flag is acknowledged with the ack word alone.

## Breakpoints

A static breakpoint is known before the run: who sends, who receives and how many
arrive. It is a write into a slot at a fixed address on the receiver, a wait until that
write has landed, and an increment of a counter the receiver waits on. There is no
header and no queue.

A dynamic breakpoint depends on data. The receiver has an inbox ring of records. A
sender fills an outbox per destination and sends packets of up to 32 records, within
the credit of free slots the receiver has granted. The number of live threads is
bounded, which limits how much can be in flight at once. When a core goes
idle it pushes its sent and consumed counters, and the run is over when the sums match
on two successive sweeps.

Both kinds are written the same way by the author; the lowering differs.

## Debugging

The watcher cannot run while the X280 runtime is open, so the runtime is built to be
debugged without it: every wait is bounded, each core has a "gave up" word, "where"
codes show where a core stopped, and the trace stamps phases per core.

A program takes and releases pages of its circular buffers only through the runtime's
wrappers (`Reader::reserve`, `Writer::wait_front`, `Compute::wait_front`,
`Compute::reserve_back`), never through the raw `cb_*` calls. A wrapper's wait is
bounded and sees the stop word, and it checks that the item it hands out stays inside
the buffer. An item of several pages is one contiguous region: tt-metal's compute side
wraps its pointer when it reaches or passes the buffer's end, the data-movement side
only when it lands exactly on it, so a straddling item would send the data-movement
pointer out of the buffer for good. The buffer must therefore be a multiple of the
item, and every push and pop on it a multiple of the item; a wrapper that sees an item
pass the end gives up with the code `bad_item` before any access.

The one kind of wait that is not bounded is the wait on the NoC itself
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
