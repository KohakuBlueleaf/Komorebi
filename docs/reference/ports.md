---
title: Ports, emit and reduce
summary: The device calls that move a value from one Tensix core to one other core (emit, arrive, take), groups of a team, and reduce by owner and by tree.
tags:
  - reference
  - device
  - breakpoints
---

# Ports, emit and reduce

Status: implemented in the Tensix runtime and run on one Blackhole card through a
`Session` and `Run`, under the dispatch kernel, and under the test feeder
([02_merge_tree](../../examples/02_merge_tree/): `python merge.py --jobs 20 --count 8
--form both`).

A static breakpoint that sends a value to one receiver. Who sends to whom, into which
slot, and how many values a wait expects all follow from the record (`first`, `count`,
the arguments) and the member's rank. Nothing is queued and nothing is addressed by
content.

```c++
#include "komorebi/device/km_port.hpp"

constexpr km::Arrival arrival = {kCbArrive, S};  // the program's arrival buffer

// reader, writer and compute, per record
const km::Group g = km::team_group(r, me);

// reader: after its own inputs, every hand-over this member waits for
if (!km::reduce_arrive(rd, r, arrival, g, form)) return;

// compute: own part in DST 0, then the parts in rank order
ok = ok && km::reduce(cp, arrival, g, form, [](uint32_t slot) {
    copy_init(kCbArrive);
    copy_tile(kCbArrive, slot, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
});

// writer
if (g.rank != 0) return km::reduce_emit(wr, r, arrival, g, form, kCbOut, 1);
if (!km::reduce_result(wr, kCbOut, 1)) return false;  // then write the result
```

## On the wire

| Step | Kernel | Call | What it does |
|---|---|---|---|
| put | writer (NoC1) | `km::emit` | the pages at the front of a buffer go into slot k of the receiver's arrival buffer; bounded wait for the write to be acknowledged |
| bump | writer | `km::emit` | an atomic increment of the receiver's port; bounded wait for its response (a write flush does not cover an atomic) |
| wait | reader | `km::arrive(port, n)` | bounded wait until the port has counted n; clears it; pushes the whole arrival buffer to compute |
| take | compute | `km::take`, `km::untake` | waits for the arrival buffer, reads slots by page index, pops it whole |

- **Ports.** Every core has `KM_PORTS` (8) port slots of 16 bytes in its control block
  at `KM_CTL_PORT`: the count senders increment, and the count a wait wants. Ports are
  cleared at start-up and by the wait that consumed them.
- **The arrival buffer** is one circular buffer of S pages, P to a slot
  (`km::Arrival {cb, S, P}`, P = 1 by default): slot k is pages k * P to k * P + P - 1,
  and a part sent with `emit` is at most P pages. Every core
  runs the same program, so slot k has the same L1 address everywhere (taken from the
  buffer's configuration). It is always handed over whole, S pages pushed and S popped,
  so its addresses never drift.
- **`KM_F_EXCL`.** A record that uses ports must carry it: two records whose teams
  overlap are then never in flight together, so nothing of one record lands in the
  next one's slots or ports.

## Groups

`km::Group {base, stride, size, rank}`: member j is core `base + j * stride`.

| Call | Group |
|---|---|
| `km::team_group(r, me)` | the whole team, in core order |
| `km::digit_group(r, me, stride, radix)` | the members whose team index differs from this core's only in the digit of weight `stride` and base `radix` (for example the K parts of one output block) |

## Reduce

| Form | Sender | Receiver | Port, slot | Slots (S / P) |
|---|---|---|---|---|
| `km::kReduceOwner` | every member but rank 0 | rank 0 waits for size - 1 on port 0, combines slots 0 to size - 2 in rank order | port 0, slot rank - 1 | at least size - 1 |
| `km::kReduceTree` | in round r, a rank that is an odd multiple of 2^r sends to rank - 2^r and is done | a multiple of 2^(r+1) waits for 1 on port r if rank + 2^r < size | port r, slot r | one per round, ceil(log2 size), at most 7 |

The combine order is fixed by rank in both forms, so a result does not depend on
arrival order; on exact-valued inputs both forms give the same bits.
`km::reduce_send(g, form)`, `km::reduce_wants(g, form, port)` and
`km::reduce_slots(g, form)` give the schedule itself (`komorebi/device/km_group.hpp`,
free of tt-metal headers). A buffer with too few slots gives up with `bad_item`.

## When a member does not send

Every wait is bounded by the start-up wait limit and sees the stop word. The port's
wait gives up first: waits that lie behind it (the writer's wait for a part that needs
arrivals, `take`, `reduce_result`) are bounded at `km::kBehindPort` (2) wait limits. A
give-up records `where` = `port` (or `emit` on the sender) with the port number in the
gave-up word's top byte, and leaves the wanted count in the port, so the host reads the
missing count (`records.waiting_ports` in the test feeder). The tile stops, the
dispatcher's or feeder's deadline stops the run, every kernel leaves, and the next
launch runs.
