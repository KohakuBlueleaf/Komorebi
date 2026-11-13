---
title: Ports, emit and reduce
summary: The device calls that move a value from one Tensix core to one other core (emit, arrive, take), groups of a group, and reduce by owner and by tree.
tags:
  - reference
  - device
  - breakpoints
---

# Ports, emit and reduce

Status: implemented in the Tensix runtime and run on one Blackhole card through a
`Session` and `Run`, under the dispatcher, and under the test feeder
([02_merge_tree](../../examples/02_merge_tree/): `python merge.py --jobs 20 --count 8
--form both`).

A static breakpoint that sends a value to one receiver. Who sends to whom, into which
slot, and how many values a wait expects all follow from the command (`first`, `count`,
the arguments) and the member's rank. Nothing is queued and nothing is addressed by
content.

```c++
#include "komorebi/device/km_port.hpp"

constexpr km::Arrival arrival = {kCbArrive, S};  // the program's arrival buffer

// reader, writer and compute, per command
const km::Group g = km::group_of(r, me);

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
  cleared at start-up and by the wait that consumed them. A sender counts the atomic's
  acknowledgement and never reads its returned value, so a lost answer
  ([atomic_answers](../../experiments/atomic_answers/)) cannot affect a port.
- **The arrival buffer** is one circular buffer of S pages, P to a slot
  (`km::Arrival {cb, S, P}`, P = 1 by default): slot k is pages k * P to k * P + P - 1,
  and a part sent with `emit` is at most P pages. Every core
  runs the same program, so slot k has the same L1 address everywhere (taken from the
  buffer's configuration). It is always handed over whole, S pages pushed and S popped,
  so its addresses never drift.
- **`KM_F_EXCL`.** A command that uses ports must carry it: two commands whose groups
  overlap are then never in flight together, so nothing of one command lands in the
  next one's slots or ports.

## Groups

`km::Group {base, stride, size, rank}`: member j is core `base + j * stride`.

| Call | Group |
|---|---|
| `km::group_of(r, me)` | the whole group, in core order |
| `km::digit_group(r, me, stride, radix)` | the members whose group index differs from this core's only in the digit of weight `stride` and base `radix` (for example the K parts of one output block) |

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

Every wait is bounded by the start-up wait limit and sees the stop flag. The port's
wait gives up first: waits that lie behind it (the writer's wait for a part that needs
arrivals, `take`, `reduce_result`) are bounded at `km::kBehindPort` (2) wait limits. A
give-up commands `where` = `port` (or `emit` on the sender) with the port number in the
gave-up word's top byte, and leaves the wanted count in the port, so the host reads the
missing count (`commands.waiting_ports` in the test feeder). The tile stops, the
dispatcher's or feeder's deadline stops the run, every kernel leaves, and the next
launch runs.

## X280 ports

A port between a Tensix core and the X280 core that serves it, without the dispatcher
(`komorebi/device/km_x280.hpp`; the X280 core's side and the declaration are in
[X280 threads](../threads.md#x280-ports)). A message is a sequence number and three
words, one 16-byte NoC write each way:

```c++
#include "komorebi/device/km_x280.hpp"

km::X280Port port(rd);  // owned by the reader; X280Port(wr, km::kStWriter): the writer
km::X280Msg m;
if (port.call(a, b, c, m)) {  // ask, then wait for its answer
    use(m.w[0]);
}
port.tell(x, y, z);          // to the X280 core; it sees the newest
if (port.told(m)) { ... }    // the X280 core's newest tell, if newer; never waits
```

| Call | Waits | Returns |
|---|---|---|
| `served()` | no | whether the program gave this core an X280 core |
| `told(m)` | no | true and the X280 core's newest tell, if newer than the last one looked at |
| `wait_told(m)` | bounded | the same, waiting for one; false: gave up (recorded) |
| `tell(a, b, c)` | for the last store to leave | false: gave up (recorded) |
| `ask(a, b, c)` | as `tell` | the ask's sequence number, 0: gave up |
| `answered(seq, m)` | no | true and the answer to ask `seq`, if it landed |
| `wait_answer(seq, m)` | bounded | the same, waiting; false: gave up (recorded) |
| `call(a, b, c, m)` | bounded | `ask`, then `wait_answer` |
| `next(first)` | bounded | the next range of a pull (`kmx_pull_run`): its count and `first`, 0 when none is left or a wait gave up (`pull_gave_up()`); asks for the range after it before returning |

**Latest wins.** A tell overwrites the line it lands in: a side that looks less often
than the other tells sees only the newest, and tells in between are lost. Ask and
answer are paired by the sequence number, one ask outstanding at a time; between the
first `next()` of a pull and the one that returns 0, that ask is the pull's.

**On the wire.** The core's `KM_CTL_X280` area (64 bytes): the X280 core's tell at
`KM_X280_TELL` (`+ 0x00`) and answer at `KM_X280_ANSWER` (`+ 0x10`), stored by the X280 core
through its window; the core's out slot `KM_X280_OUT` (`+ 0x20`), the source of its
writes; `KM_X280_TO` (`+ 0x30`): the X280 core's tile (`x | y << 16`) and the mailbox
address in its local memory, written by the host at the load (0: no X280 core). The core
writes its ask to the mailbox at `KM_BOX_ASK` (`+ 0x00`) and its tell at `KM_BOX_TELL`
(`+ 0x10`) on its kernel's NoC. Each line has one writer; no atomics. A pull's ask is
`{KM_PULL_ASK, 0, 0}`, answered `{first, count, pulls so far}` (count 0: none left).

**When a side does not answer.** Every wait is bounded by the wait limit and sees the
stop flag. A give-up commands `where` = `kWhereX280Answer` (37), `kWhereX280Tell` (38)
or `kWhereX280Send` (36) for a store that did not leave, `kWhereX280None` (39) when the
program gave the core no X280 core. The command then ends as any command whose member gave up.
