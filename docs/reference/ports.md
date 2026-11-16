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
the arguments) and the core's rank. Nothing is queued and nothing is addressed by
content.

A group's port commands do not overlap: a port's count and its arrival slots belong to
one command at a time. `run.submit` refuses a command of a program's `port_handlers` (a
class's handlers that reduce are its own) on cores that such a command without its
event still holds, naming both handles, unless it is submitted `exclusive=True`; a
thread's submits are not checked.

```c++
#include "komorebi/device/km_port.hpp"

constexpr km::Arrival arrival = {kCbArrive, kArrivePages};  // the program's arrival buffer

// reader, writer and compute, per command
const km::Group group = km::group_of(command, me);

// reader: after its own inputs, every arrival this core waits for
if (!km::reduce_arrive(reader, command, arrival, group, form)) return;

// compute: own part in DST 0, then the parts in rank order
ok = ok && km::reduce(compute, arrival, group, form, [](uint32_t slot) {
    copy_init(kCbArrive);
    copy_tile(kCbArrive, slot, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
});

// writer
if (group.rank != 0) {
    return km::reduce_emit(writer, command, arrival, group, form, kCbOut, 1);
}
if (!km::reduce_result(writer, kCbOut, 1)) return false;  // then write the result
```

## On the wire

| Step | Kernel | Call | What it does |
|---|---|---|---|
| put | writer (NoC1) | `km::emit` | the pages at the front of a buffer go into a slot of the receiver's arrival buffer; timed wait for the write to be acknowledged |
| bump | writer | `km::emit` | an atomic increment of the receiver's port; timed wait for its response (a write flush does not cover an atomic) |
| wait | reader | `km::arrive(..., port, wanted)` | timed wait until the port has counted `wanted`; clears it; pushes the whole arrival buffer to compute |
| take | compute | `km::take`, `km::untake` | waits for the arrival buffer, reads slots by page index, pops it whole |

- **Ports.** Every core has `KM_PORTS` (8) port slots of 16 bytes in its control block
  at `KM_CONTROL_PORT`: the count senders increment, and the count a wait wants. Two
  more follow (`KM_PORT_PAGES`, the reader's and the writer's), which only the page
  calls count at ([islands](islands.md)), so a reduce never shares a count with them. Ports are
  cleared at start-up and by the wait that consumed them. Word `KM_PORT_CREDIT` of a
  port on its sender counts a repeated reduce's credits (below). A sender counts the atomic's
  acknowledgement and never reads its returned value, so a lost answer
  ([atomic_answers](../../experiments/noc/atomic_answers/)) cannot affect a port.
- **The arrival buffer** is one circular buffer of `pages` pages, `per` to a slot
  (`km::Arrival {cb, pages, per}`, `per` = 1 by default): slot `slot` is pages
  `slot * per` to `slot * per + per - 1`, and a part sent with `emit` is at most `per`
  pages. Every core runs the same program, so a slot has the same L1 address everywhere
  (taken from the buffer's configuration). It is always passed to compute whole, all
  `pages` pages pushed and popped, so its addresses never drift.
- **`KM_FLAG_EXCLUSIVE`.** A command that uses ports must carry it: two commands whose groups
  overlap are then never in flight together, so nothing of one command lands in the
  next one's slots or ports.

## Groups

`km::Group {base, stride, size, rank}`: rank `rank` is core `base + rank * stride`.

| Call | Group |
|---|---|
| `km::group_of(command, me)` | the whole group, in core order |
| `km::digit_group(command, me, stride, radix)` | the cores whose group index differs from this core's only in the digit of weight `stride` and base `radix` (for example the parts of one output block along the matrix's inner dimension) |

## Reduce

| Form | Sender | Receiver | Port, slot | Slots |
|---|---|---|---|---|
| `km::kReduceOwner` | every core of the group but rank 0 | rank 0 waits for size - 1 on port 0, combines slots 0 to size - 2 in rank order | port 0, slot rank - 1 | at least size - 1 |
| `km::kReduceTree` | in each round, a rank that is an odd multiple of 2^round sends to rank - 2^round and is done | a multiple of 2^(round+1) waits for 1 on port `round` if rank + 2^round < size | port `round`, slot `round` | one per round, ceil(log2 size), at most 7 |

The combine order is fixed by rank in both forms, so a result does not depend on
arrival order; on exact-valued inputs both forms give the same bits.
`km::reduce_send(g, form)`, `km::reduce_wants(g, form, port)` and
`km::reduce_slots(g, form)` give the schedule itself (`komorebi/device/km_group.hpp`,
free of tt-metal headers). A buffer with too few slots gives up with `bad_item`.

**Repeated in one command.** A reduce over the same group may run again in the same
command, for example a turn of a loop. From its second turn on, a sender writes its
receiver's slot only after a credit: the receiver's reader first reserves the arrival
buffer (so compute has taken the last turn), then adds 1 to word `KM_PORT_CREDIT` of
that port on each of the port's senders (tree: one; owner: size - 1), and only then
waits for the port. A sender's emit of turn t waits until its own word has counted t - 1.
Without the credit, a sender that runs ahead would overwrite a slot compute is still
adding, and the receiver's clear would wipe a bump that came early. The first turn of a
command pays nothing: no credit, no wait, and the same NoC traffic as before. The turn is
counted per command and group (a static on each mover, tagged by the command's
sequence), so the count starts over after a deadline give-up, and a chain's record
starts it over too, because the chain's own credits order its records. A reduce whose
group spans islands is refused at its second turn, at both ends (`bad_item` at `port`
and `emit`), because a credit back would have to cross the link. Two reduces over
different groups that take turns on the same cores in one command reset the count at
each switch, so they are not covered
([hazards](hazards.md#a-reduce-repeated-in-a-loop-of-one-command)).

## All-reduce: the sum at every rank

The same reduce, and then its sum cast back to every rank of the group, in the same
command (`komorebi/device/km_tree.hpp`):

```cpp
#include "komorebi/device/km_tree.hpp"
// reader
km::all_reduce_arrive(reader, command, arrival, group, form);
// writer: rank 0's sum stays at the front of kCbOut (reduce_result, then pop)
if (!km::all_reduce_emit(writer, command, arrival, group, form, kCbOut, 1)) return false;
// compute, after it packed its part: every other rank takes the sum from slot 0
km::all_reduce_take(compute, arrival, group, [&] { copy_tile(arrival.cb, 0, 0); });
```

Rank 0 combines as in the reduce, then writes the sum's pages into slot 0 of every
other rank's arrival buffer, one multicast a NoC rectangle that whole runs of the
group's cores make (a run: consecutive ranks in one row of the NoC), and, once those
writes are acknowledged, sets each rank's port `KM_PORT_ALL_REDUCE` (0) to 1 by the
same multicasts, from a word at the same offset of its 16-byte line. Every
other rank's reader, after its own arrivals, waits for that count and hands the
arrival buffer to compute, whose `all_reduce_take` reads slot 0. The port is free by
then: a rank has sent its part (its own arrivals taken and cleared) before the sum
exists.

One card (p150a, AICLK 800 MHz), the tree form, one float32 tile a rank, every rank's
copy exact, the median of 5 commands on the card
(`experiments/islands/islands_tree/tree.py`; each rank then writes its copy to DRAM,
rank 0 alone in the plain reduce):

| Group | Reduce (rank 0 holds the sum) | All-reduce (every rank holds it) |
|---|---|---|
| 16 ranks | 7.62 µs | 10.48 µs |
| 118 ranks | 10.48 µs | 22.68 µs |

The count was an atomic a rank first: 10.16 and 21.98 µs, the same within the noise,
so the cast's time at 118 ranks is not its count.

Across islands the same calls: the reduce crosses as the ports' reduce does, then rank
0 writes the sum into slot 0 of each other island's first rank of the group
(`km::write_l1`, 1 at its port after), and that leader casts it to its own island's
ranks as rank 0 does to its own (`km::all_reduce_keeps(group)`: the ranks that cast and
keep the sum at their buffer's front). Card 0 split in 2, 118 ranks over both: every
rank exact, 36.6 µs.

In a class: `whole = km.all_reduce(total)` of a tile in DST, or `km.all_reduce(self.part)`
of a float32 field compute wrote, every rank's field then the group's sum
([classes](classes.md)). The split norm of a row of 2,048 over 16 cores
(`experiments/language/tile_split`, `SplitNormAll`): each core sums the squares of its
own 4 tiles, the field all-reduced, then its tiles normalized: the same output as the
norm whose every core reads the whole row (relative RMS error 1.66e-3 against float64
both), 16.46 to 17.70 µs a command against 15.92 to 17.02.

## When a core does not send

Every wait times out at the start-up wait limit and sees the stop flag. The port's
wait gives up first: waits that lie behind it (the writer's wait for a part that needs
arrivals, `take`, `reduce_result`) time out at `km::kBehindPort` (2) wait limits. A
give-up commands `where` = `port` (or `emit` on the sender) with the port number in the
gave-up word's top byte, and leaves the wanted count in the port, so the host reads the
missing count (`commands.waiting_ports` in the test feeder). A reduce given up clears its
ports' counts (a bump counted early would meet the next command's wait). The tile stops, the
dispatcher's or feeder's deadline stops the run, every kernel leaves, and the next
launch runs.

## X280 ports

A port between a Tensix core and the X280 core that serves it, without the dispatcher
(`komorebi/device/km_x280.hpp`; the X280 core's side and the declaration are in
[X280 threads](../threads.md#x280-ports)). A message is a sequence number and three
words, one 16-byte NoC write each way:

```c++
#include "komorebi/device/km_x280.hpp"

km::X280Port port(reader);  // owned by the reader; X280Port(writer, km::kStWriter): the writer
km::X280Msg message;
if (port.call(ask, 0, 0, message)) {  // ask, then wait for its answer
    use(message.w[0]);
}
port.tell(hash, total, largest);  // to the X280 core; it sees the newest
if (port.told(message)) { ... }   // the X280 core's newest tell, if newer; never waits
```

| Call | Waits | Returns |
|---|---|---|
| `served()` | no | whether the program gave this core an X280 core |
| `told(message)` | no | true and the X280 core's newest tell, if newer than the last one looked at |
| `wait_told(message)` | timed | the same, waiting for one; false: gave up (recorded) |
| `tell(first_word, second_word, third_word)` | for the last store to leave | false: gave up (recorded) |
| `ask(first_word, second_word, third_word)` | as `tell` | the ask's sequence number, 0: gave up |
| `answered(sequence, message)` | no | true and the answer to ask `sequence`, if it landed |
| `wait_answer(sequence, message)` | timed | the same, waiting; false: gave up (recorded) |
| `call(first_word, second_word, third_word, message)` | timed | `ask`, then `wait_answer` |
| `next(first)` | timed | the next range of a pull (`kmx_pull_run`): its count and `first`, 0 when none is left or a wait gave up (`pull_gave_up()`); asks for the range after it before returning |
| `next(first, word)` | timed | the same, and the word the X280 core's `pick_word` gave with the range (0 for other pulls and when none is left) |

**Latest wins.** A tell overwrites the line it lands in: a side that looks less often
than the other tells sees only the newest, and tells in between are lost. Ask and
answer are paired by the sequence number, one ask outstanding at a time; between the
first `next()` of a pull and the one that returns 0, that ask is the pull's.

**On the wire.** The core's `KM_CONTROL_X280` area (64 bytes): the X280 core's tell at
`KM_X280_TELL` (`+ 0x00`) and answer at `KM_X280_ANSWER` (`+ 0x10`), stored by the X280 core
through its window; the core's out slot `KM_X280_OUT` (`+ 0x20`), the source of its
writes; `KM_X280_TO` (`+ 0x30`): the X280 core's tile (`x | y << 16`) and the mailbox
address in its local memory, written by the host at the load (0: no X280 core). The core
writes its ask to the mailbox at `KM_BOX_ASK` (`+ 0x00`) and its tell at `KM_BOX_TELL`
(`+ 0x10`) on its kernel's NoC. Each line has one writer; no atomics. A pull's ask is
`{KM_PULL_ASK, 0, 0}`, answered `{first, count, word}` (count 0: none left).

**A word with the range.** With `pick_word` ([the pull's word](threads.md#the-pulls-word))
the X280 core answers each range with a word the core reads beside it, here the
overlay to run on the range (`examples/18_thread_split`):

```c++
uint32_t first = 0, word = 0;
while (const uint32_t count = port.next(first, word)) {
    km::Overlay overlay(reader, word);  // the overlay the X280 core named
    if (!overlay.load()) {  // nothing moves when a slot holds it
        return;
    }
    overlay.call(Range{first, count});
    port.tell(kinds(first, count), word, count);  // what the X280 core decides from
}
```

**When a side does not answer.** Every wait times out at the wait limit and sees the
stop flag. A give-up commands `where` = `kWhereX280Answer` (54), `kWhereX280Tell` (55)
or `kWhereX280Send` (53) for a store that did not leave, `kWhereX280None` (56) when the
program gave the core no X280 core. The command then ends as any command whose core gave up.
