---
title: Dynamic Handlers
summary: How a program author writes a dynamic handler, in which messages move between cores as the data decides, through per-core queues in DRAM, and how the run ends by itself.
tags:
  - reference
  - runtime
---

# Dynamic Handlers

Status: implemented on one Blackhole card, through a `Session` (the dispatcher reads
the status lines and ends the run) and under the test feeder. One dynamic run at a time.
The last cores of every dynamic run's group are counter tiles that hold the queue tails
([Counter tiles](#counter-tiles)). With them, 10^6 forwards were exact in 91 of 91 runs through
a `Session` on 120 cores and in 3 of 3 under the test feeder on 119
([05_forward](../../examples/05_forward/README.md#measured)).

A channel is known before the run: who sends what to whom and how many arrive.
A routed message is known only from the data: a ray goes to the core that holds the
next part of the scene, a finished ray to the core that owns its film tile. In a
dynamic handler the program's state travels as **messages**, fixed-size blocks of words that
move from core to core until the program consumes them. The destination of each message,
the number of arrivals at a core and the end of the run are all decided by the data.

A program has up to four dynamic handlers, and each message names the one that handles
it where it lands ([A message names its handler](#a-message-names-its-handler)). The
program writes the handlers' bodies in its compute kernel. The reader and writer
kernels of a dynamic run are generic and ship with the runtime
(`src/komorebi_native/kernels/messages/reader.cpp` and `messages/writer.cpp`). In the
other form, each handler's body is an overlay that a core loads when a message for it
arrives, and a program has up to 16 handlers
([Handlers as overlays](#handlers-as-overlays)). A message can carry data of up to
4 KiB beside its line ([A message with a body](#a-message-with-a-body)). Beside the
queues, a message can go straight into the L1 of a declared receiver
([A direct message](#a-direct-message)).

## The calls

Compute kernel, inside the runtime's loop (`#include "komorebi/device/messages/km_messages.hpp"`):

```c++
compute.run([&](const km_command_t& command) {
    if (km::handler(command) != kHandlerForward) {
        return;
    }
    km::Messages messages(compute, command);  // command: the RUN command of the dynamic handler
    if (messages.seeding()) {  // once per run and core: the first messages
        ... messages.send(destination, message) ...
        messages.seeded();
    }
    while (km::Packet packet = messages.next()) {  // up to 32 messages; empty when the run ended
        for (uint32_t index = 0; index < packet.count; index++) {
            const volatile tt_l1_ptr uint32_t* incoming = packet[index];  // the message, in L1
            ... the body, then exactly one of:
            messages.forward(destination, message);  // the message goes on at rank destination
            messages.consume();  // the message ends here
            ... and any number of:
            if (!messages.send(destination, message)) { ... no quota left ... }
        }
        messages.done(packet);  // the packet's page is free
    }
});
```

| Call | What it does |
|---|---|
| `Messages(compute, command)` | starts the dynamic part of the run on this core; all three compute threads construct it |
| `seeding()`, `seeded()` | true until `seeded()` or the first `next()`: the place for the run's first sends |
| `next()` | the next packet of up to 32 messages; empty when the run has ended, the stop flag was set or a wait gave up. Every wait inside it is timed |
| `packet.count`, `packet[index]` | the number of messages, and the L1 address of message `index` |
| `forward(destination, message)` | the message joins the queue of rank `destination` of the run's group (a rank, not a core index), with the handler of the message being handled; a forward to the core itself takes the same path |
| `forward(destination, message, handler)` | the same, and the message is handled by `handler` where it lands |
| `consume()` | the message ends; one unit of quota returns to this core |
| `send(destination, message)` | a new message at rank `destination`, with the handler of the message being handled (during seeding, the `RUN` command's); false, and nothing is sent, when this core has no quota left |
| `send(destination, message, handler)` | the same, with handler `handler` |
| `send_or_give_up(destination, message)`, `(destination, message, handler)`, `(destination, message, body)`, `(destination, message, handler, body)` | `send`, and with no quota left the run gives up by name (`quota`, the destination) instead of returning false; a class's `self.send` is this |
| `handler()` | the handler of the message taken last with `packet[index]`; before the first, the `RUN` command's |
| `done(packet)` | the packet's buffer page is free and its messages are counted as consumed |
| `quota()`, `count()` | sends left on this core; cores in the group (without its counter tiles) |
| `rank()`, `counter()` | this core's rank; true on a counter tile, which takes no message |
| `wait_stop()` | waits for the stop flag (a test of the run's timeout) |

A message is `message_bytes / 4` words. Word 0 belongs to the runtime: it holds the message's
mark and its handler ([Marks](#how-a-message-travels)), and in a program with bodies
word 1 is the runtime's too. The program's fields go in the other words; the forward
example uses words 1 to 4.

## The host side

`komorebi.program.messages_header()` gives `(constants, fields)` of
`km_messages.hpp`: its constants as a dict and the field order of `MessagesArgs`.

```python
program.dynamic(
    "forward", message_bytes=64, max_live=1 << 16, queue_messages=1 << 16, bins=16
)
```

`Program.dynamic` makes one handler dynamic, or several (`program.dynamic("lookup",
"value", "report", ...)`, up to four, by name or number; up to 16 with `parts=`, see
[Handlers as overlays](#handlers-as-overlays)), and adds the runtime's three buffers to
the program (circular buffers 26 to 28: incoming packets, outgoing messages,
open bins). All of a program's dynamic handlers share the options and the queues. The
options are those of `komorebi.program.Dynamic`:

| Option | Default | Meaning |
|---|---|---|
| `message_bytes` | 64 | size of a message: 64, 128, 192 or 256 |
| `max_live` | 65536 | the most messages alive at once in the whole group |
| `queue_messages` | 65536 | each core's queue in DRAM, a power of two |
| `bins` | 16 | open bins per writer, one destination each |
| `flush_age` | 500 | cycles: a bin whose first message is this old is sent while no line comes (0: only when full, needed, or compute waits) |
| `idle_cycles` | 20000 | how long a core must stay idle before its line says so |
| `counters` | 2 | counter tiles, 1 to 4: the last cores of each run's group |
| `hold_links` | False | across islands: False, a message to another island's rank is a delivery by the link service (any group, the links shared with pages); True, between two cards only, a RUN of every rank holds the links and the message service carries its messages, pages across waiting meanwhile (example 05's 10^6 forwards: 307.4 ms against 1,531.5 ms by deliveries; [islands](islands.md#the-run)) |
| `stamps` | False | a measurement: each core's reader and writer add up the cycles of each step of a message ([What a queued hop costs](#what-a-queued-hop-costs)) |

The launcher rejects, before any device opens:

- `max_live > queue_messages`: if every message were aimed at one core, its queue could not
  hold them;
- queues that do not fit the DRAM given to them (`cores * queue_messages * message_bytes`);
- a group with no core beside its counter tiles;
- `max_live` smaller than the number of cores (some core would start without quota);
- a ring depth below 2 (the end of a run is a second command in flight).

A run is one `RUN` command of a dynamic handler to a group, with `KM_FLAG_EXCLUSIVE`; its
handler is the one the run's seeds have. The command stays in flight for the whole run;
its deadline is the run's deadline. Its argument words are the program's.

## A message names its handler

A message carries its handler where it lands. A thread of the program, one message,
changes what it does as it moves, or without moving:

```c++
messages.send(rank, message, kHandlerKeep);  // a new message with its handler
messages.forward(rank, message, kHandlerShade);  // goes on at rank, as shade there
messages.forward(rank, message);  // the same handler as the message being handled
messages.forward(messages.rank(), message, kHandlerShade);  // this core, as shade
messages.handler();  // the handler of the message being handled
```

The compute body switches on `messages.handler()`; reading `packet[index]` makes that
message the one being handled. A forward to the core itself goes through its queue like
any other and comes back in a later packet. The writer takes the handler beside the
destination in the word compute hands it (`destination | handler << 8`) and writes it
into word 0 with the mark; the reader checks only the mark. Komorebi carries any handler
number below 256 and does not check that it is one of the program's dynamic handlers; a
handler the body has no case for is the program's to report. A program with one dynamic
handler is written as before: its messages all keep the `RUN` command's handler.

The [message handlers example](../../examples/14_message_handlers/README.md) sends 10^6
messages through three handlers, to other cores and to their own, exact against a
replay.

The kernels' and the dispatcher's test of "a dynamic run" is `KM_IS_DYNAMIC(handlers,
handler)` in `km_abi.h`: the program's dynamic handlers are one word, one a byte, the
first word of `km::MessagesArgs` and the dispatcher's `messages_handler`; a program with
one dynamic handler has its number there, as before.

## Handlers as overlays

A handler's body can be an [overlay](overlays.md): a compute part that a core loads into
a slot when a message for that handler arrives and the core does not hold it. The runtime
then owns all three kernels, the message loop among them
(`src/komorebi_native/kernels/messages/compute.cpp`), and the program supplies one source
per handler:

```c++
// lookup.cpp: the body of handler lookup
#include "komorebi/device/messages/km_messages.hpp"
#include "handlers.h"

void km_message(km::Messages& messages, km::Packet& packet) {
    for (uint32_t index = 0; index < packet.count; index++) {
        const volatile tt_l1_ptr uint32_t* incoming = packet[index];  // a lookup message
        ... forward, send, consume, as in the calls above
    }
}
```

```python
program = km.Program(
    "message_overlays",
    kernels=None,  # the runtime's reader, compute and writer
    args=lambda core: ([], [], [results.address]),  # compute argument 0, in L1
    handlers=folder / "handlers.h",
).dynamic("lookup", "value", "report", parts=folder)
program.slots(count=2)  # optional: compute slots, by default one per handler (up to 4)
run.submit("lookup", (0, count), [threads, steps], exclusive=True)
```

`km_program` is called for a command of a program; `km_message` is called for the
messages of a handler. `parts` is a folder holding `<name>.cpp` for each handler
(`lookup.cpp`, `value.cpp`, `report.cpp` here), or a dict `{name: source}`. Handler `i` of the list is overlay `i + 1`; a program whose
handlers are parts has no overlays of its own (refused either way round).

| Call | What it does |
|---|---|
| `km_message(messages, packet)` | the handler's body, once per packet for the packet's messages of this handler: `packet.count` of them, `packet[index]` as above. `messages.handler()` is this handler |
| `messages.seeding()` | true in the one call that seeds: the `RUN` command's handler is called once with an empty packet before the first packet (not on a counter tile) |
| `messages.command()` | the `RUN` command (its argument words are the program's) |
| `messages.arg(index)` | the program's compute argument `index` |

A part does not call `next()` or `done()`: the runtime takes the packet, calls each of
its handlers and frees it.

**Per packet.** When the reader has read a packet, it groups the packet's messages by
handler in the packet's header: the handlers the tile holds first, then the others, each
in the order of the program's list. It hands compute the packet, then runs `load()` of each
group's overlay in that order. Compute takes the groups in the same order: `load()`, then
`call()` with a packet that holds that group's messages. The reader and compute decide
from the same records whether the tile holds a handler, so they meet only for a handler it
does not hold; a held handler costs one look at the records per packet, nothing per
message. Messages of one packet are handled by handler, not in queue order (Komorebi
never promised an order).

**State.** A part holds nothing from one call to the next (its `.bss` is zeroed on every
call). What a handler keeps across messages (counters, sums, a table) lives in L1 that the
program declares, as any resident data: a tensor with `shard_grid` whose address is a
compute argument (`messages.arg(0)` above). The body runs on all three compute cores
alike, as before: one of them writes the state (`UNPACK((...))`).

**An overlay sent by name.** A message names its handler, so a handler's part travels
with a thread by name: where the thread lands, the reader loads the part from DRAM if the
tile does not hold it, then compute calls it. A handler that only `send` or `forward`
names (not the `RUN` command's) is loaded nowhere until a message for it arrives.
Example 23's `--send` run: each of 12 ranks loaded the chain handler once, at the
thread's first arrival, 8.32 µs against 6.51 µs for a later arrival
([23_message_body](../../examples/23_message_body/README.md#an-overlay-sent-by-name)).

**The limits.**

- Up to 16 handlers (`kMessagesHandlersMax`): the packet's 64-byte header holds a
  group's end for each. A `RUN` command names one of the first four, the handlers the
  dispatcher knows (`KM_DYNAMIC_MAX`, one a byte in one word); the others are reached
  only by `send` and `forward`. Every other overlay number (17 to 63) is free.
- Slots: by default one a handler on each compute core, as many as the L1 holds
  (`program.slots`), as for any overlay; with fewer
  slots than handlers, a load replaces the handler the tile used least recently. The
  slots lie below 1 MB of L1, and the session's L1 budget table lists them.
- A message whose handler is not one of the program's dynamic handlers, or whose program
  field is not 0, stops the run before compute sees it: the core gives up with reason 40
  (`handler`), and `seen` is the message's word 0. A part that does not load gives up
  with reason 41 (`load`); the core's gave-up record names the overlay and why.
- The compute kernel does no hardware start-up (as the forward and handler examples):
  a part that uses the matrix or vector unit calls the compute API's own `*_init` calls.

**What it costs.** Card 0, AICLK 1,350 MHz, the three handlers of the
[message handlers example](../../examples/14_message_handlers/README.md) in its form and
as parts ([20_message_overlays](../../examples/20_message_overlays/README.md#measured)):

| | 120 cores, 1,013,360 messages | 1 core, 82,938 messages |
|---|---|---|
| bodies in the compute kernel (example 14) | 10.83 ms, 93.6 M messages a second | 20.90 ms, 340 cycles a message |
| as parts, 3 slots | 10.76 ms, 94.2 M a second; 3 loads a core (the first messages) | 21.69 ms, 353 cycles; 3 loads |
| as parts, 2 slots | 10.77 ms, 94.1 M a second; 13,230 loads, 112 a core, one per 77 messages | 21.67 ms; 3 loads |
| as parts, 1 slot | 11.40 ms, 88.9 M a second; 97,394 loads | 22.20 ms, 361 cycles; 601 loads, 0.85 µs each |

Medians of 10 runs (3 for one slot); the run time is the `RUN` command's, submit to
completion. The one-core column is from card 1 on the same day (host load 4 to 5, three
runs each), after the writer's mark change. A load stops the core's message loop while
compute finishes the packet before it and the parts are read: about 0.8 µs of the run's
time on one core and on the grid, where the loads of 118 cores share the DRAM banks. The
grid's run is bound by its queues in
DRAM, not by compute, so 112 loads a core cost it nothing measurable. Serving the held
handlers of a packet first (the order above) is what keeps loads few: without it, a packet
with all three handlers and two slots loads three times, and the same runs loaded 20,249
times (2 slots, grid) and 5,137 times (1 slot, one core: 28.70 ms, 1.3 µs a load).

## A message with a body

A message can carry a body of up to one page (4 KiB) beside its line: the data that
rides with a thread as it is forwarded, too large for the line's words. The program
names the most bytes a body has; the writer sends a body with its message, and the reader
hands the handler a pointer and a length.

```python
program.dynamic("chain", max_live=128, queue_messages=128, body_bytes=4096)
```

```c++
const volatile tt_l1_ptr uint32_t* incoming = packet[index];
const km::Body body = packet.body(index);  // in L1 until done(packet); 0 bytes: none
volatile tt_l1_ptr uint32_t* words = body.words();
... read or change words[0 .. body.bytes / 4) ...
messages.forward(destination, message, body);  // the message and its body go on
messages.send(destination, message, km::Body{address, bytes});  // a new one, from L1
```

| Call | What it does |
|---|---|
| `packet.body(index)` | the body of message `index`: `address` (L1, 64-byte aligned), `bytes` (0: the message has none) and `words()` |
| `forward(destination, message, body)`, `forward(destination, message, handler, body)` | as `forward`, with `body` beside the message. A body from a buffer of the program: returns on all three compute threads once it has left, so its L1 can be written again; a body of the packet in hand: returns at once, and `done(packet)` waits until the packet's bodies have left |
| `send(destination, message, body)`, `send(destination, message, handler, body)` | the same for a new message (false, and nothing is sent, without quota) |

`body_bytes` is 0 (the default: no bodies) or a multiple of 64 up to 4096. In a program
with bodies, word 1 of a message is the runtime's too: its body's bytes. A forward or a
send without a body sets it to 0. A body sent is a buffer of the program in L1 (a tensor
with `shard_grid`, as the example's area) or a body that arrived, 64-byte aligned,
`bytes` at most `body_bytes`; the compute threads that write it may each write a part,
since the call waits until every thread has reached it.
A handler that is an [overlay part](#handlers-as-overlays) takes bodies the same way:
`packet.body(index)` follows the part's own order of the packet's messages (example 23's
`--parts`: 6.39 µs a hop on 12 cores against 5.45 with the body in the compute kernel).

**How a body travels.** Every queue position has a DRAM page of `body_bytes` beside its
line (`cores * queue_messages * body_bytes` more DRAM: 60 MiB for 120 cores, 128
positions and 4 KiB). The writer takes a message with a body as any other, then sends its
bin at once: after the reservation it writes the body to the page of the message's
position and the lines, and the packet's commit waits for both, so a reader that sees the
mark finds the body landed. The writer then counts the body as written, which is what the
pack thread waits for before `forward` returns (or `done`, for bodies of the packet in
hand). The reader reads the bodies of a packet's
messages into the packet's page after its lines; a packet of a program with bodies holds
at most 4 messages (`kMessagesBodyPacket`), so the incoming buffer's page is 18,496 bytes
for 64-byte lines and 4 KiB bodies (two pages).

A body whose bytes are above `body_bytes` or whose address is not 64-byte aligned stops
the run: the core gives up with reason 50 (`body`), as it does when the writer has not
written a body within the wait limit.

**What it costs.** Card 1, a forward chain of one thread
([23_message_body](../../examples/23_message_body/README.md#measured)), the time of the
`RUN` command over its hops:

| | 64-byte message | with a 4 KiB body | with a 4 KiB body changed at every hop |
|---|---|---|---|
| 12 cores, 1200 hops | 1.97 µs a hop | 2.39 µs | 5.45 µs |
| the grid (118 ranks), 1180 hops | 2.09 µs | 2.53 µs | 5.57 µs |

A body adds about 0.4 µs to a hop, in which it goes to DRAM and back and the writer
sends the bin at once instead of filling it; the times do not split that further.
Changing the 1024 words on the three compute threads takes the rest.

No core copies a body: the writer sends it from where it lies in L1 (the program's
buffer, or the packet it arrived in), and the reader reads it into the packet page.
Over the grid, 2048 messages forwarding their bodies as they arrived to ranks drawn
from a ray tracer's traffic move 34.6 million 2 KiB bodies a second, 33.8 million of
4 KiB ([ray_moves](../../experiments/programs/ray_moves/README.md#part-b-the-transport-card)).
The bound there is each core's writer: a body is one reservation, its writes and one
commit. The writer overlaps them across bins: it commits the last bin while the next
bin's reservation travels, and a forward of a body that arrived does not wait for its
bin (before both: 30.3 and 29.4 million, `forward` waiting about 2,100 cycles).

## A direct message

Beside its queues, a program can declare pairs of ranks between which a message goes
straight from the sender's L1 into a slot in the receiver's L1: no DRAM, no
reservation, no counter tile. The call returns false while the slot still holds the
pair's last message; the program then tries again later or forwards through the
queue.

```python
program.dynamic("ring", max_live=1024, queue_messages=1024, direct=(1,))  # r to r + 1
program.dynamic("walk", direct=[(0, 5), (0, 9), (5, 0)])  # (sender, receiver) ranks
program.dynamic("walk", direct={0: [5, 9], 5: [0]})  # the same pairs
```

```c++
if (!messages.direct(to, message)) {  // false: the slot is busy, nothing was sent
    messages.forward(to, message);
}
```

| Call | What it does |
|---|---|
| `direct(destination, message)`, `direct(destination, message, handler)` | as `forward` (quota unchanged), through this core's slot at rank `destination`; true on all three compute threads when it went, false on all three when the slot is busy |
| `direct(destination, message, body)`, `direct(destination, message, handler, body)` | the same with a [body](#a-message-with-a-body) beside the line, in a program with `body_bytes`; true once the body has left this core's L1 |

`direct=` takes rank offsets (each rank r to (r + d) mod n, n the load's message ranks:
its cores without the counter tiles; 0 is the core itself), a list of (sender,
receiver) pairs, or a mapping from a sender to its receivers. A core may have any
number of receivers and senders; the bound is the L1 it costs.

**How it travels.** Each core has a direct area in L1 that the host writes at the load
(an L1 tensor): a table of its declared receivers (each one's rank and the place of this
core's slot there) and of its senders (each one's rank and its credit line), a 16-byte
credit line per receiver, two counts per receiver, a taken line per sender, and a slot
of `message_bytes` + `body_bytes` per sender. `direct()` finds the receiver in the table (a linear
search: one L1 load and one compare an entry, not measured apart) and compares the
receiver's count on the credit line with its own count of lines sent; equal, the slot is
free: pack counts one more and hands the line to the writer. The writer writes the line
into the slot with mark 0, waits for the write, then writes its first 16 bytes with the
mark (the pair's count as a lap, as a queue position's), the queue's commit. The
receiver's reader looks at its senders' slots before its queue; it copies a slot whose
mark is the next one into a packet page, as it does the X280 ranks' inbox, and hands the
slot back by writing its own count to the sender's credit line. A body goes into the
slot after the line's place, before the line and its mark; the reader reads it by the
NoC from the slot into the packet page after the lines, as a queued body, then hands
the slot back. Compute sees a
direct message as any other: `next()`, `packet[index]`, the handler named in its word 0
(handlers as parts group it as an inbox packet: example 23's `--parts --direct`, 1199 of
1199 hops direct, 3.05 µs a hop, 6.30 with a 4 KiB body changed, against 3.13 / 8.25
through the queue).

A class's dynamic handler writes it as a name's value, as a class names any value a
line gives: `went = self.direct(rank, message)` (with `to=Class.handler` as `forward`)
is 1 when the message went, 0 when the slot was busy; the call anywhere else (alone on
its line, or inside a test) is refused, since its 0 would be lost
([direct_ring](../../experiments/programs/direct_ring/README.md)'s `ring_class.py`).

**L1.** A core's area is 48 bytes, 28 bytes a receiver and 20 + `message_bytes` +
`body_bytes` a sender, rounded to 16; the load's L1 table lists the largest core's as
"direct messages (tables, slots)". A ring of 118 ranks with 64-byte messages: 176 bytes
a core; with 2 KiB bodies, 2,224.

**The run's end.** A direct line counts in its sender's `sent` once its commit has
landed, and in its receiver's `consumed` at `done()`, as a queued one. The reader's idle
test looks at the slots too, so the two passes cannot both see every core idle while a
line is in a slot.

**What is refused.** At the load: a pair given twice, a rank outside the message ranks,
`direct=` beside X280 ranks, and a group across islands (a direct line
across islands would follow the link service's write into a far core's L1; not in this
version). The tables hold ranks of runs whose group starts at core 0: a run whose group
starts elsewhere gives up with reason 77 (`direct`), and so does `direct()` to a rank
that is not a declared receiver of this core.

**What it costs.** Card 1, 2026-11-16, AICLK 1,350 MHz, host load 3.9 to 6.2, one
token a rank, 8475 hops each (1,000,050 hops), every token replayed on the host and
exact ([direct_ring](../../experiments/programs/direct_ring/README.md)):

| 118 ranks | direct, forward when busy | queue only |
|---|---|---|
| a ring, r to r + 1 | 16.44 ms, 1.94 µs a hop; 75 % of hops direct | 97.93 ms, 11.55 µs a hop |
| pairs, r to 5r + 1 or r² + 3 (mod 118) | 26.20 ms, 3.09 µs a hop; 67 % direct | 26.93 ms, 3.18 µs a hop |
| a ring with a 2 KiB body each token | 22.32 ms, 2.63 µs a hop; 72 % direct | 43.99 ms, 5.19 µs a hop |

## How a message travels

Every core has a queue of `queue_messages` messages in DRAM. A sender never waits for the
receiver: the writer collects outgoing messages in bins (one destination each, up to 32
messages), reserves positions in the receiver's queue with one atomic add on the
receiver's tail word on its counter tile, writes the messages to DRAM and counts them as sent. The receiver's
reader takes messages from the head of its queue in order, as packets of up to 32. A bin is sent when
it is full, when its slot is needed for another destination, when the core runs
out of input, so nothing waits in a bin while the core is idle, and when no line comes
and its first message is `flush_age` cycles old.

### What a queued hop costs

`Dynamic(stamps=True)` builds the reader and writer with stamps (`KM_MESSAGES_STAMPS`,
a build apart): each core adds up, in an L1 area the load gives
(`run.backend.stamps`, words `kStamp*` of km_messages.hpp), the cycles a line waits in
its bin, the reservation's round trip, the waits for written lines, the time from a bin
written to its commit, the reader's tail reads and line reads, the reads that found no
mark at the head, and the time from a new tail to the packet's push. Card 1,
2026-11-16, 118 ranks, 10^6 hops ([direct_ring](../../experiments/programs/direct_ring/README.md)
`--stamps`, before `flush_age`), cycles at 1,350 MHz:

| | spread (pairs): 3.35 µs a hop | the ring: 11.55 µs a hop |
|---|---|---|
| lines a flush, a packet | 1.4, 1.5 | 22.6, 22.6 |
| a line in its bin | 815 | 5,360 |
| reservation, lines landed, commit | 394, 423, 329 | 394, 456, 337 |
| reader: tail read; lines read; new tail to push | 330; 592; 1,135 | 403; 875; 1,956 |
| compute: a packet in hand to `done` | 629 | 8,710 (386 a token) |

Spread traffic, as a ray tracer's, has no one large part. The ring was a convoy: its
tokens bunched into about five groups, each token waiting in its bin while its core
handled the whole packet, since a bin was sent only when full or when compute waited.
`flush_age` sends such a bin while compute is still busy, and the groups break up
(1.3 lines a flush at 1000). Card 1, 2026-11-17, AICLK 1,350 MHz, host load 3.2 to 8.0,
`flush_age` at eight values, the median of three runs each (every run exact); a hop is
the `RUN` command's time over the hops, cycles at 1,350 MHz:

| `flush_age` | 0 | 100 | 250 | 500 | 1000 | 2000 | 4000 | 16000 |
|---|---|---|---|---|---|---|---|---|
| ring, queue only (µs a hop) | 10.69 | 2.58 | 2.58 | 2.52 | 3.79 | 4.08 | 5.63 | 10.80 |
| ring, direct and forward when busy | 1.91 | 1.91 | 1.91 | 1.91 | 1.91 | 1.91 | 1.91 | 1.91 |
| pairs, queue only | 3.08 | 3.12 | 3.12 | 3.11 | 3.10 | 3.08 | 3.10 | 3.12 |
| one thread with a 4 KiB body (example 23) | 7.47 | 7.47 | 7.47 | 7.47 | 7.48 | 7.47 | 7.47 | 7.47 |
| ray_moves bodies, 2 KiB (millions a second) | 33.82 | 33.73 | 33.78 | 33.78 | 33.86 | 33.71 | 33.78 | 33.77 |
| ray_moves bodies, 4 KiB (millions a second) | 32.88 | 32.90 | 32.88 | 32.95 | 32.97 | 32.90 | 32.93 | 32.99 |

A bin is aged only while some bin besides the one awaiting its commit waits, so a lone
body's bin is committed as before.
The bodies are about 2% slower than before the stamps and the age came in (7.33 µs and
34.7 / 33.7 M a second), at either age and without stamps: the cost follows the build of
the writer, and neither taking that bookkeeping off a body's path nor moving the age scan
out of line brought it back.

Only the ring through the queues alone moves with the age: its tokens form convoys, so
each waits in its bin until the age, and a small one (100 to 500) makes it about four
times faster (2.5 µs a hop against 10.7 at 0). Above 500 the age costs it: 3.79 at
1000, 5.63 at 4000, and at 16000 (11.9 µs, longer than a hop) as much as at 0.
Traffic with many destinations (the pairs, a ray tracer's), the direct ring and bodies
stay within 1.5 % at every value, so the age neither helps nor hurts them. 0 sends a
bin only when it is full, needed for another destination, or compute waits. The
default is 500: the ring's best value, with every other row within 1.5 % of its own
best there (the default was 1000 until 2026-11-17).

**Marks.** Word 0 of a message in a queue is `mark | handler << 16 | program << 24`. The
mark, in the low 16 bits, is the position's lap of the queue, `position /
queue_messages`, counted from 1 to 65535 and round again; 0 is empty. The program field
is 0 (reserved). Every core zero-fills its own queue at start-up, before it reports
ready, and a writer reserves a position only after the reader has taken the one a lap
before it (`max_live <= queue_messages`), so the slot the reader waits on holds 0, the
previous lap's mark or the message written for it: the reader compares the low 16 bits
alone. The mark sets no limit on a run: the lap count wraps every 65535 laps (2.1 M
messages a core with a queue of 32, 1.07 G with 16384), and a run of 118 messages
forwarded 72000 times each to one core with a queue of 128 (66,375 laps) was exact
([05_forward](../../examples/05_forward/README.md#measured)). The limit is the load's, as
before: a core rejects a run once its positions have reached 2^30 (reason `full`, about
10^9 messages a core); reload the program then.

**Commit.** Only a 16-byte chunk is known to land whole, so a packet commits
explicitly: the writer sends the packet's messages with the first message's mark 0, waits
for those writes, then rewrites the first message's first 16 bytes with its mark. The
reader takes the prefix of valid marks from its head; it cannot pass an uncommitted
first message, and once that message is valid the whole packet had landed.

**Reservation.** The atomic add's answer goes to a return address programmed on every
call, to a 16-byte slot of the sender's own control block (the firmware's default
return slot is shared). A sender waits for the answer to replace a sentinel, not only
for the atomic's count; an answer that never lands ends the run with reason 32 (`lost`),
naming the destination, never with a wrong result. A program that adds its own barrier between cores at start-up
must allow more than the launch skew: under slow dispatch a timeout of 7 ms let 47 of
119 cores give up; the reservation probe allows one second.

So a full inbox cannot deadlock the grid, and while `max_live <= queue_messages` holds no
queue can overflow. The cost is that every forward goes through DRAM twice.

**Quota.** Each rank starts a run with `max_live // ranks` sends, where ranks are the RUN's
ranks: its group's cores over every island without the counter tiles, plus the load's X280
ranks. The device divides at the run's start (the compute words carry `max_live`); one
dynamic run uses the queues at a time (a dynamic `RUN` waits while another or its end is
in flight), so the quotas of a run sum to at most `max_live`. A group of 16 cores (14
ranks) at `max_live` 4,096 gives 292 sends a rank; a full card's 118 ranks give 34; a
group of one rank gets all of `max_live`. `send` takes one, `consume` gives one back to the
core where the message ends, `forward` changes nothing. Seeding sends count too. Moving
quota between cores is the program's business: a program that sends on some cores and
consumes on others runs those cores dry. A hand kernel's `send` returns false with no
quota left, and nothing is sent. A class's `self.send(rank, m)` gives the run up instead
(`quota`, the destination): the run's error is `km.MessagesGaveUp`, which names the core
(on two cards, its rank and island) and says to raise `max_live` or to write
`went = self.send(rank, m)`, which is 0 when nothing was sent.
A message is never dropped unseen. Whether a core runs out depends on timing: a fan of
3 children to depth 3 from 16 seeds on 14 ranks ran exact at 34 sends a core in one run
and ran out in the next.

## Counter tiles

The tails do not live on the receivers. The last K cores of a run's group (`counters`,
2 by default; `komorebi.layout.counter_tiles` gives the layout) are counter tiles: core
core c's tail is a 16-byte line on counter tile c mod K, at `KM_CONTROL_TAILS`. Every
reservation is an atomic add on that line; a core learns its tail by a timed
16-byte read of it. A counter tile runs the same three kernels: at the start of a run it
sets the tails of the cores it serves to their heads and shows READY (cores wait
for it, with a timeout, reason 35 `counter`), then takes no message, never submits an atomic, goes
idle like any core and leaves on the end command. `run.state()` names it
(`dynamic.counter`: the run and the cores it served).

**Why.** On the card, the returned value of a non-posted atomic add is sometimes lost
when the tile that answers it submits atomics of its own on the same NoC (with a return
address that changes from call to call): the add lands, the answer never does. When every
core kept its own tail, 2 to 12 of 50 runs of 10^6 forwards ended with reason 32 (two
builds). A tile that
only answers lost none in the 21.4 M reservations of the probe runs that found the
condition ([atomic_answers](../../experiments/noc/atomic_answers/)). This is an empirical condition of
the atomic unit; its cause is not known. Ports and group completion are not affected: they
count acknowledgements and never use a returned value.

**Cost.** K cores of the group take no message. Every core's control block grows by the
tail lines (`KM_CONTROL_TAILS`): 2,112 bytes of L1 on every core for every
program, dynamic or not. Card 0, 120 cores, X280 at 1000 MHz, 10000 messages
of 100 forwards (10^6 forwards), submit to completion: 11.8 ms with K = 2 (median of 30 runs,
11.6 to 12.0 ms), 11.6 ms with K = 1 (10 runs); 10^5 forwards 3.43 ms, 10^3 forwards 172 µs.
At the 1750 MHz opt-in: 11.8 ms (median of 91 runs), 11.6, 3.44 ms and 158 µs. In the probe
one counter tile answered 126 M reservations a second (one per 10.7 cycles) and two
answered 229 M: a program that reserves more often than the example may need K = 2 or
more.

## How the run ends

Nobody counts the messages in advance. Each core keeps two counters, messages sent (forwards and
sends) and messages consumed, and writes a 16-byte status line `{run, state, sent,
consumed}` to the submitter when it has been idle for `idle_cycles`: its queue empty, its
compute waiting in `next()`, every bin sent. Before it takes a message again it writes a
busy line and waits until that write has landed.

The submitter reads every line of the group twice. When both passes show every core idle
with the same counters, and the total sent equals the total consumed, no message is in a
queue or on the way: the submitter sends an end command (`KM_HANDLER_MESSAGES_END`) to the group,
`next()` returns an empty packet on every core, and the `RUN` command completes through
the usual group completion. Under a `Session` the submitter is the dispatcher: the
lines are in its local memory (`KMD_LIM_MESSAGES`), it checks them on every pass of its loop
while the run is in flight, and the end command goes out under the `RUN` command's
command, so the host sees only the `RUN` command's done event. One dynamic run is in
flight at a time: a second `RUN` of the dynamic handler from the host waits until the
first has ended (a `RUN` of any of the program's dynamic handlers); from the stream engine or a work cluster it is rejected busy (reason
11) and sent again after the kernel's wake, like any command of those sources that
cannot go now. A queued one (`KM_FLAG_QUEUE`) is rejected with reason 12. The status lines
take 2 KiB of the kernel's local memory (`KMD_LIM_MESSAGES`).

## When something goes wrong

Every wait has a timeout and watches the stop flag. A core that gives up writes its first
reason, the queue position and the run into its control block (`KM_CONTROL_MESSAGES +
KM_MESSAGES_GAVE`), writes a `gave up` status line, and the submitter stops the run on every
core. Under a `Session` the dispatcher ends the session with the `RUN` command's
timeout and names the core in its core block (`KMD_H_MESSAGES_GAVE`, 1 + core). No reset is needed: the next run starts on
the same loaded program.

| Code | Reason | Meaning |
|---|---|---|
| 19 | gap | a reserved position was not written within the wait limit; word 3 is the word 0 found there |
| 20 | overflow | more messages than `queue_messages` were in one core's queue |
| 21, 22, 23, 25 | atomic, flush, read, line | a reservation, the message writes, the message reads or the status line did not finish within the wait limit |
| 24 | in page | the compute kernel did not free a packet page within the wait limit (a body that stalls) |
| 26 | bad dest | `forward` or `send` named a core outside the group |
| 27, 28, 29 | next, out page, outbox | a wait between the compute kernel and the reader or writer ran out |
| 30 | stopped | the stop flag was set |
| 31 | align | the packet buffer or the message size is not a multiple of 64 bytes |
| 32 | lost | a reservation was answered but its returned position never landed; word 3 names the destination (not seen with counter tiles) |
| 33 | fill | the start-up zero fill of the queue did not finish |
| 34 | full | this core's queue positions reached 2^30 in this load |
| 35 | counter | a counter tile did not show READY within the wait limit, or the group has no core beside them |
| 40 | handler | handlers as parts: a message names a handler that is not one of the program's dynamic handlers, or a program other than 0; word 3 is its word 0 |
| 41 | load | handlers as parts: a handler's part did not load (the core's gave-up record names the overlay) |
| 77 | direct | `direct()` to a rank that is not a declared receiver of this core, or a run with direct pairs whose group does not start at core 0; word 1 is the rank |

The codes follow the runtime's where codes (0 to 18), 40 and 41 follow the overlays',
77 follows the asks', and all are `kWhereMessages*` in `km_core.hpp`.

## X280 ranks

A dynamic run can take X280 cores as ranks too: X280 cores 0 to n - 1 (n from 1 to 4)
of one work cluster's thread are ranks `count` to `count + n - 1`, after the group's
`count` core ranks. A core's `forward` and `send` to such a rank work as to a core; the
thread takes its messages and sends to core ranks with the calls of
[X280 cores as ranks](threads.md#x280-cores-as-ranks). The host declares them with the
handlers:

```python
program.dynamic(
    "grid",
    "x280",
    message_bytes=64,
    max_live=8192,
    queue_messages=8192,
    counters=1,
    x280_ranks=4,
    x280_cluster=1,
)
program.thread(cluster=1, sources=["rank.c"])
```

On a core, `messages.x280()` is the number of X280 ranks (0: none); `forward` and `send`
accept ranks below `count() + x280()`. The quota is `max_live` over every rank, the X280
ranks included.

**Cores to an X280 rank.** Each X280 rank has a queue of `queue_messages` messages in
its cluster's DRAM bank (the thread's region `x280_queues`, after its own), with the
same marks and commit as a core's queue. It borrows core index `KM_X280_RANK_CORE(r)` (124
+ r) for its tail line on the counter tiles and the writer's destination, so the
grid has at most 124 cores. The X280 rank flushes the head of its queue and takes the
prefix of valid marks, up to 8 messages at a time.

**An X280 rank to cores.** A core's messages from the X280 ranks land in an inbox of 32
messages in its L1, which its reader takes before its DRAM queue. The cluster's X280
cores share each inbox's positions through an atomic add in their local memory; the
core's reader writes its head back into a credit line there every 8 messages, and an
X280 rank waits for room for at most the line's `wait` (then it gives up, below). The
inbox is not part of `max_live`: a full inbox makes the X280 rank wait, never a core.

**The end.** Each X280 rank keeps a status line `{run, state, sent, consumed}` like a
core's, at core index `KM_X280_RANK_CORE(r)` in the dispatcher's lines, and the
dispatcher reads `count + n` lines in its two passes. It stores a run line
`{run, START, group, handler}` into the thread at the `RUN`, and `{run, END, ...}` at the
end command; `kmx_messages_next` returns no message once it sees the end. An X280 rank
that gives up writes a `gave up` line (`KMX_GAVE_*` in word 3, its queue's head in word
2) and stops taking messages, and the dispatcher stops the run as for a core that gives
up ([When something goes wrong](#when-something-goes-wrong)), naming core index
`KM_X280_RANK_CORE(r)`. A rank that gives up for room (`KMX_GAVE_ROOM`) has taken the
core's next inbox position and leaves it unfilled; the core reads its inbox in order,
so every later run of the load is refused at once, each X280 rank giving up
`KMX_GAVE_BROKEN` (5), until the next load.

**Over islands.** On two or more cards, with deliveries (`hold_links=False`, the
default), island 0 holds the X280 ranks: the other islands load the program with no
X280 ranks, their cores still counting island 0's (their thread runs as a thread
without ranks: `kmx_messages_run` returns `KMX_BAD` there at once). A core of any island forwards and sends to rank `count + r`: its
message goes to X280 rank r's queue on island 0, across the link from another island
(rank `KM_X280_FLAT(r)` = 0xFFF0 + r in the writer's word, above every core's flat
rank; a session of more ranks is refused at load). The dispatcher gives the X280 ranks
the group's message ranks over every island as `count`, each island's part less its
counter tiles as the cores count them, so `rank` agrees with the cores'. An X280 rank
forwards and sends to the core ranks of every island. To another island's rank the
message is a delivery line of the rank's own: the rank waits until the line before in
its ring place is confirmed (`KM_X280_RING_LINES`, 8 lines in flight a rank), stages
the message in its far area in LIM, and stores the line, sealed as every line of the
link service, into its ring on island 0's link core toward that island (2 KiB,
`KM_X280_RINGS_BYTES`, of every link core's L1 in such a load). That link core's lead
page CPU sends it on as a core's delivery, and the holder's lead confirms it with one
write into the rank's LIM. A place still unconfirmed after the wait gives the rank up
with `KMX_GAVE_FAR` (6; not exercised on a card), and the run ends as for a give-up.
X280 ranks over islands run between two cards, across one link: a split card has
none (its dispatchers take the work clusters), so a rank two links away is not run. A dynamic `RUN` must take
ranks of island 0 (refused at submit, by name), and `hold_links=True` with X280 ranks
is refused at load.

Limits of this version: the X280 ranks are on one cluster of one island; an X280 rank
sends only to core ranks; a program with bodies
(`body_bytes`) has no X280 ranks; messages an X280 rank's function leaves untaken when
it returns are consumed by the library. The [thread rank
example](../../examples/24_thread_rank/README.md) runs 12 cores and four X280 ranks
with every message checked, and on two cards (`islands.py`) the cores of both islands
and island 0's X280 ranks sending to each other.

## Not in this version

- A direct path the runtime chooses by itself, with DRAM as overflow: a
  [direct message](#a-direct-message) is the program's call, and a false is the
  program's to handle.
- A body that asks for a DRAM fetch decided by a message: a dynamic program keeps what its
  body reads in L1.
- Quota moving between cores; dynamic handlers with different message sizes in one
  program; more than four handlers with the bodies in the program's compute kernel.
- Dynamic handlers of two programs in one run: the program field of a message is 0.
- A body larger than one page (4 KiB). A cast of a body to a group is a broadcast whose
  block is one 4 KiB page
  ([23_message_body](../../examples/23_message_body/README.md#a-cast-of-a-body)).
- An overlay's bytes as a body: a handler that forwards its own code with its thread
  does not fit one page (example 23's chain handler is 7,232 bytes of parts). A message
  sends the overlay by name instead (below).
- Priorities or ordering between messages: a program that needs an order sorts at the
  receiver.

The [forward example](../../examples/05_forward/) is a complete dynamic program with a numpy
replay of every message.
