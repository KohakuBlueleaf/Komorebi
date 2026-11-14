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
(`src/komorebi_native/kernels/messages_reader.cpp` and `messages_writer.cpp`). In the
other form, each handler's body is an overlay that a core loads when a message for it
arrives, and a program has up to 16 handlers
([Handlers as overlays](#handlers-as-overlays)). A message can carry data of up to
4 KiB beside its line ([A message with a body](#a-message-with-a-body)).

## The calls

Compute kernel, inside the runtime's loop (`#include "komorebi/device/km_messages.hpp"`):

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
| `idle_cycles` | 20000 | how long a core must stay idle before its line says so |
| `counters` | 2 | counter tiles, 1 to 4: the last cores of each run's group |

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
(`src/komorebi_native/kernels/messages_compute.cpp`), and the program supplies one source
per handler:

```c++
// lookup.cpp: the body of handler lookup
#include "komorebi/device/km_messages.hpp"
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

**The limits.**

- Up to 16 handlers (`kMessagesHandlersMax`): the packet's 64-byte header holds a
  group's end for each. A `RUN` command names one of the first four, the handlers the
  dispatcher knows (`KM_DYNAMIC_MAX`, one a byte in one word); the others are reached
  only by `send` and `forward`. Every other overlay number (17 to 63) is free.
- Slots: 1 to 4 on each compute core (`program.slots`), as for any overlay; with fewer
  slots than handlers, a load replaces the handler the tile used least recently. The
  slots go just under 1 MB of L1, and the session's L1 budget table lists them.
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
| `forward(destination, message, body)`, `forward(destination, message, handler, body)` | as `forward`, with `body` beside the message; returns on all three compute threads once the body has left, so its L1 can be written again |
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
pack thread waits for before `forward` returns. The reader reads the bodies of a packet's
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

## How a message travels

Every core has a queue of `queue_messages` messages in DRAM. A sender never waits for the
receiver: the writer collects outgoing messages in bins (one destination each, up to 32
messages), reserves positions in the receiver's queue with one atomic add on the
receiver's tail word on its counter tile, writes the messages to DRAM and counts them as sent. The receiver's
reader takes messages from the head of its queue in order, as packets of up to 32. A bin is sent when
it is full, when its slot is needed for another destination, and when the core runs
out of input, so nothing waits in a bin while the core is idle.

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

**Quota.** Each core of the group starts a run with `max_live / cores` sends. `send` takes one,
`consume` gives one back to the core where the message ends, `forward` changes nothing. Seeding
sends count too. Moving quota between cores is the program's business: a program that
sends on some cores and consumes on others runs those cores dry.

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
condition ([atomic_answers](../../experiments/atomic_answers/)). This is an empirical condition of
the atomic unit; its cause is not known. Ports and group completion are not affected: they
count acknowledgements and never use a returned value.

**Cost.** K cores of the group take no message. Every core's control block grows by the
tail lines (`KM_CONTROL_TAILS`): 2,112 bytes of L1 on every core for every
program, dynamic or not. Card 0, 120 cores, X280 at the session's 1000 MHz, 10000 messages
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

The codes follow the runtime's where codes (0 to 18), 40 and 41 follow the overlays',
and all are `kWhereMessages*` in `km_core.hpp`.

## Not in this version

- A direct L1-to-L1 path with DRAM as overflow, and a forward to the core itself without DRAM.
- A body that asks for a DRAM fetch decided by a message: a dynamic program keeps what its
  body reads in L1.
- Quota moving between cores; dynamic handlers with different message sizes in one
  program; more than four handlers with the bodies in the program's compute kernel.
- Dynamic handlers of two programs in one run: the program field of a message is 0.
- A body larger than one page (4 KiB). A cast of a body to a group is a broadcast whose
  block is one 4 KiB page
  ([23_message_body](../../examples/23_message_body/README.md#a-cast-of-a-body)).
- An overlay sent as a body (`overlay.send`) that the target copies into its slot: a
  handler that forwards its own code with its thread does not fit one page (example
  23's chain handler is 7,104 bytes of parts, 4,608 of them pack's), and the loader
  reads parts from DRAM only.
- Priorities or ordering between messages: a program that needs an order sorts at the
  receiver.

The [forward example](../../examples/05_forward/) is a complete dynamic program with a numpy
replay of every message.
