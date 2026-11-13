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

The program writes one handler body in its compute kernel. The reader and writer
kernels of a dynamic handler are generic and ship with the runtime
(`src/komorebi_native/kernels/messages_reader.cpp` and `messages_writer.cpp`).

## The calls

Compute kernel, inside the runtime's loop (`#include "komorebi/device/km_messages.hpp"`):

```c++
cp.run([&](const km_command_t& r) {
    if (km::handler(r) != kHandlerForward) {
        return;
    }
    km::Messages dy(cp, r);  // r: the RUN command of the dynamic handler
    if (dy.seeding()) {  // once per run and core: the first messages
        ... dy.send(dest, message) ...
        dy.seeded();
    }
    while (km::Packet pk = dy.next()) {  // up to 32 messages; empty when the run ended
        for (uint32_t i = 0; i < pk.count; i++) {
            const volatile tt_l1_ptr uint32_t* in = pk[i];  // the message, in L1
            ... the body, then exactly one of:
            dy.forward(dest, message);  // the message goes on at rank dest
            dy.consume();         // the message ends here
            ... and any number of:
            if (!dy.send(dest, message)) { ... no quota left ... }
        }
        dy.done(pk);  // the packet's page is free
    }
});
```

| Call | What it does |
|---|---|
| `Messages(cp, r)` | starts the dynamic part of the run on this core; all three compute threads construct it |
| `seeding()`, `seeded()` | true until `seeded()` or the first `next()`: the place for the run's first sends |
| `next()` | the next packet of up to 32 messages; empty when the run has ended, the stop flag was set or a wait gave up. Every wait inside it is timed |
| `pk.count`, `pk[i]` | the number of messages, and the L1 address of message `i` |
| `forward(dest, message)` | the message joins the queue of rank `dest` of the run's group (a rank, not a core index); a forward to the core itself takes the same path |
| `consume()` | the message ends; one unit of quota returns to this core |
| `send(dest, message)` | a new message at rank `dest`; false, and nothing is sent, when this core has no quota left |
| `done(pk)` | the packet's buffer page is free and its messages are counted as consumed |
| `quota()`, `count()` | sends left on this core; cores in the group (without its counter tiles) |
| `rank()`, `counter()` | this core's rank; true on a counter tile, which takes no message |
| `wait_stop()` | waits for the stop flag (a test of the run's timeout) |

A message is `message_bytes / 4` words. Word 0 belongs to the runtime: it holds the message's
mark. The program's fields go in the other words; the forward example uses words 1 to 4.

## The host side

```python
program.dynamic(
    "forward", message_bytes=64, max_live=1 << 16, queue_messages=1 << 16, bins=16
)
```

`Program.dynamic` makes one handler dynamic and adds the runtime's three buffers to the
program (circular buffers 26 to 28: incoming packets, outgoing messages, open bins). The
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

A run is one `RUN` command of the dynamic handler to a group, with `KM_FLAG_EXCLUSIVE`. The command
stays in flight for the whole run; its deadline is the run's deadline. Its argument
words are the program's.

## How a message travels

Every core has a queue of `queue_messages` messages in DRAM. A sender never waits for the
receiver: the writer collects outgoing messages in bins (one destination each, up to 32
messages), reserves positions in the receiver's queue with one atomic add on the
receiver's tail word on its counter tile, writes the messages to DRAM and counts them as sent. The receiver's
reader takes messages from the head of its queue in order, as packets of up to 32. A bin is sent when
it is full, when its slot is needed for another destination, and when the core runs
out of input, so nothing waits in a bin while the core is idle.

**Marks.** Word 0 of the message at queue position `p` is `p + 1` once it is valid; 0 is
empty. Every core zero-fills its own queue at start-up, before it reports ready, and
positions never restart while the program stays loaded, so an address never holds the
mark of the position the reader is waiting for unless that message was written for it. A
core rejects a run once its positions have reached 2^30 (reason `full`); reload the
program then.

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

**Cost.** K cores of the group take no message. Every core's control block grows from
0xD40 to 0x1580 bytes for the tail lines: 2,112 bytes of L1 on every core for every
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
first has ended; from the stream engine or a work cluster it is rejected busy (reason
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

The codes follow the runtime's where codes (0 to 18) and are `kWhereMessages*` in `km_core.hpp`.

## Not in this version

- A direct L1-to-L1 path with DRAM as overflow, and a forward to the core itself without DRAM.
- A body that asks for a DRAM fetch decided by a message: a dynamic program keeps what its
  body reads in L1.
- Quota moving between cores; several dynamic handlers with different message sizes in one
  run.
- Priorities or ordering between messages: a program that needs an order sorts at the
  receiver.

The [forward example](../../examples/05_forward/) is a complete dynamic program with a numpy
replay of every message.
