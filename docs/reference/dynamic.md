---
title: Dynamic Segments
summary: How a program author writes a dynamic segment, in which items move between cores as the data decides, through per-core queues in DRAM, and how the run ends by itself.
tags:
  - reference
  - runtime
---

# Dynamic Segments

Status: implemented for runs fed by the test feeder on one Blackhole card. The dispatch
kernel does not yet end a dynamic run (it has no status lines and no two-pass end), so a
`Session` cannot run one yet.

A static crossing is known before the run: who sends what to whom and how many arrive.
A dynamic crossing is known only from the data: a ray goes to the core that holds the
next part of the scene, a finished ray to the core that owns its film tile. In a
dynamic segment the program's state travels as **items**, fixed-size records that
move from core to core until the program retires them. The destination of each item,
the number of arrivals at a core and the end of the run are all decided by the data.

The program writes one segment body in its compute kernel. The reader and writer
kernels of a dynamic segment are generic and ship with the runtime
(`src/komorebi_native/kernels/dyn_reader.cpp` and `dyn_writer.cpp`).

## The calls

Compute kernel, inside the runtime's loop (`#include "komorebi/device/km_dyn.hpp"`):

```c++
cp.run([&](const km_record_t& r) {
    if (km::seg(r) != kSegHop) {
        return;
    }
    km::Dyn dy(cp, r);  // r: the RUN record of the dynamic segment
    if (dy.seeding()) {  // once per run and core: the first items
        ... dy.spawn(dest, item) ...
        dy.seeded();
    }
    while (km::Packet pk = dy.next()) {  // up to 32 items; empty when the run ended
        for (uint32_t i = 0; i < pk.n; i++) {
            const volatile tt_l1_ptr uint32_t* in = pk[i];  // the item, in L1
            ... the body, then exactly one of:
            dy.hop(dest, item);  // the item goes on at member dest
            dy.retire();         // the item ends here
            ... and any number of:
            if (!dy.spawn(dest, item)) { ... no quota left ... }
        }
        dy.done(pk);  // the packet's page is free
    }
});
```

| Call | What it does |
|---|---|
| `Dyn(cp, r)` | starts the dynamic part of the run on this core; all three compute threads construct it |
| `seeding()`, `seeded()` | true until `seeded()` or the first `next()`: the place for the run's first spawns |
| `next()` | the next packet of up to 32 items; empty when the run has ended, the stop word was set or a wait gave up. Every wait inside it is bounded |
| `pk.n`, `pk[i]` | the number of items, and the L1 address of item `i` |
| `hop(dest, item)` | the item joins the queue of member `dest` of the run's team (a member index, not a core index); a hop to the core itself takes the same path |
| `retire()` | the item ends; one unit of quota comes back to this core |
| `spawn(dest, item)` | a new item at member `dest`; false, and nothing is sent, when this core has no quota left |
| `done(pk)` | the packet's buffer page is free and its items are counted as consumed |
| `quota()`, `count()` | spawns left on this core; members in the team |
| `wait_stop()` | waits for the stop word (a test of the run's timeout) |

An item is `item_bytes / 4` words. Word 0 belongs to the runtime: it holds the item's
mark. The program's fields go in the other words; the hop example uses words 1 to 4.

## The host side

```python
program.dynamic("hop", item_bytes=64, max_live=1 << 16, queue_items=1 << 16, bins=16)
```

`Program.dynamic` makes one segment dynamic and adds the runtime's three buffers to the
program (circular buffers 26 to 28: incoming packets, outgoing items, open bins). The
options are those of `komorebi.program.Dynamic`:

| Option | Default | Meaning |
|---|---|---|
| `item_bytes` | 64 | size of an item: 64, 128, 192 or 256 |
| `max_live` | 65536 | the most items alive at once in the whole team |
| `queue_items` | 65536 | each core's queue in DRAM, a power of two |
| `bins` | 16 | open bins per writer, one destination each |
| `idle_cycles` | 20000 | how long a core must stay idle before its line says so |

The launcher refuses, before any device opens:

- `max_live > queue_items`: if every item were aimed at one core, its queue could not
  hold them;
- queues that do not fit the DRAM given to them (`cores * queue_items * item_bytes`);
- `max_live` smaller than the number of cores (some core would start without quota);
- a ring depth below 2 (the end of a run is a second record in flight).

A run is one `RUN` record of the dynamic segment to a team, with `KM_F_EXCL`. The record
stays in flight for the whole run; its deadline is the run's deadline. Its argument
words are the program's.

## How an item travels

Every core has a queue of `queue_items` items in DRAM. A sender never waits for the
receiver: the writer collects outgoing items in bins (one destination each, up to 32
items), reserves positions in the receiver's queue with one atomic add on the
receiver's tail word, writes the items to DRAM and counts them as sent. The receiver's
reader takes items from the head of its queue in order, as packets of up to 32. A bin is sent when
it is full, when its slot is needed for another destination, and when the core runs
out of input, so nothing waits in a bin while the core is idle.

**Marks.** Word 0 of the item at queue position `p` is `p + 1` once it is valid; 0 is
empty. Every core zero-fills its own queue at start-up, before it reports ready, and
positions never restart while the program stays loaded, so an address never holds the
mark of the position the reader is waiting for unless that item was written for it. A
core refuses a run once its positions have reached 2^30 (reason `full`); reload the
program then.

**Commit.** Only a 16-byte chunk is known to land whole, so a packet commits
explicitly: the writer sends the packet's items with the first item's mark 0, waits
for those writes, then rewrites the first item's first 16 bytes with its mark. The
reader takes the prefix of valid marks from its head; it cannot pass an uncommitted
first item, and once that item is valid the whole packet had landed.

**Reservation.** The atomic add's answer goes to a return address programmed on every
call, to a 16-byte slot of the sender's own control block (the firmware's default
return slot is shared). A sender waits for the answer to replace a sentinel, not only
for the atomic's count. A program that adds its own barrier between cores at start-up
must allow more than the launch skew: under slow dispatch a bound of 7 ms let 47 of
119 cores give up; the reservation probe allows one second.

So a full inbox cannot deadlock the grid, and while `max_live <= queue_items` holds no
queue can overflow. The cost is that every hop crosses DRAM twice.

**Quota.** Each core starts a run with `max_live / cores` spawns. `spawn` takes one,
`retire` gives one back to the core where the item ends, `hop` changes nothing. Seeding
spawns count too. Moving quota between cores is the program's business: a program that
spawns on some cores and retires on others runs those cores dry.

## How the run ends

Nobody counts the items in advance. Each core keeps two counters, items sent (hops and
spawns) and items consumed, and writes a 16-byte status line `{run, state, sent,
consumed}` to the issuer when it has been idle for `idle_cycles`: its queue empty, its
compute waiting in `next()`, every bin sent. Before it takes an item again it writes a
busy line and waits until that write has landed.

The issuer reads every line of the team twice. When both passes show every core idle
with the same counters, and the total sent equals the total consumed, no item is in a
queue or on the way: the issuer sends an end record (`KM_SEG_DYN_END`) to the team,
`next()` returns an empty packet on every core, and the `RUN` record completes through
the usual team completion.

## When something goes wrong

Every wait is bounded and watches the stop word. A core that gives up writes its first
reason, the queue position and the run into its control block (`KM_CTL_DYN +
KM_DYN_GAVE`), writes a `gave up` status line, and the issuer stops the run on every
core. The run's timeout names the core. No reset is needed: the next run starts on
the same loaded program.

| Code | Reason | Meaning |
|---|---|---|
| 19 | gap | a reserved position was not written within the wait limit; word 3 is the word 0 found there |
| 20 | overflow | more items than `queue_items` were in one core's queue |
| 21, 22, 23, 25 | atomic, flush, read, line | a reservation, the item writes, the item reads or the status line did not finish within the wait limit |
| 24 | in page | the compute kernel did not free a packet page within the wait limit (a body that stalls) |
| 26 | bad dest | `hop` or `spawn` named a member outside the team |
| 27, 28, 29 | next, out page, outbox | a wait between the compute kernel and the reader or writer ran out |
| 30 | stopped | the stop word was set |
| 31 | align | the packet buffer or the item size is not a multiple of 64 bytes |
| 32 | lost | a reservation was answered but its returned position never landed; word 3 names the destination |
| 33 | fill | the start-up zero fill of the queue did not finish |
| 34 | full | this core's queue positions reached 2^30 in this load |

The codes follow the runtime's where codes (0 to 18) and are `kDyn*` in `km_dyn.hpp`.

## Not in this version

- A direct L1-to-L1 path with DRAM as overflow, and a hop to the core itself without DRAM.
- A body that asks for a DRAM fetch decided by an item: a dynamic program keeps what its
  body reads in L1.
- Quota moving between cores; several dynamic segments with different item sizes in one
  run.
- Priorities or ordering between items: a program that needs an order sorts at the
  receiver.

The [hop example](../../examples/05_hop/) is a complete dynamic program with a numpy
replay of every item.
