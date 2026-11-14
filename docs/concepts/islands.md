---
title: Islands
summary: What an island is, why ranks stay flat across islands, the island table a core holds, what a kernel can ask about it, the links between islands and what they carry, messages across a link, and what is not built.
tags:
  - concepts
  - islands
  - links
---

# Islands

Status: implemented over two Blackhole p150a cards, each card's chip an island. A group
that spans both cards runs as one part on each, every core sees its flat rank in the
whole group, and a program written for one card (one `Program`, as example 01's) runs on two as
it does on one. Nothing
crosses between the cards inside a run yet: no message, page or command goes over a
link from a kernel. The calls are in [islands](../reference/islands.md), the kernel
side in the [runtime reference](../reference/runtime.md#islands).

An **island** is a set of cores that reach the others only through links. Today it is one
card's chip: its own Tensix grid, its own dispatcher on its own X280 cluster 0, its own
DRAM banks. Cores inside an island reach each other by the NoC, as on a card opened
alone. A core of one island reaches a core of another only over a link, which is slower
than the NoC and which a program has to be able to see.

## Why the level exists

A program is written for ranks: a command names a group `first | count << 16` of them,
a core finds its place in the group with `me() - km::first(command)`, and a message
names its destination by rank. Komorebi keeps that. Ranks are **flat and contiguous by
island**, in island order: with two cards of 120 cores, ranks 0 to 119 are card 0's cores
and ranks 120 to 239 are card 1's. A core's rank is its island's first rank plus its
local core (its index in the core table). `me()` is that flat rank, so the kernels of
example 01 run on both cards without a change:

```cpp
const uint32_t index = reader.me() - km::first(command);
...
km::partition(tiles, index, km::count(command))
```

What changes is what a core must hold in order to know where a rank is. It does not
hold a table of ranks. It holds the **island table**, one entry for each island, and
looks a rank up in it. The core table of a core, the cores it can address by the NoC,
is its island's. So what a core holds about the others grows with the number of islands,
not with the number of ranks.

The same split is what the host and the dispatcher see. A dispatcher serves its own
island's cores, so the host cuts a group by island (`Graph.split`) and sends each
island's dispatcher the group's word unchanged; each dispatcher keeps the part that
lies on its island, runs it, and completes it through the first of its cores in the
group. The host waits for every part and joins them into one ticket.

## The island table

The host writes the table into every core's control block at load. An entry is 8 bytes,
little-endian: first rank (16 bits), ranks (16), next link (8, `KM_ISLAND_NO_LINK`, 0xFF,
in the holder's own entry), hops (8), zero (16). The table holds up to
`KM_ISLANDS_MAX` (16) entries. On two cards, the table of a core of card 0:

| Island | First rank | Ranks | Next link | Hops |
|---|---|---|---|---|
| 0 (this one) | 0 | 120 | 0xFF (none) | 0 |
| 1 | 120 | 120 | 0 | 1 |

A core of card 1 holds the same table with `0xFF` and 0 in island 1's entry, its own, and
next link 0 and hops 1 in island 0's. The host's values are in `komorebi.islands`: the graph of islands and links
(`Graph`, `line`, `ring`), the rank of every core, `Graph.split` for a group,
`islands.place` for a tensor's pages, and `ENTRY` for the layout. On one card the table
has one entry, `(0, cores, NO_LINK, 0)`, and every answer below is the one of a run
without islands.

## What a kernel sees

A kernel asks about islands with these calls (`km_island.hpp`, `km_group.hpp`). They only
ask: no call sends anything.

| Call | Answer | On one island | On two cards (core of card 0 / card 1) |
|---|---|---|---|
| `km::island()`, `km::islands()` | this core's island, the islands of the run | 0, 1 | 0 / 1, and 2 |
| `km::island_of(rank)`, `km::crosses(rank)` | the island that holds `rank`; whether a send to it crosses a link | 0, false | the card of `rank`; true for a rank of the other card |
| `km::resolve(rank)` | its island, then the local core (this island) or this island's next link toward it | `{0, true, rank}` | the same, with the next link for a rank of the other card |
| `km::hops(i)`, `km::neighbours()` | links crossed to island `i`; the islands one link away, a bit each | 0, 0 | 0 and 1 for the two islands; the other island's bit |
| `km::rate(i)` | bytes a second toward `i`: `km::kRateLocal` for this island, 0 (not known) for another | `kRateLocal` | `kRateLocal` for its own, 0 for the other |
| `group.local()`, `group.leaders()` | the group's part on this island; the first rank of each island part | the group; its first rank | the group's part on this card; ranks 0 and 120 for a group over both |
| `km::partition(items, part, parts)` | part `part` of `parts` of `items` | `partition(items, 0, 1)`: all items | the same call; a rank partitions by its flat rank |

`km::local_part(view, command)` is the part of a command's group on this island in local
cores, which a completion uses. On an X280 thread, `kmx_island()` and `kmx_islands()`
answer the same as `km::island()` and `km::islands()`; no thread runs on two cards yet.

The answers come from the table and from nothing else. `km::rate` is 0 for another island
because the table holds no rate, not because the link has none (below).

## Links and the cores they take

Two p150a cards are joined by four Ethernet links. A core that sends over a link holds a
connection to the router that tt-metal's fabric starts on that link's Ethernet core. The
sending core sits in the Tensix row next to the Ethernet row, under the
link's Ethernet tile: on these cards, cores 8 to 11 of row 0 under the Ethernet tiles at
x 24 to 27, one for each of the four links. A router has one channel for a core of its own
card, so one core at a time holds a link's connection; sharing a link between cores at
once needs tt-metal's mux cores.

The host's graph models this: a link that is used takes a core on each island it joins
out of that island's ranks (a link core), and ranks and thread numbers are counted
without it. A session over two cards takes each island's last `(links + 1) // 2` cores
(one with `links=0`), both data-movement CPUs of a link core holding a link each, from
the load to the run's end; a command names them as link cores, first
`KM_GROUP_LINK_BASE` (0xFE00) + k, never as ranks. Link 0, on link core 0's first CPU,
carries the dispatcher's command lines; the others carry pages
([islands](../reference/islands.md#pages-across-a-link)).

What a link carries, measured. Two Blackhole p150a cards joined by four Ethernet links,
one process holding both with tt-metal's 1-D fabric in slow dispatch, AICLK 1350 MHz.
The rate is payload bytes over the receiver's span from its first counted packet to its
last; every packet was checked exact; each row ran once. The senders are experiment
kernels, one sending core a link ([link_rate](../../experiments/link_rate/README.md),
[two_cards](../../experiments/two_cards/README.md)), not a call a program can make yet.

| What | GB/s |
|---|---|
| One link, plain writes of 4352-byte payloads, one count for a batch of 64 | 26.76 to 28.87, by version of the sender's loop (three were run) |
| Four links, one sending core a link, plain writes, one way | 107.0 (111.1 with the third version of the loop) |
| The same, both ways at once | the same each way: 107.0 and 107.0 |
| Four links from two cores, two links a core (one on each data-movement CPU) | 115.0 |
| One link, a fused write and increment with every packet, 4352-byte payloads | 10.56 |
| One link, 4 KiB packets, each a fused write and increment, a wait for the writes to leave every fourth packet | 10.03 a link, 40.1 over four links, one way and each way at once |
| One link, plain writes, a packet for a DRAM bank or a core's L1 with no kernel running on the receiving card | 26.77, and every slot exact |

By packet size, plain writes, one link, one sending core: a packet costs the same time at
1024, 2048 and 4352 bytes (about 150 ns), so the rate is a count of packets (6.6 million
a second), not of bytes. The sender never found the router's 16 slots full: the rate here is
what one core can issue, not what the link carries. Where the link itself stops, between
28.9 GB/s and the 50 GB/s of the line, was not reached.

A packet's header is 48 bytes (routes of up to 16 hops; 64 bytes for longer ones) and its
payload up to 4352 bytes, so a 64-byte message is one packet. The card address in the
header is a route (which link, how many hops) beside the core's NoC address on its own
card. One sending core sent 2.8 million 64-byte packets a second. In [islands_mesh](../../experiments/islands_mesh/README.md) a handler on one island
held a fabric connection across commands and wrote 64 bytes into the L1 of a core of the
other, exact in both directions, 1.685 µs a command from island 0 to island 1 and 1.585
the other way (1,000 pipelined to one core).

## Messages across a link

A dynamic run whose group is every rank of two islands (`Session(cards=..., links=1)`)
forwards to a rank of the other island as to any rank: the kernels do not change. The
writer flush puts a message for another island into link core 0's outbox; beside the
RUN the session starts the message service on link core 0, on the CPU that holds link 0
over the connection it opened at load. The service sends the outbox across, many
messages to a packet (at most 4352 bytes) and one count a packet, and issues what landed
from the far island to its ranks. Every wait is bounded and the link has credits, so a
full far inbox holds the sender. The run ends when both islands are quiet: each service
sends its island's quiet report across, and the dispatcher's quiet test counts link core
0's line (sent: messages issued again here; consumed: messages sent across).

[05_forward](../../examples/05_forward/README.md) runs this on two Blackhole p150a cards,
one link: 16 messages of 4 forwards over 234 message ranks of 238, exact against its
replay, 27 of the 64 forwards across. Measured before the service moved onto the link
cores (a link core of its own as a rank, one link,
[link_service](../../experiments/link_service/README.md)): 6.5 µs a crossing for a single
message, 4.7 µs in a run of 1000 round trips; about 4.7 million 64-byte messages a second
each way when messages wait to be batched, 0.6 million a packet a message.

While a run's messages are served, link 0's CPU serves nothing else: with `links=1` a
dispatcher's command line or a page on link 0 waits until the run ends.

## What is not built

- **Collectives over a link.** `collect` and `broadcast` over islands, a dynamic run on
  part of the ranks of two islands, X280 ranks over a link, and a kernel's own ask for a
  page another island holds (the host asks for it today).
- **A rate in the table**, so `km::rate` of another island is 0.
- **A card split into islands.** The host's graph describes it (`islands.split_card`);
  nothing runs on it.
- **A dict of programs, X280 threads and ports, engines and uploads over islands,
  tensors in L1.** A list of programs, one an island, loads.
- **More than two cards.** The mesh opens as 1 x n; two were tried.
