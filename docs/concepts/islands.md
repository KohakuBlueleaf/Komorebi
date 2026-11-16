---
title: Islands
summary: What an island is, why ranks stay flat across islands, the island table a core holds, what a kernel can ask about it, a card split into islands, a line of islands and what is forwarded through those between, the links between islands and what they carry, messages across a link and the end of a run over islands, acting by island, and what is not built.
tags:
  - concepts
  - islands
  - links
---

# Islands

Status: implemented over two Blackhole p150a cards, each card's chip an island, and
over one card split into two to four islands. A group that spans islands runs as one
part on each, every core sees its flat rank in the whole group, and a program written
for one card (one `Program`, as example 01's) runs on two as it does on one. Commands,
pages (asks, puts, copies), messages and the parts of a reduce or a broadcast cross
between islands over the links, sent from kernels, from a dispatcher and from an X280
thread's commands and copies; what is not built is under
[What is not built](#what-is-not-built). One card is one island is the way to run, and
the recommendation. A card split into islands is for developing and checking island
programs. X280 threads on a split card are not supported, and two cards as a line of
four islands is not in this release. The calls are in [islands](../reference/islands.md),
the kernel side in the [runtime reference](../reference/runtime.md#islands), recovery
in [how reset works](reset.md).

An **island** is a set of cores that reach the others only through links. It is one
card's chip: its own Tensix grid, its own dispatcher on its own X280 cluster 0, its own
DRAM banks; or rows of one card split into islands (below). Cores inside an island
reach each other by the NoC, as on a card opened alone. A core of one island reaches a core of another only over a link, which is slower
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
`islands.place` for a tensor's pages by its hint, and `ENTRY` for the layout. On one card the table
has one entry, `(0, cores, NO_LINK, 0)`, and every answer below is the one of a run
without islands.

## A card split into islands

An island is a set of ranks with a table entry, not a card. So one card can be several
islands: `km.Session(card=0, islands=2)` gives the session of two islands over one card,
the same object and calls as over two cards. One card as one island is the way to run;
a split card is the way to develop and check a program written for islands with the
cards at hand, and to exercise more than two islands (it is not faster: its islands
share one card). Each island's ranks run on a contiguous range of the card's cores from
its base (core = base + rank - the island's first rank), and island i's dispatcher runs
on X280 cluster i of that card. The limits: at most four islands (one a cluster), each
island past the first takes a work cluster away (X280 threads on a split card are not
supported, and the session refuses them), a split card keeps no reset lists (a core
that does not leave a given-up command ends the run), and each island is whole rows of
the grid, since a dispatcher sends commands over its own rows.

Every pair of islands of a split card is joined by a **wire**: link cores taken from
each island's last cores, serving what a link core serves between two cards (commands
across, a core's own page ask, put and L1 write, messages). Only the transport differs:
where a link core between cards hands a packet to an Ethernet router, on one card it
writes the far tile over the NoC (`km_transport.hpp`, chosen by the build), each count
or word after the writes before it, as a router keeps them. The island table's next
link names the link core toward each island. A split card's tensors are held by
island, as over two cards: a page another island holds is asked for across
(`card_dram=True` keeps one copy in the card's DRAM, every page local, for kernels that
read a tensor by its bank address).

The islands' links are a graph given as island pairs, a pair k times k links: cards
take the one the fabric reports (three or more cards in a row are a line), and a split
card takes `km.Session(card=0, islands=n, graph=islands.line_pairs(n))` or any other
connected pairs. Where only neighbouring islands are joined, island 0 reaches island 2
through island 1. A link core that receives what is not for
its island sends it on by its island's next link toward the target, to the island's
link core toward it, which sends it across; no other core of the island between does
anything. What is forwarded: a command's line and its DONE, an ask and its pages, an
L1 write (a reduce's fold, a broadcast's blocks), and a message (queued again in the
outbox of the link core toward its rank). A put goes one link only. Each island
between adds 2.3 to 3.9 µs to a command and its DONE. The calls and the numbers are in
[islands](../reference/islands.md#a-card-split-into-islands).

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
| `group.local()`, `group.leaders()` | the group's part on this island; the first rank of each island part | the group; its first rank | the group's part on this card; ranks 0 and 119 for a group over both (one link core an island) |
| `km::partition(items, part, parts)` | part `part` of `parts` of `items` | `partition(items, 0, 1)`: all items | the same call; a rank partitions by its flat rank |

`km::local_part(view, command)` is the part of a command's group on this island in local
cores, which a completion uses. On an X280 thread, `kmx_island()` and `kmx_islands()`
answer the same as `km::island()` and `km::islands()`; a thread runs on one island's X280
and reaches the others through commands and copies across.

The answers come from the table and from nothing else. `km::rate` is 0 for another island
because the table holds no rate, not because the link has none (below).

## Links and the cores they take

Two p150a cards are joined by four Ethernet links. A core that sends over a link holds a
connection to the router that tt-metal's fabric starts on that link's Ethernet core. The
sending core's place makes no difference to the rate: a corner, a middle row and the row
under the Ethernet tiles sent the same
([link_rate](../../experiments/islands/link_rate/README.md#6-where-the-sender-sits)). A router has
one channel for a core of its own card, so one core at a time holds a link's connection;
sharing a link between cores at once needs tt-metal's mux cores.

The host's graph models this: a link that is used takes a core on each island it joins
out of that island's ranks (a link core), and ranks and thread numbers are counted
without it. A session over two cards takes each island's last `(links + 1) // 2` cores
(one for one or two links), both data-movement CPUs of a link core holding a link each,
from the load to the run's end; a command names them as link cores, first
`KM_GROUP_LINK_BASE` (0xFE00) + k, never as ranks. Link 0, on link core 0's first CPU,
carries the dispatcher's command lines and the cores' asks; the others carry pages
([islands](../reference/islands.md#pages-across-a-link)).

A kernel that reads a tensor's pages calls `km::read_pages` with the tensor's
reference word, which the script passes as on one card (`tensor.address`) and which
names the tensor's entry in a table every core holds: the session placed the tensor by
the hint the script gave, in the kernel's own words (split along a dimension over a
group of ranks, like another tensor, or a copy), and the entry says which island holds
which part. The pages its island holds are read from its own DRAM, as on one card; for
pages another island holds the core puts one ask line into its island's link core 0,
which sends it on link 0 between commands, and the holder's link core pushes the pages
into the asking core's L1 and counts them at its port, where the core waits. No host
takes part. On one island the call is the local read, so a kernel written for one
island runs unchanged on two.

Writing is the same the other way: `km::write_pages` writes the pages its island holds
as the local page write does, and for pages another island holds puts one line into
its island's link core 0, whose page CPU reads them from the core's L1 and sends them
straight into the holder's DRAM; the far link core counts them once written and only
then adds them to the core's port. A thread's `kmx_fetch` and `kmx_store` name a
tensor and a range, never an island: the copy they submit looks the tensor up in a
table the load wrote and asks for, or puts, the pages another island holds.

What a link carries, measured. Two Blackhole p150a cards joined by four Ethernet links,
one process holding both with tt-metal's 1-D fabric in slow dispatch, AICLK 1350 MHz.
The rate is payload bytes over the receiver's span from its first counted packet to its
last; every packet was checked exact; each row ran once. The senders are experiment
kernels, one sending core a link ([link_rate](../../experiments/islands/link_rate/README.md),
[two_cards](../../experiments/islands/two_cards/README.md)), not a call of a program: a program
sends pages with the page calls, whose rates are in
[islands](../reference/islands.md#pages-across-a-link).

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
card. One sending core sent 2.8 million 64-byte packets a second. In [islands_mesh](../../experiments/islands/islands_mesh/README.md) a handler on one island
held a fabric connection across commands and wrote 64 bytes into the L1 of a core of the
other, exact in both directions, 1.685 µs a command from island 0 to island 1 and 1.585
the other way (1,000 pipelined to one core).

## Messages across a link

A dynamic run forwards to a rank of another island as to any rank: the kernels do not
change. By default a message to another island's rank is a delivery by the link
service: the writer hands its bin to its link core, which pushes it to the far
island's link core, which writes it into the rank's queue; any group runs, and pages
cross the same links meanwhile ([islands](../reference/islands.md#the-run)). A program
declared with `hold_links=True`, between two cards, takes the faster old path below
instead: its RUN is every rank and the links are its own while it runs. The
writer flush puts a message for another island into the outbox of its link; beside the
RUN the session starts the message service on the CPU that holds link 0, over the
connection it opened at load. With `message_links=True` (opt-in: one run of three of example 05's 10^6 forwards over four links stalled at 81 % of its visits, cause not found yet) every
link has a service, and a rank's messages take link `rank % links`, so one rank's stay
in order. A service sends its outbox across, many messages to a packet (at most
4352 bytes) and one count a packet, and issues what landed from the far island to its
ranks. A message's body crosses in its message's packet, right after it (at most 4,224
bytes for 64-byte messages), and is written into the rank's body page before the
message: example 23 takes 3.001 µs a hop with the message alone, 6.109 µs with a 4 KiB
body (one service; four: 2.990 and 6.056). Every wait is bounded and the link has credits, so a full far
inbox holds the sender. The run ends when both islands are quiet: link 0's service decides for its
island over the run's cores and the other services' state lines, sends its island's
quiet report across, and the dispatcher's quiet test counts link core 0's line (sent:
messages issued again here; consumed: messages sent across, every link's). Link 0's
service calls its island quiet once two passes alike find every core idle and every
other service empty in this run, with every service's counts balanced; both islands
quiet with the sums sent across one way equal to those issued the other, nothing is in
flight on any link.

[05_forward](../../examples/05_forward/README.md) runs this on two Blackhole p150a cards,
one link: 16 messages of 4 forwards over 234 message ranks of 238, exact against its
replay, 27 of the 64 forwards across. 10^6 forwards (`hold_links=True`): 302.0 ms
with one service (one link; by deliveries 1,531.5 ms), 80.1 to 80.4 ms with four (`message_links=True`, every link about 62,500
messages each way); one card: 11.77 ms. Measured before the service moved onto the link
cores (a link core of its own as a rank, one link,
[link_service](../../experiments/islands/link_service/README.md)): 6.5 µs a crossing for a single
message, 4.7 µs in a run of 1000 round trips; about 4.7 million 64-byte messages a second
each way when messages wait to be batched, 0.6 million a packet a message.

While a run holding the links is served, its links' CPUs serve nothing else: a
dispatcher's command line or a page waits until the run ends.

Over more than two islands the end of a run is decided once for the whole run, not a
pair at a time. Every link core serves its link. Each island's link core 0 checks its
island quiet (the run's cores idle, every link core empty, the counts balanced) and
reports {quiet, received across, sent across, round} toward island 0, which holds the
group's first rank, forwarded like any line. When two successive rounds show every
island quiet and the sum sent across equal to the sum received across, with the same
counts, island 0 sends the verdict to every link core, and each island's link core 0
shows its line idle; each dispatcher then ends its part. Every wait is bounded. Example
05 on card 0 as three and four islands in a line ran exact against its replay, each
link's messages equal to the replayed crossings (56 and 75); island 0's part took 166.1
µs and 216.9 µs, against 146.6 µs over two islands.

## Acting by island

A program written for one island runs on several unchanged: a reduce whose group spans
islands folds inside each island first, as a group of its own, and each island's first
rank then sends its fold once to the group's rank 0. Such a reduce runs once a command:
at its second turn it is refused by name at both ends (`bad_item`), since the credit a
repeated reduce waits for ([ports](../reference/ports.md#reduce)) would have to cross
the link; `km::local` keeps a repeated reduce inside each island. A broadcast whose group spans
islands is the inverse: the sender casts in its own island's part and sends each block
once to the first core of the next island's part, which casts it in that part; that
core's kernel calls `arrive` as any receiver does. A command over islands completes with
the result word of the part that holds the group's first rank, as on one island. A
program written for islands says so in its own words, and each word has an answer on one
island:

| Call | What it does | On one island |
|---|---|---|
| `km::reduce_*(..., km::local)` | the reduce stays inside this core's island | the whole group |
| `km::Broadcast(..., km::local)` | the broadcast stays inside this core's island: each part's first core sends | the same broadcast |
| `group.leaders()`, `group.leader(j)` | the first rank of each island part of the group | the group's first rank |
| `km::to_island(i).put(writer, command, arrival, cb, pages, rank, port, slot)` | pages into an arrival slot of `rank` on island `i`, one on its port | a local emit |

A part for another island goes by `km::write_l1`: a line in the queue of the sender's
link core toward the rank's island, its pages sent by the page CPUs, any number of links
away; the far link core's CPUs write them into the rank's arrival slot and add 1 to its
port, and the sender's call returns once they landed. The receiving core waits on its port as it
does for a sender of its own island ([runtime](../reference/runtime.md#islands)).

Measured on two Blackhole p150a cards joined by one link, 512 integer tiles summed, the
median of 20 commands from submit to the event on the host, every total exact:

| Program | Two cards (238 ranks) | One island |
|---|---|---|
| [02_merge_tree](../../examples/02_merge_tree/README.md), kernels unchanged, by tree | 172.6 µs | 162.8 µs (island 0's 119 ranks) |
| [26_islands_sum](../../examples/26_islands_sum/README.md), written for islands | 165.9 µs | 158.0 µs (card 0 alone, 120 ranks) |

The rows of example 02 asked for a result word (one more pass over the total at the
root); those of example 26 did not.

[03_broadcast](../../examples/03_broadcast/README.md#over-two-cards), kernels unchanged,
one flat group of 238 ranks, a 6 x 4 rectangle, 8 blocks of 2 tiles a broadcast, the
median of 8 commands, every core's checksum exact: 705.7 µs with its 6 column broadcasts
across the islands (the rows inside), 535.5 µs with its 4 row broadcasts across (the
columns inside), against 186.1 µs on card 0 alone. Each block that crosses costs two
L1 writes (the block, then its VALID line) and a marker back, each landed before the
call returns.

Two calls gather and reduce a whole tensor across the islands with nothing about
links in the caller: `km::all_gather` and `km::all_reduce`, from a kernel or from the
host ([reference](../reference/islands.md#collectives-across-islands)).

## When a core stops

Each island's dispatcher holds its own island's reset lists. A command given up on
island i resets and loads again that island's cores still in it, or whose kernels
ended at a wait's own limit; the time-out says `island i:` and names them by flat
rank and the island's own core number, and the run goes on on both islands
([recovery](../guides/recovery.md); how each step is done and what it costs,
[how reset works](reset.md)). A link core is recovered by the host, not by
the dispatcher: a link core that does not leave a given-up command is named in the
time-out (`link core k`), the commands whose traffic it held are given up at once
naming it, and `run.links_hung` lists it. `run.reset_link(island, k)` then closes its
connections at their routers, drops link core 0's asks and puts not yet sent, loads
the core again and opens them again; reads and commands across are exact after it.
The far link core needs nothing.

## What is not built

- **Over a link:** a dynamic run on part of the ranks (its RUN takes every rank), X280
  ranks over a link. Between two cards a kernel's ask and put use every page CPU of
  every link; over more than two islands an ask is forwarded (queued on by the lead page
  CPU of each island between), and a put goes one link only (a put to an island two links away is refused).
  Over more than two islands the messages take one link a neighbour.
- **A rate in the table**, so `km::rate` of another island is 0.
- **More than two islands of a split card:** one wire a pair and at most three link
  cores an island for messages. Not in this release, by decision: X280
  threads on a split card (a split card is for developing and checking island programs),
  and two cards each split into islands, which would make a line of four (fabric links
  and wires in one graph); forwarding has run over wires only.
- **Uploads over islands, tensors in L1.** An upload (a stream or a graph) runs on the
  island it is given to; its commands name any flat group. A `Program`, a list or a dict
  of groups in flat ranks, and a list of programs one an island, load as on one card,
  and `run.load` of a program some group of each island it touches holds works over
  flat ranks ([islands](../reference/islands.md#the-run)); an island's X280 ports
  serve its own ranks, and a thread of one island fetches and stores tensors the other
  holds.
- **More than two cards.** The mesh opens as 1 x n, so the fabric reports a line; two
  were tried.
- **A link core reset by its dispatcher.** The dispatcher resets only ranks; a link core
  that hangs is named by its dispatcher and reset by the host (`run.reset_link`), as
  [how reset works](reset.md#4-reset-a-link-core-from-the-host) says. Its lost traffic
  is given up and named, not sent again.
