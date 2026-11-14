---
title: Islands
summary: A session over several cards, each card's chip an island with its own grid and dispatcher, or one card split into islands; ranks flat and contiguous by island, commands split by island on the host, tensors placed by island. What it does today and what it does not yet.
tags:
  - reference
  - host
---

# Islands

Status: implemented over two Blackhole p150a cards, and over one card split into two
to four islands. A group that spans islands runs as one part on each island, and every
core sees its flat rank in the whole group, so an existing program runs unchanged.
Commands cross between the cards on link 0 and pages on the session's links, through
each island's link cores; nothing crosses between the islands of a split card.

An island is a set of cores that reach the others only through links: here one card's
chip, with its own Tensix grid and its own dispatcher on its own X280 cluster 0. The
host values (the graph of islands, ranks, the split of a group, placement) are in
`komorebi.islands`.

```python
import komorebi as km

with km.Session(cards=[0, 1]) as session:  # one session, 238 ranks
    source = session.tensor(array, "bfloat16", "tile")  # split by island
    run = session.load(program)  # on every island
    ticket = run.submit("map", group=(0, 238), args=[...])
    parts = run.wait(ticket, timeout=5.0)  # one event per island
    run.stop()
    (result,) = session.read_bf16(output)  # assembled from every island
```

## The session

`km.Session(cards=[a, b, ...], links=0, pll_mhz=None, tree=None, log=None,
grid_timeout=10.0, kernel_bytes=None, channel=256 << 20, slow_code=False)` returns an
`IslandSession`.
With one card it has the same calls over one island: no mesh and no fabric, the card's
own `Session` as `session.island(0)`, the graph `islands.one`, every page of every
tensor on the card, and each load the card's run as it is without islands. A program
written for islands runs on one card unchanged.

- **Open.** In this order: tt-metal's 1-D fabric is set, the cards open as one 1 x n
  mesh (each card's chip a 1 x 1 submesh with its own command queue), the X280 runtime
  opens once over the mesh, then each island gets a `Session` given its device
  (`session.island(i)`). The cards must be the visible ones (`TT_VISIBLE_DEVICES` is
  set to them) and must not be held by another process.
- **Ranks.** Flat and contiguous by island, in island order: island i holds ranks
  `session.graph.first(i)` to `first(i) + count(i) - 1`, its cores in core order.
  `session.ranks` is the total (238 on two cards with one link core an island, 236
  with `links=4`), at most 65,024: a group's first
  below `KM_GROUP_LINK_BASE` (0xFE00), where link cores' names start (0xFF00 on: X280
  threads).
- **Close.** Every island's run is stopped through the X280 runtime, the islands'
  sessions are closed, then the mesh, and the fabric is set off. A grid that did not
  return keeps the mesh open, as a card's session keeps its device.
- `links` 1 to 4 (two islands; 0, the default, is 1): the links the session uses.
  Each island's last `(links + 1) // 2` cores are its link cores
  (`session.link_cores`, `graph.link_cores`), not ranks; both data-movement CPUs of a
  link core send, each on its own link (`session.channels`: the Ethernet channel of
  each). Link 0 is the lines link: link core 0's CPU 0 holds it and sends the
  dispatcher's command and DONE lines (commands across), and pages only when it is
  the one link; links 1 on carry pages. The session checks that a DRAM bank sits at
  the same NoC place on both islands, and that both islands' core tables are the
  same. One card is an island of one: no link cores.
- A link core is named outside the ranks: a command's group with first
  `KM_GROUP_LINK_BASE + k` and count n names link cores k to k + n - 1 of the island
  that takes it (`graph.link_group(island, k, n)`). Such a group is never cut by
  island and never crosses; on a core, its rank is its link-core index less k. A link
  core is in no group of ranks. A timed-out command to link cores names them (its
  island's own core numbers), as for ranks.

## The run

`session.load(program, **options)` loads one `Program` on every island in turn (its
grid, then its dispatcher), or a list of Programs, one an island in island order, with
`Session.load`'s options and `island=(graph, i)`: each core gets the island table and
each dispatcher its island's range of ranks. It returns an `IslandRun`.

- `run.submit(handler, group, args)` takes a group of flat ranks. The host finds the
  islands it touches with `Graph.split` and sends each of them the group's word
  unchanged; each island's dispatcher cuts its own part and completes it through the
  first of its cores in the group, and a core's rank is its flat rank less the group's
  first, its count the whole group's. The call returns a ticket. `run.submit_many` does the same for n commands, one `submit_many` an
  island.
- `run.wait(tickets, timeout)` sends every island's queued commands, then feeds the
  islands in turn (one native round each: store what its ring has room for, take its
  events) until every part arrived, so the islands run their parts at once; all within
  the one timeout.
  A ticket completes when every part has; its result is the tuple of its parts' events
  in island order. `run.drain`, `run.stop` and `run.abort` act on every island within
  one timeout; `run.state()` lists every rank in flat order, `run.result()` every
  island's result block.
- An error from an island names it: the message starts `island i:` and the exception
  carries `error.island`; the cores a timeout lists (where each waited, those reset
  and loaded again) are that island's own core numbers. A `WaitTimeout`'s `pending`
  lists tickets.
- `run.inspect(island, cores=None)` asks island `island`'s dispatcher to read its cores
  `(first, count)` in its own core numbers (default all), as `Run.inspect` does.
- `run.send()` hands every island's queued commands to its dispatcher without waiting.
- `run.island(i)` is island i's own `Run`.
- A dynamic program (`Program.dynamic`) on two islands with `links` set: each core
  gets the message service's arguments and `KM_LINKS`, and its messages cross. Its RUN
  takes every rank (`(0, session.ranks)`; another group raises `ValueError`; so does
  `submit_many` of a dynamic handler and X280 ranks). Before the RUN the session queues
  a PAGES command of mode `KM_PAGES_MESSAGES` (a0: the RUN's group word) to link core 0
  of each island; its link 0 CPU serves the run's messages until the RUN's end, and the
  dispatcher's quiet test counts that core's line. The service grows the kernels: the
  forward example opens its session with `kernel_bytes=84 * 1024`.

## Tensors

`session.tensor(data, dtype, layout, like=None, copy=False, by="rank")` makes one
tensor of the whole shape on every island, at the same address, so a program takes its
`address` and `accessor_args()` unchanged. Its pages are placed by `islands.place`:
split by island along the leading dimension in rank order (`like`: the group the split
follows; default every rank), so a rank that partitions the tensor by rank finds its
part in its own island's DRAM; `by="island"` splits the leading dimension into even
parts by island (`partition(units, i, islands)` to island i), for a program that cuts
its work by island first and over an island's ranks inside; `copy=True` puts every
page on every island. A split
tensor's pages that another island holds are zeros on this one.

A kernel asks where a page lives with `km::island_of(placement, page)`
(`km_placement.hpp`): the session writes a tensor's placement as compile arguments
after its accessor's (`tensor.placement_args()`: the islands, copy, then each
island's first page and pages), which `km::Placement<offset>()` reads; a copy's page
answers the core's own island (`km::island()`, from the island table). On both islands' cores, for four tensors
(all on island 0, all on island 1, split, a copy): 40 answers, all as
`islands.island_of` gives them (`experiments/islands_pages/pages.py --where`).

`session.read` and `session.read_bf16` assemble a tensor from the island that holds
each page, through that island's host channel; `session.write` writes each page into
the island that holds it (a copy: into every island).

## Pages across a link

On two islands the run's kernels are built with `KM_PAGE_LINKS` and the link cores
serve the runtime's `PAGES` command (`KM_HANDLER_PAGES`, 248, `km_pages.hpp`, its
services in one place, `pages_serve`): each page CPU reads its share of the pages from
its island's DRAM and sends a page a packet with plain writes, straight into the other
island's DRAM or a core's L1; the last packet of every 64 also adds that batch's pages
to a count in a core's port. The dispatcher's own `PAGES` line (`KM_PAGES_LINE`) goes
to link core 0, whose link 0 CPU sends the packet it staged in the link area into the
other island's dispatcher; a page command and a line on that CPU run in turn.

- At the load, each link CPU of both islands opens the connection to its link and
  holds it until its kernel ends; a send never opens or closes one. It then says
  hello to the same CPU of the far link core, one word into that core's port 5 or 6,
  and the load returns once both sides have heard each other (bounded), so no page is
  sent before the far side runs.

- `session.cross(source, target, first=0, pages=None, timeout=10.0)` writes the pages
  of `source` into the same pages of `target` where another island holds them, and
  waits until the far island's count says they all landed. It returns the pages, the
  bytes and the seconds from the submit to the last count.
- `session.send_pages(source, first, pages, to, core=None, counted=None, port=None)`
  queues the send alone: into tensor `to`, or into L1 address `to` of local core
  `core` of the other island, counted at port `port` (default 7) of core `counted`.
  `session.wait_pages(island, core, pages, port=None)` queues a bounded wait for a
  count. Both return handles of the islands' own runs.
- 64 MiB from island 0 into a tensor of island 1 over four links: 2.09 ms, 32.1 GB/s,
  every page exact (`experiments/islands_pages`; measured when pages took all four
  links, before link 0 became the lines link: pages now take three of four).
- A core reads pages another island holds by asking: `session.ask(tensor, first,
  pages, island, core, to, port)` has the holder's link cores send them into L1
  address `to` of the asking core, counted at its port; the core waits on that count,
  bounded, with `km::port_wait(reader, command, port, pages)` and reads them from its
  L1. The ask travels as a command from the host (`IslandSession.ASKS_BY = "host"`);
  a core's own message to the holder is the other setting, not built yet.
- `session.ask_range(tensor, first, pages, group, to, port)` is one ask for the ranks
  of `group` (one island's): rank `group[0] + i` gets pages `first + i * pages` on into
  L1 address `to` of its core, counted at its port. One `PAGES` range command to the
  holder's link cores; their page CPUs split the ranks, and each rank's last packet
  adds its pages to its count.
- `session.wait_together([(island, handle), ...], timeout)` waits for handles of
  several islands' own runs at once. A host's commands reach an island's ring only
  while the host waits on that island, so two sides that wait on each other are
  waited together.
- Example 01's MAP on island 0's 118 ranks reading an input island 1 holds alone
  (`islands.py --across 4`, each rank's two tiles pushed into its transfer buffer,
  2026-11-14, every tile exact):

  | Ask | µs a round | Reading its own DRAM |
  |---|---|---|
  | One `ask_range` a round, one native round an island in `wait_together` | 355.9 | 181.0 |
  | One `ask_range` a round, 1 ms an island in `wait_together` | 1,340.4 | 171.2 |
  | 118 asks a round (one a rank), connections held | 3,009.1 | 170.5 |
  | 118 asks a round, a connection opened and closed by each command | 3,132.5 | 169.9 |

## A card split into islands

`km.Session(card=a, islands=n)`, n 2 to 4, returns the same `IslandSession` over one
card described as n islands: the same calls, the same `IslandRun`, so a program for
islands is developed and checked with one card. One device and one open; the graph is
`islands.split_card(cores, columns, n, rows=True, reserve=False)`.

- **Cores.** An island's ranks run on a contiguous range of the card's cores from its
  base: core = base + (rank - the island's first rank). The base is in the island
  header of every core (`KM_ISLAND_BASE`) and in the dispatcher's island word; on a
  whole card it is 0. With no link cores, rank and core are the same number.
- **Dispatchers.** Island i's dispatcher runs on X280 cluster i of the card, the same
  image as cluster 0's, with its own ring from the host, its own state area and its own
  LIM; each island's cores report to their own dispatcher. So a card splits into at
  most four islands, and each island past the first takes a work cluster: a split
  card runs no X280 threads (the session refuses them).
- **Load.** `session.load(program)` builds and launches one `Program` once on every
  core, each core given its island's table, base and dispatcher, then starts every
  dispatcher. Each dispatcher sends commands over its island's rows only, so an island
  is whole rows of the grid: on the 12 x 10 grid of a p150a, 2 islands of 5 rows (60
  cores each), 3 of 4, 3 and 3 rows (48, 36, 36), 4 of 3, 3, 2 and 2 rows (36, 36,
  24, 24).
- **Commands.** As on two cards: the host sends a flat group's word to each island it
  touches, and each dispatcher runs its part. `run.inspect(island, cores)` takes the
  card's core numbers; a give-up names its island and the run goes on.
- **Stop.** Each island stops alone: the host waits for its rows and reads its cores'
  control blocks from L1, since a buffer read would wait for the whole launch.
- **Tensors.** One tensor in the card's DRAM; its placement by island is kept, so a
  kernel's `km::island_of(placement, page)` answers as on two cards.
- **Not on a split card.** A list or dict of programs, dynamic handlers, X280 threads
  and ports, links, a reset list for a core (a given-up command is not recovered by
  a reset), queued commands (the dispatcher places them over the whole card), and
  engine streams and graphs (not checked on a split card).

Measured on one Blackhole p150a, example 01 unchanged (`islands.py --cards 0 --whole`,
then with `--islands 2`), 2026-11-14, X280 at 1350 MHz, host load about 3, the median
of 3 rounds of 20,000 NOP or 2,000 MAP, microseconds a command, submit to event,
pipelined:

| What | The whole card | Split in 2 |
|---|---|---|
| NOP on each island's own ranks | 1.190 (0, 120) | 1.218 (0, 60), 1.249 (60, 60) |
| NOP on all 120 ranks | 1.218 | 2.537 (a part on each island, joined) |
| MAP on (0, 120), the input split over the islands | 2.651 | 4.674 |

A NOP on one island of the split card costs what it costs on the whole card; a group
over both islands costs two commands of host work, as on two cards. MAP on (0, 120):
120 tiles bit-exact on both. With `--hang`: an inspect of island 1's core 70 during a
HANG on ranks 40 to 119 named its reader waiting for a port arrival; the deadline gave
the command up, `CommandTimeout` named island 1, and the run went on; 16 pages of
island 1 read inside the run were exact.

## Measured

Two Blackhole p150a cards joined by four Ethernet links, one process, example 01's
program and kernels unchanged (`examples/01_mapcollect/islands.py --whole`, and
without `--whole` for the rows of one island's groups), when every core was a rank (240
ranks, before an island always took link core 0), 2026-11-14, X280 at 1350 MHz, host load about 2 to 3 (AICLK not sampled),
microseconds a command, submit to event, pipelined (the median of 3 rounds of 20,000
NOP or 2,000 MAP, one host thread submitting to both islands):

| What | µs a command |
|---|---|
| NOP on island 0's 120 ranks | 1.195 |
| NOP on all 240 ranks (a part on each island, joined) | 2.410 |
| MAP on all 240 ranks, the input split over both islands | 5.215 |
| MAP on group (0, 120) | 2.553 |
| MAP on (0, 120) and (120, 120) in turn | 2.691 |

A NOP on one island costs what it costs on a card opened alone (1.131 to 1.159 µs
in `experiments/islands_mesh`). A 240-rank ticket costs two commands of host work, one
an island, from one host thread: 1.21 µs a part. MAP commands alternating between the
islands cost what MAP on one island costs, so the islands run at once. MAP on one
group of all 240 ranks (rank r adds input tiles 2r and 2r + 1, which lie in its own
island's part of the input): the last round's 240 tiles bit-exact against
`mapcollect.py`'s sums, both kernels ended with no error. With `--hang`: an inspect of
island 1's core 10 during a HANG on ranks 100 to 239 named its reader waiting for a
port arrival; the deadline gave the command up, `CommandTimeout` named island 1 and its
cores, and the run went on; 16 pages of island 1 read inside the run were exact. Open
to close in 14 s (the mesh with the fabric 5 s; the tensors and both loads 2 s).

## Not yet

- Links: a dynamic run's messages cross only in a RUN of every rank;
  `collect` and `broadcast` over islands are not built, and while the run's messages
  are served link 0 carries nothing else. Pages cross on the host's command
  (`send_pages`, `ask`, `ask_range`), not yet on a kernel's own request.
- A dict of programs, X280 threads and ports, engines and uploads over islands;
  tensors in L1.
- More than two cards: the mesh opens 1 x n; tried with two.
- Between the islands of a split card: no link (a NoC write with a settable rate).
  Three and four islands on one card were not run.
