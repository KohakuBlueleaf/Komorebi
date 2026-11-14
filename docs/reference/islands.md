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

with km.Session(cards=[0, 1]) as session:  # every link, 236 ranks
    source = session.tensor(array, "bfloat16", "tile")  # split by island
    run = session.load(program)  # on every island
    ticket = run.submit("map", group=(0, session.ranks), args=[...])
    parts = run.wait(ticket, timeout=5.0)  # one event per island
    before = run.crossings()
    ...  # more commands
    print(run.crossings() - before)  # what they sent over each link
    run.stop()
    (result,) = session.read_bf16(output)  # assembled from every island
```

## The session

`km.Session(cards=[a, b, ...])` returns an `IslandSession`; nothing else is needed.
Its other arguments, `links=None, islands=None, pll_mhz=None, tree=None, log=None,
grid_timeout=10.0, kernel_bytes=None, channel=256 << 20, slow_code=False`, are
optional limits and choices.
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
  `session.ranks` is the total, at most 65,024 (below), a group's first
  below `KM_GROUP_LINK_BASE` (0xFE00), where link cores' names start (0xFF00 on: X280
  threads).
- **Close.** Every island's run is stopped through the X280 runtime, the islands'
  sessions are closed, then the mesh, and the fabric is set off. A grid that did not
  return keeps the mesh open, as a card's session keeps its device.
- **Links.** Two cards use every link the fabric found between them (four on two
  p150a cards; `session.links`); `links` 1 to 4 is an optional limit. Each island's
  last `(links + 1) // 2` cores are its link cores
  (`session.link_cores`, `graph.link_cores`), not ranks; both data-movement CPUs of a
  link core send, each on its own link (`session.channels`: the Ethernet channel of
  each). Link 0 is the lines link: link core 0's CPU 0 holds it and sends the
  dispatcher's command and DONE lines (commands across), and pages only when it is
  the one link; links 1 on carry pages. The session checks that a DRAM bank sits at
  the same NoC place on both islands, and that both islands' core tables are the
  same. One card is an island of one: no link cores. The ranks an island has, on a
  card of 120 Tensix cores:

  | Links | Link cores an island | Ranks an island | Ranks on two cards |
  |---|---|---|---|
  | 1 or 2 | 1 | 119 | 238 |
  | 3 or 4 (the default on two p150a) | 2 | 118 | 236 |

- **Kernel buffer.** The kernels of a run over two cards carry the link cores'
  services, so the session opens with a kernel buffer of 112 KiB
  (`island_session.KERNEL_BYTES`: the largest such build, 90,112 bytes, and at least
  16 KiB more, in steps of 16 KiB); one card keeps the runtime's 70,656 bytes. The
  load checks each build against the buffer: one that does not fit raises
  `KernelSizeError` with its size, the buffer and the most L1 allows, before anything
  is launched. `kernel_bytes` sets another size at the open.
- A link core is named outside the ranks: a command's group with first
  `KM_GROUP_LINK_BASE + k` and count n names link cores k to k + n - 1 of the island
  that takes it (`graph.link_group(island, k, n)`). Such a group is never cut by
  island and never crosses; on a core, its rank is its link-core index less k. A link
  core is in no group of ranks. A timed-out command to link cores names them as
  `link core k (island i, core c)`.

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
- Every line of such a split carries `KM_FLAG_PART` (`km_abi.h`): its dispatcher runs
  its own part and never sends the line across, since the host gave each other island
  its part itself. A line whose group reaches past its dispatcher's island without the
  mark (a thread's `kmx_submit`, or a line the host gives one island's own
  `run.island(i)`) crosses: that dispatcher sends the other part over the link and
  completes the command when every part is back
  ([commands across islands](threads.md#commands-across-islands)). So every rank runs
  each command once either way: measured with example 01's COUNT (every core adds one
  to its count) on flat ranks (60, 120) of 238, 1,000 times from the host and 1,000
  times from island 0's thread, each rank in the group counting exactly 1,000 for
  each and the others 0 (`examples/25_islands_threads --steps count`); 7.46 µs a
  command from the host (split, pipelined), 10.89 µs from the thread (crossed, one at
  a time). `exclusive=True` on a split is allowed, each island exclusive on its part;
  an EXCLUSIVE line that would cross is rejected with reason 2 (`flags`).
- `run.wait(tickets, timeout)` sends every island's queued commands, then feeds the
  islands in turn (one native round each: store what its ring has room for, take its
  events) until every part arrived, so the islands run their parts at once; all within
  the one timeout.
  A ticket completes when every part has; its result is the tuple of its parts' events
  in island order. `run.drain`, `run.stop` and `run.abort` act on every island within
  one timeout; `run.state()` lists every rank in flat order, each with its `rank` and
  `island`, `run.result()` every island's result block.
- The run is addressed and reported by flat rank. An error from an island starts
  `island i:` and carries `error.island`; a `CommandTimeout` (a time-out or a give-up)
  names its ticket, and the cores it lists, where each waited and those reset and
  loaded again (`recovered`), by flat rank with the island and core as information:
  `rank 128 (island 1, core 10) reader waits in port`. A `WaitTimeout`'s `pending`
  lists tickets.
- `run.inspect(ranks=None)` reads what each rank of `ranks` (first, count; default
  every rank) is doing while the run goes on, each island's part asked of its own
  dispatcher, as `Run.inspect` does; each core's dict also holds its `rank` and
  `island`.
- Recovery (`load(..., recover=True)`, the default; [recovery](../guides/recovery.md)):
  each island's dispatcher gets its own island's reset lists at load, one list a kernel
  group, so a list of programs works too (a core gets the list of the program it
  holds). A command given up on island i resets and loads again that island's cores
  still in it; the `CommandTimeout` starts `island i:` and its `recovered` lists
  their flat ranks. `run.reset_core(rank)` resets the core of one rank and loads its
  kernels again; `run.recoveries` lists every recovered give-up as `{island, ticket,
  handle, cores, ranks, text}` (`cores`: the island's own core numbers); `run.recovery`
  is each island's `Run.recovery`. On two p150a cards with the map/collect example,
  each island's lists take 104,192 bytes and 3.4 to 4.0 ms of the load. Link cores
  get no list, so a give-up does not reset one.
- `run.reset_link(island, k)` resets link core k of `island` by the host
  (`session.reset_link`): each of its connections closed at its router (the close
  request, the router's teardown, the producer cursor set at the router's read
  counter), the core reset and loaded again, then its OPEN with the far link core's
  hello taken as heard. Returns `{island, core, by, seconds, reset_seconds,
  open_seconds, parts}`. Between commands only: a line or ask in flight on that core
  is lost and its command times out at its deadline.
- `run.send()` hands every island's queued commands to its dispatcher without waiting.
- `run.island(i)` is island i's own `Run`, the low-level door: its groups, inspect
  and `reset_core` take that island's own core numbers.
- A dynamic program (`Program.dynamic`) on two islands: each core
  gets the message service's arguments and `KM_LINKS`, and its messages cross. Its RUN
  takes every rank (`(0, session.ranks)`; another group raises `ValueError`; so does
  `submit_many` of a dynamic handler and X280 ranks). Before the RUN the session queues
  a PAGES command of mode `KM_PAGES_MESSAGES` (a0: the RUN's group word) to link core 0
  of each island; its link 0 CPU serves the run's messages until the RUN's end, and the
  dispatcher's quiet test counts that core's line. The service grows the kernels (the
  forward example's to 97,008 bytes), inside the kernel buffer of two cards.

## What crossed

`run.crossings()` reads, by link and direction, what the run sent over the links since
its start: a `Crossings` whose `rows[(link, source, target)]` hold the counts island
`source` sent to `target` on its link `link`:

| Field | Counted for |
|---|---|
| `messages` | a dynamic run's messages to a rank of the other island |
| `lines` | the dispatcher's command lines (commands across), 192 bytes each |
| `asks` | ask lines of `km::read_pages` for pages the other island holds, 64 bytes each |
| `pages` | pages sent: the answers to asks, `send_pages`, `cross`, `ask`, `ask_range` |
| `parts` | parts relayed by `km::to_island(i).put` |
| `bytes` | every payload byte of the above |

`after - before` is what crossed between two reads, so a command is bracketed by a
read before and one after it; `total(source, target)` sums the links. Each CPU of a
link core counts what it sends in plain counters of its link area (`KM_LINK_CROSS`,
`km_abi.h`); no other core counts anything. A read is one `PAGES` command of mode
`KM_PAGES_COUNTS` to each island's link cores, whose CPUs copy their counters into a
small DRAM tensor the host reads: it waits behind a command a link core runs, so while
a dynamic run's messages are served, link 0's counts are read after its RUN. On one
island, `crossings()` has no row.

Measured on two Blackhole p150a cards, every link (4), 2026-11-14:
[05_forward](../../examples/05_forward/README.md) (16 messages of 4 forwards over 232
message ranks) sent 15 messages from island 0 and 7 from island 1, the replay's 22
forwards across, 80 bytes each; example 01's MAP on island 0's 118 ranks reading an
input island 1 holds sent 118 asks a round on link 0 and got 236 pages a round back on
link 1 (2,048 bytes each), and nothing crossed while it read its own DRAM.

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
- A kernel reads a tensor's pages wherever they live with one call (`km_ask.hpp`):
  `km::read_pages(reader, command, placement, tensor, address, first, pages, to,
  port)`, with the tensor's `km::Placement`, its accessor, its bank address (the same
  on every island), and L1 address `to` of the core, page k at `to` + k aligned pages.
  The pages its island holds are read as `noc_async_read_page` reads them (the
  caller's read barrier waits for them); on one island that is all the call does.
  Each run of pages another island holds is one ask line of 64 bytes: the core
  reserves a line in its island's link core 0 with a local atomic, waits until that
  line is free, and writes it; link core 0's link 0 CPU sends it, between commands,
  over its held connection into the far link core 0, whose page CPU (link 1, or link
  0 when there is one link) reads the pages from its DRAM, pushes them into the
  asker's L1, adds them to the asker's port, and sends back the count of asks it
  answered. The asker waits on its port, bounded; a miss gives the command up at
  `ask` with the island asked in the word's top byte. 32 lines are in flight an
  island. A tensor of one card has `placement_args()` too (one island holding every
  page), so the same kernel builds on one card.
- `IslandSession.ASKS_BY` is `"core"` (the call asks itself) by default; with
  `"host"` (the define `KM_ASKS_BY_HOST`) the call only waits on the port, and the host
  sends: `session.ask(tensor, first, pages, island, core, to, port)` has the holder's
  link cores send the pages into L1 address `to` of the asking core, counted at its
  port, and `session.ask_range(tensor, first, pages, group, to, port)` is one ask for
  the ranks of `group` (one island's): rank `group[0] + i` gets pages
  `first + i * pages` on into L1 address `to` of its core, counted at its port; one
  `PAGES` range command to the holder's link cores, whose page CPUs split the ranks.
- `session.wait_together([(island, handle), ...], timeout)` waits for handles of
  several islands' own runs at once. A host's commands reach an island's ring only
  while the host waits on that island, so two sides that wait on each other are
  waited together.
- Example 01's MAP on island 0's 118 ranks reading an input island 1 holds alone
  (`islands.py --across`, four links, 2026-11-14, every tile exact; the rows after
  the first pushed each rank's two tiles into its transfer buffer, the first into its
  input buffer):

  | Ask | µs a round | Reading its own DRAM |
  |---|---|---|
  | Each rank's own `km::read_pages`, no ask from the host | 434.3 | 183.8 |
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
  touches, and each dispatcher runs its part. `run.inspect(ranks)` takes flat ranks as
  on two cards; a give-up names its ranks and island and the run goes on.
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
  are served link 0 carries nothing else (a core's ask waits too). A kernel's own ask
  (`km::read_pages`) is answered by one page CPU, on one link; asks of more than two
  islands are not built.
- A dict of programs, X280 threads and ports, engines and uploads over islands;
  tensors in L1.
- More than two cards: the mesh opens 1 x n; tried with two.
- Between the islands of a split card: no link (a NoC write with a settable rate).
  Three and four islands on one card were not run.
