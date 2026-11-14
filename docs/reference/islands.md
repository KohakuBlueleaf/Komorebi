---
title: Islands
summary: A session over several cards, each card's chip an island with its own grid and dispatcher, or one card split into islands (each joined to each, or a line forwarding through the islands between); ranks flat and contiguous by island, commands split by island on the host, tensors placed by island. What it does today and what it does not yet.
tags:
  - reference
  - host
---

# Islands

Status: implemented over two Blackhole p150a cards, and over one card split into two
to four islands. A group that spans islands runs as one part on each island, and every
core sees its flat rank in the whole group, so an existing program runs unchanged.
Commands cross between the cards on link 0 and pages on the session's links, through
each island's link cores; between the islands of a split card they cross on wires.

An island is a set of cores that reach the others only through links: here one card's
chip, with its own Tensix grid and its own dispatcher on its own X280 cluster 0. The
host values (the graph of islands, ranks, the split of a group, placement) are in
`komorebi.islands`.

```python
import komorebi as km

with km.Session(cards=[0, 1]) as session:  # every link, 236 ranks
    source = session.tensor(array, "bfloat16", "tile")  # split along dimension 0
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

- **Kernel buffer.** On islands with links the ranks and the link cores load
  different kernels from one program: the ranks the program's kernels with the client
  side of the page calls only (posting an ask or a put, a wait on its port, the relay's
  ask), the link cores the runtime's link kernels (`kernels/link_reader.cpp`,
  `link_writer.cpp`, `link_compute.cpp`, built with `KM_LINK_SERVICES`): the page
  CPUs' service, the lines and messages, the relay's serving side. A rank still
  serves a `PAGES` command's wait (`wait_pages` may name a rank). The runtime's buffer
  holds each CPU's largest text over the two groups. A program with messages, a card
  split into islands and a program placed by overlays keep the services in every
  kernel. The session opens with a kernel buffer of 128 KiB
  (`island_session.KERNEL_BYTES`: the largest such build, the dense linear layer's
  98,592 bytes, and at least 16 KiB more, in steps of 16 KiB); one card keeps the
  runtime's 70,656 bytes. The
  load checks each build against the buffer: one that does not fit raises
  `KernelSizeError` with its size, the buffer and the most L1 allows, before anything
  is launched. `kernel_bytes` sets another size at the open.
- **L1 of a rank.** What the session keeps on every core is the load's L1 table rows
  `session: ...`: the own words (`KM_OWN_BYTES`, 256 bytes: each data-movement CPU's
  scratch for an ask, a put line and a relay ask, and the relay's done word), the
  tensor table (48 bytes an entry on two islands, as many entries as the tensors held
  at the load, in steps of 4; a tensor made while a run lives takes a free entry) and
  a dynamic run's message region. When the link cores run the link kernels, the link
  area and the relay (48,192 bytes) are theirs alone: at the base of their circular
  buffers, zeroed by the host at the load (`write_l1`, just before the launch), the
  ranks' buffers launched apart; the link cores take no overlay slots. Otherwise (a
  program with messages, a split card, a program placed by overlays) both are
  session tensors on every core, rows of the table too.
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
  get no list, so a give-up does not reset one: a link core still in a given-up
  command is named instead (`KMD_EVENT_LINK`, `run.links_hung`), see below.
- `run.reset_link(island, k)` resets link core k of `island` by the host
  (`session.reset_link`): each of its connections closed at its router (the close
  request, the router's teardown, the producer cursor set at the router's read
  counter), the core reset and loaded again, then its OPEN with the far link core's
  hello taken as heard; on link core 0 its ask and put queues drop the lines not yet
  taken (the tail set to the taken count, those lines cleared) while it is in reset.
  Returns `{island, core, by, seconds, reset_seconds, open_seconds, parts}`.
- A link core that does not leave a given-up command (the runtime's PAGES commands on
  link cores are given up at their deadline too) is left to the host: its dispatcher
  posts `KMD_EVENT_LINK` (`Run.link_events`: `link`, `what`, `req`, `ticks`), the
  time-out's `links` holds `(k, "hung")`, and every other command whose traffic the
  core held is given up at once, its `links` `(k, "lost")`: a group that has the core,
  a crossing whose line was staged on it (the line's packet number is taken again by
  the next packet), or, on link core 0, a command whose cores wait for an ask or put.
  `run.links_hung` lists `(island, k)` until `run.reset_link(island, k)`.
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
| `pages` | pages sent: the answers to asks, puts, `send_pages`, `cross`, `ask`, `ask_range` |
| `parts` | parts relayed by `km::to_island(i).put` |
| `puts` | put lines of `km::write_pages` (64 bytes) and their confirmations (8 bytes) |
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

A tensor of an island session is one logical tensor: a script names it and says how
the program uses it, and the session decides which island holds which page. The hints
are in the kernel's own words:

| Hint | Where the pages live |
|---|---|
| none | as `split=0` over every rank |
| `split=d`, `group=(first, ranks)` | the group's ranks partition dimension `d` (its page steps: rows of a "row" tensor, tile rows or tile columns of a "tile" one) by the rule of `km::partition`, and the part rank `r` works on is held by `r`'s island; `group` defaults to every rank |
| `like=other` | `other`'s hint applied to this tensor, so the same holders part for part |
| `copy=True` | every page on every island; only when asked, never by default |

`session.tensor(data, dtype, layout="row", split=None, group=None, like=None,
copy=False)` makes it and writes it; `session.empty(shape, dtype, layout, ...)` takes
the same hints and writes nothing. `islands.place(graph, shape, layout, split, group,
copy)` is the rule on the host: a `Placement` whose pages are `outer` blocks of `steps`
steps of the split dimension, `inner` pages a step, and whose `parts[i]` are the
(first step, steps) island `i` holds of every block. `placement.where(page)` gives the
island and the page's index in that island's part, `placement.runs(first, pages)` the
runs held at consecutive indices, `placement.part(i)` the pages of island `i`'s part in
its order. A split along a dimension other than the leading one gives an island pages
that are not one run of page indices: tile column 0 of every tile row on island 0 and
column 1 on island 1, for `split=1` over a group of two ranks one on each island.

Each island allocates only its part, in its own DRAM at its own bank address, so a
tensor that does not fit one card fits two when the hint splits it. A part that does
not fit raises `KomorebiError` naming the island and the most pages of that kind its
DRAM takes now.

`tensor.address` is the tensor's reference word, what a script passes in a command and
a kernel hands to Komorebi's page calls. On one card it is the bank address, as without
islands. Over islands it is the index of the tensor's entry in the tensor table
(`KM_TENSORS_MAX`, 32, entries), which the session writes into every core of every
island at the load and again when a tensor is made while a run is loaded; the word
`KM_ISLAND_TENSORS` of a core's island area holds the table's L1 address. An entry is
the aligned page bytes, the steps of the split dimension, the pages a step and a copy
flag, then each island's part: its bank address, first step and steps. On a core,
`km::page_at(tensor, page)` (`km_placement.hpp`) gives the island that holds the page,
its part's bank address and the page's index there, and `km::island_of(tensor, page)`
the island; a word that is not an entry (a bank address) answers this island, the word
and the page, so a kernel written this way runs unchanged on one card. A copy's page
answers the core's own island. `tensor.accessor_args()` is the same on every island.

`session.write(tensor, pages, first=0)` scatters each island's pages into its part
through that island's own host channel, and `session.read` and `session.read_bf16`
gather them into one array; while a run is loaded, the islands' transfers go at once,
one host thread an island. A part that is one run of the tensor's pages moves from and
into the caller's rows themselves, without a copy of its own. A hint whose part does not
fit an island's DRAM raises `km.KomorebiError` before anything is written: the island,
the pages and shape asked, the allocator's sentence, and the most pages of that kind
that fit there now (asking island 0 alone for 48 GiB on a p150a: refused in 35 ms, at
most 16,688,832 pages of 2 KiB). Example 01's MAP on one group of the 238 ranks of two
cards, its input of 238 x 2 tiles placed with no hint, `split=0` and `split=1` over the
group (118, 2) (`examples/01_mapcollect/islands.py --whole --split 1 --group 118,2`;
2026-11-14, AICLK 800 MHz idle before the run, host load 3.8 to 8.4, one run each):

| Input hint | Island 0 holds | MAP, us | Wrong tiles |
|---|---|---|---|
| none | tile rows 0 to 118 | 6.10 | 0 of 238 |
| `split=0` | tile rows 0 to 118 | 6.13 | 0 of 238 |
| `split=1`, group (118, 2) | tile column 0 | 469.4 | 0 of 238 |

With `split=1` every rank reads one tile from its own island and asks the other island
for the other. The same script with `--cards 0` and `split=1`: one island holds every
page, the word is the bank address, 0 wrong of 120, 3.13 us a MAP. A write of 64 MiB
with the run loaded, one run each: split over both cards 1.85 to 3.69 GB/s, all on
island 0 2.89 to 3.11 GB/s, card 0 alone 2.86 to 3.13 GB/s: each island first copied
its part out of the caller's rows, and those copies did not overlap.

Without that copy (`examples/01_mapcollect/islands.py --whole --io-mib 64,512`, the
run loaded, AICLK 800 MHz, host load 4.3, one run, every byte read back compared), GB/s
of one write and one read:

| Size | Split over both cards: write, read | Island 0 alone: write, read |
|---|---|---|
| 64 MiB | 7.38, 8.39 | 6.52, 4.93 |
| 512 MiB | 13.59, 3.43 | 7.54, 5.14 |

What bounds them: island 0's cores move 64 MiB in 7.9 ms (write) and 7.6 ms (read) of
the 10.3 and 13.6 ms; the rest is the host's copy into and out of the hugepage channel,
13.1 GB/s in and 7.1 out for one thread in that run. Both cards at once, each card's
cores took 5.0 ms for its 32 MiB written (3.9 ms read), and the host's two copies at once
reached 7.3 GB/s in and 7.5 out in that run against 21.5 and 20.0 in the run before,
the machine shared with other work: the 512 MiB read over both cards is bound by the
host's copies out of the channels.

`IslandSession(host_reaches=[0])`, for a test (`km.Session(cards=[0, 1],
host_reaches=[0])`), is the machine whose second island has no host link: every
tensor's parts are written and read through island 0's channel. Island 1's part is
written into a buffer of its shape on island 0 and sent by island 0's link cores into
island 1's part; a read has island 1's link cores send it into such a buffer, read
through island 0's channel. Both run inside a loaded run with links, and a write or
read outside one raises. Example 01 with `--host-reaches 0 --split 0` (input and output
written after the load): MAP 0 wrong of 236, every read exact; GB/s with island 1's
half crossing, against island 0 alone in the same run:

| Size | Write | Read | Island 0 alone: write, read |
|---|---|---|---|
| 64 MiB | 5.12 | 2.74 | 6.52, 5.05 |
| 512 MiB | 6.11 | 3.51 | 7.58, 5.23 |

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

- `session.cross(source, target, first=0, pages=None, timeout=10.0, shape=0)` writes
  the pages of `source` into the same pages of `target` where another island holds
  them (a copy: every other island's copy), and waits until the far island's count
  says they all landed. It returns the pages, the bytes and the seconds from the
  submit to the last count.
- `session.send_pages(source, first, pages, to, core=None, counted=None, port=None)`
  queues the send alone: into tensor `to` (which the other island holds at the
  indices its part has here), or into L1 address `to` of local core `core` of the
  other island, counted at port `port` (default 7) of core `counted`.
  `session.wait_pages(island, core, pages, port=None)` queues a bounded wait for a
  count. Both return handles of the islands' own runs.
- A page CPU passes pages through its chunk of the run's transfer buffer: on two cards
  the session asks for a chunk of 64 KiB (`LINK_CHUNK`) where a core has room for two
  of them in half of what its program leaves, else the 16 KiB default. A 64 KiB chunk
  holds 15 slots of 4 KiB pages. A CPU's share of 32 pages or more is read in groups:
  4 pages of one DRAM bank a read (pages p, p + 8, ..), or pages one after another
  from L1 or one bank's range, three groups in flight; 16 pages a read when three
  groups of 16 slots fit (a 256 KiB chunk); a smaller share one page a read, 15 in
  flight. Each group's read is waited for by its own transaction id (reads from
  several banks land out of order), and a group is read again a group later, once
  its packets left L1. A send's shape (`KM_PAGES_SHAPE_*`, the top byte of its buffer
  word; `session.cross(.., shape=)`) sets pages a read and slots to measure them.
- Link 0 carries pages too when no message service holds it: the host's send names it
  (`KM_PAGES_SHAPE_LINK0`), and its CPU sends its island's next ask line, if one
  waits, after each read's packets; the session then counts four page CPUs
  (`KM_PUT_CPUS`).
- 512 MiB from island 0 into a tensor of island 1 (`session.cross`, four links, the
  default chunk): 8.93 ms by the command's own ticks, 60.1 GB/s, every page exact
  (`experiments/islands_pages`, 2026-11-15). With a 256 KiB chunk (16 pages a read)
  128 MiB took 1.68 ms, 80.0 GB/s. The host's own wait for a cross takes far longer
  than the cross (62 ms at 128 MiB, 246 ms at 512): `pages.py --shapes` reports both.
- A kernel reads a tensor's pages wherever they live with one call (`km_ask.hpp`):
  `km::read_pages(reader, command, args, tensor, first, pages, to)`, with the
  tensor's accessor arguments, its reference word, and L1 address `to` of the core,
  page k at `to` + k aligned pages.
  The pages its island holds are read as `noc_async_read_page` reads them, at their
  index in its part (the caller's read barrier waits for them); on one island that is
  all the call does. Each run another island holds at consecutive indices of its part
  is one ask line of 64 bytes: the core
  reserves a line in its island's link core 0 with a local atomic, waits until that
  line is free, and writes it; link core 0's link 0 CPU sends it, between commands,
  over its held connection into the far link core 0. Every page CPU of the far
  island (every link's CPU when no message service holds link 0, else those holding
  links 1 on, or link 0 when it is the only link) serves its share of the line's
  pages (`partition(pages, j, page CPUs)`; link 0's CPU only in a line of 64 pages or
  more, a smaller line split over the others): it reads the line (across the NoC
  when it sits on link core 1), reads its
  pages from its DRAM, pushes them into the asker's L1, each batch adding its pages to
  the asker's port once landed, and sends back its own count of asks answered
  (`KM_ASK_SERVED` + 4 j). The asker waits on its port for all of them, in whatever
  order they land, bounded; a miss gives the command up at `ask` with the island asked
  in the word's top byte. 32 lines are in flight an island; a line goes across again
  once every far page CPU answered the one 32 before it.
- A kernel writes a tensor's pages wherever they live with the counterpart, in the
  same header: `km::write_pages(core, command, args, tensor, first, pages, from,
  port)`, from the reader or the writer, page k from L1 `from` + k aligned pages. The
  pages its island holds are written as `noc_async_write_page` writes them, at their
  index in its part (the caller's write barrier waits for them); on one island that is
  all the call does. Each run another island holds at consecutive indices of its part
  is one put line (the same 64 bytes, `KM_PUT_*` in `km_abi.h`) into its own island's
  link core 0, reserved and written as an ask line is. The CPU of link 1 sends
  the line to the far link core 0; every page CPU reads its share of the pages from the
  asking core's L1 and sends them straight into the holder's DRAM; each batch's last
  packet adds its pages, once all the batch carried are written (the router flushes
  its writes before the add), to the far link core 0's count of pages landed from that
  page CPU (`KM_PUT_LANDED` + 4 j). Once every page CPU's share of the line has landed,
  the far CPU of link 1 adds them to the asker's port and sends back the count of puts
  confirmed. A put line is free again once every page CPU took it (a count each,
  `KM_PUT_TAKEN` + 4 j; the island's page CPUs in `KM_PUT_CPUS`). The call waits on
  its port, bounded, so `from` is free when it
  returns; a miss gives the command up at `ask` with the island in the word's top
  byte. 32 put lines are in flight an island.
- An ask or put line may name the asker's DRAM instead of its L1 (`KM_ASK_DRAM`: an
  interleaved buffer, or one bank's range, from page `KM_ASK_DRAM_FIRST`): the far
  page CPUs push a get's pages straight into it, or read a put's from it.
- A copy (`KM_HANDLER_COPY`, what `kmx_fetch` and `kmx_store` submit) that names a
  tensor by its reference word finds where its pages are in the tensor table; a side
  only this island holds is read at its bank address here. A copy with one side whose
  pages another island holds asks each run the other island holds (to the end of its
  part's steps, read from the table entry at once) as one line straight between the
  two DRAMs, all in flight, and waits once on the core's port; this island's pages go
  through the chunk as the copy does. A copy with both sides placed moves a chunk at a
  time, a line a run of the chunk. A thread's source names the tensor and the range
  only. Pages that cross are at most one packet (4,352 bytes). On one island, or for a
  tensor named by a bank address, the copy is what it was.
- `IslandSession.ASKS_BY` is `"core"` (the call asks itself) by default; with
  `"host"` (the define `KM_ASKS_BY_HOST`) the call only waits on its page port, and
  the host sends: `session.ask(tensor, first, pages, island, core, to, port)` has the
  holder's link cores send the pages into L1 address `to` of the asking core, counted
  at its port `port` (`KM_PORT_PAGES` for a reader's call, + 1 for a writer's), and `session.ask_range(tensor, first, pages, group, to, port)` is one ask for
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

- Example 01's MAP on one flat group of all 236 ranks over both islands, kernels
  unchanged, storing with `km::write_pages` (`islands.py --put`, four links,
  2026-11-14, every tile exact): 532.8 µs a round into an output island 0 holds alone
  (island 1's 118 ranks put theirs across), 343.2 µs into the output split over the
  islands (each rank's page in its own island's DRAM). With every page CPU serving
  (2026-11-15, `--across --put`, every tile exact, what crossed as counted): the far
  read 530.6 µs a round (its own DRAM 214.4), the put 561.2 µs (split 314.3); with
  four page CPUs and grouped reads 572.2 µs (220.2) and 597.2 µs (327.4). Two tiles a
  rank are one or no page a page CPU: these rounds are the lines' round trip, not the
  links' rate.
- Asked and put pages at the links' rate (`experiments/islands_pages/pages.py --ask`
  / `--put`, island 0's 118 ranks, each a share of the tensor in calls of `--step`
  pages of 4 KiB, island 1 holding the source or the target; one command, its event's
  span; every page read back exact; 2026-11-15, four links):

  | Move | Pages a call | 64 MiB | 512 MiB |
  |---|---|---|---|
  | `km::read_pages`, four page CPUs, 4 pages a read | 128 | 1.75 ms, 38.4 GB/s | 11.3 ms, 47.3 GB/s |
  | `km::write_pages`, the same | 128 | 1.75 ms, 38.3 GB/s | 11.1 ms, 48.4 GB/s |
  | `session.cross` (host, by its ticks) | | | 8.93 ms, 60.1 GB/s |
  | `km::read_pages`, links 1 to 3, a page a read | 128 | 2.37 ms, 28.3 GB/s | 16.7 ms, 32.1 GB/s |
  | `km::write_pages`, the same | 128 | 2.36 ms, 28.4 GB/s | 16.6 ms, 32.3 GB/s |
  | `km::read_pages`, the same | 32 | 3.04 ms, 22.1 GB/s | 22.7 ms, 23.7 GB/s |
  | One page CPU on link 1, three slots (2026-11-14, `kmx_fetch`) | | 11.0 ms, 6.1 GB/s | |

  What bounds a page CPU, measured by the host's cross of 128 MiB over links 1 to 3
  by shape (`pages.py --shapes`): a page a read 27.7 GB/s with 15, 30 or 60 slots
  alike (15 transaction ids); no read at all 42.9 (55.2 without waiting for a slot's
  packets to leave L1); 4 pages a read 48.1 with 12, 28 or 60 slots alike; 16 pages a
  read 44.2 with one group, 62.2 with three (20.7 a CPU). So neither the DRAM read's
  latency nor the slots bound it: the loop's own work a read and a packet does (a
  read's wait, the slot's wait, a header and two NoC commands a packet). A plain
  sender from L1 does 27.8 a link (`experiments/link_rate`). Asked and put pages stay
  under the cross: each line of 128 pages gives a CPU 32, so a line's round trip
  weighs on each share.
- `kmx_fetch` and `kmx_store` of island 0's work cluster 1 thread on 16 cores, of a
  tensor of 4 KiB pages each island holds (`experiments/bank_fetch/islands.py`, four
  links, 2026-11-14, every word checked, best of 3 for a fetch):

  | Call | 1 MiB | 64 MiB |
  |---|---|---|
  | `kmx_fetch`, island 0's tensor | 51 µs | 1.30 ms, 51.6 GB/s |
  | `kmx_fetch`, island 1's tensor | 204 µs | 11.0 ms, 6.10 GB/s |
  | `kmx_store`, into island 0's tensor | 47 µs | 1.30 ms |
  | `kmx_store`, into island 1's tensor | 197 µs | 10.7 ms |

  With every page CPU serving (2026-11-15, every word exact): `kmx_fetch` of island
  1's 64 MiB 12.4 ms, `kmx_store` 11.7 ms (1 MiB: 228 and 213 µs), the copy asking a
  16 KiB chunk (4 pages) a line, one line a core in flight. With a line a whole run
  between the DRAMs (2026-11-15, every word exact): `kmx_fetch` of island 1's 1 MiB
  107 µs, 64 MiB 1.125 ms (59.7 GB/s); `kmx_store` into it 102 µs and 1.118 ms; the
  local ones 51 µs and 1.30 ms.

## A kernel's page reads and writes

A kernel reads and writes a tensor's pages through Komorebi's calls, never tt-metal's
own page calls: a page read through `noc_async_read_page` reads its own island's DRAM
only, so a kernel that calls it works on one card and reads the wrong pages over
islands. The calls (`km_ask.hpp`) take the tensor as one object:

```c++
using Input = km::Tensor<0>;  // compile arguments from 0 on: tensor_args()
using Output = km::Tensor<Input::next_compile_time_args_offset()>;

const Input input(command.a0);  // the tensor's reference word
if (!km::read_pages(core, command, input, first, pages, to)) return;
noc_async_read_barrier();
if (!km::write_pages(core, command, output, first, pages, from)) return;
```

| Call | What it does |
|---|---|
| `km::Tensor<offset>` | a tensor's accessor arguments, from the compile arguments the host's `tensor.tensor_args()` gives (`accessor_args()`), and its reference word (`tensor.address`); `next_compile_time_args_offset()` for the next one |
| `km::read_pages(core, command, tensor, first, pages, to)` | pages `first` .. `first + pages - 1` into L1 `to` on, page k at `to` + k aligned pages; on the reader or the writer (`core`: its `km::Reader` or `km::Writer`); each run of pages another island holds asked for and waited for, counted at the CPU's page port (`KM_PORT_PAGES` the reader's, + 1 the writer's, past the ports a kernel names). False: gave up |
| `km::write_pages(core, command, tensor, first, pages, from)` | pages `first` .. from L1 `from` on; each run of pages another island holds one put line, waited for at the CPU's page port, so `from` is free on return. False: gave up |
| `km::Stream` | a stream over a `km::Tensor` reads and writes through these calls ([Movement](movement.md#a-stream-over-islands)) |

A program loaded on one island (a card's own session, or an island session of one
card) is built without the tensor table (no `KM_TENSOR_TABLE`): the word is the bank
address, and each call compiles to exactly tt-metal's page call it replaces,
`noc_async_read_page` or `noc_async_write_page` once a page. Over
islands each page is found by `km::page_at` and the calls are the forms by word above
(`km::read_pages(core, command, args, word, ...)`). The reader asks with the link
area's own line and answer words; the writer with the first ask line's place, which
only link core 0 holds lines in. Without page links (`links=0`) a page another island
holds gives the command up (`bad_item` at `ask`).

The examples and projects whose kernels read or write a tensor's pages use these calls:
02, 03, 04, 06, 07, 09, 11, 15, 16, 17, 19, 21, 22, 23, 26, the dense project's
resident, streamed, grid and K-split forms and its SwiGLU layer (each operand by its
own word), attention, the core reset experiment's reader and the tutorial's lessons
from 16 on. Example 09's writer writes single words of its output: a word goes to
the part of the island that holds its page (`km::page_at`); a word of a page another
island holds gives the command up. Measured on two p150a cards, one page link,
2026-11-15:

- Example 11's `islands.py`: island 0's four cores copy 192 pages island 1 holds
  alone, exact, and READ them in 275.8 µs (1.43 GB/s); with `--write-across` they copy
  island 0's input into an output island 1 holds alone (every page put across), exact,
  313.4 µs.
- The dense project's `islands.py --b-far` (7680 x 64 x 128, integer inputs): island 1's
  grid reads all of W from island 0, bit for bit, 36.7 µs a call.
- Example 26's `sum.py` (238 ranks, T = 512, the input a copy): exact, 170.3 µs a
  round. With `--far` (the input held by island 1's ranks, so island 0's ranks, rank 0
  among them, ask for all of theirs while rank 0 counts the reduce on its ports):
  exact, 529.3 µs a round.

With the link services in the link cores' kernels only, example 11's build takes
67,680 bytes (92,192 with the services in every kernel), the dense grid's 82,512
(122,752 before) and the dense linear layer's 98,592 (143,440 before); the link
reader's BRISC text is the larger of the two groups' for example 11 and the grid.

## A card split into islands

`km.Session(card=a, islands=n, line=False, card_dram=False)`, n 2 to 4, returns the
same `IslandSession` over one card described as n islands: the same calls, the same
`IslandRun`, so a program for islands is developed and checked with one card. One card
as one island stays the way to run; a split card is the way to check a program written
for islands, and to exercise more than two islands, with the cards at hand. One device
and one open.

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
- **Tensors.** Held by island, as over two cards: each island's part of a tensor (by
  its hint) in its own copy, its word naming its table entry, so a page another island
  holds is asked for across, exactly as between two cards. `card_dram=True` keeps one
  tensor in the card's DRAM instead, every page local to every island, its word the
  bank address: for kernels that read a tensor by its bank address (example 03's
  islands script opens so).
- **Wires.** Every pair of islands is joined by `links` wires (default 1; with more
  than two islands, one): link cores taken from the end of each island's rows, a
  neighbour's links ceil(links / 2) cores in neighbour order, named and served as
  between two cards (commands across on link 0, a core's own page ask and put, the
  relay, messages, `crossings()`). The build defines `KM_WIRE`: the link's transport
  (`km_transport.hpp`) writes the far tile over the NoC instead of handing the packet
  to a router, each count or word after the writes before it; `KM_WIRE_DELAY` (ticks
  of the wall clock, default 0) holds each packet back to imitate a slow link. Every
  link core is found by the island table's next link.
- **A line.** `line=True` joins only neighbouring islands (0-1, 1-2, 2-3; the islands'
  rows are in that order), as three or more cards in a row would be: island 0 reaches
  island 2 through island 1. The table's hops are the islands apart; an end island has
  one link core, an island between two (one toward each neighbour).
- **Forwarding.** Over more than two islands the build also defines `KM_FORWARD` (the
  session's; two islands build without it and their kernels stay smaller). A link core
  that receives what is not for its island sends it on by its island's next link
  toward the target: to the island's link core toward it over the card's NoC, which
  sends it across. No other core of the island between does anything. Forwarded: a
  command's line and its DONE, an ask and the pages that answer it (an ask is queued
  again in the line queue of the link core toward the holder; the pages go back to the
  asker), a relay part (a reduce's fold, a broadcast's blocks) and an island's report
  at the end of a run (below). Each link core holds a forward area (3 parts of 2
  slots of 4,480 bytes, `KM_FORWARD_*` in `km_abi.h`, its address in the island area
  at `KM_ISLAND_FORWARD`): part 0 written by the far link core across, one part by each
  other link core of its island, each with credits back. A message is not forwarded
  packet by packet: the link core that takes it in queues it in the outbox of the link
  core toward its rank, as a writer does. `crossings()` counts every link a crossing
  uses. A put (`km::write_pages`) still goes one link only; a put to an island two
  links away is refused (the kernel gives up).
- **Messages over more than two islands.** Every link core serves its link in a
  dynamic run (a PAGES command of mode MESSAGES each). The run's end is decided once
  for the whole run: each island's link core 0 checks its island quiet (every core of
  the run idle in this run and every link core of the island empty, alike over two
  passes, and the island's counts balanced) and sends {quiet, received across, sent
  across, round} to island 0's link core 0, forwarded like any line. The run ends when
  two successive rounds show every island quiet, the sum sent across equal to the sum
  received across, and the counts unchanged: island 0's link core 0 then sends the
  verdict to every link core (its island's others, and across), and each island's link
  core 0 shows its line idle with the island's totals, which ends that island's part.
  Every wait is bounded. A link core that waits for room in another link core's outbox
  keeps sending its own meanwhile, so messages that cross an island both ways do not
  wait for each other. One link between two islands, at most three link cores an
  island.
- **Commands across.** A dispatcher sends a command to every island its group
  reaches; each island writes its own part of a far dispatcher's inbox (the 16 slots
  cut in islands - 1 parts), so with three islands a dispatcher keeps 3 commands
  across in flight and with four 2 (8 with two).
- **Not on a split card.** A dict of programs, X280 threads and ports, a reset list
  for a core (a given-up command is not recovered by a reset), queued commands (the
  dispatcher places them over the whole card), and engine streams and graphs (not
  checked on a split card).

Measured on one Blackhole p150a, example 01 unchanged (`islands.py --cards 0 --whole`,
then with `--islands 2`), 2026-11-14, X280 at 1350 MHz, host load about 3, the median
of 3 rounds of 20,000 NOP or 2,000 MAP, microseconds a command, submit to event,
pipelined:

| What | The whole card | Split in 2 |
|---|---|---|
| NOP on each island's own ranks | 1.190 (0, 120) | 1.218 (0, 60), 1.249 (60, 60) |
| NOP on all 120 ranks | 1.218 | 2.537 (a part on each island, joined) |
| MAP on (0, 120), the input split over the islands | 2.651 | 4.674 |

Across the wires of card 0 split into n islands, 2026-11-14 (median from submit to
event; the two-card numbers from the examples' READMEs beside them):

| What | 2 islands | 3 | 4 | Two cards |
|---|---|---|---|---|
| Example 01, MAP on island 0's ranks reading its own DRAM / reading island 1's tiles by each core's ask (µs a round) | 201.1 / 306.9 (59 ranks) | 279.3 / 320.9 (46) | 251.6 / 285.8 (33) | 441 across (118 ranks) |
| Example 26, the sum over every rank (µs a round) | 162.18 | 174.19 | 168.37 | 165.9 |
| Example 25, NOP from island 0's ring to its ranks / to island 1's (µs) | 2.966 / 3.891 | 3.034 / 5.067 | 3.042 / 6.278 | 2.98 / 3.94 |
| Example 05, 16 messages of 4 forwards, the parts (µs) | 145.1 and 99.9 | refused | refused | 174.5 and 134.1 |

Every output exact, every kernel ended with no error. These are to check a program,
not to compare speeds: a split card's islands share one card's DRAM and NoC.

The same on card 0 split as a line (`line=True`), 2026-11-15, the far island at the
line's other end (two links from island 0 on three islands, three on four), every
output exact, every kernel ended with no error:

| What | 3 in a line | 4 in a line |
|---|---|---|
| Example 01, MAP on island 0's ranks reading its own DRAM / reading the far island's tiles by each core's ask (µs a round) | 215.96 / 376.63 (47 ranks) | 225.45 / 339.29 (35) |
| Example 01, what crossed each link of the way, one round each core | asks 141 out, pages 282 back (577,536 B), on both links | asks 105, pages 210 (430,080 B), on all three |
| Example 26, the sum over every rank (µs a round) | 183.22 | 194.17 |
| Example 25, NOP from island 0's ring to its ranks / to the far island's (µs) | 3.021 / 7.376 | 3.088 / 11.300 |
| Example 03, 8 broadcasts of 8 blocks over a 6 x 10 rectangle (µs a round, 480 checksums) | 820.62 | 931.09 |
| Example 05, 16 messages of 4 forwards, the parts, island 0 first (µs) | 166.1, 134.3, 110.2 | 216.9, 184.3, 147.1, 115.6 |
| Example 05, messages on the links, against the replay's link crossings | 56 of 56 | 75 of 75 |

What a hop costs: a NOP's command and its DONE to an island one link away take 3.9 to
5.1 µs (the mesh rows above), two links 7.4 µs and three 11.3 µs, so each island
between adds 2.3 to 3.9 µs. Example 01's 141 asks of a round, each answered by two
pages, took 160.7 µs more from two links away than from island 0's own DRAM, against
41.6 µs from one link away in the mesh of three. The end of a dynamic run over islands
(example 05, island 0's part, which holds the end): 146.6 µs over two islands, 166.1 over
three in a line, 216.9 over four. The kernels with forwarding built in, beside the 128 KiB
buffer a session takes by default: example 01's 99,280 B, example 05's 123,216 B (over two
cards, without forwarding: 92,576 B and 109,360 B, built off the card).

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
  (`km::read_pages`) and put (`km::write_pages`) are carried by every page CPU; over
  more than two islands an ask is forwarded but carried by one page CPU a link, and a
  put goes one link only. A copy across asks 16 KiB a line, one line a core in flight.
- A dict of programs, X280 threads and ports, engines and uploads over islands;
  tensors in L1.
- A write of part of a page another island holds (example 09's writer gives up), a
  tensor read by an address the kernel builds itself (`InterleavedAddrGen`: profile
  pages, job tables, the raster and ray tracer projects, the result lines of examples
  12 and 18), and an overlay's part (examples 12 and 18: an overlay takes no compile
  arguments, so no `km::Tensor`) do not cross.
- More than two cards: the mesh opens 1 x n; tried with two.
- A split card: a copy across and a put over more than one link; more than one wire a
  pair with more than two islands; two cards each split (fabric links and wires in one
  graph), so forwarding has run over wires only.
