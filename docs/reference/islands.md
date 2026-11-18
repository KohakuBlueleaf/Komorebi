---
title: Islands
summary: A session over several cards, each card's chip an island with its own grid and dispatcher, or one card split into islands (each joined to each, a line forwarding through the islands between, a ring); ranks flat and contiguous by island, commands split by island on the host, tensors placed by island. What it does today and what it does not yet.
tags:
  - reference
  - host
---

# Islands

On islands nothing differs from one card but time and L1. A program written for one
card runs on two cards, or on one card split into islands, with the same source: a
group that spans islands runs as one part on each island, every core sees its flat
rank in the whole group, and Komorebi's calls find a rank, a page or a word wherever it
lives. What differs is what a crossing costs and what the session keeps on the cores;
the few things that are still refused are listed under [the limits](#the-limits), each
with the refusal that names it, beside the one difference in values (a float reduce
whose group spans islands adds in another order).

An island is a set of cores that reach the others only through links: one card's chip,
with its own Tensix grid and its own dispatcher on its own X280 cluster 0, or rows of
one card split into islands (below). Islands have run over two Blackhole p150a cards
and over one card split into two to four islands. One card as one island is the way to
run, and the recommendation; a card split into islands is for developing and checking
programs written for islands. The host values (the graph of islands, ranks, the split
of a group, placement) are in `komorebi.islands`.

## What is the same

The calls are the same on one island and many: `km.Session(cards=[...])` or
`km.Session(card=a, islands=n)` has the calls of a card's session, a load builds one
program for every island, `run.submit` takes a group of flat ranks, and a ticket's
events read as one card's. In a kernel, `km::read_pages`, `km::write_pages`,
`km::read_bytes`, `km::put_bytes`, `km::write_l1`, `km::add_word`, a port's send, a
message, a chain's marks and the collectives reach any rank or page by its flat rank or
its tensor's word, over as many links as the way takes; on one island each compiles to
the NoC call it replaces. Commands from the host, from a dispatcher, from an X280 thread
or an engine reach any group. Tensors are placed by hint and read and written whole.

A card session's, run's and tensor's calls take the same names, keywords and defaults
on islands: `run.wait(handles)`, `run.event(handle)` (an event once every part
arrived, else None; `run.event_parts(handle)` each part's), `run.forget(handles)`,
`run.given_up(handle)` and `run.wait_any(handles)` take tickets, `run.inspect(group=)`
flat ranks, `run.reset_core(core)` a flat rank, `run.addon(name, parts, ...)` one
card's options; `session.load(program, ...)` its options by position too,
`session.read(..., out=)`, `session.read_chunks(tensor)`, `session.empty(..., own=)`,
`tensor.read_bytes()`, `tensor.nbytes` and `tensor.freed` as on one card;
`session.tensor(..., transposed_tiles=True)` stores every island's tiles transposed (set
after the host wrote the data it is refused, on one card too), `run.submit(..., block=)`
gives each island's part the block in its own DRAM, and `run.threads` and
`run.thread_results()` key every island's threads by flat number (3 x island +
cluster), `run.thread` and `run.thread_result()` the lowest's. `session.tensor(data, dtype,
shard_grid=grid)` and `session.empty(shape, dtype, shard_grid=grid, address=None)`
make an L1 tensor of one island's grid on every island at one address (a class's L1
field takes it as on one card): its rows are `[islands x cores, words]`, island i's
cores' rows `i x cores` on in the grid's order (a split card: the card's `[cores,
words]`, every island's cores among them). An L1 tensor made on one island alone
(`session.island(i)`) holds L1 the others do not, so the next one's addresses, or the
load's link areas, differ; that is refused, naming the cause. A value of one
card is each island's when every island's is the same (a split card's always), else
refused by name: `session.nocs()`, `session.l1()`, `session.dram_banks`,
`session.pll_mhz`, `session.device`,
`session.channel` and `session.launch()` (on two cards `session.island(i)`'s),
`session.card` (on two cards `session.cards`), and `run.programs`, `run.h`, `run.added`,
`run.program_field(p)`, `run.load_field(p)`, `run.field_handler(n, field)` and
`run.thread_words(...)` (one numbering on every island). Over every island:
`run.alive()`, `run.store()`, `run.keep_resets(exclude, timeout)` (`exclude` flat ranks;
each island's link cores get no list, as at the load), `run.group(ticket)` (the flat
group), `run.stalls` (each with its `island`), `run.recovery_refused` (the first island
that kept no lists, named), `run.submit_s` and `run.wait_s` (the island run's own host
time) and `run.native_s` (every island's, summed). Refused by name, the message naming
what to call: the engine calls `upload`, `start`, `restart`, `release`, `cancel`,
`wait_engine`, `run_hostfed` and `check_templates` (`run.island(i)`'s: an engine runs on
the island it is uploaded to), `run.sequence` (`run.trace().command(ticket)`, or
`run.island(i).sequence(handle)`), `run.holds()`, `run.state_words()`, `run.backend`,
`run.inspect_ticks`, `run.stored`, `run.blocks_held` and `run.dispatcher_given_up()`
(`run.island(i)`'s), `session.channel_transfer` (`session.island(i)`'s, with
`tensor.parts[i]`), and `tensor.buffer`
(`tensor.parts[i].buffer`). `tests/core/islands/test_one_unit.py` holds the list; its
script makes the same calls on one card, two cards and a split card on the mock and
compares what each returns.

## What costs time

A crossing costs microseconds where the NoC costs fractions of one. Measured on two
p150a cards joined by four links, and on one card split into islands (the sections
below give each run):

| What | On one island | Across |
|---|---|---|
| A NOP command and its DONE | 1.2 µs | 3.9 to 5.1 µs one link away; each island between adds 2.3 to 3.9 µs (8 in flight: 4.97, 6.06 and 6.25 µs one to three links away) |
| A command from the host over both islands of two cards | | 7.46 µs (split by the host, pipelined); 10.89 µs from a thread (crossed by its dispatcher) |
| A tile written into a rank's L1 (`km::write_l1`, example 32), every rank, only each island's last rank across | | 10.8 µs a command over two cards, 8.5 µs on a split card |
| The same, every rank's tile across | | 578.6 µs (118 tiles each way, about 1.7 GB/s each way) |
| A put line of one page, out and confirmed | | 5.1 µs |
| `km::read_bytes` of a range | 0.48 to 0.54 µs | 1.70 to 1.95 µs answered by the far island |
| Pages written from L1 into the other island's DRAM, both ways at once | | 72.5 GB/s each way (512 MiB) |
| `km::read_pages` of the other island's pages | | 40.6 to 48.3 GB/s |
| `session.cross`, DRAM to DRAM | | 100.4 GB/s (one link's ceiling 40.2 GB/s) |
| `kmx_fetch` of 64 MiB | 1.30 ms | 1.128 ms, 59.5 GB/s, of the other island's |
| A chain's mark to a rank of the other island (a put line and its confirmation) | | about 1.2 µs, one after another: a call waiting for every block of the call before (a covering range: "A covering range on islands" in [classes](classes.md)) over 32 and 40 ranks took 785.8 µs as a chain, 55.7 µs as the two calls alone |
| A dynamic run's message | | about 2.9 µs a delivery; 10^6 forwards 1,531.5 ms by deliveries, 307.4 ms with `hold_links=True` on one link |
| `km::all_gather` / `km::all_reduce` of 256 pages over two cards | return at once | 21.3 µs / 180.0 µs |
| A ring put of 32 pages of 4 KiB a rank from 16 ranks (2 MiB), a split card, 2026-11-17 | | 188 µs one link away (11.1 GB/s); two links away 1,085 µs (1.93 GB/s) on the 2 x 2, 1,185 µs (1.77 GB/s) on the doubled line of 3: one lead page CPU's rate |

A program that knows its islands crosses less: example 26 folds each island to its
leader and sends one part across (238 ranks summed in 165.9 µs a round), and a
tensor placed with the ranks that read it carries nothing over the links (example 01's
MAP on a split card: 201.1 µs reading its own island's DRAM, 306.9 µs asking across).

## What costs L1

What the session keeps on the cores is a row of every load's L1 table:

| Where | Row | Bytes |
|---|---|---|
| Every island with links | link cores, not ranks: `ceil(links / 2)` cores toward each neighbour, from the end of its cores | 119 or 118 ranks a p150a card instead of 120 |
| Every core | the kernel buffer: an island session opens at what its script's last run needed and an eighth more, else 160 KiB | against one card's 70,656 |
| Every core | the own words (each mover's ask and put lines, its put lines' words, its `km::put_bytes` batch) | 8,640 |
| Every core | the tensor table: 48 bytes an entry on two islands, 80 on four (32 entries by default), or kept in DRAM with 32 rows a mover cached | 1,536 on two islands; 4,800 cached |
| A link core | the link area, the put queue | 10,752; 84,032 |
| A link core | two chunks a link CPU, for its sends and its put ring | 192 KiB to 512 KiB each |
| A link core, over more than two islands | the forward rooms (8,992 bytes a room a level) and the far places | a line or ring of four 18,304 + 8,896; the 2 x 2 with wrap 36,544 + 8,896 |
| A rank, a split card or a program whose messages cross | the link area as a session tensor | 10,752 |
| A rank's overlay slot, for code added by `run.addon` | a part that calls a page read or write builds the far page path in | example 30's SCALE writer part 11,392 against one island's 1,920 (its 2,048-byte slot refuses it on islands) |

On a split card the link cores also run the program's kernels and hold its buffers and
slots; their put queue and chunks are a row of the table there, and a load that fits
only without the chunks is built without them ([the session](#the-session)).

## The options

- `graph=`: the islands' links as pairs, a pair given k times is k links
  ([the graph](#the-graph)); `islands.line_pairs`, `ring_pairs` and `grid_pairs` give
  such lists.
- `tree=`: a tree of the user's for the collectives, on the host (`tree=[parents]`) or in
  a kernel (`km::tree(low, high)`); by default each root's tree is the routes' tree
  ([collectives](#collectives-across-islands)).
- `Dynamic(hold_links=True)`: a dynamic run whose messages hold the links (a RUN of
  every rank of two cards), against the default deliveries that share the links with
  pages over any group.
- `// km: own island` on a line of a mover, or `Program(own_island_pages=True)` for every
  line of a program: a tt-metal page call that means the core's own island's bank,
  which an island load otherwise refuses ([a kernel's page reads and
  writes](#a-kernels-page-reads-and-writes)).

## The limits

What is still refused because two ranks sit on different islands. A host refusal raises
an error that names it at load or submit; a device refusal gives the command up, and its
give-up names the call's place (`bad_item` at `ask`, as `run.inspect` and the error
read it), not the call.

Not built yet (one island does it):

- `hold_links=True` with X280 ranks is refused at load (`X280 ranks with
  hold_links=True are not done yet`); with deliveries, island 0's X280 ranks and the
  cores of every island send to each other ([X280 ranks](dynamic.md#x280-ranks)),
  run between two cards across one link (a split card has no X280 ranks: its
  dispatchers take the work clusters). An X280 rank that gives up in such a run
  ends it with `MessagesGaveUp` and the X280 closes clean (it once needed a card
  reset: [the hazard](hazards.md#an-x280-ranks-give-up-in-a-run-across-islands)).
- A split card loads one `Program` of the kernel form: more than one `Program`, a plan
  or a list (programs in the runtime's form) and `run.addon` are refused at load and
  at the addon ([a card split into islands](#a-card-split-into-islands)).
- `Dynamic(hold_links=True)` runs on two cards only and over every rank: a split card or
  more than two islands refuse it at load, another group at submit.
- A split card resets no core of a given-up command: its part's cores end at their own
  wait limit and take no later command there. The stop ends cleanly when a give-up named
  each of them and lists them in the result (`result["ended"]`, a line an island).
- A split card refuses `run.submit(..., queued=True)` by name: no run has proven a
  queued command on a split card yet (`run.island(i).submit(..., queued=True)` queues
  on island i's ranks).
- The host's page helpers `session.ask_range`, `session.send_pages` and `session.cross`
  serve one holder and one island's ranks, a link apart: a range over two holders is one
  call a holder (`tensor.holders` names them), a far island one call a hop, or
  `session.read` and `session.write`, which reach any page.

Bounds of size (cores, L1, a field):

- A split card: at most four islands, a dispatcher an X280 cluster; so X280 threads,
  X280 ports and X280 ranks are refused at load there (the clusters are taken).
- When any island runs an X280 thread, every island's dispatcher takes the work rings'
  layout (no ring in use where it has none), so the upload regions and link areas agree:
  an island without a thread then has 74,240 B less upload region (522,112 B against
  596,352 B with example 09's link area).
- A page read across by `km::read_bytes` or `km::ask_bytes` is at most a link packet
  (`bad_item` at `ask`).
- With `hold_links=True` a dynamic message and its body cross in one link packet of
  4,352 bytes: `body_bytes` is at most 3,904 with 192-byte messages and 4,032 with
  256-byte ones (one island and the default deliveries: 4,096); a larger one is refused
  at load (`body_bytes ... over a link: at most ... bytes`). The way around: the default
  deliveries, which carry a body in its own slot.
- A `km::put_bytes` batch needs the page CPUs' rings: on a split card whose load fits
  only without the link cores' chunks, a batch to them gives up (`bad_item` at `ask`).

One difference in values, not a refusal: a reduce in groups of S may span the edge, and
its float sums are rounded in another order than on one island (integers and exact
combines give the same words).

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
grid_timeout=10.0, kernel_bytes=None, channel=256 << 20, slow_code=False,
host_reaches=None, card_dram=False, packet=None, message_links=False, graph=None,
tensors=None`, are optional limits and choices: `packet` is the bytes of a link packet,
whole 64-byte lines (default 8,192, from 4,352 to 15,232), `tensors` the tensor table's
entries (default 32, more as L1 allows: [tensors](#tensors)), `graph` the islands'
pairs ([the graph](#the-graph)), and `host_reaches` and `card_dram` are
described [below](#tensors) and under [a card split into
islands](#a-card-split-into-islands).
With one card it has the same calls over one island: no mesh and no fabric, the card's
own `Session` as `session.island(0)`, the graph `islands.one`, every page of every
tensor on the card, and each load the card's run as it is without islands. A program
written for islands runs on one card unchanged.

- **Open.** In this order: tt-metal's 1-D fabric is set, the cards open as one 1 x n
  mesh (each card's chip a 1 x 1 submesh with its own command queue), the X280 runtime
  opens once over the mesh, then each island gets a `Session` given its device
  (`session.island(i)`; `session.sessions` is every island's, in island order, the
  same `Session` n times on a card split into n islands). The cards must be the visible ones (`TT_VISIBLE_DEVICES` is
  set to them) and must not be held by another process.
  A process that opened cards as a mesh may not open a card alone, nor the
  reverse: the open is refused, and the other kind runs in a new process
  ([hazard](hazards.md#a-card-opened-alone-in-a-process-that-opened-a-mesh)).
  Sessions over the same cards open one after the other in one process, as on one
  card: a close closes each island's device before the mesh, so a closed session or
  a tensor still held keeps no device open, which tt-metal would refuse at the next
  mesh. Measured on cards 0 and 1, 2026-11-18: two sessions in turn, the first kept
  alive, each with its one command's 236 pages right
  (`experiments/islands/islands_calls/sessions.py`).
- **Ranks.** Flat and contiguous by island, in island order: island i holds ranks
  `session.graph.first(i)` to `first(i) + count(i) - 1`, its cores in core order.
  `session.ranks` is the total, at most 65,024 (below), a group's first
  below `KM_GROUP_LINK_BASE` (0xFE00), where link cores' names start (0xFF00 on: X280
  threads).
- **Close.** Every island's run is stopped through the X280 runtime, the islands'
  sessions are closed, then the mesh, and the fabric is set off. A grid that did not
  return keeps the mesh open, as a card's session keeps its device.
- **Links.** Cards are joined as the fabric joins them ([the graph](#the-graph)): two
  cards use every link the fabric found between them (four on two p150a cards);
  `links` 1 to 4 is an optional limit a pair, and `session.links` the fewest links of
  any pair. A neighbour's links take its island's next `ceil(links / 2)` cores from
  the end as link cores (`session.link_cores`, `graph.link_cores`), not ranks; both
  data-movement CPUs of a link core send, each on its own link (`session.channels`:
  each link's Ethernet channel, two slots a link core). More than two cards are
  built and checked on the host, not tested on three cards. Link 0 is the lines link: link core 0's CPU 0 holds it and sends the
  dispatcher's command and DONE lines (commands across), and pages only when it is
  the one link; links 1 on carry pages. A core's bytes into a rank's L1 on the other
  card (`km::write_l1`, `km::emit` across) go as a line of the put queue, its pieces by
  the page CPUs. Every packet of a line, a message or a forward leaves by the send the pages
  use (`LinkSender::send_stateful`; a header alone by `send`). The session checks that a DRAM bank sits at
  the same NoC place on both islands, and that both islands' core tables are the
  same. One card is an island of one: no link cores. The ranks an island has, on a
  card of 120 Tensix cores:

  | Links | Link cores an island | Ranks an island | Ranks on two cards |
  |---|---|---|---|
  | 1 or 2 | 1 | 119 | 238 |
  | 3 or 4 (the default on two p150a) | 2 | 118 | 236 |

- **Kernel buffer.** On islands with links the ranks and the link cores load
  different kernels from one program: the ranks the program's kernels with the client
  side of the page calls only (posting an ask, a put or an L1 write, a wait on its port), the link cores the runtime's link kernels (`kernels/link/reader.cpp`,
  `link/writer.cpp`, `link/compute.cpp`, built with `KM_LINK_SERVICES`): the page
  CPUs' service, the lines and messages. A rank still
  serves a `PAGES` command's wait (`wait_pages` may name a rank). The runtime's buffer
  holds each CPU's largest text over the two groups. A program with messages and a
  card split into islands keep the services in every kernel (see a link core's L1). The session opens at the kernel buffer its script's last run needed and an
  eighth more (at least 8 KiB), else at 160 KiB (`islands.parts.load.KERNEL_BYTES`: the largest such build, a dynamic program with a
  message service on every link, example 23's 159,360 bytes, in steps of 16 KiB); it
  costs every core 32 KiB of L1 more than 128 KiB did (1,367,936 bytes left for
  circular buffers and L1 tensors against 1,400,704). One card keeps the runtime's
  70,656 bytes and follows its kernels. A program that sets its own buffer keeps it.
  The
  load checks each build against the buffer: one that does not fit raises
  `KernelSizeError` with its size, the buffer and the most L1 allows, before anything
  is launched. `kernel_bytes` sets another size at the open; with less than 160 KiB,
  `message_links=True` falls back to link 0's service alone (logged).
- **L1 of a rank.** What the session keeps on every core is the load's L1 table rows
  `session: ...`: the own words (`KM_OWN_ALL_BYTES`, 8,640 bytes: each data-movement
  CPU's scratch for an ask and a put line, its put
  lines' words, its put port (`KM_OWN_PUTS`: count, want, in flight), the headers it builds for a piece of a put into the rings, and its
  batch of byte records for `km::put_bytes`, 2,048 bytes), the
  tensor table (48 bytes an entry on two islands, 80 on four, `tensors=` entries: 32
  by default, 1,536 B a core on two islands) and
  a dynamic run's message region. When the link cores run the link kernels (a kernel-form
  program, a class, a plan or a dict of classes), the link area (10,752
  bytes) is theirs alone: at the base of their circular buffers, zeroed by the host at
  the load (`write_l1`, just before the launch), the ranks' buffers launched apart; the
  link cores take no overlay slots. Otherwise (a program with messages, a split card)
  it is a session tensor on every core, a row of the table too. Before the first load
  on two or more cards, the own words, the island table and the tensor table are rows
  already ("made at the first load"), so a class made then gets the room a class made
  after a load does; on a split card its link area too (a session tensor there).
- **L1 of a link core.** A program's circular buffers, its overlay slots and a plan's
  switch area are launched on the ranks alone when the link cores run the link kernels,
  so none of a link core's L1 goes to them. What it holds on two p150a cards with packets of 8,192 bytes:

  | What | Bytes |
  |---|---|
  | The kernel buffer (every core's) | 128 KiB by default (`kernel_bytes`) |
  | What the session keeps (above: own words, tensor table) | as on a rank |
  | The run's control block, its share a core | 17,024 |
  | The transfer buffer (the host's transfers) | 32,848, or 131,152 with room |
  | The link area | 10,752 |
  | The put queue (`KM_QUEUE_BYTES`) | 84,032 |
  | Two chunks a link CPU (its own sends and its put ring) | 2 x 512 KiB |
  | Left (`LINK_CHUNK_MARGIN` at least) | 64 KiB or more |

  The chunk is at most 512 KiB, whole 64 KiB, two in what is left after the rows above
  and the margin, and at least the least that holds a put ring of 8 slots
  (`komorebi.session.kernels.link_chunk_least(packet)`: 192 KiB for 8,192-byte packets); below that the link
  CPUs have no chunk of their own and the puts go one line at a time. Two loads still
  run the program's kernels on the link cores, with its buffers, slots and link area
  there as on a rank: a dynamic program whose messages cross (link core 0 serves
  them with the program's message kernels) and a card split into islands (one launch
  for the whole card). Their link CPUs take two chunks of that least (no more) where
  they fit after the program's buffers, the transfer buffer, what the session keeps,
  the put queue and the margin, so a `km::put_bytes` batch has its ring. There the put
  queue and what follows it (the chunks included) lie above the slots on the link
  cores alone, and the load's L1 budget has a row for them ("link cores: their put
  queue"): the slots are fitted around it. On a split card a load that does not fit
  with the chunks is built once more without them (the log says "the link cores have
  no put ring: a km::put_bytes batch to them is refused", and such a batch is refused
  by name); one that still does not fit is refused by name with the table.
- **The put queue at one place.** A put line's sender writes its arrival into the far
  link core's queue at the place of its own, so the queue must sit at the same offset on
  every island. The link cores take no overlay slots when they run the link kernels:
  each island's ranks keep slots of their own size, and the queue is the first of the
  link cores' buffers. A load whose islands would put their queues at different places
  (offset, chunk) is refused by name (`check_link_queues`).
- A link core is named outside the ranks: a command's group with first
  `KM_GROUP_LINK_BASE + k` and count n names link cores k to k + n - 1 of the island
  that takes it (`graph.link_group(island, k, count)`). Such a group is never cut by
  island and never crosses; on a core, its rank is its link-core index less k. A link
  core is in no group of ranks. A timed-out command to link cores names them as
  `link core k (island i, core c)`.

## The run

`session.load(program, **options)` loads one `Program` on every island in turn (its
grid, then its dispatcher), or a list of Programs, one an island in island order, with
`Session.load`'s options and `island=(graph, i)`: each core gets the island table and
each dispatcher its island's range of ranks. It returns an `IslandRun`. The table has an
8-byte entry an island: its first rank and ranks, the next link toward it and the links
crossed (hops), and the mask of the ports (neighbour p of `Graph.neighbours`, bit p) on
any shortest path to it (`Graph.ports_toward`), the links bulk pages may take. The
table is a tensor in the L1 of every core, sized by the island count (its address in
the island header, `KM_ISLAND_TABLE`), so no constant of the cards bounds the islands:
`Graph` takes up to 256 (an island is a byte), and a load of more than 16 islands builds
with `KM_ISLANDS_MAX` set to its count (the link services' per-island arrays). The
island area keeps the port table instead (`KM_ISLAND_PORTS`): for each port (neighbour
p) and each of its links, the link core, its CPU, the far island, the port's links and
the far link core's tile.

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
- `run.submit(handler, cluster=n, args=..., result=True)` names an island's own X280
  thread by its flat number n (`KMX_THREAD(island, c)` = 3 x island + c, as a thread
  names one: [commands across islands](threads.md#commands-across-islands)); that
  island's run gives it to its work cluster c, one ticket of one part. A number no
  island serves, or a cluster of that island with no thread, is refused by name. One
  island's run, `run.island(i).submit(handler, cluster=c)`, names island i's own
  cluster c the same way.
- `run.submit(handler, count, args, queued=True)` takes a count, as one card's does, and
  sends the whole command to one island: the host picks the island with the fewest of
  this run's queued commands not yet done (the lowest of them on a tie) among those with
  `count` ranks, and that island's dispatcher queues it and chooses its cores in its
  own ranks, as one card's kernel does over the card. One ticket, one part;
  `run.group(ticket)` is `(None, count)` until the done event names the first core,
  then the flat group. A count larger than every island's ranks, or `exclusive=True`,
  is refused. The island is chosen without looking at where the command's tensors
  live: a page that the other island holds is asked for across a link, as for any
  command (`km::read_bytes` 1.70 to 1.95 µs answered there against 0.48 to 0.54 µs at
  home, [what costs time](#what-costs-time)). A command that reads much of one
  island's part runs cheaper on that island's ranks, submitted on a fixed group. The
  values equal one card's: 64 queued commands of 8 cores on two cards, each core adding
  to word 0 of its own page of a tensor split by rows over both islands, went 32 to
  each island with every page right, as on card 0 alone
  (`experiments/islands/islands_calls/queued.py`).
- `run.wait(tickets, timeout)` sends every island's queued commands, then feeds the
  islands in turn (one native round each: store what its ring has room for, take its
  events) until every part arrived, so the islands run their parts at once; all within
  the one timeout.
  A ticket completes when every part has; its result is the tuple of its parts' events
  in island order, which also reads as one card's event (`kind` "done" only if every
  part is, else the first other part's; `microseconds` the slowest part's; `reason`).
  A run loaded with `trace=` gives `run.trace().command(ticket)`, the ticket's parts'
  rows in flat rank order, each in its island's clock (`.island(i)`: that island's
  own trace, for phases and leads). `run.drain`, `run.stop` and `run.abort` act on
  every island within one timeout; `run.state()` lists every rank in flat order, each
  with its `rank` and `island`, as one card's lists every core.
- `run.result()` and `run.stop()` give one result block with one card's keys, so a
  script written on one card reads it unchanged (`result["error_name"]`); `.islands`
  holds each island's own block in island order (also `run.island(i).result()`). A key
  every island holds alike is that value. Otherwise: `state` and `state_name` are
  "done" only if every island's is, else the first island's that is not; the error
  and where it was (`error`, `where`, their `_name`s, `slot`, `sequence`, `req`,
  `seen`, `abandoned`, `x280_cores`) are the first failing island's, named by the new
  key `island` (None when none failed); the counts (`host_commands`, `commands`,
  `events`, `rejected`, `held`, `no_room`, `checked`, `bad_lines`, `results`, `idle`,
  `full`, `passes`) are summed, a ticket counting once a part; `ticks`, `last`,
  `boot_ticks` and `max_busy` are the largest (each island in its own clock); `busy`
  is every island's slots together (a bit set on any island); `cleaned` and
  `result_ready` hold if every island's does; `failed` and `ended` join each island's,
  each line starting `island i:`; `bad_first` is the first island's whose words did
  not match. Any other key that differs (`control`, `result`) is None: read
  `.islands[i]`.
- A tensor's own `read()` and `write()` are refused while a run is loaded, as on one
  card, naming `session.read(tensor)` and `session.write(tensor, pages)`, which read
  and write its pages live on every island.
- The run is addressed and reported by flat rank. An error from an island starts
  `island i:` and carries `error.island`; a `CommandTimeout` (a time-out or a give-up)
  names its ticket, and the cores it lists, where each waited and those reset and
  loaded again (`recovered`), by flat rank with the island and core as information:
  `rank 128 (island 1, core 10) reader waits in port`. A given-up command's
  `CommandTimeout` lists the cores of every island that held a part: each other
  island's part as its own dispatcher's inspect of the deadline read it. This covers a
  group sent through one island's `run.island(i)`, whose far part that island's
  dispatcher crossed. A `WaitTimeout`'s `pending` lists tickets.
- `run.inspect(group=None)` reads what each rank of `group` (first, count; default
  every rank) is doing while the run goes on, each island's part asked of its own
  dispatcher, as `Run.inspect` does; each core's dict also holds its `rank` and
  `island`.
- Recovery (`load(..., recover=True)`, the default; [recovery](../guides/recovery.md)):
  each island's dispatcher gets its own island's reset lists at load, one list a kernel
  group, so a list of programs works too (a core gets the list of the program it
  holds). A command given up on island i resets and loads again that island's cores
  still in it; the `CommandTimeout` starts `island i:` and its `recovered` lists
  their flat ranks. `run.reset_core(core)` resets the core of flat rank `core` and
  loads its kernels again; `run.recoveries` lists every recovered give-up as `{island, ticket,
  handle, cores, ranks, text}` (`cores`: the island's own core numbers); `run.recovery`
  reads as one card's (lists, bytes and seconds of every island, `cores` by flat
  rank; None when an island kept none) and `run.recovery.islands[i]` is island i's. On two p150a cards with the map/collect example,
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
- Programs of the runtime's form load as on one card: a list, or a dict of groups in
  flat ranks, is planned over every rank and each island runs its part, the runtime's
  kernels built with the island's defines (example 19, `--far`; example 16 `--cards
  0,1`, a dict whose group is cut at the island boundary, exact).
- `run.load(program, group)` on an `IslandRun` takes flat ranks and sends each island
  it touches its part, as `submit` does (one ticket). It loads only a program that
  some group of each of those islands holds: a program's parts are built against the
  kernels of a group that holds it (its buffers' formats are in them), so a program
  whose groups are all on the other island is refused with a sentence, before any part
  is sent. Example 16 `--cards 0,1 --phase live`: sum cut at the boundary, its island
  1 half switching to scale and widen and back 200 times, 600 commands exact, the load
  onto island 0 refused.
- `run.addon(program)` on an `IslandRun` adds a class to the live run on every island
  at once, all or none, with one program field (classes.md, "Another class added to a
  live run", item 8); its tensors may be made before the load or while the run lives
  (each later tensor's table row is written by the ranks, below). Its parts must fit
  the slots the load kept, which the loaded programs and the classes named ahead
  (`Program.ahead(program)`) size. An instance made later of a class named ahead (the
  same class and name, its tensors made after the load) takes the named one's own
  overlay numbers (`KM_OVERLAY_SHIFT`), so its parts build byte for byte as the load
  built them. A part over its slot is refused, naming both ways to keep more.
- `run.addon(name, parts=...)` on an `IslandRun` adds hand-written overlay code as on
  one card: every island's checks and build first (its number free there, its parts
  in the load's reserve), then each island's parts and entries, one overlay number on
  every island (island 0's lowest free, or `number=`), returned; a handler's body is
  then submitted over flat ranks. Each island's entry goes to its ranks alone (its
  link cores run no program kernels on two cards). A split card refuses it, as it
  refuses a class. Added code is larger on islands: its page calls build the far page
  path in, so a slot sized on one island may refuse it (example 30's SCALE writer part:
  1,920 bytes on one island, 11,392 on two cards, against the example's 2,048-byte
  writer slot); the refusal names the part, its bytes and the slot, and the loaded
  program keeps more with `Program.slots(bytes={"writer": n})`.
- An add (code or class) whose commit a later island's device refuses after earlier
  islands took it cannot be taken back there (an entry command only adds): the error
  names the island that refused and says so, and the overlay number, or the class, is
  marked unusable on every island: a command naming it is refused on the host (by the
  islands' run and by each island's own, `run.island(i)`), and
  the number is given out again on no island. Every check and build of every island
  comes before the first write, so a refusal on the host leaves nothing written.
- `run.send()` hands every island's queued commands to its dispatcher without waiting.
- `run.island(i)` is island i's own `Run`, the low-level door: its groups, inspect
  and `reset_core` take that island's own core numbers.
- A dynamic program (`Program.dynamic`) declared with `hold_links=True`, on two cards
  only (a split card or more than two islands refuse it at load): each core
  gets the message service's arguments and `KM_LINKS`, and its messages cross. Its RUN
  takes every rank (`(0, session.ranks)`; another group raises `ValueError`; X280
  ranks are refused at load). In `run.submit_many` such a RUN goes as `submit` in its
  row's place: the rows before it as one `submit_many`, then its PAGES command and the
  RUN, then the rows after it; the tickets are one range in the rows' order, and each
  island queues its parts in that order (its dispatcher takes them so; their events
  come as each part ends). Before the RUN the session queues
  a PAGES command of mode `KM_PAGES_MESSAGES` (a0: the RUN's group word) to link core 0
  of each island; its link 0 CPU serves the run's messages until the RUN's end, and the
  dispatcher's quiet test counts that core's line. With
  `Session(cards=..., message_links=True)` (opt-in: one run of three of example 05's 10^6 forwards over four links stalled at 81 % of its visits, cause not found yet) and a kernel
  buffer of 160 KiB or more (the default) every link has a service (link k:
  CPU k % 2 of link core k / 2, its own region, a second region on every core for the
  second CPUs; built with `KM_MESSAGE_SERVICES`): a rank's messages take link
  `rank % links`, so one rank's stay in order; with less, link 0's alone (logged).
  Example 05's 10^6 forwards: four services, 80.1 to 80.4 ms, every link about 62,500
  messages each way; one service (one link), 302.0 ms. Link 0's service alone decides the island
  quiet: its own outbox
  empty, then two passes alike over the RUN's cores and the other services' state lines
  (each empty in this run), balanced over every service's counts; the two islands'
  reports carry those sums. The other services write no status line; one that gives up
  makes link 0's give up. A body crosses in its message's packet (at most 4,224 bytes
  for 64-byte messages; the load refuses more): example 23, one service, 3.001 µs a hop
  with the message alone, 6.109 µs with a 4 KiB body; four services 2.990 and 6.056. The services grow the kernels,
  built off the card (four services; one service: 138,656 and 140,384): the forward
  example's two-card build 159,568 bytes, example 23's
  161,296 (on the card 157,632 and 159,360), the forward example on a split card 139,136,
  each inside the 160 KiB buffer; one card's builds do not change (44,880 and 43,040).
- By default (`hold_links=False`) a dynamic program runs over any group, on a split
  card, two cards or more islands: a message for a rank of another island is a delivery by the link service (`km_deliver.hpp`, `KM_DELIVER`):
  the writer hands the bin to its link core (op 5, confirmed), the lead page CPU pushes
  it into a staging slot of the far island's link core, which reserves on the rank's
  counter tile, writes the queue and commits by the mark; the links carry pages
  meanwhile, and the dispatchers' waves (`waves.h`) end the run. A direct pair
  (`direct=`) to a rank of another island goes by `far_write` (its body, the line with
  mark 0, then its mark) and its slot comes back by a posted add to the sender's credit
  word; the direct tables number the message ranks of every island. Example 23 by
  `direct()` (card 1, 2026-11-16): split in two 3.556 µs a hop without a body, 6.891
  with 4 KiB; in a line of three 3.900 and 7.406, exact.
- Which path a dynamic run takes is the program's: `Dynamic(hold_links=...)`. False
  (the default): deliveries, any group, the links shared with pages. True: the message
  service above, between two cards only, a RUN of every rank (another group is refused
  at submit, naming `hold_links`), and while it runs the links are its own (a page
  across waits). Example 05 on cards 0 and 1, one link (2026-11-16): 10^6 forwards
  1,531.5 ms by deliveries, 307.4 ms with `hold_links=True` (80.1 ms with four links
  and `message_links=True`); a group of 150 ranks with 64 pages across in the same run,
  by deliveries, 371.9 ms; all exact. Deliveries are slower because one lead page CPU an
  island sends them and the far one delivers each, about 2.9 µs a delivery.
- A put line's confirmation is its number + 1 with the holder island in the top byte
  (`KM_OWN_CONFIRM`): an island with two link cores numbers each queue alone, and a bare
  number from one queue met the wait for a line of the other (a delivery too many on a
  line of three; exact 3 of 3 since). At a dynamic run's deadline the dispatcher names
  the island whose wave did not come (`KM_EVENT_ISLAND` + it: the leader the lowest
  island that did not answer its round, another island the leader), and the error
  says "island N did not answer". A link core that gives up between commands while its
  island's run is open (its service of the lines and deliveries) writes its status
  line with the run's sequence, which the dispatcher reads every pass with the X280
  ranks': the run ends at once with `MessagesGaveUp` naming "link core k (island i)"
  and its state word's where and code, not at its deadline (not exercised on a card).

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
| `puts` | put lines of `km::write_pages` (64 bytes) and their confirmations (8 bytes) |
| `bytes` | every payload byte of the above |

A row also holds the put ring's counters of the link CPU, as cycles and turns (32-bit,
a difference is taken modulo 2^32): `ring_sending` (its turns that sent), of them
`ring_router` (waiting for the router's slots), and between its turns `ring_late` (the
next slot ready, the CPU elsewhere), `ring_filling` (reserved, not yet written),
`ring_empty` (not reserved) and `ring_turns` (the turns that sent: a count);
`Crossings.IDLE` names `ring_late`, `ring_filling`, `ring_turns` and `ring_empty`, and
`Crossings.CYCLES` those after `ring_sending` and `ring_router`;
"Where a put's time goes" below reads them.

`run.crossings(puts=True)` also gives each row its link core's put words, read in the
same command: `put_tail` (the put lines' numbers handed out), `put_taken` (the lines
each page CPU took), `put_done` and `put_confirmed` (lines sent across, and the far
island's lines taken here), `put_bad`, `queue_pending` and `queue_last`. Link core 0
holds the queue: its rows show every page CPU's count, and the other link cores' rows
show only their own. A tail ahead of every page CPU's count, unchanged across two reads,
means a line was numbered and never written: every later line waits behind it.

`after - before` is what crossed between two reads, so a command is bracketed by a
read before and one after it; `total(source, target)` sums the links. Each CPU of a
link core counts what it sends in plain counters of its link area (`KM_LINK_CROSS`,
`km_abi.h`); no other core counts anything. A read is one `PAGES` command of mode
`KM_PAGES_COUNTS` to each island's link cores, whose CPUs copy their counters into a
small DRAM tensor the host reads: it waits behind a command a link core runs, so while
a dynamic run's messages are served, its links' counts are read after its RUN. On one
island, `crossings()` has no row.

The first command of each (handler, group) on an island run is bracketed so, without
waiting: `run.submit` reads the counts before it queues the command and returns, and
`run.wait` (or `run.drain`) reads them again when it has the command's event; when
pages, asks or puts crossed the session's log then says so as a fact, `cross.cross on
ranks 0..39 moved 40 pages across the link (asks 0, puts 80); if that is not meant, pass
its tensors split=0, group=(0, 40) for cross.cross's group, or copy=True if every rank
reads all of it` (a gather is meant to cross; a weight placed over another group than
the handler's is not). The line names the hint that makes this command's reads local;
the default placement stays the first ranks' (see [Tensors](#tensors)). The first use costs two reads (a `COUNTS` command to each
island's link cores, each) and later commands of the same (handler, group) nothing.
`run.crossing_line` is `"log"` (the default), `"error"` (raised by that wait after the
command has run, its events on the error's `events`) or `"off"`; a run whose messages
cross is not read (its counts wait behind its RUN). A run loaded with `trace=` starts at
`"off"`, so that its trace holds the commands it was given: set to `"log"` or `"error"`,
each first use takes two of the trace's slots on every island. A command past the
trace's slots is refused by `trace.command`, which says how many slots the trace holds and
what took them. The counts are a link's, so the line
cannot name the tensor or the ranks that fetched across; commands in flight in the
same span are counted with it, and the line says how many (`with 2 other commands in
flight`).

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
| `split=d`, `group=(first, ranks)` | the group's ranks partition dimension `d` (its page steps: rows of a "row" tensor, tile rows or pages along a row of a "tile" one: k tiles a step with `page_tiles=k`, so `group` is sized in pages: a group sized in tiles leaves them on the first islands, and the uneven-split warning names it; proven on the card on one island and on two) by the rule of `km::partition`, and the part rank `r` works on is held by `r`'s island; `group` defaults to every rank |
| `like=other` | `other`'s hint applied to this tensor, so the same holders part for part |
| `copy=True` | every page on every island; only when asked, never by default |

`session.tensor(data, dtype, layout="row", split=None, group=None, like=None,
copy=False, page_tiles=1)` makes it and writes it; `session.empty(shape, dtype, layout, ...)` takes
the same hints and writes nothing; `t.read()` and `t.write(data)` do what a one-card
tensor's do, each island's part through its own channel. `islands.place(graph, shape, layout, split, group,
copy)` is the rule on the host: a `Placement` whose pages are `outer` blocks of `steps`
steps of the split dimension, `inner` pages a step, and whose `parts[i]` are the
(first step, steps) island `i` holds of every block. `placement.where(page)` gives the
island and the page's index in that island's part, `placement.runs(first, pages)` the
runs held at consecutive indices, `placement.part(i)` the pages of island `i`'s part in
its order. A split along a dimension other than the leading one gives an island pages
that are not one run of page indices: tile column 0 of every tile row on island 0 and
column 1 on island 1, for `split=1` over a group of two ranks one on each island.

Each island holds the steps its own ranks take, so when `km::partition` gives the
islands shares out of proportion to their ranks (168 tile columns over 236 ranks: one
to each of ranks 0 to 167, 118 on island 0 and 50 on island 1), a command that splits
the group by island reads the difference across the link. The session says so when it
makes the tensor: a log line and a warning with the shape, the hint, each island's
steps and ranks, and the steps (and pages) such a command would read across. The
spelling that keeps every step local is a group whose ranks each take the same count,
centred on the islands' edge, as `experiments/language/dense_class/linear_class.py`
does: `split=1, group=(edge - tiles // 2, tiles)` with `edge = session.graph.first(1)`.

A tensor with no hint, of fewer steps than the session's ranks, lies wholly on island
0, because its steps go to the first ranks. A command whose group straddles the edge
then reads every page of island 1's ranks across the link, one ask a page; the session
prints the placement warning, then "shift on ranks 100..135 moved 360 pages across the
link (asks 144, puts 144); if that is not meant, pass its tensors split=0,
group=(100, 36) for shift's group, or copy=True if every rank reads all of it". Placed
on the command's own group (`split=0, group=<the command's group>`, and `copy=True`
for a small tensor every rank reads), a command that reads only its own
island's pages costs what it costs on one island
([forms](../../experiments/forms/nn/README.md#where-a-two-card-commands-time-goes)),
cards 0 and 1, 2026-11-18, us a command:

| Command | One card | Group on island 0 | Straddling, no hint | Straddling, placed |
|---|---|---|---|---|
| 3x3 shifts (0/1 products, 36 ranks) | 11.6 | 12.3 | 569.6 (360 pages across) | 10.7 (0) |
| 3x3 conv (32 ranks) | 32.1 | 38.7 | 2,938.3 (1,184) | 137.2 (48) |
| layer norm (4 ranks) | 67.8 | 67.6 | 284.9 (576) | 67.1 (0) |
| texel fetch from DRAM, 8 x 1,024 single-page reads (8 ranks) | 580.7 | 700.5 | 13,875.2 (4,104) | 634.5 (0), texture copied |

A command whose group crosses the edge costs nothing measurable by itself (the placed
shifts and layer norm equal the island-0 ones; the placed conv still reads 48 pages
across, its input placed on the shifts' 36 ranks, not its own 32). The reads across the link are the cost:
the fetch's 4,100 single-page asks moved 17.1 MB in 13,875 us, about 1.2 GB/s, against
40.6 to 48.3 GB/s for bulk `km::read_pages` of the other island's pages (above). A page
asked for one at a time across the link is the slow form.

Rows a command needs from just across the edge (a conv's halo) cross whatever the
placement. The way to handle them is one step that brings them over in bulk, then
local reads: the tensors hold the halo rows twice, once on each island, and one rank
of each island reads the other island's halo rows with one `km::read_pages` and writes
them into its own. For the conv above: the step 11.6 us (16 pages in 2 asks), then the
conv 42.6 us with 0 pages across, 54.2 us in all against 137.2 us
([halo_exchange](../../experiments/forms/nn/README.md#a-halo-brought-over-first)).

Each island allocates only its part, in its own DRAM at its own bank address, so a
tensor that does not fit one card fits two when the hint splits it. A part that does
not fit raises `KomorebiError` naming the island and the most pages of that kind its
DRAM takes now ("island i: its part of N pages of shape (...) does not fit its DRAM
(...); at most M pages of this kind fit there now: split it over more islands' ranks
(`group`) or free tensors").

`tensor.address` is the tensor's reference word, what a script passes in a command and
a kernel hands to Komorebi's page calls. On one card it is the bank address, as without
islands. Over islands it is the index of the tensor's entry in the tensor table
(`km.Session(cards=..., tensors=n)` entries, 32 by default: `KM_TENSORS_MAX` splits the
word's values into entries below it and bank addresses above, in every kernel; a session
of n over 32 builds its kernels, link kernels and X280 threads with `KM_TENSORS_MAX=n`,
and those of 32 or fewer are built as without the knob. Every DRAM bank address lies
above the allocator's base, over 1 MiB, so n may go to 65,536. Where it takes less L1,
the table lives in each island's DRAM instead, a row a page of whole 64 B lines, and
the kernels are built with `KM_TABLE_CACHE=K`: the reader and the writer of each core
keep K rows each in L1 (32 by default) and read a row they lack from DRAM, about
0.33 us on the card (120 ranks at once as one alone). That costs a core 4,800 B on two
islands whatever n is against 48 B an entry flat, so the host takes it from 101
tensors on; `km.Session(..., tensor_cache=K)` forces it with K rows (a multiple of 4)
and `tensor_cache=False` keeps the table flat, and the load's L1 table names the form
it took (`tensor table, cached (32 rows a RISC)` or `tensor table, flat (n
entries)`). A word's set is the word mod K / 4, of 4 ways, the least recently used
taken: a kernel that looks up five words of one set in turn (such as 0, 8, 16, 24, 32
at K 32) misses on every lookup. `session.table_misses()` reads each core's reader and
writer misses since the load from the cores' L1 after `run.stop()`; a long run shows
its thrashing so only when stopped (no live word carries the count yet). Measured on
two cards (2026-11-16), `km::page_at` on five words in turn on every rank, 199 rounds:
98.8 cycles a lookup flat, 142.7 cached on five sets (hits: 44 cycles more, 6 misses
a reader for the run), 1,554 cached on one set (every lookup a miss, 1,001 misses a
reader, 236 ranks reading rows at once); the Linear class forced cached
(`tensor_cache=32`) ran in 66.7 to 68.8 us against 65.3 flat), which the session writes into every core of every island at the load; the word
`KM_ISLAND_TENSORS` of a core's island area holds the table's L1 address. An entry is
the aligned page bytes, the steps of the split dimension, the pages a step and a copy
flag, then each island's part: its bank address, first step and steps. On a core,
`km::page_at(tensor, page)` (`km_placement.hpp`) gives the island that holds the page,
its part's bank address and the page's index there, and `km::island_of(tensor, page)`
the island; a word that is not an entry (a bank address) answers this island, the word
and the page, so a kernel written this way runs unchanged on one card. A copy's page
answers the core's own island. A page call of a tensor word (`km::read_pages`,
`km::write_pages`, a stream's block) looks the entry up once for each run of pages
that one island holds at consecutive indices (`km::page_run`), not once a page.
`tensor.accessor_args()` is the same on every island.
A part's fourth word is its bytes in each bank (an interleaved buffer's pages over the
8 banks, rounded up, times the aligned page bytes), the bound of `km::SizedTensor`.

A whole model over islands holds more than 32 tensors at once: open the session with
`tensors=N`. Without it the 33rd tensor is refused with `32 tensors at once: the tensor
table is full (free one first, or open with a larger tensors=)`, and a `tensors=`
outside 1 to 65,536 with `tensors=N: 1 to 65536, the tensor table's entries ...`. At 310
entries the table is cached, 4,800 B a core on two islands, where flat it would take
14,880 B; each N over 32 builds its own kernels (`KM_TENSORS_MAX=N`), so a prebuild
opens with the same N.

A tensor (or a bank range) made while a run lives takes a free entry, and each island's
ranks write its row themselves before the call returns: the host writes the row into a
page of each island's DRAM (at the row's own offset mod 64), sends each island's ranks
one command of the service overlay (`KM_HANDLER_ENTRY` with `KM_ENTRY_TABLE`, a1 the
word, a2 the row's DRAM address), hands every island's command before it waits for any,
and returns the tensor once every island is done. No command can name the word before,
and every page call reads the table after an L1 cache invalidate, so a kernel on any
island sees the new row; a freed word keeps its row until a later tensor rewrites it.
The service overlay's reader part grows by 384 B in island loads (40,896 to 41,280 B
for example 26) and not at all on one card. Measured on two cards (2026-11-16): a
tensor made while a class's run lived took 3.7 to 6.9 ms to make (its parts, its
writes and its row); the class then wrote 40 pages of it from each island into the
other's part, and every page read back exact, again after it was freed and its word
taken by a second tensor. With a table in DRAM the host writes the row into that table
itself and the same command, a2 0, empties the word's slots in both caches of every
rank, so the next lookup reads the new row.

`session.bank_range(island, bank, address, size)` names `size` bytes of one DRAM bank of
one island from bank address `address` (a multiple of 64), such as an X280 thread's
region (`run.island(i).threads[cluster].regions.address(name)` gives its bank and
address): it takes an entry of the table as a tensor does and returns an object whose
`address` is its word and whose `free()` frees the entry (the bank's bytes stay);
`bank` is kept beside it. Its entry is the holder island, the bank, the bytes and the kind 2
(`KM_TENSOR_BANK`), then the holder's part with the bank address; under
`km::page_at` it names no island's pages. A kernel on any island reads and writes it by
that word with `km::BankRange`. A session of one island, or a split card's one DRAM,
makes no entry: the word is the bank address itself. `kmx_fetch` and `kmx_store` do
not take the word (a copy reads a word as an interleaved tensor's); an X280 thread
names its own region by its bank and address as before.

`session.write(tensor, pages, first=0)` scatters each island's pages into its part
through that island's own host channel, and `session.read` and `session.read_bf16`
gather them into one array; while a run is loaded, the islands' transfers go at once,
one host thread an island. A part that is one run of the tensor's pages moves from and
into the caller's rows themselves, without a copy of its own. An L1 tensor is read or
written live by its cores' handlers, so a live transfer whose pages reach a link core
(an island's L1 tensor over its whole grid) is refused before anything is sent, naming
the page, the core and the island: make the tensor over the island's ranks (`group=`),
or read it after `run.stop()`. A hint whose part does not
fit an island's DRAM raises `km.KomorebiError` before anything is written: the island,
the pages and shape asked, the bytes a bank is asked and has left, and the most pages of
that kind that fit there now (asking island 0 alone for 48 GiB on a p150a: refused in
35 ms, at most 16,688,832 pages of 2 KiB). Example 01's MAP on one group of the 238 ranks of two
cards, its input of 238 x 2 tiles placed with no hint, `split=0` and `split=1` over the
group (118, 2) (`examples/01_mapcollect/islands.py --whole --split 1 --group 118,2`;
2026-11-14, AICLK 800 MHz read idle before the job, the session's 1,350 MHz from its open
to its exit; host load 3.8 to 8.4, one run each):

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
run loaded, AICLK 1,350 MHz from the open: the 800 MHz once written here was read idle
before the job; host load 4.3, one run, every byte read back compared), GB/s
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
  (`KM_PAGES_SHAPE_LINK0`); the session then counts four page CPUs
  (`KM_PUT_CPUS`).
- 512 MiB from island 0 into a tensor of island 1 (`session.cross`, four links,
  packets of 8,192 bytes, the link cores' own chunks): 5.35 ms by the command's own
  ticks, 100.4 GB/s, every page exact (`experiments/islands/islands_pages`, 2026-11-15). The
  host's own wait adds about 0.45 ms whatever the size (64 MiB in 1.05 ms by the
  host's clock, 512 MiB in 5.85 ms): `pages.py --shapes` reports both.
- A kernel reads a tensor's pages wherever they live with one call (`km_ask.hpp`):
  `km::read_pages(reader, command, args, tensor, first, pages, to)`, with the
  tensor's accessor arguments, its reference word, and L1 address `to` of the core,
  page k at `to` + k aligned pages.
  The pages its island holds are read as `noc_async_read_page` reads them, at their
  index in its part (the caller's read barrier waits for them); on one island that is
  all the call does. Each run another island holds at consecutive indices of its part
  is one get line of 64 bytes in the link service's one queue, the queue put lines
  take too (`KM_QUEUE_*`, 512 lines, on link core 0 of the link toward the holder):
  the core reserves a line with a local atomic, waits until every page CPU took the
  line 512 before it, and writes it, its kind word (`KM_ASK_RING`) saying op 4, a read.
  The lead page CPU sends it, between commands, over its held connection into the far
  link core 0's asks, numbered there in the order sent. Every page CPU of the far
  island (every link's CPU when no message service holds link 0, else those holding
  links 1 on, or link 0 when it is the only link) serves its share of the line's
  pages (`partition(pages, j, page CPUs)`; link 0's CPU only in a line of 64 pages or
  more, a smaller line split over the others): it reads the line (across the NoC
  when it sits on link core 1) and takes it only once its last word checks (a line's
  last word is its number folded with its other words, so a copy read as the line
  lands is read again: [hazards](hazards.md#a-line-read-across-as-it-lands)), reads its
  pages from its DRAM, pushes them into the asker's L1, each batch adding its pages to
  the asker's port once landed, and sends back its own count of asks answered
  (`KM_ASK_SERVED` + 4 j). The asker waits on its port for all of them, in whatever
  order they land, bounded; a miss gives the command up at `ask` with the island asked
  in the word's top byte. 32 get lines are in flight a link; the lead sends the next
  once every far page CPU answered the one 32 before it, and the lines behind it in
  the queue wait meanwhile.
- A kernel writes a tensor's pages wherever they live with the counterpart, in the
  same header: `km::write_pages(core, command, args, tensor, first, pages, from,
  port)`, from the reader or the writer, page k from L1 `from` + k aligned pages. The
  pages its island holds are written as `noc_async_write_page` writes them, at their
  index in its part (the caller's write barrier waits for them); on one island that is
  all the call does. Each run another island holds at consecutive indices of its part
  is one put line (the same 64 bytes, `KM_PUT_*` in `km_abi.h`): each data-movement
  CPU of a core owns its put lines, two of them (a parity each), and writes the next
  once the far island confirmed the last one on that parity (its own words,
  `KM_OWN_PUT_*`); a local atomic gives the line its place in link core 0's put queue
  (`KM_QUEUE_*`, 512 lines, on the link cores alone). Where the page CPUs have rings
  (the link cores' own chunks), the putting core writes the pages and their headers
  into the page CPUs' ring slots itself, a piece of up to 16 pages at a time, each
  piece's last packet adding its pages to the far link core 0's count for that line
  once written; a page CPU's ring loop sends the ready slots and nothing else. Without
  rings every page CPU reads its share of the line's pages from the asking core's L1
  and sends them. The far island's lead page CPU confirms each line on its own, in
  whatever order its pages landed: the pages to the asker's port, then the line's
  number into the asker's own words. The call waits on its port, bounded, so `from` is
  free when it returns; a miss gives the command up at `ask` with the island in the
  word's top byte. `km::put_pages` sends the same line without waiting, its pages
  counted at the CPU's put port in its own words (`KM_OWN_PUTS`) until `km::put_wait`,
  so a read or ask across between them counts at the page port apart and passes on
  its own pages alone. A putting core that gives up after it reserved ring slots leaves
  them unfilled: the ring sends in order, so it stays stopped there until the cores'
  recovery or the next load.
- While a core waits in an ask or a put line, its live words (a give-up's "waits in")
  name the wait: `ask_confirm` (the far island's word for its line before on that
  parity), `ask_number` (link core 0's answer, the line's number), `ask_place` (the
  line's place in the queue: every page CPU took the line 512 before),
  `ask_slot` (room in a page CPU's ring), `ask_sent` (its writes leaving), `ask_count`
  (its port's count of the pages); the give-up word itself names `ask`.
- A page line refused before it holds anything (toward the core's own island, a size
  or an alignment that does not fit, a port out of range), under `guard` in an island
  build, gives up that command alone (`refused` at `ask`, as a deadline's give-up is
  recorded). The core's kernels do not end, so it is never reset. The command is given
  up at its deadline, the row names the refusal, and the core takes its next command
  and the stop's EXIT. Without the guard it is `bad_item`, and the core's kernels end.
- A command given up while a page line is out (under `guard`): each reader and writer
  counts the pages its lines asked or put and no wait counted yet (the word
  `KM_PORT_FLIGHT` of its page port and of its put port). At the given-up command's
  end it waits, bounded by the wait limit, until that count landed and its put lines
  were confirmed, with the pages kept in their buffers. Then it resets the port's count and want and the put confirmation
  words, so the next command neither passes early nor waits for a line that is not
  coming. A line not served in time stops the core rather than reuse a buffer the link
  core may still read or fill: the row names the command, the island, the first page
  and the pages left (live words `kLvLost*`, the tensor's bank address in the port's
  `KM_PORT_LOST_TENSOR`). A give-up between taking a line's number and writing the
  line is named ("took line N of the link core toward island I and never wrote it"):
  that link core serves its lines in order, so its later lines wait. The settle does
  not race the dispatcher's 20 ms after the abort: a core clears its live sequence
  before it settles, so the slot is freed and the host gets the `CommandTimeout` at
  once. The core takes its next command only after the settle, so that command starts
  late against its own deadline. A line never served ends the core's kernels at the
  wait limit (`limit` in `clear`), and the next command's time-out names the pages left.
  All of it is built only where a kernel can ask or put across (`KM_PAGE_LINKS`, the
  island builds): a one-card program pays nothing. The stop's EXIT,
  if every core it waits for reported or ended by its own give-up, is left done at its
  deadline (`KMD_ERROR_ENDED`). `run.stop` then raises naming those cores, not a time-out
  and a failed finish ([hazards](hazards.md#a-page-call-across-given-up-with-its-lines-still-out)).
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
  only. Pages that cross are at most one packet (the session's, 8,192 bytes by
  default). On one island, or for a
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
  four page CPUs and grouped reads 572.2 µs (220.2) and 597.2 µs (327.4); with put
  lines a CPU's own and packets of 8,192 bytes (every tile exact, what crossed as
  counted) the far read 606.5 µs (214.1) and the put 401.1 µs (split 342.6). Two
  tiles a rank are one or no page a page CPU: these are the lines' latency, not the
  links' rate.
- Pages at the links' rate (`experiments/islands/islands_pages/pages.py`, island 0's 118 ranks,
  or both islands' at once, four links, packets of 8,192 bytes, pages of 4 KiB, every
  page read back exact, 2026-11-15). The put is the linear kernel's: a core's run of
  pages from its own L1 (`km::put_pages` and `put_piece`, pieces of 4 pages into the
  rings). One link's ceiling, a plain sender from L1 slots at the same packet size
  (`experiments/islands/link_rate`): 40.2 GB/s (204 ns a packet, the router's pace), 155.6 on
  four links with a sender a core, 144.6 with two senders a core.

  | Move | 64 MiB | 512 MiB |
  |---|---|---|
  | Puts from L1 into the other island's DRAM, both islands at once, each way | 1.08 ms, 62.1 / 62.3 GB/s | 7.41 ms, 72.5 GB/s |
  | `session.cross`, DRAM to DRAM (host, by its ticks) | | 5.35 ms, 100.4 GB/s |
  | the same, by the host's clock | 1.05 ms, 64.1 GB/s | 5.85 ms, 91.7 GB/s |
  | `km::read_pages`, 128 pages a call | 1.95 ms, 34.5 GB/s | 13.2 ms, 40.6 GB/s |
  | `km::read_pages`, 256 pages a call | | 11.1 ms, 48.3 GB/s |

  The 64 MiB row was measured again on 2026-11-16, with the put queue first in the link
  cores' buffers (1.081 and 1.077 ms, every page exact). An earlier 1.00 ms, 66.9 GB/s
  has no log and has not been seen again; 2026-11-16 08:00 gave 61.6 / 62.3 GB/s.

  Where a put's time goes (512 MiB both ways, counters on the cores): a putting core
  spends 0.87 of its cycles waiting for its turn at a ring (its slots reserved, room
  shown), 0.07 writing its pages and headers into the slots, 0.04 reading its own
  pages, under 0.01 waiting for a put line or the run's last confirmation. A link CPU
  sends 0.33 to 0.48 of the time (about 180 ns of its own work a packet) and waits
  for the router's slots 0.22 to 0.32; between its turns, 0.15 its ring's next slot is
  reserved but not yet written and 0.08 to 0.15 it serves its other services (lines,
  answers, confirmations). The host's send through the same page CPUs reads DRAM:
  without the reads its loop does 117.7 GB/s (29 a CPU), with them 99.3.
- `kmx_fetch` and `kmx_store` of island 0's work cluster 1 thread on 16 cores, of a
  tensor of 4 KiB pages each island holds (`experiments/x280/bank_fetch/islands.py`, four
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
  local ones 51 µs and 1.30 ms. With packets of 8,192 bytes and the link cores' own
  chunks (every word exact): `kmx_fetch` of island 1's 64 MiB 1.128 ms (59.5 GB/s),
  1 MiB 115 µs; `kmx_store` 1.14 ms and 112 µs.

## A kernel's page reads and writes

A kernel reads and writes a tensor's pages through Komorebi's calls, never tt-metal's
own page calls: a page read through `noc_async_read_page` reads its own island's DRAM
only, so a kernel that calls it works on one card and reads the wrong pages over
islands; a load of islands refuses a mover's source line that names one (the file and
line), unless the line says `// km: own island` because the call means the core's own
island's bank (`Program(own_island_pages=True)`: every such call of the program). The calls
(`km_ask.hpp`) take the tensor as one object:

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
| `km::put_bytes(core, command, tensor, page, offset, bytes, source)` | exactly `bytes` bytes at byte `offset` of page `page` become the bytes at L1 `source`; nothing else of the page is touched, so words other cores write into the same page stay. A page this island holds: one NoC write (`source` kept until it is written). A page another island holds: a record (the page, the offset and the bytes, 16 bytes of head and the bytes padded to 16) appended to the CPU's batch (2,048 bytes in its own words), `source` free on return; the batch is sent when the next record does not fit, when a record is for another island, or at `km::put_wait`: one put line, the batch as one ring piece onto its landing page in that island's DRAM (a page for each holder, source island and put line owner: `n * n * 512` pages of 2 KiB, n² MiB on each island of n islands, one copy on a split card; a session whose DRAM cannot hold it is refused by name, "the put batches' landing"), and one confirmation once that island's lead page CPU has written each record with one plain NoC write at its page's address plus the offset. One batch of a CPU is in flight at a time, so one CPU's records apply in the order it put them. A range of up to 2,000 bytes is one record (a longer one, several). Does not wait; the batch is sent only when full or at `km::put_wait`, so a command ends with it (a batch left by an earlier command is dropped; under `KM_GUARD` a body that returns with its batch unsent gives up by name, `bad_transfer` in `putunsent` with its handler). Across, the page CPUs need their rings and the range ends below 1 MiB of its page (else `bad_item` at `ask`). Offset and bytes are multiples of 4 and `source` has the offset's low 4 bits (a NoC write's two ends share them); a range that leaves its page, or breaks these, is refused (`bad_transfer` at `ask`). Two puts to the same bytes from different cores: no order. False: gave up |
| `km::write_bytes(core, command, tensor, page, offset, bytes, source)` | `km::put_bytes`, then `km::put_wait(core, command)`: the bytes written on return |
| `km::put_wait(core, command)` | sends the CPU's batch, then waits until its NoC writes are written and every batch it sent was confirmed by the other island (its records written there). False: gave up |
| `km::read_bytes(core, command, tensor, page, offset, bytes, into)` | exactly `bytes` bytes at byte `offset` of page `page` land at L1 `into`, and nothing else of L1 is written; returns once they have. A page this island holds: one NoC read. A page another island holds: one get line (64 bytes); that island's page CPUs take these lines in turn, read the range from DRAM and write it into `into`: the words before `into`'s first 16-byte line and after its last, one inline word write each, then the 16-byte lines between as one write that adds 1 to the CPU's page port. Offset and bytes are multiples of 4 and `into` has the offset's low 6 bits (a read from DRAM has the same low 6 bits at both ends on Blackhole, and pages start 64-aligned); a range that leaves its page, or breaks these, is refused (`bad_transfer` at `ask`). Across, any number of links away (the get line forwarded by the link cores of each island between, the range sent back by the holder's page CPU the same way) and a page a packet holds (else `bad_item` at `ask`). False: gave up |
| `km::ask_bytes(core, command, tensor, page, offset, bytes, into, asked)` | the same, not waited for: a range asked across adds 1 to `asked`. Any number may be in flight, each into its own `into`; the queue holds 512 lines and 32 gets a link are across at once, and an ask past them waits. `km::bytes_wait` before the CPU's next page call |
| `km::bytes_wait(core, command, asked)` | waits until the CPU's NoC reads have landed and its page port has counted `asked` (then 0). False: gave up |
| `km::SizedTensor(word, page_bytes)` | an interleaved session tensor (its reference word) in pages of `page_bytes` given at run time, such as a command's; every call above takes it as it takes a `km::Tensor`. Page `i` lies in bank `i` % 8 at `i` / 8 x `page_bytes`. Over islands the tensor is a copy (each island its own part) or held whole by one island (a `group` of one island's ranks); refused (`bad_item` at `ask`): a tensor split over islands, `page_bytes` not a multiple of 64, a page past its part's bytes in each bank. A page call across takes pages a packet holds; a byte call across goes as the range's own place in its bank, so its page may be larger than a packet. Built for one island it is the bank address and each call compiles to the NoC call written by hand with `InterleavedAddrGen`, unchecked |
| `km::BankRange(word, bank, page_bytes)` | a range of one DRAM bank (the word of `session.bank_range`) in pages of `page_bytes` (a multiple of 64): page `q` at its bank address + `q` x `page_bytes`; every call above takes it. Over islands the bank and address come from the entry (a word that names no entry is a bank address in `bank` here); a page past the range's bytes is refused (`bad_item` at `ask`). Across, a page call is one line a page. Built for one island the word is the bank address in `bank`, read with `get_noc_addr_from_bank_id` |
| `km::Stream` | a stream over a `km::Tensor` reads and writes through these calls ([Movement](movement.md#a-stream-over-islands)) |

A program loaded on one island (a card's own session, or an island session of one
card) is built without the tensor table (no `KM_TENSOR_TABLE`): the word is the bank
address, and each call compiles to exactly tt-metal's page call it replaces,
`noc_async_read_page` or `noc_async_write_page` once a page. Over
islands each page is found by `km::page_at` and the calls are the forms by word above
(`km::read_pages(core, command, args, word, ...)`). The reader asks with the link
area's own line and answer words; the writer with its own block after the reader's,
both into the one queue. Without page links (`links=0`) a page another island
holds gives the command up (`bad_item` at `ask`).

The examples and projects whose kernels read or write a tensor's pages use these calls:
02, 03, 04, 06, 07, 09, 11, 15, 16, 17, 19, 21, 22, 23, 26, the dense project's
resident, streamed, grid and K-split forms and its SwiGLU layer (each operand by its
own word), attention, the core reset experiment's reader and the tutorial's lessons
from 16 on. Example 09's writer writes single words of its output, each with
`km::put_bytes` wherever its page is, then waits with `km::put_wait`. Measured on two
p150a cards, one page link, 2026-11-15:

- Example 09's `islands.py` (65,536 keys, 4,096 buckets, island 0's 118 ranks and its
  thread): the output held by island 0, PLACE 0.128 ms; held by island 1 (every key a
  32-byte record, 1,062 batches across, 61.7 keys each, a line, a ring piece and a
  confirmation a batch), PLACE 5.99 ms, 0.091 µs a key for the island, both exact.
  The same with a line of its own for every key (12 bytes inside the line) was
  112.9 ms.
- `experiments/islands/islands_bytes` (four links): island 0's 118 ranks each read 64 ranges,
  each at a page and offset of its own, of a 1 MiB tensor island 1 holds (4,096-byte
  pages), of 4, 64, 128 and 2,048 bytes, one in flight and 32: every word exact and
  the 16 bytes either side untouched. Island 1 answers a range in 1.70 µs (4 bytes) to
  1.95 µs (2,048 bytes) however many a rank keeps in flight, so at this load a rank
  waits 190 to 220 µs a range; read from island 0's own DRAM, a range takes 0.48 to
  0.54 µs one in flight and 0.18 to 0.30 µs with 32.
- `experiments/islands/islands_bytes/forms.py` (four links): a 1 MiB tensor island 1 holds,
  read and written by all 236 ranks of both islands with `km::SizedTensor`: two pages
  a rank in pages of 1,024 and of 192 bytes (`km::read_pages`, then `km::write_pages`
  xor a mark, 0.51 and 0.50 ms a command), and four 128-byte items a rank of pages of
  8,192 bytes (`km::ask_bytes`, `km::put_bytes`, 0.92 ms); and 32 KiB of bank 6 of
  island 0 (`session.bank_range`, `km::BankRange` in 64-byte pages) of which each rank
  reads two pages and 40 bytes and writes the pages back (1.29 ms). Every word the
  cores checked was exact (183,608), and so was every word of both tensors read back
  by the host, the other banks of the range's tensor unchanged.

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

The link kernels built for size were measured and are not kept: their text sets an
island's kernel buffer for every core, and smaller text cost the link rate (two cards,
example 12 as a rank, 2026-11-16):

Smaller text cost the link page CPUs' loops even with the put line's four kept at O2:
the helpers they call stay at -Os and are not inlined into an O2 caller.

| Link reader / writer built | Text (B) | Example 12's kernels | Puts both ways, 64 MiB | Gather, 2 / 272 pages |
|---|---|---|---|---|
| O2 (kept) | 50,724 / 42,632 | 97,152 B | 1.00 ms, 66.9 GB/s | 12.74 / 22.62 µs |
| -Os | 36,052 / 30,252 | 70,096 B | | 13.56 / 25.34 µs |
| -Os, the put line's four loops O2 | 38,640 / 32,832 | 75,264 B | 1.15 ms, 58.3 GB/s | 13.18 / 24.76 µs |

An island session opens at the size its script's last run needed and an eighth more
([the kernel buffer](host.md#the-kernel-buffer)), so example 12 opens at 110,464 B and
has 53,376 B more a core than at the fixed 160 KiB.

## The graph

The islands' links are a graph given as island pairs: `graph=[(0, 1), (1, 2), ...]`,
a pair given k times (either order) is k links. The helpers `islands.line_pairs(n)`,
`islands.ring_pairs(n)` and `islands.grid_pairs(rows, columns, wrap=True)` return
such lists (`links=k`: each pair k times); `session.graph.pairs()` gives a session's
back, and a script's flag "0-1,1-2" becomes one by `islands.parse_pairs`.

- **Cards** take the graph the fabric reports at the open: each pair of cards it
  joins directly, as many links as it found (at most 4, and `links`). `graph=` takes
  a part of it, each pair checked against what the fabric has (refused by name past
  it). The mesh opens as 1 x n, so the fabric reports a line.
- **A split card** (below) joins every pair once by default (`links` wires each);
  `graph=` joins the pairs given. `links` and `graph` are not given together.
- **Refused by name** only what cannot be wired (`check_wired`, and the graph itself):
  a graph that is not connected, a pair of one island, an island with more
  neighbours than a card's four ports, more than 4 links a pair (a byte each in a
  PAGES command's link word).
- **Links a pair.** Each pair uses all its links; `session.links`, the fewest any pair
  has, is the island's page CPUs a pair (from link 1 when a pair has more, so link
  core 0 holds one).
- **Link cores.** An island's link cores are in neighbour order, `ceil(links / 2)`
  toward each neighbour (`graph.link_cores_toward(island, neighbour)`), a link a CPU;
  the PAGES command toward a neighbour carries that neighbour's channels alone.

## A card split into islands

`km.Session(card=a, islands=n, graph=None, card_dram=False)`, n 2 to 4, returns the
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
  most four islands, and each island past the first takes a work cluster: X280 threads
  on a split card are not supported (the session refuses them).
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
- **Wires.** The pairs of the graph are joined by wires (default every pair, `links`
  wires each, default 1): link cores taken from the end of each island's rows, a
  neighbour's links ceil(links / 2) cores in neighbour order, named and served as
  between two cards (commands across on link 0, a core's own page ask, put and L1
  write, messages, `crossings()`). The build defines `KM_WIRE`: the link's transport
  (`km_transport.hpp`) writes the far tile over the NoC instead of handing the packet
  to a router, each count or word after the writes before it; `KM_WIRE_DELAY` (ticks
  of the wall clock, default 0) holds each packet back to imitate a slow link. Every
  link core is found by the island table's next link.
- **Any graph.** `graph=islands.line_pairs(n)` joins only neighbouring islands (0-1,
  1-2, 2-3; the islands' rows are in that order), as cards in a row would be: island 0
  reaches island 2 through island 1, the table's hops the islands apart.
  `islands.ring_pairs(n)` closes the line (of three islands, the triangle);
  `islands.grid_pairs(2, 2)` is four islands as a 2 x 2 grid with wrap (islands 0 1 /
  2 3, each pair two wires, the direct one and the wrap, so every island has four
  links); a star `[(0, 1), (0, 2), (0, 3)]` or three wires to one neighbour and one to
  the next are as good. Refused only as [the graph](#the-graph) says.
- **Forwarding.** Over more than two islands the build also defines `KM_FORWARD` (the
  session's; two islands build without it and their kernels stay smaller). A link core
  that receives what is not for its island sends it on by its island's next link
  toward the target: to the island's link core toward it over the card's NoC, which
  sends it across. No other core of the island between does anything. Forwarded: a
  command's line and its DONE, a get and the pages that answer it (a get is queued
  again in the queue of the link core toward the holder; the pages go back to the
  asker), an L1 write (a reduce's fold, a broadcast's blocks) and an island's report
  at the end of a run (below). Each link core holds forward rooms in its own buffer,
  after its put queue (`KM_FORWARD_*` in `km_abi.h`): a room of 2 slots of 4,480 bytes
  for each writer (the far link core across, each other link core of its island), each
  sending CPU and each hop level, with credits back. A packet on its h-th link waits
  only for room of level h + 1, so no circle of links can wait on itself; the levels
  are the graph's diameter less one. A message is not forwarded
  packet by packet: the link core that takes it in queues it in the outbox of the link
  core toward its rank, as a writer does. `crossings()` counts every link a crossing
  uses. A put (`km::write_pages`) to an island two or more links away (`KM_FAR_PUTS`,
  built when the diameter is over one) is sent by the asker's link core: its pages
  forwarded to the holder's island, then its line into one of `KM_FAR_LINES` (8) far
  places a source island there, with credits back; the holder confirms to the asker
  as for a neighbour's. A ring put (`km::put_pages` with a run) two links or more
  away is such a line a piece, each waited for, so it moves at the far put's rate of
  one lead page CPU, not the ring's. A `km::put_bytes` batch two links or more away is
  one far line whose page is the batch: it lands on its own page of the holder's
  landing (a page a holder, source island and far place: `n * n * 8` pages of 2 KiB
  past the neighbours' pages, n islands), and the
  holder's lead page CPU writes its records before it confirms; the putting CPU waits
  for that confirmation before it takes its next record. On a split card
  the last link's write is made by that link core's CPU 0 (the one that takes the
  rooms' packets on), so a forwarded page's DRAM address is built for that CPU's NoC
  (a bank's NoC place differs between the two NoCs).
- **L1 of a link core.** The forward rooms and far places, beside the put queue and
  the chunks: a line or ring of four 18,304 + 8,896 bytes, the 2 x 2 grid (two wires a
  pair) 36,544 + 8,896, a 4 x 4 grid with wrap 108,288 + 8,896, a 4 x 8 grid with wrap
  (32 islands, diameter 6) 180,288 + 17,792 with one wire a pair, 360,576 + 17,792 with
  two. Ranks hold none of it, and a session of one card or two none at all. A level
  costs 8,992 bytes a room (a writer and CPU: 2 slots and its words), so the largest diameter is the
  one whose rooms fit a link core's L1; a load past it is refused by name (L1).
- **Messages over more than two islands.** A dynamic run over more than two islands
  (a line, a grid, a split card) takes the deliveries (the default, `hold_links=False`,
  below): a message for a rank of another island is a delivery by the link service,
  whichever island the rank is on, and the dispatchers' waves end the run.
  `hold_links=True` (the message service, every rank) runs between two cards only: the
  load refuses it on more than two islands or a split card, by name ("runs between two
  cards only").
- **Commands across.** A dispatcher sends a command to every island its group
  reaches, over as many links as the way takes; each island writes its own part of a
  far dispatcher's inbox, a part of 16 packets for each other island, so a dispatcher
  keeps 8 commands across in flight with any number of islands
  ([commands across islands](threads.md#commands-across-islands)).
- **Not on a split card.** More than one `Program` (a list or a dict of programs, and
  programs in the runtime's form: the load refuses them), X280 threads, ports and ranks
  (not supported: the load refuses them), a reset list for a core (`run.recovery_refused`
  says `a card split into islands`: a given-up command is not recovered by a reset and
  a core that does not leave it ends the run), queued commands (refused by name: not
  proven on a split card), and engine streams and graphs (not checked on a split
  card).

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

The same on card 0 split as a line (`graph=islands.line_pairs(n)`), 2026-11-15, the far island at the
line's other end (two links from island 0 on three islands, three on four), every
output exact, every kernel ended with no error:

| What | 3 in a line | 4 in a line |
|---|---|---|
| Example 01, MAP on island 0's ranks reading its own DRAM / reading the far island's tiles by each core's ask (µs a round) | 215.96 / 376.63 (47 ranks) | 225.45 / 339.29 (35) |
| Example 01, what crossed each link of the way, one round each core | asks 141 out, pages 282 back (577,536 B), on both links | asks 105, pages 210 (430,080 B), on all three |
| Example 26, the sum over every rank (µs a round) | 183.22 | 194.17 |
| Example 25, NOP from island 0's ring to its ranks / to the far island's (µs) | 3.021 / 7.376 | 3.088 / 11.300 |
| Example 03, 8 broadcasts of 8 blocks over a 6 x 10 rectangle (µs a round, 480 checksums) | 820.62 | 931.09 |
| Example 05, 16 messages of 4 forwards, the parts, island 0 first (µs) | 179.4, 141.3, 107.8 (medians of 12 runs) | 216.9, 184.3, 147.1, 115.6 |
| Example 05, messages on the links, against the replay's link crossings | 56 of 56 | 75 of 75 |

What a hop costs: a NOP's command and its DONE to an island one link away take 3.9 to
5.1 µs (the mesh rows above), two links 7.4 µs and three 11.3 µs, so each island
between adds 2.3 to 3.9 µs. Those were with 3 and 2 commands across in flight; with 8 (2026-11-16,
example 25's `hops` step, pipelined NOPs from island 0's dispatcher on a line of four):
4.97 µs one link away, 6.06 two, 6.25 three. Example 01's 141 asks of a round, each answered by two
pages, took 160.7 µs more from two links away than from island 0's own DRAM, against
41.6 µs from one link away in the mesh of three. The end of a dynamic run over islands
(example 05, island 0's part, which holds the end): 146.6 µs over two islands, 179.4 over
three in a line (the median of 12 runs, 169.9 to 200.9), 216.9 over four. The kernels with forwarding built in, beside the 128 KiB
buffer a session took by default then: example 01's 99,280 B, example 05's 123,216 B (over two
cards, without forwarding: 92,576 B and 109,360 B, built off the card).

A NOP on one island of the split card costs what it costs on the whole card; a group
over both islands costs two commands of host work, as on two cards. MAP on (0, 120):
120 tiles bit-exact on both. With `--hang`: an inspect of island 1's core 70 during a
HANG on ranks 40 to 119 named its reader waiting for a port arrival; the deadline gave
the command up, `CommandTimeout` named island 1, and the run went on; 16 pages of
island 1 read inside the run were exact.

## Collectives across islands

Four blocking calls over the islands, each on pages of a copy tensor (`copy=True`),
called by every rank of a command (a reader or a writer, one of them):

| Call | Before | After |
|---|---|---|
| `km::all_gather(core, command, tensor, first, pages)` | island c wrote its share, `partition(pages, c, islands)` | every island's copy holds every share |
| `km::all_reduce(core, command, tensor, first, pages, scratch, kind, buffer, room)` | each island's copy is its partial | every island's copy holds the sum |
| `km::reduce_to(core, command, tensor, first, pages, scratch, kind, buffer, room, root)` | each island's copy is its partial | island `root`'s copy holds the sum; the others hold partial sums |
| `km::broadcast_from(core, command, tensor, first, pages, root)` | island `root`'s copy holds the pages | every island's copy holds them |

`kind` is `km::Sum::Int32` (modulo 2^32), `Float32` or `Bfloat16` (each pair added in
float32 and rounded to nearest even); `scratch` is a copy tensor of ports x pages of the
same page size (ports: the most neighbours an island has), `buffer` and `room` an L1
buffer of two pages or more. From the host, `run.all_gather(tensor)`,
`run.all_reduce(tensor, scratch)`, `run.reduce_to(tensor, scratch, root)` and
`run.broadcast_from(tensor, root)` (the kind from the tensor's format) send GATHER and
REDUCE to every rank, served by the runtime's service overlay with the same calls. On
one island all four return at once. `all_reduce` and `reduce_to` refuse, with
`km.KomorebiError`, a tensor of another format than `int32`, `uint32`, `float32` or
`bfloat16` ("all_reduce adds int32, uint32, float32, bfloat16: not ...", "reduce_to adds
['bfloat16', 'float32', 'int32', 'uint32']: not ..."), a scratch whose pages differ in size from the tensor's ("the
scratch's pages differ in size"), and a run whose transfer buffer holds no page of
the tensor ("the run has no transfer buffer of a page"). In a class's chain, `km.all_gather(copy, first,
pages)` between two calls is a record of its own, its leaders the same as these
([a gather in a chain](chain.md#a-gather-in-a-chain)).

Each call runs over a tree of the islands: by default the routes' tree of each root
(island i's parent toward root r is the island across its next link toward r, the same
table every core holds; its children are the neighbours whose next island toward r is
i, a mask a root in the island table). Inside a call the first of each island's ranks
of the command leads; the others meet it, share its adds and wait. All-gather: share c
travels down island c's tree, each leader putting it into its children's copies as soon
as it has it (a put line from DRAM a child), counted in the leader's counts table, a
word an island after the island table. All-reduce: share c of the pages is summed up
island c's tree (each island adds its children's partials, put into its scratch at the
port they arrive by, into its own copy, then puts that into its parent's scratch), then
travels down the same tree: every island roots the tree of its own share, so the adds
and the crossings spread over the islands and links. `reduce_to` sends every page up
the tree of `root` and stops there; `broadcast_from` sends every page down it. A
leader takes the shares farthest from their root first on the way up and nearest first
on the way down, so no leader waits on one that waits on it.

A tree of the user's: `km::tree(low, high)` as the last argument of `km::all_reduce`,
`km::reduce_to` or `km::broadcast_from`, island i's parent in the 4 bits at 4 i of
`low` (islands 0 to 7) and `high` (8 to 15), the root its own parent; on the host
`tree=[parents]`, checked there (a parent an island, each one link away, one root, no
loop). Every page then goes up that tree to its root (and down it, for the all-reduce);
`root` must be the tree's. `km::all_gather` keeps the routes' trees: each share starts
at its own island. On the ring of 4, the chain `tree=[1, 2, 3, 3]` (3 links deep where
the routes' tree of island 3 is 2), 64 pages, every copy exact from a kernel and from the
host: broadcast_from 77.5 µs (the routes' tree: 51.4), reduce_to 629.7 µs (606.5), the
all-reduce 704.6 µs (331.8: one tree up and down, against four trees of a quarter each).

Measured on two p150a cards joined by four links, 2026-11-15, AICLK 1350 MHz, the
median of 10 calls, every copy exact, the call's time on the card (submit to done, its
slowest island) and on the host (from the call to its return):

| Call | Pages (float32 tiles) | On the card | On the host |
|---|---|---|---|
| all-gather | 2 | 53.7 µs | 322.9 µs |
| all-gather | 256 (1 MiB) | 62.9 µs | 326.6 µs |
| all-reduce | 2 | 134.6 µs | 400.1 µs |
| all-reduce | 256 (1 MiB) | 222.1 µs | 491.8 µs |
| all-gather | 4,096 (16 MiB) | 132.4 µs | |
| all-gather | 16,384 (64 MiB) | 381.7 µs | |
| all-reduce | 4,096 (16 MiB) | 1,488.3 µs | |
| all-reduce | 16,384 (64 MiB) | 5,556.8 µs | |

Over the trees (2026-11-16, AICLK 1350 MHz, every copy exact): two cards, all-gather of
256 pages 21.3 µs, all-reduce 180.0 µs (examples 27 and 28, the same as along the line);
card 0 split in 3 (a line), 256 pages: all-gather 84.7 µs (the line form: 158.7),
all-reduce 959.4 µs (978.2); 64 pages on a split line of 3 and a ring of 4
(`experiments/islands/islands_tree/pages`), from a kernel and from the host alike:

| Islands | all-gather | broadcast_from | all-reduce | reduce_to |
|---|---:|---:|---:|---:|
| 3, a line (root 2) | 30.7 µs | 51.7 µs | 328.5 µs | 330.0 µs |
| 4, a ring (root 3) | 27.0 µs | 51.4 µs | 331.8 µs | 606.5 µs |

`reduce_to` on the ring adds every page at the root, from both its children: the
all-reduce spreads the same adds over four roots.

The rows of 16 and 64 MiB are from a kernel, the median of 3 calls, each island's copy
checked by sample (the first, middle and last 8 pages of every share)
(`experiments/islands/islands_collectives/account`). An all-gather of 64 MiB moves 32 MiB each
way in 381.7 µs, 87.9 GB/s each way; from 16 to 64 MiB, 25.2 MB more each way take
249.3 µs, 101 GB/s each way (the links' rate the host's sends reach, 100.4 GB/s): the
bytes cross at the path's rate, and the rest, about 49 µs, is the fixed part of 2
pages. An all-reduce adds 80.8 µs a MB of tensor (12.4 GB/s): its two crossings at that
rate take about 0.66 ms of the 5.56 ms at 64 MiB, and the adds, about 4.8 ms (each
island adds half the tensor over its 118 ranks, about 90 cycles a float32 word with
its DRAM reads and writes), are the rest: the adds in integer arithmetic on the
movers are the reduce's largest term, and ours.

Where the time goes, from the calls' step stamps (`--stamps` of the account
experiment: each rank's wall clock at each step of the call, compiled in on request),
the host's form, two cards, the median of 3 calls (43.2, 52.3, 124.3 and 210.4 µs),
microseconds of the island that finishes last, each column summing to the call:

| Term | Gather, 2 pages | Gather, 1 MiB | Reduce, 2 pages | Reduce, 1 MiB | Kind |
|---|---|---|---|---|---|
| the command around the ranks | 2.5 | 2.5 | 2.4 | 2.5 | ours |
| the ranks' entries and their meeting | 1.4 | 1.4 | 1.4 | 1.4 | ours |
| the leader's put lines out | 2.0 | 2.0 | 2.2 | 2.2 | ours: the line's steps |
| waiting for the other island's share (the reduce: its partial) | 32.4 | 41.9 | 34.9 | 43.5 | about 28 ours (the other island starts later); the rest its line |
| the team let go to add | | | 4.4 | 4.4 | ours |
| the adds | | | 69.4 | 137.8 | ours: about 90 cycles a float32 word, in integer code on the reader |
| waiting for the other island's sums, own lines confirmed | 0.1 | 0.2 | 5.1 | 14.3 | hardware and the line's steps |
| the release of the island's 117 other ranks | 4.4 | 4.4 | 4.4 | 4.4 | ours |

What crosses is a small part: a put line of one page goes out and is confirmed in
5.1 µs, one of 128 pages (512 KiB) in 15.0 µs, of which about 5 µs is the links' rate
(101 GB/s each way) and the rest the line's pushes, one line at 53 GB/s. The largest
terms are ours: the second island's command starts about 28 µs after the first's (the
other island waits 0.2 µs for the share; the host stored the islands' commands one after
the other, fixed since: below), the adds of the reduce, the release and the go (117
atomics each, one multicast each since: below). The adds take ceil(pages of the share / ranks) pages on the
slowest rank: 1 page at 2 pages, 2 at 1 MiB. The first call of a run costs about
15 µs more (the service overlay's first load on every rank).

From a kernel, the form a chain of layers calls between layers, two cards, a gather of
2 pages took 46.6 µs while the host stored each island's command in turn and waited
for it before the next (the first island waited 36.3 µs for the other's share; storing
the other island first moved the wait to it). Handing every island's command first and
storing them back to back from Python gave 19.7 µs: the command 2.4, the entries and
the meeting 1.4, the lines out 2.0, the wait for the other share 9.5 (about 4.3 the
share's line, about 5 the second island's store), the confirm 0.1 and the release 4.4
(117 atomics).

Two changes remove those two terms. The leader releases a whole island with one
multicast: it writes a marker (the command's sequence number and the count of its
releases in the command, 1 to 7 and again: so a second `km::all_gather` or
`km::all_reduce` in one command never finds the first's marker still in the word, and
two releases in a row always differ) into a word of its own control block, then multicasts those 4
bytes to the same word on the island's other cores, one rectangle each for the rest of
its row, the other rows of the grid's first panel and the second panel; each rank
waits for the marker. The reduce's go before the adds is the same release. A team that
is not a whole island, or an island the host gave no rectangles, keeps an atomic a
rank. And `IslandRun.send` (the wait sends first) hands every island's lines to one
native call that writes every dispatcher's ring and read count, then one fence.

Measured again from a kernel (`account.py --cards 0,1 --sizes 2,144,272 --calls
gather,reduce --forms kernel,host --store wait,old --rounds 3 --stamps`, two cards,
every share exact, 2026-11-15), microseconds of the island that finishes last, the
call's last round (each column sums to it; the medians of 3: 12.74, 22.62 and 91.04):

| Term | Gather, 2 pages | Gather, 272 pages | Reduce, 2 pages | Kind |
|---|---|---|---|---|
| the command around the ranks | 2.3 | 2.2 | 2.2 | ours |
| the ranks' entries and their meeting | 1.4 | 1.4 | 1.4 | ours |
| waiting for the other island's partial | | | 7.3 | its line |
| the team let go to add | | | 0.9 | ours: one multicast |
| the adds | | | 70.1 | ours, not changed here |
| the leader's put lines out | 2.0 | 2.0 | 2.0 | ours: the line's steps |
| waiting for the other island's share (the reduce: its sums) | 5.3 | 16.2 | 5.6 | its line: hardware and the line's steps |
| the confirm | 0.1 | 0.1 | 0.1 | |
| the release of the island's 117 other ranks | 0.9 | 1.0 | 0.9 | ours: one multicast |

The release took 4.4 µs and takes 0.9: the ranks see the marker 0.3 to 0.9 µs after
the confirm, the leader's last 0.9 waiting for its write to be acknowledged. The wait
for the other share went from 9.5 to 5.3 µs and is now the same on both islands (5.3
and 5.2): the second island's store is gone. Stored island by island from Python
(`--store old`) the first island waits 36.0 µs instead. The host's form takes the same
times (12.4 µs, 2 pages). On one card split into three islands in a line (`--cards 0
--islands 3`), the release takes 0.8 to 0.9 µs on every island, including those whose
first core is not the card's.

Measured again on 2026-11-16 with the put queue first in the link cores' buffers (the
same command, kernel form, 3 rounds): 12.56 µs at 2 pages and 22.82 µs at 272, both
exact.

The larger gather, 272 pages (557 KB each way): 22.6 µs, 24.6 GB/s each way for the
call. The term that holds it is the share's put line: 16.2 µs from the lines out to the
share in, of which 5.3 is a line's own time (one page) and 10.9 the other 135 pages,
51 GB/s. Each share is one line from the leader, from DRAM; the link cores' page CPUs
split it (three or four of them; every one at 64 pages or more), each reading 4 pages a
group with 3 groups in flight. The puts that reach 72.5 GB/s on the same links come
from 118 cores at once, each from its own L1. The call already spreads its line over
every page CPU, so what remains is the link cores' engine (its reads in flight a CPU),
not the call: about 10.9 µs at 557 KB. At 144 pages (295 KB each way) the line takes
13.1 µs, 7.8 for its last 71 pages (37 GB/s), and the call 20.2 µs, 14.6 GB/s.

On the host every call costs about 270 µs more than on the card: the host's submit and
wait (ours). On one card split into three islands in a line, 1 MiB: all-gather from a
kernel 79 to 159 µs on the card, exact; all-reduce 978 µs, exact from a kernel and from
the host. The all-gather from the host moved no page there (2.4 ms on the card, no
error): its handler number, 243, is also the dispatcher's own inspect, and a
dispatcher whose cores hold the whole group (each island's on a split card; between two
cards each island holds half of it) read its cores and answered as an inspect, the
ranks never running the call. The dispatcher's own commands now carry a header flag
(`KMD_FLAG_DISPATCHER`) and are taken by it, not by their number.
Four islands were not run. On one card as one island both calls return at once,
exact: from a kernel the command's own 2.2 µs.

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
in `experiments/islands/islands_mesh`). A 240-rank ticket costs two commands of host work, one
an island, from one host thread: 1.21 µs a part. MAP commands alternating between the
islands cost what MAP on one island costs, so the islands run at once. MAP on one
group of all 240 ranks (rank r adds input tiles 2r and 2r + 1, which lie in its own
island's part of the input): the last round's 240 tiles bit-exact against
`mapcollect.py`'s sums, both kernels ended with no error. With `--hang`: an inspect of
island 1's core 10 during a HANG on ranks 100 to 239 named its reader waiting for a
port arrival; the deadline gave the command up, `CommandTimeout` named island 1 and its
cores, and the run went on; 16 pages of island 1 read inside the run were exact. Open
to close in 14 s (the mesh with the fabric 5 s; the tensors and both loads 2 s).

## Classes on islands

A class (`km.Program` written as a class) is made with the island session as its
`session` (its L1 table is island 0's) and loaded on every island, a copy an island.
`session.grid` is each island's grid, the same on every island. What its lines mean:

| A class's line | On two islands |
|---|---|
| `self.rank`, `self.size`, `self.leader` | the flat rank less the command's first, the group's count: as on one card |
| `self.core` | the core's flat rank: its island's first rank plus its index there |
| a stream of a tensor's tiles or pages | the page wherever its island holds it (`split`, `copy`): a far page is asked across the link |
| `km.reduce(..., group=S)`, `km.all_reduce(..., group=S)` | each group of S consecutive ranks, wherever its ranks lie: a group that spans islands folds each island's part as its own group, then each part's first rank sends once to the group's rank 0 (`all_reduce`: rank 0 sends the sum back to each part's first rank, which casts it to its part). The arrival has the most slots any group of S takes wherever a command puts it, on islands only: a group on one island its rounds, one across the root part's rounds (at most S - 1 ranks) and a slot each further island; on two islands max(rounds(S), rounds(S - 1) + 1), one slot more than one island's unless S - 1 is a power of two. The parts are added in another order than one island's tree: an integer or exact combine gives the same words, a float sum may differ by rounding |
| `km.Broadcast`, `self.x`, `self.y` | a rectangle that fits one island is each island's own: its command's group is that island's rectangle in flat ranks (island i's first rank plus the rectangle's cores), and a broadcast's command on another group is refused naming the group. A `rect=` taller than island 0 is one rectangle over the islands from island 0's first row (below): its group the flat ranks of its cells, a broadcast's command on any other group refused, saying so. With no `rect=`, the rectangle is one island's in the flat ranks (below): rank r at (r % width, r // width) wherever it is, its group from flat rank 0 (a broadcast's command on another group refused, saying so). On a split card a `rect=` is in each island's own rows, refused naming an island whose ranks do not hold it. A load, at the start or `run.load`, needs each island's part of its group to hold the rectangle in flat ranks, and is refused naming that part |

A command's group may start and end anywhere in the flat ranks. A class that works by
`self.rank` and `self.size` is exact across an island's edge (the Linear class's split on
ranks 104..131 across rank 118, two cards, 2026-11-16), and so is a reduce in groups of
S, but for the order of its additions: a group that spans islands folds each island's
part, then crosses once, so its float sums are rounded in another order than one
island's tree (integers and exact combines are equal). An example, not a bound: the decode layer class on two cards with its groups started at rank 0 (groups of 15 across rank 118 in attend's merge, o and down): qkv equal, its attention output 366 of 65,536 words differing (largest 0.0050 relative), the layer's output 996 (largest 0.0065), 2026-11-17. `self.core` is the core's flat rank (its island's first rank plus its index there), so
it is defined across the edge.

A `rect=` counts its rows from the island of the command's first rank, and rows past
that island's last row continue on the next island, the islands standing one below the
other in the island order. Two cards of 12 x 10 cores, each with its 118 ranks:

```text
             columns 0 .. 11
island 0  row 8   . . X X X X . . . . . .   rect=(2, 8, 4, 4): rows 8 and 9 of
          row 9   . . X X X X . . . . L L   island 0, rows 0 and 1 of island 1
island 1  row 0   . . X X X X . . . . . .   (L: island 0's link cores, never in a
          row 1   . . X X X X . . . . . .   rectangle); its group: flat ranks 98..135
```

A rectangle that fits one island behaves as on one island: each island has its copy,
and a command that starts on a later island counts from that island's first row. A
`rect=` at the make taller than island 0 is one rectangle from island 0's first row, its
group the flat ranks of its cells; a broadcast's command on another group is refused,
naming that one rectangle. `km.UPTO(rect=)` counts from the island of the group's first
rank and may run on, over as many islands as it reaches (a cell on a link core is
refused by name). `self.x` and `self.y`
are places in the whole rectangle (above, island 1's row 1 is `self.y` 3). A broadcast's
column that crosses an edge is a part an island: each part's first core multicasts to
its part, and the part before it feeds it each block across the link (`km::write_l1`),
the form `layout.island_broadcast_args` writes by hand (`examples/03_broadcast/
islands.py`). Why so: a program written for one island runs unchanged, and a command
that starts on a later island counts from there, as before. A column broadcast of one
float32 tile down 4 columns of 8 rows, the median of 8 commands from submit to event
(2026-11-17): 5.2 µs on one island; 80.6 µs over two cards (island 0's rows 6 to 9,
island 1's 0 to 3); 120.5 µs over two islands of a card split in 3 (5 rows); over all
three islands of that card 120.2 µs (8 rows) and over the four of a card split as a
2 x 2 grid 171.4 µs, every page equal to one island's.

A class made with no `rect=` takes one island's rectangle wherever its ranks are: rank r
is at column r % W and row r // W (W the grid's width), so `self.x`, `self.y` and the
rectangle's columns and rows are one island's for every rank, on any number of islands.
Over two cards it is the whole 12 x 10 grid, the group (0, 120), its last row island 0's
row 9 and island 1's first two ranks; with fewer ranks than the grid (a card split in 3:
116) it is every rank for `self.x` and `self.y`, their whole rows (the group (0, 108))
for a broadcast. Two cards of 118 ranks, each cell's flat row:

```text
                 columns 0 .. 11
island 0  row 9   9 9 9 9 9 9 9 9 9 9 L L   a: flat row 10, b: 11; L: island 0's
island 1  row 0   9 9 a a a a a a a a a a   link cores, in no flat row
          row 1   a a b b b b b b b b b b
```

A broadcast is one island's (the same senders; rows cut at island 0's panels) in parts:
a part ends at an island's edge, a physical row's end and a panel, and the part before
it feeds its first core each block (`km::write_l1`, across the link or on its own
island). The parts of a broadcast have no bound (each is fed by the one before it); a
part has at most 16 cores with the next part's row, as a broadcast on one island, which
the default never reaches (a row or column of 12 or 10 cores). What changed: a command
of such a class that starts on a later island no longer counts from that island's first
row; its group is the flat one. Times, the median of 8 commands from submit to event, a
float32 tile a core (2026-11-17), one island / two cards / a card split in 3 / split as
a 2 x 2 grid: a row broadcast 5.66 / 23.02 / 56.66 / 120.58 µs, a column broadcast
6.36 / 35.58 / 149.44 / 387.9 µs, `self.x` and `self.y` alone 5.1 / 12.98 / 272.86 /
256.3 µs, every page equal to one island's; the attention class (t1) on two cards
33.62 µs against 14.39 on one island, word-equal.

The output dimension sharded, the activation a copy (`experiments/language/
dense_class/linear_class.py --cards 0,1`): the split handler over the flat group of
its 112 ranks centred on island 1's first rank, so each island runs four groups of 14
on its own half of W's columns; X `copy=True`, W and Y `split=1` over the 128 ranks
centred there (a column tile a rank: a hint over the command's 112 ranks would put 72
of the 128 on island 0, and part of W would be read across the link). At 32 x 4096 x
4096 it is exact in 65.3 us (2026-11-16; 65.1 before the put queue's move), against 94.2
on one card.

## What is not built

The refusals that remain are under [the limits](#the-limits). Beside them:

- Deliveries spread over the link cores' CPUs and links as pages are (speed work): page
  CPU j would send the lines numbered j modulo the CPUs over its own link, into its own
  slot ring of the far island's places, and the far CPU j would confirm ring j, so the
  far chain runs on as many CPUs as links. A kernel's own ask (`km::read_pages`) and put
  (`km::write_pages`) are already carried by every page CPU.
- Uploads over islands (an engine's stream or graph runs on the island it is uploaded
  to, its commands to any flat group:
  [commands across islands](threads.md#commands-across-islands)).
- The host's simulators of a graph and a stream (`komorebi.engines`) check a node's
  group against one card's cores (`core_count`, 119 by default), not against the
  islands' flat ranks.
- A tensor read or written by an address the kernel builds itself
  (`InterleavedAddrGen`: profile pages, job tables, the raster and ray tracer projects)
  does not cross, and neither do its ranges of a page. The raster passes' reads of the
  X280 thread's regions in one DRAM bank can be named by `km::BankRange`, and the ray
  tracer's packet streams, whose page bytes change with the handler and the stage, by
  `km::SizedTensor`; neither project uses them yet. An overlay of the kernel form is
  built with its kernel's defines and compile arguments, so its `km::Tensor` crosses
  (examples 12 and 18, `--far`).
- More than two cards: the mesh opens 1 x n, so the fabric reports a line (a 2-D
  mesh shape is not built); tried with two.
- A split card: a copy across more than one link; X280 threads (not supported, by
  decision: a split card is for developing and checking island programs).
- A split card's five limits, in the order a project meets them, with what each costs
  (each would be its own build with a card proof; none is planned):
  1. One `Program` of the kernel form a split card ([the limits](#the-limits)): a mixed
     program runs as one `Program`, or on two cards.
  2. `hold_links=True` on two cards only: a split card's dynamic run takes the
     deliveries, 1,531.5 ms against 307.4 ms held for 10^6 forwards on two cards (about
     5 times).
  3. Messages over more than two islands: each message for another island is a delivery
     by the link service, one link a pair (about 2.9 µs a delivery).
  4. The host's page helpers (`ask_range`, `send_pages`, `cross`) serve one holder and
     one island a link apart: a call a holder or a hop (`session.read` and
     `session.write` reach any page).
  5. `host_reaches`: an island with no host link has its tensors written and read only
     inside a loaded run with links, so its inputs are written after the load.
- Two cards each split into islands, as a line of four (fabric links and wires in one
  graph): not in this release, so forwarding has run over wires only.
- A link core reset by its dispatcher: not built; a hung link core is named by its
  dispatcher and reset by the host (`run.reset_link`; [how reset
  works](../concepts/reset.md#4-reset-a-link-core-from-the-host)).
