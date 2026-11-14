---
title: Islands
summary: A session over several cards, each card's chip an island with its own grid and dispatcher; ranks flat and contiguous by island, commands split by island on the host, tensors placed by island. What it does today and what it does not yet.
tags:
  - reference
  - host
---

# Islands

Status: the host side is implemented over two Blackhole p150a cards. Nothing crosses
between the cards inside a run yet: a group that spans islands runs as one part on
each island, and each part's cores see the ranks of their own part.

An island is a set of cores that reach the others only through links: here one card's
chip, with its own Tensix grid and its own dispatcher on its own X280 cluster 0. The
host values (the graph of islands, ranks, the split of a group, placement) are in
`komorebi.islands`.

```python
import komorebi as km

with km.Session(cards=[0, 1]) as session:  # one session, 240 ranks
    source = session.tensor(array, "bfloat16", "tile")  # split by island
    run = session.load(program)  # on every island
    ticket = run.submit("map", group=(0, 240), args=[...])
    parts = run.wait(ticket, timeout=5.0)  # one event per island
    run.stop()
    (result,) = session.read_bf16(output)  # assembled from every island
```

## The session

`km.Session(cards=[a, b, ...], links=0, pll_mhz=None, tree=None, log=None,
grid_timeout=10.0, kernel_bytes=None, channel=256 << 20)` with two or more cards
returns an `IslandSession`; with one card it is the card's `Session` as before.

- **Open.** In this order: tt-metal's 1-D fabric is set, the cards open as one 1 x n
  mesh (each card's chip a 1 x 1 submesh with its own command queue), the X280 runtime
  opens once over the mesh, then each island gets a `Session` given its device
  (`session.island(i)`). The cards must be the visible ones (`TT_VISIBLE_DEVICES` is
  set to them) and must not be held by another process.
- **Ranks.** Flat and contiguous by island, in island order: island i holds ranks
  `session.graph.first(i)` to `first(i) + count(i) - 1`, its cores in core order.
  `session.ranks` is the total (240 on two cards).
- **Close.** Every island's run is stopped through the X280 runtime, the islands'
  sessions are closed, then the mesh, and the fabric is set off. A grid that did not
  return keeps the mesh open, as a card's session keeps its device.
- `links` is 0: no core is taken for a link, every core is a rank. The graph already
  takes a link core out of the ranks on each island a used link joins; a session will
  take them there once the link service exists.

## The run

`session.load(program, **options)` loads one `Program` on every island in turn (its
grid, then its dispatcher), with `Session.load`'s options, and returns an `IslandRun`.

- `run.submit(handler, group, args)` takes a group of flat ranks. The host splits it
  with `Graph.split` and submits each part to its island's dispatcher; the call returns
  a ticket. `run.submit_many` does the same for n commands, one `submit_many` an
  island.
- `run.wait(tickets, timeout)` sends every island's queued commands first, so the
  islands run their parts at once, then waits for every part within the one timeout.
  A ticket completes when every part has; its result is the tuple of its parts' events
  in island order. `run.drain`, `run.stop` and `run.abort` act on every island within
  one timeout; `run.state()` lists every rank in flat order, `run.result()` every
  island's result block.
- An error from an island names it: the message starts `island i:` and the exception
  carries `error.island`. A `WaitTimeout`'s `pending` lists tickets.
- `run.island(i)` is island i's own `Run`, in its own ranks.

## Tensors

`session.tensor(data, dtype, layout, like=None, copy=False)` makes one tensor of the
whole shape on every island, at the same address, so a program takes its `address`
and `accessor_args()` unchanged. Its pages are placed by `islands.place`: split by
island along the leading dimension in rank order (`like`: the group the split
follows; default every rank), so a rank that partitions the tensor by rank finds its
part in its own island's DRAM; `copy=True` puts every page on every island. A split
tensor's pages that another island holds are zeros on this one.

`session.read` and `session.read_bf16` assemble a tensor from the island that holds
each page, through that island's host channel; `session.write` writes each page into
the island that holds it (a copy: into every island).

## Measured

Two Blackhole p150a cards joined by four Ethernet links, one process, example 01's
program and kernels unchanged (`examples/01_mapcollect/islands.py --hang`):

2026-11-14, X280 at 1350 MHz, host load about 3 (AICLK not sampled),
microseconds a command, submit to event, pipelined (the median of 3 rounds of 20,000
NOP or 2,000 MAP, one host thread submitting to both islands):

| What | µs a command |
|---|---|
| NOP on island 0's 120 ranks | 1.192 |
| NOP on all 240 ranks (a part on each island, joined) | 2.427 |
| MAP on group (0, 120) | 2.623 |
| MAP on (0, 120) and (120, 120) in turn | 2.695 |

A NOP on one island costs what it costs on a card opened alone (1.131 to 1.159 µs
in `experiments/islands_mesh`). A 240-rank ticket costs two commands of host work, one
an island, from one host thread: 1.21 µs a part. MAP commands alternating between the
islands cost what MAP on one island costs, so the islands run at once. The last MAP
round's 240 tiles (each island's group reading its own part of the split input) were
bit-exact against `mapcollect.py`'s sums; the input read back from both islands was
exact; a HANG given up on island 1 raised `CommandTimeout` naming island 1 and the run
went on; both kernels ended with no error (129,006 and 63,006 commands). Open to close
in 14 s (the mesh with the fabric 5 s; the tensors and both loads 6 s).

## Not yet

- **A group across islands with the whole group's ranks.** A part's cores see the
  ranks of their part (`me - first`, `count` of the part), so example 01's MAP over a
  group that spans islands partitions over each part, not over the group. The
  encoding is decided: every island of the split is sent the group's word unchanged in
  flat ranks, its dispatcher cuts its own part and a core's rank is its island's first
  rank plus its own index less the group's first. It needs the island table on the
  cores; the host side has the switch (`IslandRun.flat_groups`).
- Links: no message, page or command crosses between islands inside a run; `forward`
  to another island's rank, `collect` and `broadcast` over islands, and a read of a
  page another island holds need the link service.
- A list or a dict of programs, X280 threads and ports, engines and uploads over
  islands; tensors in L1.
- More than two cards: the mesh opens 1 x n; tried with two.
