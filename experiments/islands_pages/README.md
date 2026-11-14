# Pages across islands

Two cards opened as the islands of one session (`km.Session(cards=[0, 1], links=n)`).
Can the cores that a session takes for its links write a tensor held by one island into
a tensor held by the other, straight into the other island's DRAM, exact, and how fast?

## The page sender

A session with `links` 1 to 4 takes the last `(links + 1) // 2` cores of each island
as link cores. They are not ranks: a command names them by first `KM_GROUP_LINK_BASE`
(0xFE00) + k (`graph.link_group`). Both data-movement CPUs of a link core send, each on
its own link: CPU c of link core r sends on link 2r + c. Link 0 also carries the
dispatcher's command lines, so with more than one link pages go on links 1 on (since
the rows below were measured; they used all four). The run's kernels are built
with the define `KM_PAGE_LINKS`, which adds the runtime's `PAGES` command
(`KM_HANDLER_PAGES`, `km_pages.hpp`) to the reader and writer loops. A run of a session
without links is built as before.

A `PAGES` send names a source (an interleaved DRAM tensor of the sending island, its
first page and page count), a target (the same pages of a tensor in the other
island's DRAM, or an L1 address of one of its cores), the core and L1 address of a
count, and the Ethernet channel of each link. Each CPU takes its share of the pages,
reads them from DRAM into slots of its L1 (its half of the run's transfer buffer, 16
KiB: three slots of 4096 B), and sends each page as one packet with a plain write. The
last packet of every 64 is a fused write and add of that batch's pages to the count.
A slot is read again once its packet has left L1. Each CPU opens its connection once,
at the load (a `PAGES` open, then a hello both ways with the far link core's same
CPU), and holds it until its kernel ends, so a link never has two senders at once. A `PAGES` wait on
the other island waits, bounded, until the count has reached the pages sent, then
takes them off it.

`session.cross(source, target)` queues both and waits for both: the send on the
holder's link cores, the wait on the other island's first link core (its port 7).
`send_pages` and `wait_pages` queue them alone.

## Result

`pages.py`: a uint32 tensor of 4096-byte pages, every page held by island 0, is copied
into a tensor whose pages are all held by island 1 (`like=` the island's ranks). The
host reads the target back from island 1 and compares every page. 2026-11-14, AICLK
1350 MHz, both runs once:

| Links | Link cores (CPUs) | Bytes | Submit to the last count | Rate | Pages wrong |
|---|---|---|---|---|---|
| 1 (channel 4) | 1 (1) | 1 MiB | 0.848 ms | 1.24 GB/s | 0 of 256 |
| 4 (channels 4 to 7) | 2 (4) | 64 MiB | 2.091 ms | 32.09 GB/s | 0 of 16,384 |

The time is the host's, from the submit of both commands to the wait's event. The
1 MiB run is the host's command time; 64 MiB over four links is 8.0 GB/s a CPU, a
quarter of what one CPU sends from L1 alone (28.9 GB/s, `experiments/link_rate`). With
three slots a CPU has at most three DRAM reads in flight, so the read latency sets the
rate; more L1 a CPU, more reads in flight, is the next step. (Both rows ran before the
writer took the `WHERE` handler below, which the page sender does not use, and before
the connections were held: each command then opened and closed its own.)

## Which island holds a page, asked on a core

`pages.py --where` makes four tensors before the load (every page on island 0, every
page on island 1, split by rank, a copy) and asks the first and the last rank core of
each island, with the handler `WHERE`, `km::island_of(tensor, page)` (the tensor by
its reference word, which the session's tensor table resolves) for the first and last
page of every part. 2026-11-14, links 4, 1 MiB tensors (256 pages; the split one 138
and 118), when the placements were the writer's compile arguments: 40 answers, 0
differ from `islands.island_of` on the host; the copy answers 0 on island 0 and 1 on
island 1. Not run since the tensor table.

## Running it

Both cards in one job, through the hang watcher, with tt-metal's environment set and
the repository's `src` on the path:

    python pages.py --links 4 --mib 64
    python pages.py --links 4 --mib 1 --where

It prints one JSON line and exits nonzero if a page is wrong or a kernel ended with an
error.
