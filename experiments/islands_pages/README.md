# Pages across islands

Two cards opened as the islands of one session (`km.Session(cards=[0, 1], links=n)`).
Can the cores that a session takes for its links write a tensor held by one island into
a tensor held by the other, straight into the other island's DRAM, exact, and how fast?

## The page sender

A session with `links` 1 to 4 takes the last `(links + 1) // 2` cores of each island
as link cores. They are not ranks. Both data-movement CPUs of a link core send, each on
its own link: CPU c of link core r sends on link 2r + c. The run's kernels are built
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
A slot is read again once its packet has left L1. Each command opens its connection
and closes it at the end, so a link never has two senders at once. A `PAGES` wait on
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
rate; more L1 a CPU, more reads in flight, is the next step.

## Running it

Both cards in one job, through the hang watcher, with tt-metal's environment set and
the repository's `src` on the path:

    python pages.py --links 4 --mib 64

It prints one JSON line and exits nonzero if a page is wrong or a kernel ended with an
error.
