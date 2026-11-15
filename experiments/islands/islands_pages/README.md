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
into a tensor whose pages are all held by island 1 (`group=` the island's ranks). The
host reads the target back from island 1 and compares every page. 2026-11-14, AICLK
1350 MHz, both runs once:

| Links | Link cores (CPUs) | Bytes | Submit to the last count | Rate | Pages wrong |
|---|---|---|---|---|---|
| 1 (channel 4) | 1 (1) | 1 MiB | 0.848 ms | 1.24 GB/s | 0 of 256 |
| 4 (channels 4 to 7) | 2 (4) | 64 MiB | 2.091 ms | 32.09 GB/s | 0 of 16,384 |

The time is the host's, from the submit of both commands to the wait's event. The
1 MiB run is the host's command time; 64 MiB over four links is 8.0 GB/s a CPU, a
quarter of what one CPU sends from L1 alone (28.9 GB/s, `experiments/islands/link_rate`). With
three slots a CPU has at most three DRAM reads in flight, so the read latency sets the
rate; more L1 a CPU, more reads in flight, is the next step. (Both rows ran before the
writer took the `WHERE` handler below, which the page sender does not use, and before
the connections were held: each command then opened and closed its own.)

With 15 slots a CPU (a 64 KiB chunk of the transfer buffer, asked by a session over
two cards) and pages on links 1 to 3 (link 0 carries the lines), 2026-11-15:
`pages.py --mib 512`, 16.857 ms, 31.85 GB/s, 0 of 131,072 pages wrong. That is 10.6
GB/s a CPU: more reads in flight lifted a CPU from 8.0, and the rest of the gap to
28.9 is not the slots (below).

## Pages a kernel asks for and puts

`pages.py --ask MIB...` and `--put MIB...` have island 0's 118 ranks move a tensor
with one `MOVE` command a size (the reader's handler): each rank takes its share of the
pages in calls of `--step` pages, `km::read_pages` into its L1 then `km::write_pages`
out of it. `--ask`: island 1 holds the source, island 0 the target, so every page is
asked for across; `--put`: island 0 holds the source, island 1 the target, so every page
is put across. The time is the command's event (the dispatcher's span, submit to
completion), the rate payload over it; each target is read back and every page
compared. Every page CPU of the holder (asks) or of island 0 (puts) serves its share
of each line: four of them, 32 pages each of a 128-page line, read four pages a
transaction (the first three rows); before, the CPUs of links 1 to 3 read a page a
transaction (the others). Four links, 4 KiB pages, 2026-11-15, each row one run:

| Move | Pages a call | 64 MiB | 512 MiB | Pages wrong |
|---|---|---|---|---|
| ask | 128 | 1.747 ms, 38.42 GB/s | 11.347 ms, 47.31 GB/s | 0 |
| put | 128 | 1.752 ms, 38.31 GB/s | 11.089 ms, 48.42 GB/s | 0 |
| ask, a page a read and 32-page shares | 128 | 2.321 ms, 28.91 GB/s | 16.456 ms, 32.62 GB/s | 0 |
| ask | 32 | 3.044 ms, 22.05 GB/s | 22.694 ms, 23.66 GB/s | 0 |
| ask | 128 | 2.368 ms, 28.34 GB/s | 16.732 ms, 32.09 GB/s | 0 |
| put | 128 | 2.363 ms, 28.40 GB/s | 16.610 ms, 32.32 GB/s | 0 |

(The 32-page row and an earlier 128-page pair, 32.14 GB/s asked and 31.84 GB/s put at
512 MiB, ran with an `ASK` and a `PUT` handler of the same loop, before the tensor table
named tensors by their reference word.) Before every page CPU served, one page CPU on
link 1 with three slots moved a thread's 64 MiB across in 11.0 ms (6.1 GB/s).

## What bounds a page CPU: the push's shapes

`pages.py --mib 128 --shapes SHAPE...` has the host cross the source into island 1
once a shape of the push (`KM_PAGES_SHAPE_*`: pages a read 4 ** bits 0-1, slots 15 <<
bits 2-3 - 1, bit 4 no read, bit 5 no wait for a slot's packets to leave L1, bit 6
link 0 too), each into a target of its own read back after all (a shape with no read
writes garbage into a target of its own, not checked). The time is the send
command's own ticks (its X280's 50 MHz count, submit to completion), beside the host's
wall time, which waits far longer (62 ms at 128 MiB, 246 ms at 512 MiB). `--chunk
KIB` asks a CPU's L1 chunk. 2026-11-15, four links, links 1 to 3 sending, 256 KiB
chunk, one run, every read shape exact:

| Shape | Reads | Slots | GB/s (three CPUs) |
|---|---|---|---|
| 0x30 | none, no wait | 15 | 55.18 |
| 0x10 | none | 15 | 42.85 |
| 0x00 (then) | 1 page | 15 | 27.74 |
| 0x08, 0x0C | 1 page | 30, 60 | 27.58, 27.74 |
| 0x05, 0x09, 0x0D | 4 pages | 12, 28, 60 | 48.05, 48.16, 48.13 |
| 0x0A, 0x0E | 16 pages | 16, 48 | 44.19, 62.16 |

Slots beyond 15 give a page a read nothing (15 transaction ids; a wait on a shared id
waits for both reads), and 4 pages a read the same at 12, 28 or 60 slots: not the
DRAM read's latency. What a CPU does a read and a packet bounds it (a read's wait,
the slot's wait, a header and two NoC commands a packet): 16 pages a read in three
groups gives 20.7 GB/s a CPU. With link 0 sending too (bit 6), 128 MiB took 1.678 ms,
79.98 GB/s at 16 pages a read and 2.189 ms, 61.31 GB/s at 4. The default push reads 4
pages a transaction in a 64 KiB chunk, 16 in a 256 KiB one; at the default chunk 512
MiB crossed in 8.932 ms, 60.1 GB/s (four links).

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
    python pages.py --mib 512
    python pages.py --ask 64 512 --step 128
    python pages.py --put 64 512 --step 128

It prints one JSON line and exits nonzero if a page is wrong or a kernel ended with an
error.
