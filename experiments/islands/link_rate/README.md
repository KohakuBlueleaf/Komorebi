# Link rate

Two Blackhole p150a cards are joined by four Ethernet links. An earlier experiment
here sent 4 KiB packets from one Tensix core a link and got 10 GB/s a link, 40 GB/s
over four. A rougher measurement with tt-metal's own sender kernel got about 30 GB/s
a link and about 120 GB/s over four. This experiment finds out which difference
between the two senders costs what, and asks four more things:

1. What does a link carry with plain unicast writes of 4352-byte payloads, one
   sending core a link, and what do the three differences between the senders cost:
   where the sender sits, a fused write and increment with every packet against plain
   writes with one count for a batch, and a wait after every fourth packet?
2. Do four links, senders spread over the grid on NoC0, carry four times one link,
   one way and both ways at once?
3. Is a packet whose target is a DRAM address or a core's L1 on the other card exact
   when no kernel on that card does anything? What does DRAM take?
4. Do two data-movement CPUs of one core, each holding a connection to a different
   link, run two links, and on which NoC?

Everything here runs on tt-metal's 1-D fabric in slow dispatch, one process holding
both cards. Every table says its conditions; every row ran once.

## The program

`host/link_rate.cpp` opens a 1 x 2 mesh with the fabric, places one sending core a
link on the sending card and a receiving core on the other, and launches every core's
program as one workload. Options (`--name value`):

| Option | Meaning |
|---|---|
| `--from` | the sending card (a mesh column), or `both` for both directions at once |
| `--links`, `--first-link` | how many links, from which |
| `--packets`, `--payload`, `--window` | packets a core, bytes in each (at most 4352), slots in L1 the packets cycle through |
| `--fused` | 1: every packet is a fused write and increment of the receiver's counter. 0: plain writes, and only the last packet of each batch (and the last packet) is fused |
| `--batch` | packets a count in plain mode (64 here) |
| `--wait-every` | the sender waits until its NoC writes have left every this many packets (0: never) |
| `--place` | the sending cores: `under` (the Tensix core of row 0 or 1 nearest at or right of the link's Ethernet tile), `row` (along row 0), `spread` (over rows and columns), `at:x,y;x,y` (the given logical cores in order, a core a link, or a pair of links with `--dual`) |
| `--rx-place` | the receiving cores: `top` (the top rows) or as `--place` |
| `--noc` | the sender kernels' NoC |
| `--target` | `l1` (a receiver kernel counts and checks), `bare` (a core's L1, no kernel there) or `dram` (a DRAM bank, no kernel there) |
| `--dual` | 1: the links pair up on one core, the second of each pair on the other CPU |

`kernels/sender.cpp` writes every packet header once before it sends (one plain and
one fused header a slot), then sends each packet as a payload from L1 and its header.
Every wait is polled with a bound. `kernels/receiver.cpp` waits up to 10 s for its
counter and checks every slot it received against the pattern.

The rate is a count of payload bytes over a span of the Tensix wall clock (AICLK
1350 MHz on both cards). At the receiver, "rx" is the bytes after the first counted
packet over the span from the first count to the last. At the sender, "tx" is all the
bytes over the span from its first send to its last flush. Every run was checked: the
receiver's own check of every slot, and the host's check of the receiver's copy of its
first slots, are exact in every row below.

Three versions of `kernels/sender.cpp` ran, and a table says which one made its
numbers (the committed one is C):

- A: the first. It bounded every wait with a helper that also read the wall clock on
  every packet. Questions 1 to 3.
- B: reads the clock only when no slot is free, and counts those cycles (words 7 to 9
  of its status page). Question 4, and question 5 except where it says C.
- C: B plus `--space-every`, a check for free router slots once for eight packets
  instead of one. The runs that name it.

One packet costs 162.6 ns in A, 150.7 ns in B and 156.6 ns in C with a 4352-byte
payload, one link, plain writes: a few instructions move the rate by up to 7 %.
Question 5 comes back to that.

## 1. One link, and what each difference costs

Sender A. 16384 packets of 4352 bytes (71 MB), window 64 slots, batch 64, one link
(link 0), card 0 to card 1, a receiver kernel on card 1. GB/s, rx (tx within 0.02):

| Sender at | plain writes, no wait | plain, wait every 4 | fused, no wait | fused, wait every 4 |
|---|---|---|---|---|
| row 0, core (0, 0) | 26.76 | 23.60 | 10.56 | - |
| under the link's Ethernet tile, core (11, 0) | 26.76 | 23.60 | 10.56 | 10.56 |
| spread, core (0, 1) | 26.76 | - | - | 10.56 |

The same program with 256 packets in the first run (row 0, plain, no wait) read 28.65
GB/s rx: a short run reads higher.

Where the sender sits costs nothing for one link: three places, the same 26.76 to
four digits. The fused write and increment costs the most: 26.76 to 10.56 GB/s, with
the wait after every fourth packet on or off. The wait costs 3.2 GB/s (12 %) with
plain writes and nothing once every packet is fused. The earlier sender had the fused
packet and the wait, which gives the 10.56 here. Sender B on the same link, from a
spread core, reads 28.87 GB/s plain, 25.22 with the wait, 10.57 fused (question 5), so
the 26.76 is sender A's own clock read; the order of the three differences is the
same. Why the fused packet costs so much is shown by question 5.

## 2. Four links, senders spread, NoC0

Sender A for all rows but the last two, which are sender C. 16384 packets of 4352 bytes a link, plain writes, batch 64, no wait, one sending core
a link on NoC0, links 0 to 3. "Spread" puts the senders at (0, 1), (3, 3), (6, 5),
(9, 7); "row" at (0, 0) to (3, 0). GB/s, rx for each link and the sum:

| Senders | Receivers | Direction | link 0 | link 1 | link 2 | link 3 | sum |
|---|---|---|---|---|---|---|---|
| spread, NoC0 | top rows | card 0 to 1 | 26.76 | 26.76 | 26.74 | 26.76 | 107.03 |
| spread, NoC0 | top rows | both at once, card 0 to 1 | 26.76 | 26.76 | 26.74 | 26.76 | 107.03 |
| | | both at once, card 1 to 0 | 26.76 | 26.76 | 26.75 | 26.76 | 107.04 |
| spread, NoC0 | spread | card 0 to 1 | 26.76 | 26.76 | 26.74 | 26.76 | 107.03 |
| spread, NoC0 | spread | both at once, card 0 to 1 | 26.76 | 26.76 | 26.74 | 26.76 | 107.03 |
| | | both at once, card 1 to 0 | 26.76 | 26.76 | 26.74 | 26.76 | 107.03 |
| spread, NoC0 (sender C) | top rows | card 0 to 1 | 27.78 | 27.78 | 27.76 | 27.78 | 111.10 |
| spread, NoC0 (sender C) | top rows | both at once, card 0 to 1 | 27.78 | 27.78 | 27.76 | 27.78 | 111.10 |
| | | both at once, card 1 to 0 | 27.78 | 27.78 | 27.76 | 27.78 | 111.10 |
| row 0, NoC0 | top rows | card 0 to 1 | 26.60 | 26.59 | 26.72 | 26.76 | 106.67 |
| spread, NoC1 | top rows | card 0 to 1 | 26.38 | 26.64 | 26.61 | 26.64 | 106.27 |

Four links carry four times one link: 107.0 GB/s one way with sender A and 111.1 with
sender C (one link with C: 27.78, question 5), and both ways at once carry the same
each way, 214 to 222 GB/s in sum. The links do not share anything that this load
fills. Where the senders sit and which NoC they use move the sum by under 1 %: senders
along one row lose 0.4 %, NoC1 0.7 %. The sum does not reach the 120 GB/s of the
rougher measurement because one sending core a link does not push a link past
about 29 GB/s here (question 5). Sender B on two cores with two links each reached
115.0 GB/s (question 4).

## 3. A packet for a card with no kernel running

The packet's header names the target on the other card: a core's L1 address (`bare`)
or the address in DRAM bank 0 (`dram`: the DRAM tile's NoC coordinates and the bank
address of a buffer allocated on that card). Card 1 runs no kernel at all. Both cards'
target memory was zeroed before the launch. The last packet of each batch is a fused
write and increment of a counter in the L1 of a core of card 1. After the sender
ended, the host read back every slot and the counter. One link, link 0, sender at
(0, 1), 4352-byte payloads, window 64, batch 64:

| Target | Packets | Slots exact (64 of 64) | Counter | Sender GB/s |
|---|---|---|---|---|
| core L1 | 256 | yes, 69,632 words, 0 wrong | 4 of 4 | 28.55 |
| DRAM bank 0 | 256 | yes, 69,632 words, 0 wrong | 4 of 4 | 28.54 |
| core L1 | 16384 | yes, 69,632 words, 0 wrong | 256 of 256 | 26.77 |
| DRAM bank 0 | 16384 | yes, 69,632 words, 0 wrong | 256 of 256 | 26.77 |

Both targets are exact with nothing running on the receiving side, and DRAM takes the
same rate as L1: 26.77 GB/s from one sending core, 2663.7 µs for 71.3 MB. That rate is
the sender's time from its first send to its last flush, a time the link sets, not
the DRAM's: the sender cannot get ahead of the link by more than a few packets of
buffer. The check shows every slot holds the pattern after being zeroed, and that
every batch's closing packet arrived; the 16384-packet runs reuse each slot 256 times
with the same bytes, so the check cannot tell one lost packet in the middle from none.

## 4. Two data-movement CPUs of one core, a link each

One core, its CPU 0 (NoC0) sending on link 0 and its CPU 1 (NoC1) sending on link 1,
a different connection each, and the same two CPUs on a second core for links 2 and 3.
Sender B, 16384 packets of 4352 bytes a link, plain writes, batch 64, no wait, card 0
to card 1, receiver kernels on card 1, every slot exact. GB/s, rx:

| Cores | Links | link 0 | link 1 | link 2 | link 3 | sum |
|---|---|---|---|---|---|---|
| (0, 1): CPU 0 on NoC0 link 0, CPU 1 on NoC1 link 1 | 2 | 28.80 | 28.73 | - | - | 57.53 |
| (0, 1) as above, and (3, 3): CPU 0 links 2, CPU 1 link 3 | 4 | 28.80 | 28.73 | 28.78 | 28.73 | 115.04 |

Two CPUs of one core each hold a connection to a different link at the same time and
each runs as fast as a core alone (28.8 against 28.87 for one link from one core), and
the NoC1 CPU is no slower than the NoC0 one. Four links run from two cores at 115.0
GB/s one way. CPU 0 of a Tensix core is on NoC0 and CPU 1 on NoC1, the default NoC
of each. A first attempt put both CPUs' packet headers, slots and status in the same
L1 addresses: the second CPU's headers overwrote the first's, so both CPUs sent to the
first CPU's receiver (it counted 8 of 4) and the second receiver timed out after its
bound, 10 s; no hang. Two kernels on one core need separate L1 regions: the second
CPU's status, headers and slots sit one region above the first's.

## 5. What the sender waits on

Sender B, one link, link 0, sender at (0, 1), 16384 packets, plain writes, batch 64,
card 0 to card 1; "wait" is the cycles the sender spent with no free slot in the
router's channel (which has 16 slots) of its 3.3 million:

| Payload, sender | GB/s | ns a packet | wait for a free slot |
|---|---|---|---|
| 4352 B plain | 28.87 | 150.7 | 0 cycles |
| 2048 B plain | 13.58 | 150.7 | 0 |
| 1024 B plain | 6.79 | 150.7 | 0 |
| 4352 B, wait for sends every 4 packets | 25.22 | 172.5 | 0 |
| 4352 B fused write and increment, every packet | 10.57 | 411.8 | 64 % of the time |
| 4352 B plain, sender C | 27.78 | 156.6 | 0 |
| 4352 B plain, sender C, space check once for 8 packets | 28.16 | 154.5 | 5 % |
| 1024 B plain, sender C, space check once for 8 packets | 6.93 | 147.7 | 1 % |

- The link is not what the plain sender waits on: it never found the router's 16
  slots full, and the router took packets as fast as the core could issue them.
- A packet costs the same time at 1024, 2048 and 4352 bytes: the rate is a count of
  packets, 6.6 million a second, not of bytes. Each packet is a payload write, a
  header write and a credit write by one RISC-V core (from the adapter's code), about
  203 cycles.
- A few instructions move it (162.6, 150.7, 156.6 ns for three versions of the loop),
  and four links on four cores, two links on one core, either NoC, all run at that
  same one-core rate: the core's issue rate sets the 28 to 29 GB/s, not the link.
- Two things do slow the sender: the fused write and increment (its header carries a
  flush flag, which was not tried off) makes the router the limit, the sender then
  waiting for slots 64 % of the time at 411.8 ns a packet; and a wait for the NoC
  writes to leave every fourth packet adds 21.8 ns a packet.
- Where the link itself stops, between 28.9 and the 50 GB/s of the line, is not
  reached with one sending core a link. A second core on the same channel of this 1-D
  router hangs. Outside this repository a second sender ran exact on VC2 of the 2-D
  fabric (two senders 27.79 GB/s in all) and on channel 1 of a patched 1-D router
  (31.38 GB/s); neither router setting is in the runtime here.

## 6. Where the sender sits

Does the place of a sending core cost rate? Sender C unchanged, plain writes, 16384
packets of 4352 bytes a link, batch 64, card 0 to card 1, receiver kernels on card 1,
every slot exact; then the session's page path (`experiments/islands/islands_pages`,
`pages.py --links 4 --mib 64`, 4096-byte pages read from DRAM through three L1 slots
a CPU, a count every 64), with its two link cores in the far corner (the default) and
in the row under the Ethernet tiles (a session option written for this one run and not kept: it measured no gain), both on the
same commit. The NoC is a torus of 17 x 12 places: NoC0 runs toward higher x and y,
NoC1 toward lower, both wrapping; CPU 0 sends on NoC0, CPU 1 on NoC1. The routers of
links 0 to 3 sit at NoC0 (3, 1), (4, 1), (13, 1), (12, 1); Tensix row 0 is NoC y = 2.

| Path | Sender cores, logical (NoC0) | Links | Hops a link (its NoC) | GB/s | Million packets a second | Checked |
|---|---|---|---|---|---|---|
| link rate | (11, 9) (16, 11), the corner | 0 | 6 (NoC0) | 27.78 | 6.38 | every slot and count |
| link rate | (2, 4) (3, 6), the router's column, a middle row | 0 | 7 (NoC0) | 27.78 | 6.38 | every slot and count |
| link rate (section 1, sender C) | (0, 1) | 0 | - | 27.78 | 6.38 | every slot and count |
| link rate, two links a core | (10, 9) (15, 11) and (11, 9) (16, 11), the corner | 0 to 3 | 7, 21, 16, 14 | 27.77, 27.28, 27.75, 27.27: 110.07 | 25.3 | every slot and count |
| link rate (section 2, sender C) | spread, a core a link | 0 to 3 | - | 111.10 | 25.5 | every slot and count |
| page path, the default | (10, 9) (15, 11) and (11, 9) (16, 11), the corner | 0 to 3 | 7, 21, 16, 14 | 28.25 | 6.90 (pages) | 16,384 of 16,384 pages |
| page path, link cores first | (0, 0) (1, 2) and (1, 0) (2, 2), under the Ethernet row | 0 to 3 | 13, 15, 22, 8 | 28.32 | 6.91 (pages) | 16,384 of 16,384 pages |

The link-rate GB/s are over the receiver's span, the page path's over the host's,
submit to the last count (as in `experiments/islands/islands_pages`). Every packet carries
4352 bytes, every page 4096.

- The place costs nothing: from the corner, a middle row or (0, 1), at 6 to 21 hops,
  one sender sends 27.78 GB/s, the same 6.38 million packets a second; the 1.7 % less
  on links 1 and 3 is NoC1's, at 21 hops and at 14 alike. The page path's link cores
  under the Ethernet row run at the rate of the corner (28.32 against 28.25).
- The row under the Ethernet tiles is not near on the NoC: NoC0 runs down, so from
  y = 2 to the routers at y = 1 it wraps through every row (13 hops to link 0), while
  the corner at y = 11 is 6 hops away.
- The page path's four CPUs send 7.1 GB/s each against 27.8 from L1: the gap is its
  DRAM read, three reads in flight a CPU, and not where the link cores sit. Its count,
  one fused write a batch of 64, is the plain sender's own (every plain rate above
  counts so), so it is inside the 27.8; a count every packet would cost 10.56.

## Running it

Every job runs through a hang watcher and holds both cards. Build the host program
against a configured tt-metal source tree with `kmr build`, naming this folder's
`kernels` (an absolute path), then run it with both cards visible and
`TT_METAL_SLOW_DISPATCH_MODE=1`:

    kmr build host build/link_rate host/link_rate.cpp \
        -D 'KM_LINK_KERNELS="<this folder>/kernels"'
    build/link_rate --from 0 --packets 16384 --place spread          # question 1, 5
    build/link_rate --from both --links 4 --packets 16384            # question 2
    build/link_rate --from 0 --target dram --packets 16384           # question 3
    build/link_rate --from 0 --links 4 --dual 1 --packets 16384      # question 4
    build/link_rate --from 0 --links 1 --packets 16384 --place at:11,9   # question 6
    python rates.py <the logs of the runs>

Each run prints one JSON line with every pair's status and the host's checks;
`rates.py` turns it into the rates above. Receivers and senders bound every wait
(`--limit-s`, 10 s by default).
