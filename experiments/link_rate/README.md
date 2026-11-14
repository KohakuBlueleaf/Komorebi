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
| `--place` | the sending cores: `under` (the Tensix core of row 0 or 1 nearest at or right of the link's Ethernet tile), `row` (along row 0), `spread` (over rows and columns) |
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

## 1. One link, and what each difference costs

16384 packets of 4352 bytes (71 MB), window 64 slots, batch 64, one link (link 0),
card 0 to card 1, a receiver kernel on card 1. GB/s, rx (tx within 0.02):

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
packet and the wait, which gives the 10.56 here.

## 2. Four links, senders spread, NoC0

16384 packets of 4352 bytes a link, plain writes, batch 64, no wait, one sending core
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
| row 0, NoC0 | top rows | card 0 to 1 | 26.60 | 26.59 | 26.72 | 26.76 | 106.67 |
| spread, NoC1 | top rows | card 0 to 1 | 26.38 | 26.64 | 26.61 | 26.64 | 106.27 |

Four links carry four times one link, 107 GB/s one way, and both ways at once
carry the same 107 GB/s each way (214 GB/s in sum): the links do not share anything
that this load fills. Where the senders sit and which NoC they use move the sum by
under 1 %: senders along one row lose 0.4 %, NoC1 0.7 %. Four links do not reach the
120 GB/s of the rougher measurement because a link here carries 26.8 GB/s, not 30.

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
