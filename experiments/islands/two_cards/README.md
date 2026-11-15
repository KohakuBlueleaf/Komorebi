# Two cards

Komorebi runs a program on one card. This experiment asks three things before
anything for more than one card is built:

1. Can two cards each run a Komorebi program today, with the host passing commands to
   both, and what does each card's command cost while the other one also runs?
2. Can a Tensix core of one card write 64 bytes into the L1 of a Tensix core of the
   other card over the Ethernet links between them, in slow dispatch?
3. How fast are those links, from one sending core a link, one way and both ways?

Answers, measured on two Blackhole p150a cards joined by four Ethernet links:

1. Yes, with one process a card. Example 01's NOP costs 1.387 and 1.390 µs a command
   on either card alone and 1.398 and 1.399 µs with both running: under 1 %. Two
   sessions in one process do not open (see below).
2. Yes, through tt-metal's 1-D fabric, in slow dispatch, with all the cards in one
   process: 1 packet and 16 packets each way arrive exact.
3. 10.03 GB/s a link a direction with 4 KiB packets, the same on each of the four
   links; 40.1 GB/s one way over four links, and 40.1 GB/s each way at once.

## 1. Two sessions, the host passing the commands

`child.py` loads example 01's program (its kernels, buffers and handlers) on the card
it sees, prints `READY`, waits for a line on its standard input, then times rounds of
commands to the whole grid of 120 cores (`submit_many`, then `drain`): 20 rounds of
50,000 NOP, then 10 rounds of 20,000 MAP (each core adds one tile). The last MAP
round's 120 tiles are checked exactly. `two_processes.py` starts one child per card,
each with `TT_VISIBLE_DEVICES` set to its card, and sends the go line to all of them
once all are loaded. The timed windows of the two cards overlapped 99.8 %.

2026-11-14, host load 11.6 to 12.1, AICLK 1350 MHz on both cards while running (800
idle), microseconds a command (submit to event, pipelined; the median of the rounds):

| Case | card 0 NOP | card 0 MAP | card 1 NOP | card 1 MAP | wrong tiles |
|---|---|---|---|---|---|
| card 0 alone | 1.387 | 1.564 | - | - | 0 |
| card 1 alone | - | - | 1.390 | 1.560 | 0 |
| both at once, one process a card | 1.398 | 1.565 | 1.399 | 1.568 | 0 / 0 |

`one_process.py` opens `Session(card=0)` and then `Session(card=1)` with both cards
visible. Card 0 opens. Card 1 raises `ValueError: card 1 is not visible (0)`.
Several things keep two sessions out of one process:

- `Session.open` chooses its card by narrowing `TT_VISIBLE_DEVICES` to that card.
- The native `Device` opens a 1 x 1 mesh of the first visible chip and takes no chip
  number.
- tt-metal keeps one context a process.
- tt-metal's X280 runtime binds one MeshDevice a process. A second mesh cannot bind
  until the first closes, so two 1 x 1 sessions cannot both have a dispatcher.

## 2. A packet over the link

`host/link_packet.cpp` is a tt-metal host program. It calls `SetFabricConfig(FABRIC_1D)`
and opens a 1 x N mesh of every visible card in slow dispatch. In that open, tt-metal
starts its fabric routers on the active Ethernet cores of the links. The program then
places one sending core a link on the sending card: the Tensix core in row 0 nearest at
or right of the link's Ethernet tile. Each sending core gets a receiving core on the
peer card. The peer card and the hop count are parameters.

- `kernels/sender.cpp` fills its slots with a pattern of seed, sender, slot and word.
  It opens its connection to the link's router and sends each packet with a write and
  an increment of the receiver's counter.
  It then closes the connection. The router's free-slot wait, the flushes and the
  closing answer are polled with a bound of 10 s each, which the header's own calls do
  not do.
- `kernels/receiver.cpp` waits up to 10 s for its counter, zeroed by the host before the
  launch, to count every packet, checks every slot against the pattern, and writes
  its status and a copy of its first 1 KiB to DRAM.
- The host builds every program before it launches any, launches them all as one
  workload, reads both cards' status pages from DRAM, checks the copies itself, closes
  the mesh (which stops the routers) and prints one JSON line.

| Run | packets | bytes | result |
|---|---|---|---|
| card 0 to card 1 | 1 | 64 | exact (receiver and host check), sender 0.29 µs from first send to flush |
| card 0 to card 1 | 16 | 64 each | exact |
| card 1 to card 0 | 16 | 64 each | exact |

After every two-card job, example 01 ran on each card alone in its own job, and each
run was correct.

What the runs found:

- In slow dispatch, a second `EnqueueMeshWorkload` waits until the cores of the
  first are idle (`sd_mesh_command_queue.cpp`). With the receiving card's program
  launched first and the sender's second, the sender started only after the
  receiver had given up (its 10 s bound) and its packet came too late. Both cards'
  programs go in one workload; a persistent grid on each card does too.
- The open of both cards with the fabric took 3 to 6 s. The fabric's own start took
  3.2 s the first time (the routers were built) and 0.2 s after that.
- Every bound expiring is reported as a status code, not a hang: no job hung and no
  reset was needed.

## 3. The link's rate

4096-byte packets (the fabric's maximum is 4352), 8192 packets a sending core (32
MiB), a window of 256 slots on each side. The sender waits 20 ms after opening its
connection, so that the receiver is already counting. The rate is the bytes after the
first packet over the receiver's span from the first arrival to the last, from its
wall clock at AICLK 1350 MHz. Every slot was checked exact in every run.

| Links (one sending core each) | one way, each link | one way, sum | both ways, each direction |
|---|---|---|---|
| link 0 alone, link 1, link 2, link 3 | 10.03 GB/s each | 10.03 | 10.04 / 10.03 (link 0) |
| links 0 and 1 | 10.03, 10.03 | 20.06 | - |
| links 0 to 3 | 10.02 to 10.03 | 40.09 (card 0 to 1), 40.10 (1 to 0) | 40.10 / 40.10 |

By packet size, link 0, one way: 64 B 0.18 GB/s (2.8 M packets a second), 1 KiB 2.77
GB/s (2.7 M a second), 4 KiB 10.03 GB/s, 4352 B 10.56 GB/s. Below about 4 KiB this
sender is the limit, not the link. It waits for its writes to the router after every
fourth packet so that it can reuse its four packet headers.

Kohaku-TT's all-reduce on the same cards moves 16.8 MB each way in 288 µs over the
four links, with several cores draining what arrives. Computed from those two figures
on its page, that is about 58 GB/s each way, against 40 GB/s each way here from one
sending core a link.

Two or four sending cores on one link were not run. A router has one channel for a
core of its own card (`fabric_builder_config.hpp`, `fabric_builder_helpers.hpp`).
A connection takes over the previous one's position when it opens. So the API
supports one connected core a link at a time, and sharing one link between cores at
once needs tt-metal's mux cores.

## What it says for the design

- **One process or two.** One process a card works today with no change, but its
  cards can only talk through the host. A link between the cards needs the cards in one
  process: the process that opens the mesh starts the fabric on that mesh's chips, and
  refuses a mesh of fewer than two (`fabric_firmware_initializer.cpp`). tt-metal's X280
  runtime is written for a mesh of several chips (it binds the mesh and orders the
  X280 tiles by chip), so the smallest two-card shape with the link is one process
  with one 1 x N fabric mesh, one X280 runtime over it, and a dispatcher on each
  card's X280 cluster 0. Read, not run: no X280 image was started on a fabric mesh
  here. Komorebi's Session would then hold the mesh instead of one card.
- **Where a card's sending core sits.** It sits under a link's Ethernet tile, in the
  Tensix row next to the Ethernet row: here the cores (8 to 11, 0) under the Ethernet
  tiles at x 24 to 27. There is one per link and four links between these two cards.
  Each link is 10 GB/s a direction with 4 KiB packets.
- **What a card address costs in a command.** A packet is a header of 48 bytes (routes
  of up to 16 hops; 64 bytes for longer ones) and up to 4352 bytes of payload. A 64-byte command
  for a core of the other card is one packet. The card address in it is a route
  (which link, how many hops) beside the core's NoC address on its own card. One core
  sent 2.8 M such packets a second.
- **What the dispatcher needs.** It needs a sending core for each link toward a peer,
  one connection each, and a route table. Its first write must come after the peer's
  receiving side is up; in slow dispatch that means launching both cards' grids in one
  workload.

### More than two cards (from reading tt-metal's fabric; only two cards were run)

- **A chip that is not a neighbour.** In the 1-D fabric (`FABRIC_1D`, `FABRIC_1D_RING`)
  the cards are a line or a ring. A sender sets `to_chip_unicast(hops)` and picks the
  link in the direction of the target, so it reaches card k by the number of hops along
  the line. The 48-byte header encodes up to 16 hops, the longer ones up to 64. A ring
  adds the closing link. The 1-D fabric cannot reach a chip off its line. The 2-D
  fabric (`FABRIC_2D` and its torus forms) routes by destination chip and mesh id
  (`fabric_set_unicast_route`) through route tables that tt-metal writes into every
  chip at the open. It reaches any chip of the mesh, and other meshes through their
  exit chips.
- **Cards in between.** The routers on a card in between forward without that card's
  Tensix cores. A 1-D router has two sender channels: one for a core of its own card
  and one for packets that pass through. Two cards cannot show this: every packet here
  made one hop.
- **What a sender needs for card k.** For the 1-D fabric, the link (the direction
  toward card k) and the hop count; for the 2-D fabric, card k's chip and mesh id. A
  per-card route table holds, for each of the N cards, the link to use and the hop
  count: N entries of about two bytes (a link of four, hops up to 64). It holds one
  entry a link if traffic is spread over the links. The NoC address on card k is the
  same as on a single card.
- **What two cards test and what they do not.** They test one hop, both directions,
  each of the four links, all four at once, and both ways at once. They do not test
  forwarding through a card in between, the ring's closing link, the 2-D routes, or
  two flows meeting in one router.

## Running it

Every job runs through the hang watcher. A job that opens both cards holds both cards'
locks for its whole run.

Part 1, with tt-metal's environment set and the repository's `src` on the path:

    TT_VISIBLE_DEVICES=0,1 python one_process.py
    python two_processes.py --cards 0,1 -- --nop-rounds 20 --nop-jobs 50000 \
        --map-rounds 10 --map-jobs 20000
    python two_processes.py --cards 0 -- ...   # one card alone

Parts 2 and 3: build the host program against the tt-metal tree with `kmr build`,
naming this folder's `kernels` (an absolute path), then run it with both cards visible
and `TT_METAL_SLOW_DISPATCH_MODE=1`:

    kmr build host build/link_packet host/link_packet.cpp \
        -D 'KM_LINK_KERNELS="<this folder>/kernels"'
    build/link_packet --cards 2 --from 0 --packets 1 --payload 64
    build/link_packet --cards 2 --from 1 --packets 16 --payload 64
    build/link_packet --cards 2 --from both --links 4 --packets 8192 \
        --payload 4096 --window 256 --delay-us 20000
    python rates.py <the logs of the runs>

`--from` is a mesh column or `both`. `--to` and `--hops` name the peer and the hop count
(default: the next column, one hop). `--first-link` and `--links` choose the links.
Each run prints one JSON line with every pair's status.
