# Five processors of one Tensix core

What each of a Blackhole Tensix core's five processors can do: the two data movers
(BRISC, where Komorebi loads the reader, on NoC 0; NCRISC, the writer, on NoC 1) and the
three TRISCs of the compute kernel (unpack, math, pack). Whether a TRISC reaches the
NoC, the same integer loop on each, NoC rates from one core, the wall clock against the
host's, and round trips through L1 between two processors. It is the data a dispatcher
that runs on one Tensix core needs.

## The program

A program in the runtime's form with hand-written kernels; every handler runs on core 0
(rank 0, logical (0, 0)) and writes its words into the core's row of an L1 tensor
(`shard_grid`, 2,304 words a core), which the host reads while the run is loaded.

| part | file |
|---|---|
| Handlers, processors, the words of a row | `kernels/five.h` |
| The loop and the two sides of a round trip, built alike for all five | `kernels/loop.hpp` |
| The movers' handlers (the reader and the writer run one body) | `kernels/mover.hpp`, `kernels/reader.cpp`, `kernels/writer.cpp` |
| The TRISCs: the loop, round trips, and NoC requests written into a command buffer's registers | `kernels/compute.cpp` |
| Driver: one JSON line per part, every NoC case checked word by word | `five.py` |

A TRISC has no NoC API of its own, so `compute.cpp` writes the request into the
registers of the movers' write (or read) command buffer of NoC 0 or 1, as
`ncrisc_noc_fast_write` does, and waits until the NoC's own acknowledgement (or read
response) counter moves. Before it starts, both movers flush their requests and say they
are ready; after it ends, the mover of that NoC takes the NoC's counters as its own
(`take_counters` in `mover.hpp`), since its barriers compare them with its count of
requests. The TRISC keeps and puts back every register it set.

## Results

Card: one Blackhole p150a, 12 x 10 = 120 cores, slow dispatch, AICLK 1,350 MHz (sysfs,
in every job), host load 0.4 to 0.8, 2026-11-19. The NoC figures of three jobs agree within 12 ticks; the table gives one job's. Kernels built at `-O2` for all five
(`opt`). Ticks are wall-clock ticks; the clock runs at AICLK (X4), so 1,350 ticks are 1 us.

| | What | How | Figure |
|---|---|---|---|
| X1 (a) | A TRISC reads a NoC register | TRISC 0, 1 and 2 read `NOC_NODE_ID` and `NOC_CFG(NOC_ID_LOGICAL)` of NoC 0 and NoC 1 | works on all three: the values BRISC reads (NoC 0 0x10611081 / 0x81, NoC 1 0x61124f / 0x81), 22 to 25 ticks for the two reads |
| X1 (b) | A TRISC writes 16 B to a neighbour's L1 | TRISC 0 on NoC 0 and NoC 1, TRISC 2 on NoC 0; target core (1, 0); the host reads the target's words | works: every word landed; acknowledged 206 to 217 ticks after the request |
| X1 (c) | A TRISC reads 16 B | TRISC 0 on NoC 0 and NoC 1, TRISC 1 on NoC 1, from core (1, 0) | works: every word right; answered 205 to 215 ticks after the request |
| X2 | One loop on each processor | 1,000,000 steps over a 4 KiB block in L1 (load, shift, add, xor, store); 7 runs alone and 7 with all five at once; median | alone 15,009,206 (TRISC 1) to 15,079,183 (NCRISC) ticks, about 15.0 ticks a step on every processor, the same sum on all five; all five at once (two jobs of 7 runs, medians): BRISC, NCRISC and TRISC 0 1.12 and 1.09 times slower (single runs 1.004 to 1.40), TRISC 1 and 2 1.04 and 1.03 (1.003 to 1.13) |
| X3 (a) | One 64-byte write, issue to flushed | 100 repetitions, median; neighbour (1, 0), farthest (11, 9), DRAM banks 0 and 7 | BRISC 0.184 / 0.264 / 0.274 / 0.275 us; NCRISC 0.179 / 0.259 / 0.270 / 0.270; TRISC 0 (NoC 0) 0.197 / 0.279 / 0.279 / 0.279 |
| X3 (b) | One 64-byte read | the same | BRISC 0.180 / 0.260 / 0.345 / 0.346 us; NCRISC 0.173 / 0.252 / 0.340 / 0.339; TRISC 0 0.197 / 0.279 / 0.360 / 0.360 |
| X3 (c) | N writes of 64 B to N cores, one flush | ranks 1 to N, N = 16, 64, 119 | BRISC 0.727 / 2.201 / 3.886 us (0.033 us a core at 119); NCRISC 0.590 / 1.647 / 2.856 (0.024); TRISC 0 0.899 / 3.359 / 6.047 (0.051) |
| X3 (c) | One multicast of 64 B | the largest rectangle inside one panel of consecutive NoC 0 columns: 60 cores (columns 6 to 11) | BRISC (NoC 0) 0.547 us; NCRISC (NoC 1) 0.481; TRISC 0 (NoC 0) 0.573; to the 54 cores of columns 0 to 5, rows 1 to 9: 0.438 / 0.568 / 0.458 |
| X3 (d) | Reading 64 B from each of N cores | N reads, one barrier | BRISC 0.623 / 1.778 / 3.104 us; NCRISC 0.533 / 1.410 / 2.414; TRISC 0 0.996 / 3.342 / 6.031 |
| X3 (e) | BRISC and NCRISC each writing half of the cores at once | one start time, the later end | 0.450 / 1.232 / 2.062 us for N = 16 / 64 / 119: 0.53 times BRISC alone, 0.72 times NCRISC alone at 119 |
| X4 | Wall clock against the host | two stamps on core 0, 2.0006 host seconds apart, while the other 119 cores run the loop on all five processors | 1,349.99 and 1,350.05 MHz in two jobs (bounds from the host's timing 1,349.78 to 1,350.28); sysfs AICLK 1,350 MHz at every stamp |
| X5 | Round trip through L1 | one processor writes k, the other answers k; 10,000 round trips | BRISC to NCRISC 79.9 ticks; BRISC to TRISC 0 79.5; TRISC 0 to BRISC 77.5; TRISC 0 to TRISC 2 62.1 |

Every NoC case was checked: each target's 64 bytes against the sender's, each read
against the target's words, the DRAM page read back; no line was wrong.

With the movers' NoCs swapped (`--swap-nocs`: BRISC on NoC 1, NCRISC on NoC 0), BRISC
still issues more slowly: 119 writes 3.550 us against NCRISC's 2.762, 119 reads 2.780 against
2.406, both at once 1.890. The difference between the movers belongs to the processor
(40 to 44 against 31 to 32 ticks a write in a batch of 119), not to the NoC; a multicast's
time follows the NoC and the rectangle's side (BRISC on NoC 1: 0.487 us to the 60 cores,
0.573 to the 54).

TRISC 0 takes 68 ticks a write in a batch of 119. It writes six to eight registers a
request (nothing kept in the command buffer between requests); whether that or a TRISC's
path to the registers sets the figure was not separated.

## Running it

On a card with slow dispatch, from the repository root:

    python experiments/core/five_processors/five.py probe --probe register:0:0
    python experiments/core/five_processors/five.py probe --probe write:0:0 read:1:0
    python experiments/core/five_processors/five.py loop ping clock
    python experiments/core/five_processors/five.py noc
    python experiments/core/five_processors/five.py noc --swap-nocs

A probe is `mode:noc:trisc` (mode `register`, `write` or `read`; TRISC 0 unpack, 1 math,
2 pack). Each part prints progress lines and one JSON line.
