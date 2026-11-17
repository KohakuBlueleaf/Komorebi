# DRAM reads along a row and down a column, on each NoC

Blackhole's NoC 0 carries a packet right, then down; NoC 1 carries it up, then left
(tt-isa-documentation, BlackholeA0/NoC/RoutingPaths.md). A DRAM read's response comes
back on the reader's NoC, so on NoC 1 its last leg runs along the reader's row and on
NoC 0 down the reader's column. This experiment measures what that means when several
cores read at once: N readers in one row against N in one column, on each NoC.

`reader.cpp` reads 2,048 pages of 2,048 bytes of an interleaved DRAM tensor into its own
L1 shard, 16 reads before each barrier, after the run's cores meet at a counter so that
they start together (skew under 0.5 us). It is launched on RISC-V 0 with NoC 0 (mover A)
or RISC-V 1 with NoC 1 (mover B), as the movers run. `routes.py` reports each reader's
rate over the time every reader of the run was reading.

## Result

One Blackhole p150a, AICLK 1350 MHz, the 12 x 10 grid (logical coordinates). GB/s a
reader (min, median, max) and the run's sum.

| readers | mover A (NoC 0) | sum | mover B (NoC 1) | sum |
|---|---|---|---|---|
| row 0, 10 cores | 27.4 / 27.5 / 27.6 | 274.8 | 10.2 / 10.3 / 10.9 | 105.2 |
| column 0, 10 cores | 12.1 / 12.2 / 13.5 | 123.6 | 23.4 / 23.6 / 23.7 | 235.5 |
| rows 0 and 5, 10 each | 15.3 / 15.4 / 15.5 | 307.4 | 7.9 / 8.2 / 11.7 | 189.3 |
| columns 0 and 6, 10 each | 10.6 / 11.0 / 13.2 | 227.6 | 12.4 / 12.6 / 12.7 | 250.9 |
| row 0, 12 cores | 24.2 / 24.4 / 24.5 | 292.5 | 8.9 / 9.0 / 11.4 | 110.1 |
| rows 0 and 5, 12 each | 13.4 / 13.5 / 13.7 | 324.4 | 6.7 / 8.2 / 9.9 | 190.8 |
| row 0, 12 cores, bank 0 only | 5.3 / 5.3 / 5.4 | 64.1 | 2.2 / 2.6 / 19.6 | 62.4 |

What these rates mean for decoding: one token reads every layer's weights once, 27.2
GB for the 62 layers of the assumed shapes in bfp8_b (products only), so a card
reading its weights at R GB/s decodes at most R / 27.2 tokens a second. At the card's
495.6 GB/s that is 18.2 tok/s; with every read on NoC 1 (at most about 250 GB/s, below)
9.2 tok/s; on NoC 0 alone (427.9 GB/s over the whole grid, movement.md) 15.7 tok/s.

- A row of readers on NoC 1 shares the row's leftward links: about 105 to 110 GB/s a
  row, whatever the number of readers past four. A column of readers on NoC 0 shares
  its downward links: about 124 GB/s a column. The other pairing does not share a
  link and each reader runs near its own rate (23 to 28 GB/s at 16 reads in flight).
- Every NoC 1 response first climbs one of the two DRAM columns. Readers spread so
  that no row is shared (columns 0 and 6 on NoC 1) still total 251 GB/s: the two DRAM
  columns are NoC 1's limit, the same as the whole grid's 247.2 GB/s on mover B
  (movement.md).
- One bank gives about 63 GB/s on either NoC; eight banks at that rate are the card's
  DRAM limit (about 500 GB/s, reached with both movers).
- A shared link is not shared evenly: on NoC 1, readers in columns 1 to 5 get 20 to
  30 % less than those in 6 to 10, and on one bank the share falls from 19.6 to 5.2
  GB/s along the row. A command that waits for its slowest reader waits for these.

## Running it

On a card, with tt-metal's environment set (14 launches, about 14 s with the open):

    python experiments/noc/dram_routes/routes.py

It prints one JSON line a run; a run whose readers did not all finish is not ok.
