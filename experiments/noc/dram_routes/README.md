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

What these rates mean for a step of the chosen shapes: one step reads every layer's
weights once, 27.2 GB for the 62 layers in bfp8_b (products only), so a card reading
its weights at R GB/s runs at most R / 27.2 steps a second. At the card's
495.6 GB/s that is 18.2 steps/s; with every read on NoC 1 (at most about 250 GB/s, below)
9.2 steps/s; on NoC 0 alone (427.9 GB/s over the whole grid, movement.md) 15.7 steps/s.

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

## The decode's reads: every core, the two movers by a pattern

`decode.py` puts the reader on all 120 cores as a decode does: each core reads its own
1,024 pages of 1,088 bytes (bfp8_b tiles, 1.1 MB), blocks of 16, the blocks shared by
the two movers by a pattern (AAAB: mover B every fourth block, the default of a
class's read stream). A command waits for its last core, so the card's rate is every
byte over the time to the LAST core's end; steps/s is that rate over 27.2 GB a step.

| pattern | card GB/s | steps/s | a core's end, us (first / median / last) | the late cores |
|---|---|---|---|---|
| A | 403.6 | 14.84 | 81.2 / 231.1 / 331.3 | rows 0 to 2 (row 9: 83 us) |
| B | 246.7 | 9.07 | 380.1 / 488.6 / 541.9 | columns 1 to 5 |
| AB | **479.5** | **17.63** | 213.6 / 253.3 / **278.8** | column 5 |
| AAB | 469.0 | 17.24 | 159.4 / 223.2 / 285.1 | rows 0 and 1 |
| AAAB | 452.6 | 16.64 | 123.7 / 227.7 / 295.4 | rows 0 to 2 |
| AAAAAAAB | 427.1 | 15.70 | 100.6 / 229.8 / 313.1 | rows 0 to 2 |

Two banks only, every core reading them on one mover:

| banks | mover A: card GB/s, last core (us) | latest row | mover B: card GB/s | latest column |
|---|---|---|---|---|
| 0 and 4 | 101.3, 1,320.0 | row 0 (row 9: 208 us) | 106.4 | column 10 |
| 1 and 5 | 114.7, 1,165.6 | row 2 (row 0: 229 us) | 101.1 | column 10 |

- On NoC 0 the late cores are a ROW: the cores just downstream of where the DRAM
  endpoints' traffic turns down the columns. Banks 0 and 4 have both NoC 0 endpoints
  in the chip's bottom row of DRAM endpoints, and their traffic reaches logical rows 0
  to 2 first after the wrap: those rows end last (A alone: 331 us against 83 at row
  9). With banks 1 and 5 alone, whose endpoints are in rows 0 and 1, row 2 is last.
- On NoC 1 the late cores are a COLUMN (columns 1 to 5), as the row limit above says.
- Splitting the blocks evenly (AB) sets the two against each other: the last core
  ends 10 % after the median one (AAAB: 30 %), and the card reads at 479.5 GB/s
  against 452.6.

## A pattern chosen by the core's place

`decode.py --by-place` gives each core its own pattern of 16 blocks, more of them on
mover A where A's row is served early and B's column late, on pages of one tile
(1,024 of 1,088 bytes, blocks of 16) and of two (512 of 2,176 bytes, blocks of 8).
`place` sets A's share from each mover's end alone (pattern_A by row, pattern_B by
column), `place_centred` the same differences about one half; `--by-place mixed` from
each mover's own end under AB (`mixed`), and half of those differences (`mixed_half`).
The ceiling is 495.6 GB/s, 18.22 steps/s.

| pattern | tile pages: card GB/s | steps/s | last / median core | two tiles: card GB/s | steps/s | last / median core |
|---|---|---|---|---|---|---|
| AB (two runs) | 477.8 / 478.3 | 17.57 / 17.58 | 1.04 to 1.11 | **497.2** / 492.8 | **18.28** / 18.12 | 1.05 to 1.07 |
| AAAB | 451.9 | 16.61 | 1.32 | 465.9 | 17.13 | 1.19 |
| place | 460.2 | 16.92 | 1.27 | 472.7 | 17.38 | 1.19 |
| place_centred | 466.8 | 17.16 | 1.10 | 485.6 | 17.85 | 1.11 |
| mixed | 484.8 | 17.82 | 1.23 | 492.1 | 18.09 | 1.15 |
| mixed_half | 493.4 | 18.14 | 1.13 | 496.8 | 18.27 | 1.09 |

- Rates do not stay put when the shares move: more A in rows 8 and 9 slows rows 0 to
  2's A, which shares their column link; the rates of a mover alone overrate A
  everywhere, and that table falls to AAAB's level.
- AB for every core already ends the last core within 1.11 of the median, its rows
  flat; on pages of two tiles it reads at the DRAM limit. A pattern by place adds at
  most 3 % on tile pages and nothing on pages of two tiles.

## The decode chain's reads, one difference at a time

`chain.py` reads as a decode product's cores do, adding one difference to the reads
above at a time. The steps:
- `even`: equal runs of pages.
- `w`: each core's own W pages: its K rows, 7 pages of two tiles a row, in blocks of 2
  rows on AB.
- `w_x`: plus X read in step, 2 tiles a block. Every group reads the same X: in gate,
  48 cores a page. X is on AAAB, a stream's default.
- `w_x_ab`: the same with X on AB.
- `--consumer` (`_c`): a consumer takes the blocks in order, 126 cycles a W tile. A
  mover's block waits until the consumer took the block 4 of its own blocks before it,
  as a read pair holding 4 blocks a half does.

GB/s is W's bytes, and all bytes, over the time to the last core's end. steps/s is W's
rate over 14.4 GB (bfp4_b) or 27.2 GB (bfp8_b) a step. The ceiling is 495.6 GB/s.

| gate (96 cores) | W GB/s | all GB/s | steps/s | last / median core | rows 0 to 7, us |
|---|---|---|---|---|---|
| bfp4_b even | 469.5 | 469.5 | 32.60 | 1.12 | 126 to 129 |
| bfp4_b w | 429.4 | 429.4 | 29.82 | 1.08 | 125 to 137 |
| bfp4_b w_x | 378.9 | 475.1 | 26.31 | 1.08 | 143 to 168 |
| bfp4_b w_x + consumer | 345.4 | 433.1 | 23.99 | **1.23** | **135 to 180 (row 0)** |
| bfp4_b w_x_ab + consumer | 375.5 | 470.9 | 26.08 | 1.08 | 150 to 164 |
| bfp8_b w | 439.7 | 439.7 | 16.17 | 1.11 | 229 to 255 |
| bfp8_b w_x | 426.0 | 483.2 | 15.66 | 1.09 | 246 to 278 |
| bfp8_b w_x + consumer | 422.1 | 478.8 | 15.52 | 1.10 | 234 to 272 |

- X re-read by every group costs W 11 to 20 % of the card. Every byte moves at 467 to
  486 GB/s, but W gets 378 (bfp4_b) and 412 to 426 GB/s (bfp8_b).
- Reads alone stay even. With the consumer, X on AAAB ties mover A to compute's pace
  in bfp4_b: its X half fills first, and NoC 0's late rows (0 to 2) then set the
  core's end. With X on AB the rows stay flat. In bfp8_b a block takes longer to read
  than to compute, so the consumer changes nothing.

## Running it

On a card, with tt-metal's environment set (about 14 s each with the open):

    python experiments/noc/dram_routes/routes.py   # rows and columns, 14 launches
    python experiments/noc/dram_routes/decode.py   # the decode's reads, 10 launches
    python experiments/noc/dram_routes/decode.py --by-place   # by place, 8 launches (mixed: 6)
    python experiments/noc/dram_routes/chain.py   # the chain's reads, 11 launches
    python experiments/noc/dram_routes/chain.py --consumer   # with a consumer, 8

Each prints one JSON line a run; a run whose readers did not all finish is not ok.
