# raster_passes

How much faster the rasterizer's per-triangle passes (COUNT and SETUP) and its fill
can run on a Tensix core when each part goes to the unit that suits it: the vector
unit (SFPU), the pack thread's RISC-V core and its vector extension, or a second
RISC-V core. Five probes, each against the rasterizer of `projects/raster` as it is
today, run on the same card the same day.

Everything here is a copy: `kernels/` and `thread/` are `projects/raster/kernels` and
`projects/raster/thread` with the probes' changes behind compile-time switches;
`lanes.*` and `kernels/lanes_*` are a one-core program of their own.

```
python experiments/raster_passes/frames.py --scene s1m --frames 5 --image --lists 400
python experiments/raster_passes/lanes.py --scene s1m --tiles 16 --side 4
```

`frames.py` runs the thread-driven frame of `projects/raster/card_thread.py` built
from this folder's copies and prints one JSON line: the frame's passes in µs, every
core's per-part counters (unless `--no-profile`), the image against the reference,
`--lists N` sampled list pages against the host's. Its switches: `--define NAME=VALUE`
(the kernels' switches below), `--ranges equal|history|predicted` and `--weights`.
`lanes.py` fills tiles on one core (see probe 4). The `results/` folder holds the JSON
of every run quoted here (the scene `s1m` is 10^6 triangles, `s100k` 10^5).

Conditions of every number: Blackhole p150a (card 1), AICLK 1350 MHz throughout (min =
mean in every run), X280 at 1000 MHz, host load 4 to 10. Every run below was exact:
the image equal to the reference bit for bit, and the sampled list pages equal to the
host's.

## Today

`projects/raster/bench_thread.py --scene s1m --runs 3 --no-balance` (`results/main-s1m.json`):
a frame of 12,124 µs, of which COUNT 3,826, SETUP 5,064 and FILL 2,108. At 10^5
(`results/main-s100k.json`): 2,557 µs, COUNT 450, SETUP 672.

## Probe 0: where COUNT and SETUP spend their time

The copy with a wall-clock counter around every part of a block of 32 triangles on each
thread (`results/profile-s1m.json`). The counters cost about 4 %: the copy without the
probes' changes runs a frame in 12,645 µs with no profile page written
(`results/copy-s1m.json`) against main's 12,124. 10^6, µs, slowest core / mean core:

| Part | COUNT | SETUP |
|---|---|---|
| The pass | 3,980 / 3,158 | 5,352 / 3,925 |
| Copy in (2,304 bytes into Dst) | 129 / 129 | 129 / 129 |
| Vector unit | 144 / 140 | 234 / 230 |
| Records out (Dst to L1, strided) | 1,203 / 1,203 | 1,209 / 1,208 |
| Lane scan | 178 / 178 | 177 / 178 |
| Per triangle (keep test, finishing) | 1,171 / 887 | 1,294 / 966 |
| Per entry (tiles, counters or slots) | 1,109 / 533 | 2,184 / 1,047 |

- The pack thread does all the work; the other threads wait: unpack 99 % of the pass,
  the reader 95 %, the writer 87 to 100 %. The math thread runs no code in these passes.
- The slowest core takes 1.26 (COUNT) and 1.36 (SETUP) times the mean: the cores get
  equal numbers of blocks, not equal work (8,297 kept triangles on the slowest core,
  4,578 on the mean).
- Reading the vector unit's results out of Dst costs about 6 cycles a word through
  strided or segment vector loads; a scalar load from L1 costs 8 cycles or more.

## Probe 1: block ranges by predicted work (kept)

The host cuts the blocks into ranges of equal predicted cost instead of equal counts.
A least-squares fit over the 120 cores of probe 0 (residuals 1 to 2 %) gives the cost of a
block: 24,056 cycles + 445 per kept triangle + 212 per (triangle, tile) + 17,697 per
clipped triangle. One cut serves both passes, since SETUP's slots come from COUNT's
counts per core.

10^6 (`results/balance-s1m.json`): slowest / mean 1.26 to 1.07 (COUNT) and 1.36 to 1.06
(SETUP); frame 12,633 to 10,793 µs. The rasterizer's own cut from the last frame's
ticks gives 1.09 and 1.04 from the second frame on (`results/history-s1m.json`). The
X280 thread cannot know this frame's counts before COUNT, so in the program the cut
would come from the last frame's counts (same camera) or a cheap count first.

## Probe 2: finishing each triangle on the vector unit (kept)

`PASSES_FINISH=1`: the vector unit computes, per lane, the keep and crossing flags, the
box as integers and as float16 columns, the top-left thresholds, the triangle number and
the record's last words (`finish_lanes` in `kernels/setup.hpp`: compares that write -1
or 0, AND, OR, XOR, shifts and integer subtractions, no condition codes). The pack
thread reads only the flags and boxes, as pairs of Dst rows by unit-stride loads (192
words a block), and in SETUP each kept triangle's record from Dst by segment loads
(`PASSES_RECORDS=0`; copying all 32 records out and transposing them with scalar code
was slower: per entry 4,280 against 2,488 µs on the slowest core).

Cycles per triangle on the mean core (vector part / scalar part): COUNT 239 / 266 to
49 / 146, SETUP 254 / 363 to 70 / 344. 10^6 (`results/finish-s1m.json`): COUNT 4,035 to
1,929 µs, SETUP 5,356 to 4,452, frame 9,620.

## Probe 3: a second stage on the writer core (kept)

`PASSES_SECOND_STAGE=1`: the pack thread hands each kept triangle to the writer core
(RISCV_1) through the out pages, one box word (first and last tile column, first and
last row) in COUNT, the box word and the 128-byte record in SETUP. The writer counts
(counters and two-row histograms in its L1, then the counters to DRAM) or sends the
record to each item's next slot by NoC (the record is not copied per entry).

The hand-over costs the pack thread 75 cycles per kept triangle (COUNT) and 139 (SETUP,
with the copy); the writer's stage takes 504 µs on the mean core (COUNT) and 764
(SETUP). 10^6 (`results/second-stage-s1m.json`): COUNT 1,929 to 1,094 µs, SETUP 4,452
to 2,826, frame 7,157.

## Probes 1 to 3 together

Weights refitted on the new costs (6,347 per block + 408 per kept triangle + 22 per
entry + 19,433 per clipped triangle), without the profile page:

| Scene | Today (main) | Copy, unchanged | Probes 1 to 3 | COUNT | SETUP |
|---|---|---|---|---|---|
| 10^6 | 12,124 µs | 12,645 | 6,486 | 3,826 to 1,163 | 5,064 to 2,079 |
| 10^5 | 2,557 | 2,613 | 1,974 | 450 to 196 | 672 to 340 |

(`results/all-s1m.json`, `results/all-s100k.json`, `results/copy-s100k.json`). FILL
(2,108 µs at 10^6) is now a third of the frame.

## Probe 4: small triangles as lanes in the fill (kept, size rule found)

Today the fill takes one triangle at a time across a whole tile: 32 lanes are 32
columns, and a row costs the same whether the triangle covers one pixel of it or all
32. `lanes.py` fills single tiles on one core four ways and checks each against
`reference.raster.fill_tile` (all five planes, every pixel):

- small triangles (box clipped to the tile at most `side` x `side`) by rows, today's code;
- the same small triangles as lanes: 32 triangles in a 4 KB page, one per lane, in the
  order of the Dst window. For each offset of the page's box (at most 4 x 4) the vector
  unit computes the box and edge tests, w and the three colours with the fill's
  operations in the fill's order, and sets bit k of a per-lane word where the lane's
  pixel at offset k is inside. The pack thread then tests each inside pixel against the
  tile's depth in Dst: larger w wins, and equal w with a lower triangle number, so the
  result does not depend on the order in which pages come;
- every triangle by rows;
- the larger by rows, then the small as lanes.

Side 4, cycles per small triangle (16 tiles drawn at random from 10^6, 6,788 small of
11,154 entries; the 8 busiest tiles, 55,267 of 57,053):

| Tiles | By rows | As lanes | Copy in | Vector unit | Depth test | Pixels inside |
|---|---|---|---|---|---|---|
| random 16 | 425 | 281 | 36 | 34 | 211 | 2.21 |
| busiest 8 | 408 | 198 | 34 | 20 | 144 | 1.49 |

The depth test costs 96 cycles a pixel: a scalar load from the Dst window takes about
10 cycles, and loads do not overlap (64 dependent loads 771 cycles, 64 independent ones
unrolled 614; a window store 4.6 cycles; a dependent L1 load 11).

The size rule, from a sweep of `side` (larger by rows plus small as lanes, divided by
all by rows; `results/lanes-*-side*.json`):

| Side | random 16 | busiest 8 |
|---|---|---|
| 1 | 0.987 | 0.949 |
| 2 | 0.904 | 0.669 |
| 3 | 0.820 | 0.505 |
| 4 | 0.778 | 0.490 |

Lanes win at every size measured, and side 4 is the largest that fits: 16 offsets of 4
vectors each fill Dst from vector 192 on. Rule: an entry whose box, clipped to its tile,
is at most 4 x 4 pixels goes to lanes (73 % of the entries at 10^6). The frame does
not use it yet: SETUP would have to write lane pages and the lists carry them. If every
tile behaved like the random sample, FILL at 10^6 would drop from 2.11 to about 1.6 ms
(an estimate, not measured).

## Probe 5: a scalar into all lanes through SFPCONFIG (dropped)

The fill sets up each triangle by loading 22 numbers into the vector unit as immediates,
two `SFPLOADI` a number. The question was whether a record read from Dst could reach
all 32 lanes without immediates. The ISA documentation limits the routes:
`SFPCONFIG` writes only LRegs 11 to 14, and only from lanes 0 to 7 of LReg 0, repeated
(lane & 7) to the 32 lanes (`WormholeB0/TensixTile/TensixCoprocessor/SFPCONFIG.md`,
which the Blackhole page refers to); `SFPSHFT2` moves lanes only by one within groups
of eight (`BlackholeA0/TensixTile/TensixCoprocessor/SFPSHFT2.md`). So a number must
first sit in lanes 0 to 7. With `LANES_BROADCAST=1` the pack thread writes eight copies
of each number (two numbers in one row of Dst, even and odd columns), then `SFPLOAD`
and `SFPCONFIG` put it in every lane of LReg 11.

The bits are exact: 64 patterns (zeros, denormals, infinities, NaN, all ones, a
sequence) through this path differ in 0 of 2,048 lanes, and the 16 tiles' images are
exact. The cycles are not: 644 per triangle against 243 today
(`results/broadcast-*.json`), of which 399 are the eight copies of 21 numbers (176 words
into Dst, 2.3 cycles a word) and 245 the vector unit's instructions, as many as before
(`SFPLOAD` and `SFPCONFIG` are two instructions a number, like two `SFPLOADI`). The
other compute RISC-V cores have no vector extension to take the copies over.

## What the rasterizer should adopt

1. Finishing on the vector unit (probe 2) with the records read from Dst by segment
   loads for kept triangles only.
2. The second stage on the writer core for counting and for sending records to their
   slots (probe 3).
3. Block ranges cut by predicted cost, the cost from the last frame's counts (probe 1).
4. Small triangles as lanes in the fill, for entries whose box clipped to the tile is at
   most 4 x 4 (probe 4); SETUP and the lists must carry lane pages first.

Probes 1 to 3 together halve the frame at 10^6 (12.1 to 6.5 ms) and take a quarter off
at 10^5 (2.56 to 1.97 ms).

## Files

- `frames.py`: the frame driver. `lanes.py`: the one-core fill driver.
- `kernels/setup_compute.cpp`, `kernels/setup.hpp`, `kernels/setup_writer.cpp`: the
  passes with the switches `PASSES_FINISH` (probe 2), `PASSES_RECORDS` (0: kept
  records from Dst; 1, the default: all records out) and `PASSES_SECOND_STAGE` (probe 3).
- `thread/frame.c`: the X280 thread, keeping ranges given by the host (`--ranges
  predicted`).
- `kernels/lanes.h`, `kernels/lanes_*.cpp`: the one-core fill, `LANES_BROADCAST` (probe 5).
