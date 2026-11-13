# Rasterization on the card

The rasterizer of `ref/raster.py` on a Blackhole card's Tensix cores, on the resident
worker runtime (`komorebi/device/km_worker.hpp`), loaded and driven through the host
package (`komorebi.Session` and `Run`, the dispatch kernel issuing the records).
Three ways in: host-binned (the host does setup and binning with the reference's own
functions; the cores fill tiles); the per-triangle stages on the cores too (COUNT and
SETUP, below; the host takes the step between them); and the whole frame on the card,
conducted by a thread on the X280 cores beside the dispatch kernel, which takes that
step itself (below). Every scene's image is bit-exact against the reference.

## The thread program

```
host:    records = setup(scene); lists = bin(records)        # the reference's own functions
         assign tiles to cores; write per core, to DRAM: its tile list (tile id, count, then the
         triangle records in triangle order, 128 bytes each: a, b, c of three edges, top-left bits,
         the 1/z plane, three colour planes, the pixel box, the triangle number)
card:    one record FILL(frame) to the team of all worker cores
core:    for each of my tiles:
             depth = 0; colour = background                  # four planes of 32 x 32 float32
             for each triangle record, in order:             # reader streams the list from DRAM
                 inside = edges and box, per pixel           # all 1,024 pixels of the tile are lanes
                 w      = 1/z plane, per pixel
                 win    = inside and w > depth               # strict: the lower index keeps a tie
                 depth  = where(win, w, depth)
                 colour = where(win, colour planes, colour)  # same value as shading the winner last
             writer: the tile's colour and depth to the frame in DRAM
```

A fifth plane holds the winner's triangle number (as a float, -1 where empty), so the
frame carries the same `tri` image as the reference.

## Who does what on a core

| Kernel | Thread | Work |
|---|---|---|
| `reader.cpp` | data mover 0 | reads the core's directory page (first list page, page count, tile ids), streams the list pages (2 KB, 16 slots) into the input buffer |
| `compute.cpp` | unpack | waits for each list page (only this thread can wait on a buffer) and passes its address to the pack thread through the mailbox; frees it when the pack thread answers |
| `compute.cpp` + `fill.hpp` | pack | issues every vector-unit instruction: tile init, per triangle the broadcast constants, per row the edge tests and the plane updates; then copies the five planes from the destination register to five output pages |
| `compute.cpp` | math | idle |
| `writer.cpp` | data mover 1 | sends each tile's five pages to the frame; writes tiles without triangles itself from a constant buffer; at the end writes the profile page |

`raster.h` holds the layouts shared by the host and the kernels: the 128-byte record,
the tile header and end slot, the directory page, the frame (per tile five 4 KB planes
in the destination register's window order). `lists.py` builds them; `card.py` loads the
program and runs a segment's records through the host package; `run.py` runs and checks one scene; `bench.py` times a loop
of frames and samples the card's power; `arith.py` is the arithmetic check;
`pictures.py` draws the card's image, its difference from the reference and the busy
map. `--out` writes the card's result in the layout of `reference/README.md`, so the
reference's comparison, picture and table scripts take it with `--external`.

```
python -m projects.raster.arith --n 1000000          # arithmetic check, bit for bit
python -m projects.raster.run --scene tiny --tiles 367 --cores 1
python -m projects.raster.run --scene s1m --out results/card --json s1m.json
python -m projects.raster.bench --scene s10k --seconds 5 --out results/card
python -m projects.raster.pictures --scene s1m --card results/card --prof s1m_prof.npz \
    --out pictures
python -m projects.raster.reference.agree --scenes s1m --external results
```

## Decisions, each with its measurement

Cycles are the pack thread's wall clock (1.35 GHz), summed over cores by the kernels'
own counters; frame times are a record's issue-to-acknowledge time, from the dispatch
kernel's event (its 50 MHz timer). The tables below were measured with an earlier
driver, a test fixture on a 120th core issuing the records to 119 workers; the same
kernels under the host package give the same frames (see "The driver").

**Exactness.** The vector unit's multiply-add rounds once; a product is taken as
`a * px + (-0)` and a sum as `x * 1 + y`, which are exact float32 operations. On
1,000,448 inputs (900,404 from the 10^4 scene's records, 100,044 edge cases), all six
per-pixel outputs (two products, the edge sum, the 1/z sum, the edge and depth
comparisons) equal numpy bit for bit on the scene inputs; every difference on the edge
cases is the unit flushing a denormal result to zero. Comparisons use the total order
of the bit patterns (`-0 < +0`); the top-left rule `e >= -0` becomes `e > 0x80000001`.

**Tile to lanes.** One pixel row of a tile is one 32-lane vector; a triangle runs over
the rows of its box inside the tile, one step per row. Each plane is 32 vectors.

**Where the planes live.** In the destination register for the whole tile (five planes
of 32 vectors, the py of each row, and the triangle's constants: 214 of its 256
vectors); they go to L1 once per tile through the register's memory window. Read-out:
16.9 k cycles per tile with scalar copies, 5.7 k with the pack core's vector extension
(32 words per load and store, 1.1 cycles per word).

**How a triangle's numbers reach the lanes.** The pack thread pushes them as
immediates (two 16-bit loads per float32), forms `a * px` once per triangle and stores
each broadcast constant as a vector in the destination register, from where each row
loads it. Cycles per (triangle, tile) for this setup, measured during development on
intermediate kernel versions (not reproducible from the repository; the last row is the
committed kernel):

| change | cycles per triangle | 10^4 frame (us) |
|---|---|---|
| first version: record read word by word from L1, branches per immediate | 620 | 329.6 |
| record copied to local memory, branch-free immediates, read-out on the vector extension, empty tiles by the writer | 341 | 203.4 |
| the row's edge tests and 1/z plane (30 instructions) replayed from the replay buffer | 341 | 166.4 |
| record read straight from L1 by the compiler (tried, dropped) | 449 | 176.3 |
| record copied by one vector load and store | 282 | 161.2 |
| box columns as float16 immediates (one load each), edge thresholds and triangle number kept in registers over the rows | 261 | 158.3 |

What is left is the core's issue of the immediates: 44 data-dependent load
instructions at 3 core instructions each (shift or mask, add the opcode, store to the
instruction buffer), plus about 35 fixed instructions; the vector unit itself runs about
80 instructions per triangle.

**The alternative: the numbers as data through the unpacker** (a probe that is not in the
repository; one core, 2,048 triangles, pack-thread cycles per triangle including its rows). The host
lays out a triangle's 21 broadcast vectors and four register values as one 4 KB
float32 tile, already per lane (`a * px` included); the unpacker writes it straight
into the destination register (`copy_tile`, fp32 unpack-to-dest, full-sync); the pack
thread copies the 21 vectors to where the rows read them, loads the four registers and
frees the register for the next tile. The bits arrive exact (0 differing words, the
`-0` neighbour threshold included). Mode 0 is the fill's own `load_triangle` from a
record.

| rows per triangle | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| immediates (cycles per triangle) | 272.5 | 332.5 | 392.4 | 494.4 | 721.5 |
| unpacker (cycles per triangle) | 182.3 | 182.3 | 230.4 | 350.4 | 590.4 |

On the core the unpacker wins by 90 to 160 cycles per triangle (1.5 times at 0 rows, 1.7
at 2, where 10^6 sits). The tile stays in L1 in the probe; 182 cycles is the
unpacker's 4 KB tile, and the rows overlap it only partly. The cost is elsewhere: 4 KB
per (triangle, tile) instead of a 128-byte record, 32 times the list bytes. At 10^6
that is 654 k entries x 4 KB = 2.7 GB per frame, which the host would have to build and
the cores read from DRAM. At an assumed 0.5 TB/s that is about 5 ms, more than today's
2.0 ms frame (an estimate, not measured). It pays only if the card itself expands a
record into its tile, which was not tried.

**Skipping outside the box.** Rows outside the triangle's box are not issued (the
core clips the box to the tile: a few instructions per triangle). Columns outside the
box fail edge 0, whose `a * px` is set to -inf there once per triangle, so a row costs
nothing extra for the box. Rows issued per (triangle, tile): 23.6 (tiny), 10.1 (10^4),
4.8 (10^5), 2.65 (10^6), against 32 without the skip. A row is 54 vector instructions
and costs 59.5 cycles (64.6 at 10^6, where most triangles have one or two rows).

**Reader page size.** 2 KB pages (16 records), 8 in flight. The pack thread's wait for
list pages stays below 1 % of its time at every size, so other sizes were not tried.

**Tile assignment.** Balanced: tiles by estimated cost, largest first, each to the
least-loaded core. The cost is measured: 260 cycles per (triangle, tile), 60 per row,
6,800 per tile with triangles (init, read-out), 2,000 per empty tile (writer only).

**Heavy tiles become bands of rows.** When one tile is estimated above a core's mean
load, every tile above a third of that mean is cut into bands of even rows of about
equal cost (at most 16). A band is an ordinary item: its header and its id carry its
rows, the pack thread runs and reads out only those rows, and the writer writes only
their bytes of the tile's pages (two rows of a plane are 256 contiguous bytes). Every
pixel still sees the same triangles in the same order, so no merge is needed and the
frame stays exact. A triangle that touches two bands is set up twice: (triangle, tile)
entries grow by 8.6 % at 10^5 and 4.4 % at 10^6. Below 10^5 no tile is above the
mean and nothing is split (splitting anyway made the estimate worse at tiny and 10^4).

## The driver

All 120 cores are workers. `card.py` writes the lists, the directory and empty frame
and profile tensors to DRAM, then loads the program (`Session.load`): with slow
dispatch a tensor write waits for every core to be idle, which a resident grid never
is, so nothing is written while the program is loaded. A FILL record goes to every core
(a0 directory, a1 lists, a3 pages per item, a4 profile), one in flight at a time
(mailbox depth 1), so its event's time is that frame alone. The frame is eleven tensors
of at most 2 MB (510 pages), whose addresses every directory page holds; after the stop
they and the profile are read by DRAM banks (`Session.read`), one tensor at a time.

Single frames through it (2026-11-13, card 1, 1,350 MHz, host load 5 to 8; `python -m
projects.raster.run --scene <s>`): tiny 119.3 us, 10^4 139.1 us, both exact, against
119.5 and 139.4 us with the earlier driver. Reading the frame back (20 MiB) took 11 to
14 s, apart from the frame time.

## COUNT and SETUP on the cores

The per-triangle stages run on the same 120 cores as a program of their own
(`kernels/setup_reader.cpp`, `setup_compute.cpp`, `setup_writer.cpp`; host side
`passes.py`, driver `card_passes.py`), two passes over the triangles:

```
upload:  the mesh de-indexed, 32 triangles a block: 18 vectors (x, y, z and colour of three
         corners) already in the order of the vector unit's register window, 2,304 bytes
COUNT    core c, its contiguous share of the blocks:
             vector unit: camera space, screen (1/z by the exact reciprocal), area, box
             pack thread, per triangle: keep rule; a triangle across the near distance is
                 clipped and its pieces go through the vector unit at its place
             entries and rows of every tile it touches += 1, += its rows there;
             in L1, per tile, how many of its entries start and end in each 2-row group
         writes its two counters (1,024 words each) to DRAM
host:    tile costs from the counters; the tiles above a third of a core's mean load
HIST     core c writes those tiles' start and end counts (64 bytes a tile)
host:    their bands: cut where the rows' estimated cost reaches j / k of the tile's, each
         band's entries = those starting before its last row less those ending before its
         first; items (tiles and bands) to cores; each core's list (header, records, end
         slot, as the fill reads it) laid out with headers, end slots and the directory
         written; the slot of every (core, item) = the item's header + 1 + the records of
         lower cores
SETUP    core c, the same share again: the same steps plus edges, 1/area and the 12 planes;
         every kept record goes once to each item it reaches, at that item's next slot
FILL     as above, over the lists in place
```

The float32 operations are the reference's, in its order, one rounding each, on the
vector unit with 32 triangles as lanes; the box's ceil and floor clamp first and then
round by adding 1.5 x 2^23. The pack thread finishes each kept record (top-left
thresholds, the box as integers and float16 columns, the triangle number), and clips a
crossing triangle with the cores' scalar floats (rounded to nearest, one operation at
a time); those cores have no divide, so the clip's one division is done in integers,
rounded to nearest even. Lists stay in ascending triangle order: each core's share is
a contiguous range and lower cores' records come first.

Exact on every scene of the package's list, with the fill unchanged (2026-11-13,
card 1, 120 cores, host load 5 to 11, card clock not sampled in these check runs;
`python -m projects.raster.card_passes --scene <s> --frame [--lists 400]`). Times are
one record's event, issue to the last core's acknowledgement. This table is the
host-stepped kernels, without bands, with the X280 at 1,750 MHz (the session's default then):

| scene | mesh blocks | list pages | COUNT us | SETUP us | FILL us | image differs | list pages differ |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny | 13 | 356 | 73.6 | 327.2 | 119.3 | 0 | 0 of 356 |
| 10^4 | 375 | 1,181 | 96.8 | 279.7 | 139.9 | 0 | 0 of 1,181 |
| 10^5 | 3,198 | 5,913 | 430.2 | 844.4 | 473.7 | 0 | 0 of 5,913 |
| 10^6 | 31,182 | 39,233 | 3,813.1 | 6,960.6 | 2,960.5 | 0 | 0 of 400 sampled |
| stress-big | 7 | 966 | 247.0 | 2,439.1 | 235.3 | 0 | 0 of 966 |
| stress-heavy | 1,885 | 3,123 | 277.1 | 820.2 | 10,930.3 | 0 | 0 of 3,123 |
| stress-degenerate | 10 | 240 | 33.0 | 93.2 | 119.2 | 0 | 0 of 240 |

The same path with this head's kernels (bands, HIST), X280 at 1,000 MHz, AICLK 1,350
MHz (one sample at 1,343 on 10^4): the host-stepped frame of each `bench_thread --host`
run of the next section, one frame each, image exact:

| scene | heavy tiles / items | COUNT us | SETUP us | FILL us |
|---|---:|---:|---:|---:|
| tiny | 0 / 1,024 | 102.3 | 397.6 | 118.2 |
| 10^4 | 0 / 1,024 | 118.8 | 334.6 | 138.8 |
| 10^5 | 112 / 1,168 | 467.8 | 950.8 | 423.1 |
| 10^6 | 130 / 1,204 | 4,051.3 | 8,032.3 | 2,104.8 |
| stress-heavy | 2 / 1,052 | 330.1 | 977.8 | 1,027.9 |

The counters equal the reference's on every scene, and the records as the vector unit
leaves them were checked word for word against `raster.setup` for every triangle of
every scene (`card_passes.py` without `--frame`; 10^6: 549,341 kept, 0 words differ).

Between the two passes the host stops the program, reads the counters (983 KB, 0.46 to
1.3 s), takes its step (7 to 12 ms), writes the lists' frame and the slots (0.12 to
0.67 s) and loads the program again (0.1 to 0.4 s): seconds against the milliseconds
of the passes. The next section moves the step to a thread on the X280 cores, which
reads the counters from DRAM itself.

Bands cost COUNT its histograms (64 KB of L1 a core; 10^6 3,813 to 4,051 us, +6 %) and
SETUP a test per item a record reaches (10^6 6,961 to 8,032 us, +15 %), and save the
fill what the slowest core would otherwise carry alone: stress-heavy 10,930 to 1,028
us, 10^5 474 to 423 us, 10^6 2,961 to 2,105 us (the two tables above; their X280 clocks
differ, the Tensix cores' do not).

## The frame on the work cluster's thread

One program holds both the passes and the fill (`kernels/frame_reader.cpp`,
`frame_compute.cpp`, `frame_writer.cpp`: each includes the fill's and the passes' kernel
file, each in a namespace of its own and without its entry, and calls one or the other
by the record's segment). A thread on work cluster 1 (`thread/frame.c`, the step in
`thread/middle.c`, plain C) conducts the frame; the host uploads the mesh once per scene,
gives the camera with the load, waits for the thread's result and reads the image back
(`card_thread.py`):

```
thread frame(camera):                                    # X280, work cluster 1
    issue COUNT; wait                                    # counters to a tensor
    issue SUM; wait              # cores: totals per tile, entries before each core
    take totals                                          # drop the cache's lines of them
    tile costs (four harts), tiles to cut
    if any: write their list; issue HIST; wait; issue HSUM; wait; take their totals
    bands and items; items by falling cost (four harts sort quarters, merged);
    each to the least loaded core (a heap); list layout and base slots as words
    give the items region                                # 37 KB written back
    issue PLACE; wait            # cores: directory pages, list headers, end slots
    issue SETUP; wait            # each core its next slots from the prefixes, then
                                 # each record to its slot
    issue FILL; wait
    result: what it found, each part's time
```

The counting and the prefix sums are a reduction over the cores, so the cores do them:
SUM gives each core of 64 a slice of 16 tiles, reads every core's counters for it,
writes the totals for the thread and, into each core's slots page, the entries of the
cores before it; HSUM does the same for each heavy tile's 32 group counts, summed over
the groups as well, so that a core's start of any band of rows is two words apart.
SETUP's reader then makes its own core's next slots: the item's base slot (from the
thread) plus the entries before it. The thread reads 8 KB of totals instead of 983 KB
of counters, writes 37 KB of words instead of the directory pages, headers and next
slots (2 MB), and keeps what is serial: the costs, the cuts, the sort and the heap.
PLACE expands its words into the directory pages and the list headers.

The pages only the cores read and write are interleaved tensors, the thread's are
regions in its own bank. All 120 cores reading one bank at once got what that bank
gives: SUM took 247 us with its pages in the bank and 59 us with them spread over the
banks.

Exact on tiny, 10^4, 10^5, 10^6, stress-big, stress-heavy and stress-degenerate: the
thread's counts, items, directory and lists equal the host step's on the reference's
counts, and the image the reference's (`python -m projects.raster.card_thread --scene
<s> [--lists 64]`); Sponza and Bistro interior (overview) images exact in the runs below.

Frames in a row (2026-11-13, card 1, 120 cores, X280 at 1,000 MHz, AICLK 1,350 MHz
(samples at 1,343 on some scenes); `python -m projects.raster.bench_thread --scene <s>
--runs 5 --frames 0 [--no-balance]`, results in `results/frame/` and
`results/frame/repeated/`): 5 loads of the program, each running as many frames as fill
about 1 s (64 to 906), median per frame, us. Two rows per scene: "first" is a frame
without history (COUNT and SETUP on equal block shares: what a single image costs);
"repeated" is a frame from the second on, same camera, with COUNT's and SETUP's ranges
cut by the frame before's costs (below; the first frame of each load counts in its
mean). Inside a frame the host does nothing: 0 round trips, 0 bytes. Power is the card's
mean over the threads' runs; it draws 60.2 to 63.0 W with the program resident and idle. The V100 column is the
whole frame of `reference/bench.py` on one idle V100 (its "ms per frame";
`reference/README.md`, Sponza and Bistro in `results/reference/`):

| scene | row | heavy tiles | COUNT | SUM | HIST + HSUM | the step | PLACE | SETUP | FILL | frame | V100 | V100 / card | card W |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny | first | 0 | 104 | 59 | - | 436 | 103 | 228 | 121 | 1,051 | 290 | 0.28 | 64.0 |
| tiny | repeated | 0 | 104 | 59 | - | 436 | 103 | 228 | 121 | 1,051 | 290 | 0.28 | 61.8 |
| 10^4 | first | 0 | 118 | 59 | - | 440 | 103 | 231 | 141 | 1,091 | 250 | 0.23 | 65.8 |
| 10^4 | repeated | 0 | 127 | 59 | - | 453 | 103 | 157 | 141 | 1,041 | 250 | 0.24 | 65.4 |
| 10^5 | first | 112 | 450 | 59 | 95 | 751 | 107 | 667 | 425 | 2,555 | 790 | 0.31 | 67.2 |
| 10^5 | repeated | 112 | 448 | 59 | 95 | 771 | 107 | 547 | 425 | 2,454 | 790 | 0.32 | 65.9 |
| 10^6 | first | 130 | 3,825 | 59 | 160 | 804 | 108 | 5,060 | 2,108 | 12,125 | 4,420 | 0.36 | 67.7 |
| 10^6 | repeated | 130 | 3,375 | 59 | 160 | 823 | 108 | 3,942 | 2,108 | 10,575 | 4,420 | 0.42 | 67.3 |
| Sponza (262,266) | first | 0 | 2,221 | 59 | - | 456 | 103 | 2,790 | 1,593 | 7,223 | 2,310 | 0.32 | 65.7 |
| Sponza (262,266) | repeated | 0 | 1,265 | 59 | - | 466 | 103 | 1,797 | 1,593 | 5,285 | 2,310 | 0.44 | 66.1 |
| Bistro interior (1,020,903) | first | 82 | 4,808 | 59 | 92 | 733 | 107 | 7,475 | 2,095 | 15,373 | 5,990 | 0.39 | 65.3 |
| Bistro interior (1,020,903) | repeated | 82 | 3,594 | 59 | 92 | 752 | 107 | 4,273 | 2,096 | 10,974 | 5,990 | 0.55 | 65.2 |

Each column is the thread's time from issuing a record to its event (the step: from
SUM's event to PLACE's issue, HIST and HSUM apart), so the frame is the sum plus the
thread's own bookkeeping. The step's parts (10^6, us, harts): taking the totals 73 (1),
tile costs and the tiles to cut 88 (4, then 1), bands and items 202 (1), the sort 118 (4
then 2 then 1), each item to the least loaded core 186 (1: each choice needs every
earlier one), the layout's words and base slots 76 (1), the items page 18 (1), the
hand-over back 33 (1). Before the counting moved to the cores the step was 4.5 to 11.6
ms (counters 983 KB read by the thread, next slots of every (core, item) written by it).

SETUP's pack thread copies each entry's record (128 bytes) into the out page with one
vector load and store instead of 32 scalar words: SETUP on tiny went from 453 to
229 us, on 10^6 from 8,186 to 5,062 and on Bistro from 15,427 to 7,476.
It is still the largest part of every large frame, set by its slowest core.

COUNT and SETUP give each core a range of the mesh's blocks; with equal ranges some hold
far more triangles in view than others, and the pass waits for the slowest. From the
second frame on, each core's writer leaves its COUNT and SETUP ticks in the camera's
region, the thread cuts the next frame's ranges into equal parts of them (each core's
ticks spread evenly over its blocks), and the reader brings each core its range with the
camera. Scenes with fewer blocks than cores (tiny) keep equal shares. The cut is history:
a frame from a new camera starts from the last camera's costs.

Per frame the host still loads the program (the camera goes with the load: 0.5 to 0.8
s) and reads the image back (20 MB, 10 to 18 s); both are outside the frame times above.

## Results

Scenes of the reference (`ref/scenes.py`): tiny, 10^4, 10^5 and 10^6 triangles,
1024 x 1024 pixels, 32 x 32 tiles; 119 worker cores; balanced assignment unless named.

The runs behind the tables of this section: 2026-11-13, card 1 (a p150a), AICLK 1,350
MHz unless marked, host load 5.9 to 10 (`/proc/loadavg`), the same kernels issued by the
earlier driver (a test fixture on a 120th core in place of the dispatch kernel; its
issue-to-acknowledgement time in place of the event's). Commands, in this folder's
terms: `python -m projects.raster.bench --scene <s> --seconds 5` for frame time and
power, `python -m projects.raster.run --scene <s> --json <f>` for the profile and the
tile assignment (`--assign round-robin` for that column). The C++ and V100 columns are
the committed table of [`reference/README.md`](reference/README.md), produced by
`reference/bench.py` (C++ on this host's 96 threads; CUDA on one idle V100).

**Agreement** through the reference's own comparison
(`reference/agree.py --external`, the card's files in the results layout): against
numpy at every size, 0 differing pixels in colour, depth and winner, depth error 0,
tile counts equal.

**Stress scenes** (`stress.py`; run by hand, each against numpy:
`python -m projects.raster.run --scene stress-big` and likewise `stress-heavy`, with
and without `--split 0`, and `stress-degenerate`), all exact:
- screen-filling triangles, triangles reaching 10^5 pixels off screen, slivers across
  the screen, equal-depth ties;
- 32,946 and 13,226 small triangles in two tiles. Run with bands, the frame takes 1.01
  ms. Run without them it takes 10.45 ms: one core reads 3,121 list pages;
- zero-area triangles, triangles off screen, behind the camera, across the near
  distance, between pixel centres, and with edges on pixel centres.

The V100 version is the same algorithm written plainly and not tuned, with no
hardware rasterizer; the card's numbers sit beside it stage for stage.

**Time, milliseconds.** The card runs the fill; setup, binning and the lists run on the
host (numpy and Python) and are listed as host stages, not compared. Card: the frame
from a FILL record's issue to its acknowledgement, which comes after the record's
output (the frame's tiles) is written; over a loop of FILL records back to back
(`bench.py`: 48,000 / 40,000 / 16,000 / 2,462 frames; median, min and max; host load
5.9 to 8.7). Cycles are converted at the clock read when the device opens (1,350 MHz);
AICLK was sampled every 50 ms through each loop (min / mean in the table). Host: the
reference's numpy `setup` and `bin_triangles`, and packing the records, the tile costs,
the assignment and the lists (medians of five). Upload of the lists: one run, slow
dispatch. References: the medians of the committed table in
[`reference/README.md`](reference/README.md) (fill stage; not rerun here).

| scene | card frame (min-max) | AICLK MHz min / mean | card W mean | host setup | host binning | records + lists | upload | C++ 1 thread fill | C++ 96 threads fill | V100 fill | V100 fill / card frame |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tiny | 0.1195 (0.1154-0.1232) | 1343 / 1349.9 (*) | 75.1 | 0.46 | 0.27 | 8.0 | 119 | 11.12 | 1.01 | 0.06 | 0.50 |
| 10^4 | 0.1394 (0.1386-0.1406) | 1350 / 1350 | 79.0 | 4.01 | 1.19 | 10.5 | 187 | 15.65 | 1.42 | 0.16 | 1.15 |
| 10^5 | 0.4258 (0.4254-0.4270) | 1350 / 1350 | 75.8 | 33.0 | 6.6 | 48.2 | 227 | 27.87 | 1.90 | 0.57 | 1.34 |
| 10^6 | 2.0310 (2.0302-2.0324) | 1350 / 1350 | 75.1 | 374 | 47.0 | 206.6 | 735 | 69.86 | 3.06 | 3.34 | 1.64 |

(*) The tiny loop's clock left 1,350 MHz for at least one sample (1,343 MHz); its
frame times are cycles at 1,350 MHz, so they may be short by up to 0.5 % there.

10^5 and 10^6 are with heavy tiles in bands. These rows are the kernels with the
masked record fields and the bounded waits; the same scenes before both took 0.4192 and
1.9940 ms (bands) and 0.4518 and 2.8226 ms (no bands). The bands cost host time in the
lists step (the band filter is plain Python per band).

The whole frame is not on the card: setup and binning run on the host in numpy. With
the host stages and the upload a frame takes 128, 203, 315 and 1,364 ms today (9, 16,
88 and 629 ms without the upload). That is a description of how much still runs on the
host, not a comparison: the V100's frame does not include the same host process.

**Power** of card 1 (`komorebi.power.PowerSampler`), counted over the stretch in which
the loop's records ran (5.0 to 7.0 s per scene). Idle: the closed card before the run,
and the open card between programs (after the first program, nothing running). The
V100 columns are from the committed reference table.

| scene | idle closed / open (W) | mean (W) | peak (W) | mean of the 150 W limit | J per frame | J per frame above closed idle | AICLK MHz min / mean | V100 W idle / mean / peak, J per frame |
|---|---|---|---|---|---|---|---|---|
| tiny | 20.0 / 62.3 | 75.1 | 78 | 50 % | 0.0090 | 0.0066 | 1343 / 1349.9 | 55 / 89 / 91, 0.026 |
| 10^4 | 20.0 / 61.5 | 79.0 | 80 | 53 % | 0.0110 | 0.0082 | 1350 / 1350 | 57 / 112 / 114, 0.028 |
| 10^5 | 20.0 / 62.3 | 75.8 | 78 | 51 % | 0.0323 | 0.0238 | 1350 / 1350 | 65 / 110 / 114, 0.087 |
| 10^6 | 20.6 / 63.0 | 75.1 | 78 | 50 % | 0.1526 | 0.1107 | 1350 / 1350 | 55 / 110 / 115, 0.484 |

At every size the card draws about half its limit and only 12 to 18 W above the open
card at rest: the cores are underused, waiting on their own instruction issue (setup)
or on each other (the slowest core), not drawing near the limit.

Calibration points on the same sensor (a loop that is not in the repository, 5 to 7 s, 119 cores): the closed
card 21.1 W; the open card at rest 62.6 to 63.0 W; a spin-only segment, every compute
core spinning on the clock, 63.4 W; the arithmetic check, 1,000,448 inputs per record
at 286.6 us, 70.5 W mean and 72 W peak. Neither loop loads the vector unit fully (the
check moves its data through the register window by the cores), nor the matrix unit or
the NoC; the fill draws more than both. The spin segment writes no output, and its
records are acknowledged about 5 us after issue although compute spins for 1 ms; its
power counts the stretch from the first issue to the last, and no time in this README
comes from the acknowledgement of a record without output.

The read-back runs at about 1 MB/s through slow dispatch. The frame is 20 MiB: five
float32 planes (depth, three colours, winner) per pixel. Depth and colour must stay
float32 to be compared bit for bit; dropping the winner plane saves 4 MiB, half-float
colour another 6 MiB but no longer agrees exactly. The winner plane alone (4 MiB, or
int32) would be enough: the host can recompute colour and depth exactly from the
winner's record with the reference's own float32 arithmetic. Not done; read-back only
lengthens correctness runs.

**Profile.** Measured on the kernels before the masked fields and bounded waits (1 to 2
% faster). Pack-thread cycles summed over the cores (one frame). Per (triangle, tile):
the triangle's setup and its rows. Busy: everything the pack thread does, including the
tile init, the read-out and its waits.

| scene | (triangle, tile) entries | rows per entry | setup per entry | cycles per row | setup + rows per entry | busy per entry | busy per covered pixel | setup / rows / read-out share of busy |
|---|---|---|---|---|---|---|---|---|
| tiny | 4,047 | 23.6 | 263 | 59.5 | 1,665 | 3,241 | 19.9 | 8 / 43 / 29 % |
| 10^4 | 17,121 | 10.1 | 261 | 59.5 | 860 | 1,185 | 30.3 | 22 / 51 / 19 % |
| 10^5 | 92,778 | 4.8 | 260 | 60.7 | 552 | 652 | 86.1 | 40 / 45 / 7 % |
| 10^6 | 626,033 | 2.65 | 260 | 64.6 | 431 | 484 | 406 | 54 / 35 / 1 % |
| 10^5, bands | 100,742 | 4.43 | 264 | 61.1 | 535 | 633 | 90.9 | 42 / 43 / 7 % |
| 10^6, bands | 653,873 | 2.54 | 262 | 64.9 | 427 | 481 | 421 | 55 / 34 / 1 % |

**Tile assignment**: frame (us) and per-core busy cycles (max / mean / min).

| scene | round-robin | balanced by count + 8 | balanced by measured cost | the same, heavy tiles in bands |
|---|---|---|---|---|
| tiny | 126.5 (165 k / 110 k / 71 k) | 127.2 (166 k / 110 k / 83 k) | 118.7 (154 k / 110 k / 91 k) | no tile split |
| 10^4 | 225.2 (298 k / 171 k / 81 k) | 158.3 (209 k / 170 k / 111 k) | 137.4 (178 k / 170 k / 165 k) | no tile split |
| 10^5 | 644.6 (867 k / 508 k / 201 k) | 476.9 (637 k / 509 k / 401 k) | 452.7 (609 k / 508 k / 487 k) | 419.1 (560 k / 536 k / 520 k) |
| 10^6 | 3,492.9 (4.71 M / 2.55 M / 1.29 M) | 2,829.1 (3.82 M / 2.55 M / 2.29 M) | 2,829.2 (3.82 M / 2.55 M / 2.45 M) | 1,993.9 (2.69 M / 2.64 M / 2.57 M) |

**Per unit** (pack-thread cycles, one frame): the two units of work and their share of
the busy time, and the mean core's busy time over the slowest core's.

| scene | cycles per row | share | cycles per triangle setup | share | read-out share | mean busy / slowest |
|---|---|---|---|---|---|---|
| tiny | 59.5 | 43 % | 263 | 8 % | 29 % | 0.71 |
| 10^4 | 59.5 | 51 % | 261 | 22 % | 19 % | 0.95 |
| 10^5 | 61.1 | 43 % | 264 | 42 % | 7 % | 0.96 |
| 10^6 | 64.9 | 34 % | 262 | 55 % | 1 % | 0.98 |

(10^5 and 10^6 with bands; before them the mean over the slowest was 0.83 and 0.67.)

**What bounds each size.**

- tiny (V100 / card 0.50): the read-out and the frame writes, and the balance. The mean
  core is busy 0.71 of the slowest one's time; the slowest core (154 k cycles) waited
  52 k cycles for free output pages while its writer sent 20 KB per tile; read-out is
  29 % of busy time. The card draws 75 W, 50 % of its limit, 13 W above the open card
  at rest: the cores wait, they do not run out of power.
- 10^4: the rows on the vector unit. Cores are within 5 % of each other (178 k against
  170 k mean); rows are 51 % of busy time at 59.5 cycles for 54 instructions.
- 10^5 (1.34): setup and rows on the vector unit, evenly spread. With the heavy tiles
  in bands the cores are within 4 % (560 k against 536 k); setup is 42 % and rows 43 %
  of busy time. Without bands it was the largest tile (609 k measured).
- 10^6 (1.64): triangle setup, bound by the core issuing immediates: 55 % of all busy
  time, cores within 2 % (2.69 M against 2.64 M). Without bands it was one tile of
  8,780 triangles on one core (3.82 M cycles, 2.83 ms).

## Not on the card yet

Triangle setup, the per-tile counts and the scatter into the tile lists run on the
Tensix cores (COUNT and SETUP, above), the step between them on the X280 thread. The
irregular parts, with the host-binned path's times (10^5 / 10^6, ms):

- Prefix sums of the per-tile counts: on the cores (SUM, HSUM, above). Tile costs,
  bands and the assignment of tiles and bands to cores: on the X280 thread, 0.43 to 0.80
  ms a frame at 1,000 MHz; on the host-binned path still inside the records-and-lists
  step (48 / 207 ms).
- Tiles beyond 16 bands (not built). A band holds every triangle touching its rows, so
  a tile of 33,000 small triangles (the stress scene) still costs 4.7 times the mean
  core in 16 bands; past that, a tile's triangles are split by range across cores and
  merged by depth and triangle number. Irregular: which tiles need it, into how many
  pieces, and the merge pairs results from different cores.
- Big triangles (not built). A triangle over many tiles is copied into every tile's
  list (128 bytes each, up to 1,024 times) and is filled with pixels as lanes, while
  small triangles would be filled with triangles as lanes. Irregular: the choice is by
  size, per frame, and moves work between the two forms and the cores.

The scene's upload (once per scene) and the frame's read-back stay on the host by
nature. Today the host uploads the lists instead (10^6: 735 ms), and the read-back
runs at about 1 MB/s.

## Next on the fill

- Triangle setup: the unpacker path wins on the core but needs 4 KB per (triangle,
  tile). Triangles as the lanes for small triangles (a 4 KB tile holds 32 triangles'
  records) is being costed; a broadcast of a few scalars to all lanes would be the
  other road.

## What the framework made hard

- **A triangle's numbers reach the vector unit's lanes as instruction immediates, or
  as a whole tile.** As immediates, each float32 costs two load-immediate
  instructions that the core builds and stores into the instruction buffer, 3 core
  instructions each. Triangle setup is then bound by the core, at about 260 cycles for
  22 numbers, while the vector unit runs about 80 instructions; at 10^6 this is half of
  all the time. The other road, data through the unpacker, writes a whole 4 KB tile and
  has no broadcast for 32-bit data: on one core it beats the immediates by 90 to 160
  cycles per triangle, but every number has to arrive already copied to all 32 lanes,
  which multiplies the list bytes by 32. Missing in between: a way to bring a few
  scalars into all lanes of a register from L1.
- **The pack thread's hardware float has no divide.** With its vector extension on, its
  scalar float32 compiles to the cores' float instructions, division included; a kernel
  that compiles one stops its core without a report (the clip's division is done in
  integers).
- **Only the unpack thread can wait on a circular buffer**, but the pack thread is the
  one with the vector extension and the destination register window. Every list page
  goes through a mailbox hand-off between the two threads.
- **A multi-page item written contiguously into a circular buffer has a hidden
  alignment assumption** the runtime does not check: the buffer's size has to be a
  multiple of every item size, and every push a whole item. Breaking it gave wrong
  arithmetic-check outputs (3-page items in a 10-page buffer) and later a misaligned
  profile item from the second record on. Worse, the two sides wrap differently:
  compute's push wraps its pointer when it reaches or passes the end, the writer's pop
  only when it lands exactly on it, so after one straddling item the writer's read
  pointer walked on through L1 and a 4,000-record run did not return. The kernels now
  check every item against the buffer's end before touching it (give-up word and stop
  word on failure).
- **The destination register in 32-bit mode has its own lane order** in the memory
  window: vector v, lane l at word 64 (v / 2) + 16 (l / 8) + 2 (l mod 8) + v mod 2. The
  frame goes out in that order and the host permutes it.
- **Waits in the compute kernel are unbounded** in the runtime's calls, and a run that
  goes wrong this way did not end through the stop word: the feeder's deadline set it,
  but no thread blocked inside a record looks at it. The kernels now poll every wait of
  their own with a wall-clock bound (0.8 s) and set a give-up word: list pages, the
  hand-over between the compute threads, output space, the NoC barriers. Masking the
  record fields and bounding the waits cost 1 to 2 % of the frame (10^6: 1.994 to 2.031
  ms).
- **Waits software cannot bound.** The vector unit's drain (`tensix_sync`), the
  hardware semaphore waits inside the destination-register hand-over
  (`tile_regs_acquire`, `copy_tile`) and a `SEMWAIT` block the thread in hardware; no
  wall clock is looked at. An instruction word the core should never have issued
  (a data value spilled into the opcode) can wedge a compute thread there for good. The
  only defences are before the push: every record field masked to its width where it
  becomes part of an instruction, host checks of every word written ahead of time, and
  a ladder of runs from one record and one core up.
- **Read-back**: about 1 to 2 MB/s, 11 to 20 s for one frame.
- **A NoC read from DRAM has to start on a 64-byte boundary.** Records 144 bytes apart
  in a region came back from the line's start for three of every four, with no error;
  they are 192 bytes apart now.
- **A thread sees only regions of its cluster's DRAM bank**, so whatever the grid reads
  from interleaved tensors has to be copied out by grid cores (PLACE above). A region
  read used to read every bank's pages of the regions' tensor in one go (fixed in the
  host package since).
- **No `memset` in a thread image**: a zero initialiser of a local array compiles to a
  call that does not link.
- **The camera goes with the load**: a thread's words and regions are written at the
  load and the host writes no DRAM while the program is loaded, so each new camera is a
  load (0.5 to 0.8 s).
- **Small things that cost time**:
  - a load-immediate of the lower half right after a multiply-add to the same register
    is unpredictable;
  - the replay buffer holds 32 instructions, less than one row (54), so only the
    27-instruction edge block is replayed;
  - the vector unit flushes denormal results to zero, which the arithmetic check had to
    model;
  - a 32-bit value pushed as a 16-bit immediate is not masked by the instruction
    macros: it carries into the opcode, and the core runs whatever instruction that
    makes. A probe that fed random words to the fill's triangle loader hung its core
    this way. The rasterizer's records hold float16 there, checked on the host, and
    the fill masks every record field where it becomes part of an instruction (about
    4 cycles per triangle).
- **What I would have asked of the runtime (not changed here)**:
  - a bounded wait for compute;
  - a contiguous-reserve call on a circular buffer that refuses a wrapping item;
  - a per-record cycle counter that compute can write, so kernels need no profile page
    of their own.
