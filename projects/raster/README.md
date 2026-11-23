# Rasterization on the card

The rasterizer of `reference/raster.py` on a Blackhole card's Tensix cores, on the persistent kernel runtime (`komorebi/device/km_worker.hpp`), loaded and driven through the host
package (`komorebi.Session` and `Run`, the dispatcher submitting the commands).
Three ways in: host-binned (the host does setup and binning with the reference's own
functions; the cores fill tiles); the per-triangle stages on the cores too (COUNT and
SETUP, below; the host runs the middle pass between them); and the whole frame on the card,
driven by a thread on the X280 cores beside the dispatcher, which runs that
pass itself (below). Every scene's image is bit-exact against the reference.

On a Wormhole chip, the host-binned fill: its comparisons in Wormhole's words and the
tile read out of Dst by the packer (`km::dst`, whole planes); `run.py --scene tiny` writes
Blackhole's frame word for word. The arithmetic handler and `--spread` are refused there
by name (they write Dst in part tiles); the stages on the cores do not build there yet
(Blackhole's reciprocal and 24-bit multiply words), and the card's own frame needs the X280.

## What each core does (host-binned)

1. The host (`run.py`, `lists.py`): runs the reference's own `raster.setup(scene)` and
   binning, assigns tiles to cores, and writes to DRAM for each core a directory page
   and its tile list: the tile ids, a count, then the triangle records in triangle
   order, 128 bytes each (a, b, c of three edges, top-left bits, the 1/z plane, three
   colour planes, the pixel box, the triangle number; `lists.pack_records`).
2. The card (`card.py`): one command `FILL` to the group of all worker cores.
3. Each core, for each of its tiles:
   - `kernels/reader.cpp`: reads the core's directory page, then streams the list
     pages from DRAM into the input buffer.
   - `kernels/compute.cpp` and `kernels/fill.hpp` (the pack thread): `init_tile` sets
     the tile's planes of 32 x 32 float32 to depth 0, the background colour and no
     triangle; then
     for each triangle record, in order, `row` takes the 32 pixels of each row as the
     lanes of the vector unit, tests them against the three edges and the box, computes
     the 1/z plane, and where a pixel is inside and its 1/z is greater than the stored
     depth it stores the new depth, the triangle number and the colour planes. The comparison is strict, so
     the lower triangle index keeps a tie.
   - `kernels/writer.cpp`: sends the tile's finished planes to the frame in DRAM.

A fifth plane holds the winner's triangle number (as a float, -1 where empty), so the
frame carries the same `tri` image as the reference.

## Who does what on a core

The reader and the writer are the two roles of the easy layout, not "the CPU that
reads" and "the CPU that writes": either data-movement CPU can read, write and send to
other cores. What differs is the NoC each uses, that each circular buffer has one
kernel filling it and one draining it, and that the writer reports the command.

| Kernel | Role | Work |
|---|---|---|
| `reader.cpp` | reader (RISCV_0, NoC0) | reads the core's directory page (first list page, page count, tile ids), streams the list pages (2 KB, 16 slots) into the input buffer |
| `compute.cpp` | unpack | waits for each list page (only this thread can wait on a buffer) and passes its address to the pack thread through the mailbox; frees it when the pack thread answers |
| `compute.cpp` + `fill.hpp` | pack | submits every vector-unit instruction: tile init, per triangle the broadcast constants, per row the edge tests and the plane updates; then copies the five planes from the destination register to five output pages |
| `compute.cpp` | math | idle |
| `writer.cpp` | writer (RISCV_1, NoC1) | sends each tile's five pages to the frame; writes tiles without triangles itself from a constant buffer; at the end writes the profile page |

`raster.h` holds the layouts shared by the host and the kernels: the 128-byte record,
the tile header and end slot, the directory page, the frame (per tile five 4 KB planes
in the destination register's window order). `lists.py` builds them; `card.py` loads the
program and runs a handler's commands through the host package; `run.py` runs and checks one scene; `bench.py` times a loop
of frames and samples the card's power; `arith.py` is the arithmetic check;
`pictures.py` draws the card's image, its difference from the reference and the busy
map. `--out` writes the card's result in the layout of `reference/README.md`, so the
reference's comparison, picture and table scripts take it with `--external`.

```
python -m projects.raster.arith --n 1000000          # arithmetic check, bit for bit
python -m projects.raster.run --scene tiny --tiles 367 --cores 1
python -m projects.raster.run --scene tiny --cards 0,1 [--far] --save frame.npy  # two cards
python -m projects.raster.run --scene s1m --out results/card --json s1m.json
python -m projects.raster.bench --scene s10k --seconds 5 --out results/card
python -m projects.raster.pictures --scene s1m --card results/card --prof s1m_prof.npz \
    --out pictures
python -m projects.raster.reference.agree --scenes s1m --external results
```

**Two cards.** The host-binned fill reads its directory and list pages and writes its
frame and profile pages with the page calls (`km::read_pages`, `km::write_pages`), so
`--cards 0,1` runs the same kernels over every flat rank of a session of two cards,
a page another card holds crossing; `--far` puts the frame on card 1 alone. A band (an
item of some rows of a tile) writes a part of each of its pages: on two cards with
`km::put_bytes` (a far page's rows go as a record across the link), on one by address
as before. Each frame tensor is split over k ranks on each side of the edge, k the most
that divides its pages evenly; the list pages take the same hint when their count is
even, and when it is odd (3,849 for `stress-heavy`) no group splits them in proportion,
so the session's warning stands for them (each core reads its own list pages wherever
they lie). The scene `tiny` on cards 0 and 1, the frame split or on card 1,
gave the frame of one card bit for bit (2026-11-15); it makes no bands at 0.33 on 120 or
238 cores. `stress-heavy` on cards 0 and 1 (2026-11-18, AICLK 1,350 MHz while open)
makes 30 bands; its frame equals the reference and one card's frame bit for bit. Its
FILL command took 10,282.5 us on two cards against 1,027.1 us on card 0. Across the link: 2,630 frame
writes at about 0.41 ms each, 1,784 list reads at 0.12 ms (238 cores queue on one link,
about 1.6 GB/s): a cost of this placement. With `tiny`,
the first FILL command took 4,552.9 us on two cards against 120.4 us on one, and
7,979.3 us with the frame on card 1: its page calls cross the link. The passes and the
frame program's own steps (COUNT, SETUP, PLACE and the X280 thread's regions) run on one card only.

## Decisions, each with its measurement

Cycles are the pack thread's wall clock (1.35 GHz), summed over cores by the kernels'
own counters; frame times are a command's submit-to-acknowledge time, from the dispatcher's event (its 50 MHz timer). The tables below were measured with an earlier
driver, a test fixture on a 120th core submitting the commands to 119 workers; the same
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

What is left is the core's submit of the immediates: 44 data-dependent load
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

**Skipping outside the box.** Rows outside the triangle's box are not submitted (the
core clips the box to the tile: a few instructions per triangle). Columns outside the
box fail edge 0, whose `a * px` is set to -inf there once per triangle, so a row costs
nothing extra for the box. Rows submitted per (triangle, tile): 23.6 (tiny), 10.1 (10^4),
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
dispatch a tensor write waits for every core to be idle, which a persistent grid never
is, so nothing is written while the program is loaded. A FILL command goes to every core
(`directory_address`, `lists_address`, `pages_per_item`, `profile_address`), one in flight at a time
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

1. Upload (host, `passes.mesh_blocks`): the mesh de-indexed, 32 triangles a block, 18
   vectors a block (x, y, z and colour of three corners) already in the order of the
   vector unit's register window, 2,304 bytes.
2. `COUNT` (`kernels/setup_reader.cpp`, `setup_compute.cpp`, `setup_writer.cpp`): each
   core takes a contiguous partition of the blocks. The vector unit computes camera
   space, screen position (1/z by the exact reciprocal), area and box; the pack thread
   applies the keep rule, and a triangle across the near distance is clipped and its
   pieces go through the vector unit at its place. The writer counts, per tile a kept
   triangle touches, one entry and the rows it covers there, and keeps in L1, per tile,
   how many entries start and end in each 2-row group; at the end it writes its two
   counters (1,024 words each) to DRAM.
3. Host (`passes.middle`, `passes.heavy_tiles`): tile costs from the counters; a tile
   above a third of a core's mean load is cut into bands.
4. `HIST`: each core writes the start and end counts of those tiles (64 bytes a tile).
5. Host (`passes.cuts`, `passes.items_of`): cuts the bands where the rows' estimated cost
   reaches j / k of the tile's; a band's entries are those starting before its last row
   less those ending before its first; assigns the items (tiles and bands) to cores;
   lays out each core's list (header, records, end slot, as the fill reads it) and
   writes the headers, end slots and the directory; the slot of every (core, item) is
   the item's header plus 1 plus the records of the lower cores.
6. `SETUP`: the same partition of the blocks again, with the same steps and also the
   edges, 1/area and the 12 plane coefficients (1/z and three colour channels, three
   each); every kept record goes once to each item it reaches, at that item's next
   slot.
7. `FILL`, as above, over the lists in place.

The float32 operations are the reference's, in its order, one rounding each, on the
vector unit with 32 triangles as lanes; the box's ceil and floor clamp first and then
round by adding 1.5 x 2^23. The vector unit also finishes the 32 lanes (`finish_lanes`
in `kernels/setup.hpp`: the keep rule's flags, top-left thresholds, the box as integers
and float16 columns, the triangle number); the pack thread reads the flags and boxes as
pairs and each kept record from Dst by segment loads, and clips a crossing triangle with
the cores' scalar floats (rounded to nearest, one operation at a time); those cores
have no divide, so the clip's one division is done in integers, rounded to nearest even.
The pack thread then hands each kept triangle's box word (tile columns and rows) to the
writer, in SETUP with its record: the writer counts entries, rows and the 2-row
histograms (COUNT) or sends the record to every item it reaches (SETUP), so the pack
thread goes on to the next block. Lists stay in ascending triangle order: each core's
partition is a contiguous range and lower cores' records come first.

Exact on every scene of the package's list, with the fill unchanged (2026-11-13,
card 1, 120 cores, host load 5 to 11, card clock not sampled in these check runs;
`python -m projects.raster.card_passes --scene <s> --frame [--lists 400]`). Times are
one command's event, submit to the last core's acknowledgement. This table is the
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
of the passes. The next section moves the middle pass to a thread on the X280 cores, which
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
by the command's handler). A thread on work cluster 1 (`thread/frame.c`, the middle pass in
`thread/middle.c`, plain C) drives the frame; the host uploads the mesh once per scene,
gives the camera with the load, waits for the thread's result and reads the image back
(`card_thread.py`). `frame_once` in `thread/frame.c` submits each command and waits for
its event before the next:

1. `COUNT`: the counters of every core go to a tensor.
2. `SUM`: the cores total the counters per tile and compute, for each core, its entries
   before it. The thread drops its cache's lines of the totals (`kmx_invalidate`) and
   reads them.
3. Tile costs, on the four cores of the cluster (`costs_part`), and the tiles to cut
   (`rm_pick`). If there are any: the thread writes their list and submits `HIST` and
   `HSUM`, then reads their totals.
4. The middle pass (`thread/middle.c`): bands and items (`rm_items`); the items by
   falling cost, sorted in quarters on the four cores and merged (`rm_sort`,
   `rm_merge`); each item to the least loaded core, from a heap (`rm_owners`); the lists'
   layout and base slots as words (`rm_compact`, `rm_base`); the thread writes the items
   region back (`kmx_flush`, 37 KB).
5. `PLACE`: the cores write the directory pages, list headers and end slots from those
   words.
6. `SETUP`: each core makes its next slots from the prefixes, then sends each record to
   its slot.
7. `FILL`. The thread submits it, (when the frame balances by block ranges) prepares the next
   frame's ranges while the grid fills, and waits.
8. The result: what the thread found and each part's time.

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
thread's counts, items, directory and lists equal the host middle pass's on the reference's
counts, and the image the reference's (`python -m projects.raster.card_thread --scene
<s> [--lists 64]`); Sponza and Bistro interior (overview) images exact in the runs below.

Frames in a row (2026-11-14, card 1, 120 cores, X280 at 1,000 MHz, AICLK 1,350 MHz
(samples at 1,343 on some scenes), host load 2.9 to 12.9 (`/proc/loadavg`, one minute);
`python -m projects.raster.bench_thread --scene <s> --runs 5 --frames 0 [--no-balance]`,
results in `results/frame/` (first) and `results/frame/settled/`): 5 loads of the
program, each running as many frames as fill about 1 s (112 to 954), median per frame,
us. Two rows per scene. "first" is a scene's first frame: COUNT and SETUP on equal block
partitions, what a single image costs. "settled" is a frame from the second on, same
camera, with COUNT's and SETUP's ranges cut by the frame before's predicted costs (below;
the first frame of each load counts in its mean). Inside a frame the host does nothing:
0 round trips, 0 bytes. Power is the card's mean over the threads' runs. The V100
columns are `reference/bench.py` on one idle V100, its stages and its whole frame ("ms
per frame"; the committed rows of `reference/README.md`, Sponza and Bistro in
`results/reference/`, not re-run: the reference's scripts did not change):

| scene | frame | heavy tiles | COUNT | SUM | HIST + HSUM | middle pass | PLACE | SETUP | FILL | card frame | V100 setup | V100 binning | V100 fill | V100 frame | V100 / card | card W |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny | first | 0 | 83 | 59 | - | 436 | 98 | 189 | 121 | 987 | 70 | 170 | 60 | 290 | 0.29 | 61.9 |
| tiny | settled | 0 | 84 | 59 | - | 436 | 98 | 189 | 121 | 987 | 70 | 170 | 60 | 290 | 0.29 | 63.4 |
| 10^4 | first | 0 | 79 | 59 | - | 440 | 98 | 164 | 141 | 982 | 30 | 50 | 160 | 250 | 0.25 | 67.0 |
| 10^4 | settled | 0 | 69 | 59 | - | 453 | 98 | 132 | 141 | 952 | 30 | 50 | 160 | 250 | 0.26 | 65.8 |
| 10^5 | first | 112 | 232 | 59 | 95 | 753 | 101 | 384 | 425 | 2,050 | 120 | 80 | 570 | 790 | 0.39 | 68.0 |
| 10^5 | settled | 112 | 190 | 59 | 95 | 772 | 101 | 324 | 425 | 1,967 | 120 | 80 | 570 | 790 | 0.40 | 68.2 |
| 10^6 | first | 130 | 1,027 | 59 | 160 | 807 | 102 | 2,632 | 2,107 | 6,894 | 770 | 270 | 3,340 | 4,420 | 0.64 | 68.9 |
| 10^6 | settled | 130 | 1,157 | 60 | 153 | 633 | 101 | 1,994 | 2,104 | 6,202 | 770 | 270 | 3,340 | 4,420 | 0.71 | 72.1 |
| Sponza (262,266) | first | 0 | 1,680 | 59 | - | 457 | 98 | 1,881 | 1,592 | 5,770 | 460 | 480 | 1,690 | 2,310 | 0.40 | 68.8 |
| Sponza (262,266) | settled | 0 | 694 | 60 | - | 357 | 98 | 912 | 1,591 | 3,711 | 460 | 480 | 1,690 | 2,310 | 0.62 | 76.9 |
| Bistro interior (1,020,903) | first | 82 | 2,142 | 59 | 91 | 736 | 102 | 3,535 | 2,095 | 8,762 | 850 | 350 | 5,610 | 5,990 | 0.68 | 66.9 |
| Bistro interior (1,020,903) | settled | 82 | 1,318 | 60 | 88 | 579 | 101 | 2,095 | 2,092 | 6,336 | 850 | 350 | 5,610 | 5,990 | 0.95 | 73.4 |

The same command run again on 2026-11-16, card 1, on a tree whose runtime and headers had
changed since (AICLK 1,350 MHz, X280 1,000 MHz, same 5 loads each): the images are exact
(0 pixels differ) and the frames are 2 to 3 % shorter than the table's: 10^6 settled 6,196 us
(first, `--no-balance`, 6,792), Sponza settled 3,692, Bistro interior settled 6,311. The
table keeps the 2026-11-14 figures for the "first" rows; no run behind them was changed.
The 10^6, Sponza and Bistro interior "settled" rows are from a new run of the same command
on 2026-11-18, card 0, commit 6396ff8e: AICLK 1,350 MHz at the open and in every sample
(Bistro: 1,343 once of 117, mean 1,349.9), host load 93 to 130 (`/proc/loadavg`, one
minute, at the jobs' start and end), 5 loads each of 144 (10^6), 173 (Sponza) and 115
(Bistro) frames, 0 pixels differ in all three. HIST + HSUM is the sum of the two medians;
the heavy-tile counts and the V100 columns are those of the older rows; the watts are the
card's mean over the threads' runs.

Each column is the thread's time from submitting a command to its event (the middle pass: from
SUM's event to PLACE's submit, HIST and HSUM apart), so the frame is the sum plus the
thread's own bookkeeping. The middle pass's parts (10^6, us, cores): taking the totals 73 (1),
tile costs and the tiles to cut 88 (4, then 1), bands and items 202 (1), the sort 118 (4
then 2 then 1), each item to the least loaded core 186 (1: each choice needs every
earlier one), the layout's words and base slots 76 (1), the items page 18 (1), the
flush back 33 (1). Before the counting moved to the cores the middle pass was 4.5 to 11.6
ms (counters 983 KB read by the thread, next slots of every (core, item) written by it).

Three changes took the 10^6 frame from 12,125 to 6,894 us (first) and 10,575 to 6,339
(settled); each was measured on its own, with the frame unprofiled (`bench_thread --runs
3`, results in `results/steps/`):

| change | 10^5 first | 10^5 settled | 10^6 first | 10^6 settled |
|---|---:|---:|---:|---:|
| before (main, 2026-11-14) | 2,557 | - | 12,124 | - |
| the vector unit finishes the lanes; kept records from Dst by segment loads | 2,255 | - | 9,313 | - |
| the writer counts and sends the records to their slots | 2,047 | 1,989 | 6,892 | 6,371 |
| ranges from the last frame's predicted block costs | 2,042 | 1,963 | 6,886 | 6,336 |

- The vector unit finishes all 32 lanes of a block (keep flags, top-left thresholds,
  box); the pack thread had done it lane by lane in scalar code after copying every
  record out of Dst. COUNT 3,826 to 1,843 us, SETUP 5,064 to 4,236 (10^6).
- The pack thread hands each kept triangle's box word to the writer (in SETUP with its
  record) instead of walking the tiles the box reaches itself; the writer, otherwise
  waiting, keeps the counters and histograms and gives every item its next slot. COUNT
  1,843 to 1,045, SETUP 4,236 to 2,619.
- COUNT and SETUP give each core a range of the mesh's blocks; with equal ranges some
  hold far more triangles in view than others, and the pass waits for the slowest.
  COUNT's pack thread adds up a predicted cost per block (at first 6,347 a block, 408 a
  kept triangle or piece, 22 a tile entry, 19,433 a clipped triangle: a fit to measured
  passes; now SETUP's own fit, "What was sped up" below) and keeps its sums at 31 evenly spaced places of its range; the thread cuts the
  next frame's ranges into equal parts of the total, linear between the places, and
  writes them while the grid runs FILL (COUNT and SETUP are done by then; the cut reads
  about 1,000 words of the camera's region, 30 ns each, and would otherwise cost 40 us
  before COUNT). Before, each core's COUNT and SETUP ticks were spread evenly over its
  range: settled 6,371 against 6,336. Scenes with fewer blocks than cores (tiny) keep
  equal partitions. The cut is history: a frame from a new camera starts from the last
  camera's costs.

Per frame the host still loads the program (the camera goes with the load: 0.5 to 0.8
s) and reads the image back (20 MB, 10 to 18 s); both are outside the frame times above.

### SHADE (`live.py --shade`)

On a frame loaded shaded the thread runs SHADE between FILL and SHOW: each fill core
colours its tiles in place from the materials and the textures in 4 KB pages
(`kernels/shade.h`, `shade.py`). Each piece of a triangle the near plane cuts is
shaded with its own u/z, v/z and light/z planes, while the reference uses the first
kept piece's planes for all of them; both are the same plane in exact arithmetic, and
on Sponza's frame 0 (33,339 pixels on later pieces, clipped corners up to 9.2e5 px off
screen) the two float64 (u, v) agree to 0.0000 texel at the 99th percentile, while
float32 rounding moves the own piece's by 0.64 texel median (2.16 at the 99th
percentile) and the first piece's by 0.80 (1.03). So `--check-shade` counts those
pixels apart and takes its share (ok under 1 %) over the others.

A texture level is stored in blocks of 32 x 32 texels, a 4 KB page each. For a tile
SHADE finds the pages its pixels fetch, reads them into 128 page slots in L1 and
fetches the texels there; a tile needing more pages is shaded in several passes of
rows. Lit materials are shaded as textured (no lighting; `--light` below lights them).

**Holes (the masked layer).** A scene with masked triangles is loaded as two meshes:
the triangles whose material is not `masked` and those whose material is, each with its
own mesh, shade, lists (sized from its own triangles), directory and frame tensors
(`shade.prepare`, `card_thread.layer_tensors`). The thread runs its frame (COUNT to
FILL) once per mesh, the second time on the masked layer's words (`RF_W_M_*` in
`thread/frame.h`), with COUNT's and SETUP's ranges and costs in its own area of the
camera's region (`RF_AREA_MASKED`; at first on equal partitions). Then two SHADE commands: the opaque layer's
as before, then the masked layer's (a6 the opaque layer's directory). For each masked
item it reads its five planes and the opaque layer's five of the same tile, drops the
pixels where the opaque layer is nearer (FILL's rule, larger 1/z; a tie goes to the
opaque layer, where the reference gives it to the lower triangle number), shades the
rest, takes each texel's alpha byte (the mask texture's, keyed beside the albedo's at
the same place and level, or the albedo's) and writes into the opaque layer's colour
planes its own colour where the byte is at least 128, the opaque layer's elsewhere. A
hole shows the opaque layer, never a second masked surface. A scene without masked
triangles runs as before.

Card 1, 2026-11-22 23:31 and 23:32, AICLK 1,350 MHz, 20 frames each, `live.py --scene
<s> --frames 20 --push --shade --check-shade` (ms, medians; pages read by the opaque
layer's SHADE in a frame). FILL is both layers' FILL; SHADE both commands, the masked
layer's in brackets:

| scene | opaque layer's frame | masked layer's frame | FILL | SHADE | SHOW | card frame | host frame | pages |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sponza (34,940 masked) | 3.54 | 1.35 | 1.64 | 4.52 (0.46) | 0.79 | 10.49 | 11.28 | 20,454 |
| Bistro interior (19,876 masked) | 6.45 | 1.19 | 2.20 | 5.11 (0.15) | 0.83 | 13.88 | 14.59 | 10,491 |
| Bistro exterior (765,306 masked) | 10.10 | 9.49 | 4.51 | 11.07 (4.37) | 0.86 | 31.83 | 32.62 | 15,192 |

Frames 0, 10 and 19 against the reference's shaded frame with only `lit` made textured:
pixels differing by more than 2 in a channel, at most 0.0006 % (Sponza) and 0.0059 %
(Bistro interior) of those checked; the masked kind's own share 0.011 % at most (Sponza frame 0, 1
pixel), 0 on Bistro interior. Bistro exterior (card 1, 2026-11-23 02:41, the same command): 2, 3 and
5 of 1,048,576 pixels (0.0005 % at most), the masked kind 0; the most pages one item read 74, no item
in more than one pass (SHADE's profile words `kShadeProfItemPages`, `kShadeProfItemPasses`). The step before (masked shaded as textured, one layer;
the page's earlier run): Sponza FILL 1.59, SHADE 4.04, SHOW 0.79, card frame 8.87, host
9.61; Bistro interior 2.11, 4.68, 0.83, 12.41, 13.12.

### Surface planes and LIGHT (`live.py --light`)

A frame loaded lit (`--light`, which implies `--shade`) runs FILL, SHADE, LIGHT, SHOW.
SHADE becomes the material stage: when its command carries the surface's directory
(a7, with the camera in a5) it writes per pixel the surface planes instead of a colour
(`kernels/shade_lit.hpp`), and LIGHT, one command to the fill's cores, turns each tile's
planes into the three colour planes SHOW reads (`kernels/light_compute.hpp`,
`light_reader.hpp`, `light_writer.hpp`; `light.py` holds the host's side and checks).
Without `--light` SHADE runs its old path unchanged and gives the textured frame.

- **The planes.** In the frame: 1/z (merged: where the masked layer's pixel is taken,
  its 1/z) and, in the triangle's plane, the material word `material | kind << 16` (an
  empty pixel keeps FILL's -1, bit 31 set). In two more frames of five planes laid out
  as the fill's (the surface's directory, `light.surface_tensors`): `a` (albedo x base,
  linear; `emissive` x emissive_scale), the normal (scene, unit), the visibility (the
  light plane's value, the corners carrying the baked visibility), the specular colour
  x its texel (0 without a specular term). A lit item leaves SHADE as three five-page
  parts in kCbOut (frame, surface 0 to 4, 5 to 9), so no push is larger than FILL's.
- **The normal without per-triangle data.** For each triangle that wins a pixel of the
  item (a table of the item's winners from its triangle plane; other records are
  skipped), its first record's planes of 1/z, u/z, v/z and the camera give in scalar
  code the triangle's normal `-(A f, -B f, A cx + B cy + C)` (planes taken about the
  centre) and the tangent frame through `M = (m0, m1, m2)`: dP/du ~ diag(1, -1, f)(m2 x
  m0) sign(det M), the bitangent toward +v (`light.frame_from_planes` is the same in
  numpy). The slot (16 words: material word, lit base, specular colour, frame) is
  gathered per pixel with the three texels (albedo, normal, specular: each keyed at its
  own level, SETUP's record word 30 now holds three levels, the shade block four words a
  triangle) and the vector unit forms `a`, the specular and `normalize(T tx + B ty + N
  tz)` (rsqrt: estimate and three Newton steps); a frame that is not valid, a triangle
  the host marks as having none (the reference's `frame_ok` false: bit 28 of the shade
  block's material word, kept in the record word), or no normal texture, leaves T = B = 0
  and the white texel: the triangle's normal.
- **LIGHT.** Per pixel, as vector arithmetic over a row's 32 lanes in Dst, a quarter tile
  (8 rows) at a time, the reference's `light_layer` with every term: the scene point
  from the pixel and the merged 1/z (`world_points`); `E = max(0, sky(n)) x vis +
  ambient + sun x max(0, n.s) x shadow + the lamps`; colour `a / pi x E` plus, where the
  specular plane is not 0, the Blinn-Phong term for the sun and for each lamp, each times
  that light's irradiance over pi (the exponent in the material word's bits 20 to 30, the
  power by squaring); `emissive` `a`; empty the background; then `x = exposure x
  colour`, `x / (1 + x)` by the exact reciprocal. The numbers are immediates from a 4 KB
  table (`light.light_table`: the sky's products, the sun, the shadow camera, the
  background's turn, the maps' page numbers, up to 32 lamps; the reader adds the
  camera), which the reader takes from the frame thread's sun region (LIGHT's a1), so
  that the host can rewrite it between frames ("The sun" below).
- **Shadow and background as pages.** The shadow map (float32 1/z, its side from the
  lighting file: 1,024 or 2,048) and the
  octahedral map (its side from the lighting file: 512 x 512 or 2,048 x 2,048, one map per
  channel) are uploaded as pages of 32 x 32 values, pages row by row
  (`light.light_pages`: 1,024 or 4,096 + 3 x 256 or 3 x 4,096 pages; LIGHT's table
  carries the sides and page counts). Per pixel the
  vector unit forms the four texel keys: the point moved one shadow texel along `n`
  into the sun's camera, its 1/z there, the 2 x 2 filter's corner and fractions (a tap
  outside the map gets key 0, a texel of 1/z 0: lit); an empty pixel's keys are the
  background texel of its view direction turned by `-turn` about y (the map is stored
  in the environment's own frame; the turn's cosine and sine are table words, each
  product and sum rounded on its own as in `raster.background`) in each channel
  (`octahedral_texel`'s arithmetic). Then SHADE's mechanism, shared (`page_passes`, `serve_ask`): the pages a
  pass of rows needs into the 128 slots, the texels gathered into Dst, the colour rows.
  A row needs at most 128 pages (32 pixels x 4 taps).
- **The ring.** LIGHT's item is 12 planes (surface ten, 1/z, material word) and takes 15
  pages of the 20-page plane ring, pushed and popped five at a time: the ring's place
  stays a multiple of five pages, which SHADE's items of 5 and 10 pages need. One item
  is in flight at a time.

Check A (host, `python -m projects.raster.light --check-frame 5000`): the frame from a
record's float32 planes against the reference's `shade_setup` frame on 5,000 whole
records: normal median / 99th percentile / most 0.0000 / 0.031 / 0.038 degrees (Sponza),
0.0061 / 0.034 / 0.147 (Bistro interior); a handful of records where the reference's uv
determinant is exactly 0 have a non-zero float32 det M; those triangles are now marked
by the host (Sponza 314 triangles, Bistro interior 2,120), whose `frame_ok` equals the
reference's on every triangle of the three public scenes.

Card 1, 2026-11-23 01:47 to 01:51, AICLK 1,350 MHz, 20 frames each, `live.py --scene <s>
--frames 20 --push --light --check-surface --check-light` (every term) and the same with
`--shade --check-shade` (ms, medians; SHADE both commands, the masked layer's in
brackets):

| scene | mode | opaque frame | masked frame | FILL | SHADE | LIGHT | SHOW | card frame | host frame |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sponza | `--light` | 3.55 | 1.36 | 1.64 | 4.79 (0.83) | 2.09 | 0.78 | 12.83 | 13.50 |
| Sponza | `--shade` | 3.55 | 1.35 | 1.64 | 4.57 (0.46) | - | 0.79 | 10.55 | 11.26 |
| Bistro interior (4 lamps) | `--light` | 6.75 | 1.19 | 2.20 | 4.94 (0.44) | 2.89 | 0.82 | 16.88 | 17.60 |
| Bistro interior | `--shade` | 6.75 | 1.18 | 2.20 | 5.17 (0.15) | - | 0.83 | 14.22 | 14.89 |
| Bistro exterior (sun, sky map, no lamps) | `--light` | 10.10 | 9.49 | 4.51 | 8.87 (2.58) | 2.32 | 0.86 | 31.83 | 32.67 |
| Bistro exterior | `--shade` | 10.10 | 9.49 | 4.51 | 11.07 (4.37) | - | 0.86 | 31.83 | 32.62 |

LIGHT's pack-thread cycles a tile (the profile's mean over the tiles; each part ends
with a drain of the vector unit). The keys, the page scan, the wait for pages and the
gather serve the shadow's four taps and the background's three texels together; the
scan (a hash lookup per tap per pixel in scalar code; since in the vector unit, "What
was sped up") is the largest part:

| scene | keys | page scan | wait for pages | gather | shadow filter | lamps | specular (sun) | rest (sky, ambient, sun, tone) | planes in, out, waits | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sponza (no lamps) | 52,963 | 144,853 | 4,346 | 16,043 | 5,051 | 864 | 20,648 | 32,325 | 19,810 | 301,272 |
| Bistro interior | 46,326 | 127,837 | 3,877 | 14,035 | 4,670 | 79,967 | 18,917 | 28,268 | 17,973 | 345,840 |
| Bistro exterior (no lamps) | 52,984 | 145,037 | 4,248 | 16,043 | 5,046 | 862 | 20,644 | 32,328 | 19,794 | 301,356 |

A lamp costs 19,992 cycles a tile on Bistro interior, its specular term included. Part
A's LIGHT (sky, ambient, unshadowed sun) took 0.35 to 0.37 ms and 45 to 50 thousand
cycles a tile. Pages read in a frame: 8,989 (Sponza), 8,963 (Bistro interior), in one
pass per quarter tile.

Check B, frames 0 and 19 (pixels the reference covers, later pieces of cut triangles
apart): material words (with the exponent) equal, visibility within 1/255 everywhere;
`a` off by more than 2/255 on 6 and 3 of 998,723 pixels (Sponza) and 59 and 62 of
1,015,272 (Bistro interior), each a neighbouring texel chosen at a texel's edge; the
normal's angle median 0.013 / 99th percentile 0.030 / most 0.73 degrees (Sponza), 0.002
/ 0.029 / 0.30 (Bistro interior), none over 2 degrees (part A: 709 pixels on Sponza's
frame 0, on triangles without a frame). Check C against `rasterize_lighted`'s default
frame (every term, the 2 x 2 filter), frames 0, 10, 19: share of pixels more than 2 / 4
/ 8 off in a channel at most 0.018 % / 0.013 % / 0.009 % (Sponza) and 0.0034 % /
0.0015 % / 0.0003 % (Bistro interior): `ok`. On Sponza the differing pixels are at the
sun's shadow edges: in the band of pixels whose reference filter is strictly between 0
and 1, and their neighbours (270 thousand pixels), 0.044 % to 0.048 % are more than 4
off (at most 94), outside it 0.0001 % to 0.0007 %; a tap flips where the point's 1/z in
the sun's camera is within a rounding of the texel's less the bias (the card's fused
multiply-adds round the point and the projection once where numpy rounds twice). Bistro
interior has no such band (the sun's share is 0 or 1 at every pixel). On Sponza's later
pieces (33,600 pixels) 35 % to 67 % differ by more than 2, as with SHADE.

**Bistro exterior** (2,833,330 triangles, 765,306 of them masked; 133 materials, 208
textures; the authored sun and sky map, no lamps). One size bound: SHADE's table (L1
buffer `kCbShadeTable` and its tensor, `kShadeTableBytes`) needs 816 + 14 a material + 2
a (texture, level) words, 7,254 for the exterior, and held 4,096; it holds 8,192 now (32
KB; the frame program's L1 1,134,448 of 1,398,336 bytes), and `shade.table` refuses a
scene beyond it by name. The texture pages are one interleaved tensor of 246,778 pages of
4 KB (1,010,802,688 bytes), each level of each texture in 32 x 32 blocks row by row; a
texel's key holds 2^20 - 1 pages (`shade.KEY_PAGES`, checked at the load). Nothing else
bound: the lists, the heavy tiles (68 at most), the items, the ring and LIGHT's tables
took the scene as they were. Card 1, 2026-11-23, AICLK 1,350 MHz, 20 frames each unless
said, `live.py --scene bistro_exterior_<view> --push`:

- vertex colours (02:36): 0 pixels differ (frames 0, 10, 19); card frame 13.29 ms (COUNT
  2.47, SETUP 4.87, FILL 3.99, SHOW 0.87), host 14.04 ms, 71 frames a second;
- `--light --check-surface --check-light` (02:43, the tables above): check C over 2 / 4 / 8
  at most 0.053 % / 0.022 % / 0.019 %, `ok`; in the shadow-edge band (512 thousand pixels:
  the trees' shadows) 0.040 % to 0.044 % more than 4 off, outside it 0.0002 % to 0.0007 %;
  check B: `a` off by more than 2/255 on 2 and 5 pixels, the normal's angle median 0.009,
  99th percentile 0.031 degrees, 5 and 4 pixels over 2 degrees (most 24.8); lit SHADE read
  34,143 pages, at most 217 for one item, one item in two passes; LIGHT 7,693 pages;
- `bistro_exterior_close --light --check-light` (02:45): `ok`, over 2 / 4 / 8 at most
  0.035 % / 0.022 % / 0.018 % (later pieces, 6,261 to 6,368 pixels, apart); card frame
  33.40 ms (masked layer 13.26), host 34.25; lit SHADE at most 816 pages for one item, 22
  items in passes of rows;
- 200 frames along the camera path (`--frames 200 --light`, 02:46): every result word 0,
  card frame median 32.26 ms, worst 38.26; host median 33.09, worst 39.43.

The exterior's frame was the two layers' passes: the masked layer (trees) ran COUNT and
SETUP on equal partitions (9.5 ms of the 31.8; "What was sped up" below).

### The sun (`live.py --sun AZ,EL`, `--sun-path N`)

The sun of a lit frame moves (`lights.moved` in `reference/lights.py`, the reference's
"Moving the sun"): the host computes the sun's numbers, the card renders its own shadow
map. Everything in LIGHT that depends on the sun is in LIGHT's table, which now lives in
the frame thread's sun region (`thread/frame.h` `RF_R_SUN`: the sun's camera page, then
the table; LIGHT's a1), so the host rewrites it between frames (`running.thread.write`)
without a load. `light.sun_lights` is `moved` without its shadow map: the turn,
`sh_turned`, `sun_direction`, `sun_camera`, the texel by `shadow_map`'s rule and the
bias, each by the reference's own function or line (`sun_camera` reads only the box,
so it gets the box's two corners: 1 ms instead of 125 on the exterior); at a scene's own
angles its table equals the loaded one word for word (Sponza, Bistro interior), on the
exterior but for four words of 3e-17 against -0 (the reference's note).

`Loop.set_sun(azimuth, elevation)` writes the region and submits **SUN** to the thread
(`serve_sun`), which renders the shadow map with the frame's own stages, once a
quadrant (below; side 1,024: once): each layer's
COUNT to FILL from the sun region's camera, each layer's ranges and costs in its own
area of the camera's region (`RF_AREA_SUN`; the first pass on equal partitions), the masked layer's lit SHADE (its merge
leaves the merged 1/z in the opaque layer's frame: leaves cast the shadow of their
leaves), then SHOW in a raw mode (`kShowRaw`: plane 0 word for word, rows from the top)
into the quadrant's pages of the light pages tensor, the layout LIGHT fetches. The eye's
frame and surface tensors are its scratch; the next FRAME writes them again. At or
below the horizon there is no pass and the sun's irradiance is 0. Check A (host):
SETUP's and FILL's float32 arithmetic at the sun's camera (1,000 scene radii away, 1/z
about 1e-7) meets no denormal on Sponza, Bistro interior or exterior and gives the
reference's map at every covered texel bit for bit.

- **Lists.** The sun sees the whole scene: at the load (`live.sun_slots`) each layer's
  entries in the sun's view are counted at the scene's own sun, four azimuths at
  elevations 25 and 80 and the zenith (box tiles, an upper bound; 0.2 / 0.9 / 2.6 s on
  Sponza / interior / exterior), and the lists hold 1.25 times the most plus 65,536 if
  that is more than the eye's. Most: Sponza 230,611 + 24,281, interior 484,980 + 12,986,
  exterior 653,337 + 265,699 (opaque + masked); the eye's default slots are larger on
  all three. A sun beyond the lists ends SUN with the result word 6 (`RF_LISTS_SMALL`),
  never an overrun; heavy tiles past 256 and items past 2,048 have their words too.
- **The ring.** SUN adds no item size: FILL's items (5 pages) and SHADE's (5 a part, 10
  merging, pushed five at a time) keep `kCbOut` (15) and the plane ring (20) at
  multiples of five, and SHOW's raw items are its own 3 pages in `kCbShow`. The four
  quadrant passes are four of these passes: no item changed.
- **A 2,048 map: four quadrants** (the lighting entry's `shadow_side`, default 2,048; the
  reference's step 3). The sun region holds a camera page a quadrant (64 bytes apart,
  quadrant (i, j) the (j n + i)-th; `light.quadrant_cameras`: the 2,048 camera with
  its principal point moved by (-1024 i, -1024 j)) and n, the quadrants a side (1 or 2,
  `RF_SUN_ACROSS_AT`); SUN runs the pass above once a quadrant. The 4,096 shadow pages
  are one 64 x 64 grid row by row, the map's own `map_pages` layout, so LIGHT's lookup
  did not change (its table already carries the side, the pages a row and the first
  page); SHOW raw writes a tile at row r, column c to page a5 + r a6 + c (a5 the
  quadrant's first page, 32 i + 2,048 j; a6 64). Each quadrant has its own cost history
  (two areas a quadrant: `RF_AREAS` 4 to 10, the camera region 196,608 to 491,520
  bytes), so a first SUN pass deals all four on equal partitions. The lists hold the
  most entries of one quadrant over the sampled suns. Side 1,024 (an entry with
  `shadow_side` 1024) is one quadrant and gives the earlier map bit for bit.

Card 1, 2026-11-23 03:24 to 03:36, AICLK 1,350 MHz, `live.py --scene <s> --frames 20
--push --light --check-light --check-sun --sun=AZ,EL` (one SUN pass, then 20 frames; ms
from the thread's sun row). Check B: the card's map against `rasterize_shaded` at the
sun's camera; check C: shares of pixels over 2 / 4 / 8 off against `rasterize_lighted`
with `lights.moved`, later pieces apart, frames 0, 10, 19 (the most of the three):

| scene | sun (az, el) | SUN pass | opaque frame | masked frame | merge | copy | B: texels over 1e-5 / one side empty | C over 2 / 4 / 8 % |
|---|---|---:|---:|---:|---:|---:|---|---|
| Sponza | own (144.08, 74.53) | 6.77 | 4.92 | 1.21 | 0.52 | 0.12 | 0 / 0 | 0.018 / 0.013 / 0.009 (= fixed sun) |
| Sponza | 90, 74.53 | 5.46 | 3.54 | 1.21 | 0.58 | 0.13 | 0 / 0 | 0.0033 / 0.0018 / 0.0014 |
| Sponza | 144.08, 25 | 5.91 | 4.18 | 1.18 | 0.42 | 0.13 | 0 / 0 | 0.0022 / 0.0010 / 0.0007 |
| Bistro exterior | own (-90, 61.18) | 20.60 | 12.45 | 6.31 | 1.68 | 0.15 | 0 / 0 | 0.053 / 0.022 / 0.019 (= fixed sun) |
| Bistro exterior | 90, 61.18 | 20.31 | 12.40 | 6.26 | 1.51 | 0.13 | 0 / 0 | 0.032 / 0.011 / 0.008 |
| Bistro exterior | -90, 25 | 18.87 | 11.61 | 5.80 | 1.31 | 0.15 | 0 / 0 | 0.108 / 0.036 / 0.029 |
| Bistro interior | own (144.08, 74.53) | 8.56 | 6.77 | 1.10 | 0.53 | 0.15 | 0 / 0 | 0.0034 / 0.0015 / 0.0003 |

Every check C is `ok`; the moved suns' differences sit in the shadow-edge band as at the
fixed sun (exterior at elevation 25: 0.13 % of the band over 4, outside 0.0007 %). The
frames after a pass are the fixed sun's (Sponza card frame 12.89 ms, exterior 31.84,
interior 16.92, as without a pass). In a loop (`--sun-path 10 --frames 200`: the
azimuth from the scene's own in steps of 15 degrees, the elevation 25 to 80 by a sine
over 24 steps; 03:35): every frame's and every pass's word 0; Sponza SUN pass median
5.99 ms, worst 6.76 (host, numbers and command, 6.86 / 7.63), card frame median 12.82,
worst 15.08; exterior SUN pass 18.41 / 20.39 (host 19.34 / 21.59), frame 32.27 / 38.26.

Limits: the pass is the eye's frame cost again; the elevation moves the light's sun, not the
background's sun disc (the reference's known inconsistency); the lists are sized from
sampled suns, so a far other sun may need `--list-slots`.

**At side 2,048** (four quadrants). Card 1, 2026-11-23 15:46 to 16:04, AICLK 1,350 MHz,
the same command with the lighting files at side 2,048 (side 1,024 from files with
`shadow_side` 1024, rebuilt: their maps equal the earlier ones bit for bit). One SUN
pass, the first (each quadrant on equal partitions); ms; check B by quadrant:

| scene | sun (az, el) | side | SUN pass | opaque | masked | merge | copy | B: texels over 1e-5 / one side empty, each quadrant | C over 2 / 4 / 8 % |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| Sponza | own | 1,024 | 6.60 | 4.84 | 1.20 | 0.45 | 0.12 | 0 / 0 | 0.016 / 0.012 / 0.009 |
| Sponza | own | 2,048 | 20.47 | 14.49 | 4.32 | 1.14 | 0.51 | 0 / 0 | 0.052 / 0.022 / 0.013 |
| Sponza | 90, 74.53 | 2,048 | 17.17 | 10.81 | 4.48 | 1.37 | 0.51 | 0 / 0 | 0.045 / 0.012 / 0.010 |
| Sponza | 144.08, 25 | 2,048 | 17.39 | 12.05 | 3.98 | 0.85 | 0.50 | 0 / 0 | 0.009 / 0.002 / 0.001 |
| Bistro exterior | own | 1,024 | 18.42 | 10.91 | 5.88 | 1.48 | 0.15 | 0 / 0 | 0.053 / 0.022 / 0.018 |
| Bistro exterior | own | 2,048 | 61.69 | 39.90 | 17.24 | 3.86 | 0.69 | 0 / 0 | 0.424 / 0.069 / 0.028 |
| Bistro exterior | 90, 61.18 | 2,048 | 61.20 | 39.29 | 17.34 | 3.86 | 0.71 | 0 / 0 | 0.145 / 0.036 / 0.018 |
| Bistro exterior | -90, 25 | 2,048 | 58.73 | 35.53 | 18.31 | 4.14 | 0.75 | 0 / 0 | 0.442 / 0.127 / 0.049 |
| Bistro interior | own | 1,024 | 8.00 | 6.29 | 1.11 | 0.44 | 0.15 | 0 / 0 | 0.0033 / 0.0014 / 0.0003 |
| Bistro interior | own | 2,048 | 29.92 | 23.82 | 4.24 | 1.23 | 0.63 | 0 / 0 | 0.0033 / 0.0014 / 0.0003 |
| Bistro interior | 90, 74.53 | 2,048 | 35.54 | 29.47 | 4.16 | 1.34 | 0.57 | 0 / 0 | 0.0032 / 0.0013 / 0.0003 |
| Bistro interior | 144.08, 25 | 2,048 | 28.14 | 22.08 | 4.09 | 1.26 | 0.70 | 0 / 0 | 0.0081 / 0.0059 / 0.0045 |

Every check C is `ok`. At 2,048 more pixels in the shadow-edge band differ (exterior's
own sun: 0.97 % of the band over 2 against 0.11 % at 1,024; outside the band 0.001 %
at both): the bias is three texels, now half the 1/z, so the card's and numpy's
roundings flip a tap more often. The lit frame (fixed sun, 16:09 to 16:11, against
1,024 at 15:46 to 15:58): card frame Sponza 11.69 ms (11.55), interior 15.14 (15.09),
exterior 24.45 (24.38); LIGHT 1.63 (1.50), 2.25 (2.19), 1.64 (1.57) ms, its cycles a
tile 222,641 (211,081), 268,414 (262,244), 215,114 (205,502), all of the difference in
the page scan (Sponza 64,822 against 53,468); pages read a frame 12,610 (9,281), 12,567
(9,844), 10,827 (7,978). Along `--sun-path 10 --frames 200` (16:03 to 16:04), every
word 0: Sponza SUN pass median 17.04 ms, worst 19.43 (the first; host 17.86 / 20.63),
against 5.23 / 6.28 at 1,024; frame median 11.65, worst 13.81; exterior 45.36 / 61.33
(host 46.28 / 62.74) against 13.67 / 18.02, frame 24.77 / 34.17. The exterior's pass
parts (medians): COUNT 8.52, middle pass 4.44, SETUP 17.80, FILL 9.71 against 2.45,
1.17, 4.47, 3.86: four passes cost four times one, the middle pass's thread work
included. The viewer (`projects/viewer/check.py` on card 1, 16:06 to 16:09): every
case `ok`; the sun moved on each lit scene, the state's SUN pass 21.6 ms (Sponza),
30.5 (interior), 59.2 (exterior).

### What was sped up (2026-11-23)

Profile first (`live.py --profile-pages`: the masked layer has its own FILL, COUNT and
SETUP, SHADE profile tensors, and the times row its own parts, words 32 to 52). Card 1,
AICLK 1,350 MHz, `live.py --scene <s> --frames 20 --push --light --check-light`, ms,
medians; before: the commit's kernels (04:54, 04:55; interior 05:32), after: 05:44 to
05:45. Bistro exterior's frame, 31.84 ms before: SETUP 10.4 (the masked layer's
slowest core 5.2 times the mean, on equal partitions), SHADE 8.9, FILL 4.5, COUNT 3.3,
LIGHT 2.3.

1. **The masked layer's ranges by cost.** The camera's region holds four areas
   (`RF_AREAS`: the eye's two layers, a SUN command's two), each a camera page (the
   thread copies the camera into areas 1 to 3), ranges and costs; every layer's COUNT
   and SETUP are dealt by its own last costs. Exterior masked layer 9.49 to 5.87 ms
   (SETUP 5.37 to 2.45, COUNT 1.35 to 0.66).
2. **The cost knots.** The old weights' predicted cost correlated 0.24 with COUNT +
   SETUP per core on the exterior (SETUP's slowest core held 8,576 kept triangles,
   the mean 2,422). SETUP keeps COUNT's ranges (SUM's prefixes are per core), so one cut
   serves both; of the weightings tried, SETUP's own fit (`kCostBlock` 4,050, kept 400,
   entry 40, clipped 9,250) gave the shortest COUNT + SETUP.
3. **The page scans** (`row_addresses` in `kernels/shade_compute.hpp`). A row's 32 keys
   in the vector unit: page and validity as vectors, `4 key + (slot - 1 - page) x 4096`
   added on the lanes of each distinct page, one `slot_of` a distinct page (LIGHT: the
   last page tried first, the next by a minimum reduction; SHADE: the first pending
   lane's). The same pages, slots and passes as the scalar scan. LIGHT 145,035 to
   47,585 cycles a tile (46 a pixel of four taps), lit SHADE 177,959 to 79,275 (77 a
   pixel of three textures). About 12 vector instructions a key at roughly one element
   a cycle is the floor of this form (an estimate from the cycles, not measured alone).
4. **The texture levels in SETUP** (`shade_word`): the loop over k in closed form, equal
   to it on 10.9 million inputs (every exponent, the edges). Exterior opaque SETUP 4.18
   to 2.99 ms.

| scene | | opaque frame | masked frame | FILL | SHADE | LIGHT | SHOW | card frame |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Bistro exterior | before | 10.10 | 9.49 | 4.51 | 8.87 (2.57) | 2.33 | 0.86 | 31.84 |
| Bistro exterior | after | 8.46 | 5.43 | 4.51 | 7.82 (2.21) | 1.55 | 0.86 | 24.36 |
| Sponza | before | 3.55 | 1.35 | 1.64 | 4.82 (0.87) | 2.10 | 0.78 | 12.88 |
| Sponza | after | 3.44 | 1.31 | 1.64 | 4.19 (0.77) | 1.49 | 0.78 | 11.54 |
| Bistro interior | before | 6.76 | 1.19 | 2.20 | 4.97 (0.45) | 2.91 | 0.82 | 16.92 |
| Bistro interior | after | 6.51 | 1.06 | 2.20 | 4.17 (0.38) | 2.19 | 0.82 | 15.07 |

Exterior parts after (opaque / masked): COUNT 2.48 / 0.81, middle pass 0.36 / 0.63,
SETUP 2.84 / 1.86, FILL 2.62 / 1.89. Every check as before: check C over 2 / 4 / 8 at
most 0.053 / 0.022 / 0.019 % (exterior), 0.018 / 0.013 / 0.009 (Sponza), 0.0034 /
0.0015 / 0.0003 (interior); check B's `a` over 2/255 on 2 and 5 (exterior), 6 and 3
(Sponza), 59 and 62 (interior) pixels, materials equal; `--shade --check-shade` on
Sponza 0.0006 % at most. The SUN pass along `--sun-path 10 --frames 200` on the
exterior (05:37, 05:38): median 18.41 to 13.65 ms, worst 20.40 to 18.03 (the first
pass, on equal partitions); frames 32.27 / 38.26 to 24.69 / 34.07. With a 2,048 x
2,048 octahedral map in the lighting file the exterior is `ok`, LIGHT 1.57 ms, 7,978
pages: the map's side and the page counts come from the file.

## The interactive loop

`live.py` loads the scene and the frame program once, with the thread built to serve
(`RF_SERVE`): `kmx_main` sets up and returns, and each `FRAME` command to the thread runs one
frame from the camera the host wrote into the thread's `camera` region (16 words), then `SHOW`,
then answers. `SHOW` is one command to the fill's cores: each packs its tiles' colour into one
page a tile of the `display` tensor (32 x 32 pixels of `R, G, B, 255`, by `reference/png.py`'s
rule, in scalar code on the writer; a band writes its rows). With `--push` the thread also
issues the display's push into the host channel before it answers, so the host reads nothing.

```bash
python -m projects.raster.live --scene sponza_overview --frames 200 --push --pictures DIR
```

Card 0, 2026-11-22, AICLK 1,350 MHz, 200 frames a scene, the cameras stepping forward along
the view; frames 0, 100 and 199 equal the reference rasterizer's 8-bit image of that camera
(0 pixels differ) in every run:

| scene | path | card frame ms | host ms a frame, median (5th, 95th) | frames a second |
|---|---|---|---|---|
| tiny | read | 1.807 | 3.147 (3.011, 3.287) | 317.8 |
| tiny | push | 2.094 | 2.723 (2.588, 2.869) | 367.2 |
| 10^6 | read | 7.315 | 8.606 (8.510, 8.727) | 116.2 |
| 10^6 | push | 7.603 | 8.234 (8.132, 8.351) | 121.5 |
| Sponza | read | 4.437 | 5.731 (5.547, 6.081) | 174.5 |
| Sponza | push | 4.723 | 5.350 (5.181, 5.686) | 186.9 |
| Bistro interior | read | 7.184 | 8.515 (8.424, 8.636) | 117.4 |
| Bistro interior | push | 7.471 | 8.129 (8.028, 8.255) | 123.0 |

`SHOW` is 0.80 to 1.14 ms of a card frame (107 to 114 thousand cycles a tile) and the push
0.287 ms. The lists are sized at the load: a camera that puts more triangles in view than
they hold ends its frame with the result word 6 (`--list-slots` makes them larger; the
masked layer's lists grow by the same factor).

## Results

The whole frame runs on the card: the table of the previous section (settled frame,
V100 / card: 10^6 0.70, Sponza 0.61, Bistro interior 0.92; a scene's first frame 0.64,
0.40, 0.68). The rest of this section is the host-binned path, where the host does setup
and binning and the cores run FILL alone; it is where the fill's own measurements come
from.

Scenes of the reference (`reference/scenes.py`): tiny, 10^4, 10^5 and 10^6 triangles,
1024 x 1024 pixels, 32 x 32 tiles; 119 worker cores; balanced assignment unless named.

The runs behind the tables of this section: 2026-11-13, card 1 (a p150a), AICLK 1,350
MHz unless marked, host load 5.9 to 10 (`/proc/loadavg`), the same kernels submitted by the
earlier driver (a test fixture on a 120th core in place of the dispatcher; its
submit-to-acknowledgement time in place of the event's). Commands, in this folder's
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

**Time of the host-binned path, milliseconds.** The card runs FILL; setup, binning and
the lists run on the host (numpy and Python) and are listed as host stages, not compared. Card: the frame
from a FILL command's submit to its acknowledgement, which comes after the command's
output (the frame's tiles) is written; over a loop of FILL commands back to back
(`bench.py`: 48,000 / 40,000 / 16,000 / 2,462 frames; median, min and max; host load
5.9 to 8.7). Cycles are converted at the clock read when the device opens (1,350 MHz);
AICLK was sampled every 50 ms through each loop (min / mean in the table). Host: the
reference's numpy `setup` and `bin_triangles`, and packing the records, the tile costs,
the assignment and the lists (medians of five). Upload of the lists: one run, slow
dispatch. References: the medians of the committed table in
[`reference/README.md`](reference/README.md) (fill stage; not rerun here).

| scene | card FILL (min-max) | AICLK MHz min / mean | card W mean | host setup | host binning | records + lists | upload | C++ 1 thread fill | C++ 96 threads fill | V100 fill | V100 fill / card FILL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tiny | 0.1195 (0.1154-0.1232) | 1343 / 1349.9 (*) | 75.1 | 0.46 | 0.27 | 8.0 | 119 | 11.12 | 1.01 | 0.06 | 0.50 |
| 10^4 | 0.1394 (0.1386-0.1406) | 1350 / 1350 | 79.0 | 4.01 | 1.19 | 10.5 | 187 | 15.65 | 1.42 | 0.16 | 1.15 |
| 10^5 | 0.4258 (0.4254-0.4270) | 1350 / 1350 | 75.8 | 33.0 | 6.6 | 48.2 | 227 | 27.87 | 1.90 | 0.57 | 1.34 |
| 10^6 | 2.0310 (2.0302-2.0324) | 1350 / 1350 | 75.1 | 374 | 47.0 | 206.6 | 735 | 69.86 | 3.06 | 3.34 | 1.64 |

(*) The tiny loop's clock left 1,350 MHz for at least one sample (1,343 MHz); its
frame times are cycles at 1,350 MHz, so they may be short by up to 0.5 % there.

10^5 and 10^6 are with heavy tiles in bands. These rows are the kernels with the
masked record fields and the timed waits; the same scenes before both took 0.4192 and
1.9940 ms (bands) and 0.4518 and 2.8226 ms (no bands). The bands cost host time in the
lists step (the band filter is plain Python per band).

On this path setup and binning run on the host in numpy: with the host stages and the
upload a frame takes 128, 203, 315 and 1,364 ms (9, 16, 88 and 629 ms without the
upload). The thread-driven program above runs those stages on the card instead.

**Power** of card 1 (`komorebi.bench.power.PowerSampler`), counted over the stretch in which
the loop's commands ran (5.0 to 7.0 s per scene). Idle: the closed card before the run,
and the open card between programs (after the first program, nothing running). The
V100 columns are from the committed reference table.

| scene | idle closed / open (W) | mean (W) | peak (W) | mean of the 150 W limit | J per frame | J per frame above closed idle | AICLK MHz min / mean | V100 W idle / mean / peak, J per frame |
|---|---|---|---|---|---|---|---|---|
| tiny | 20.0 / 62.3 | 75.1 | 78 | 50 % | 0.0090 | 0.0066 | 1343 / 1349.9 | 55 / 89 / 91, 0.026 |
| 10^4 | 20.0 / 61.5 | 79.0 | 80 | 53 % | 0.0110 | 0.0082 | 1350 / 1350 | 57 / 112 / 114, 0.028 |
| 10^5 | 20.0 / 62.3 | 75.8 | 78 | 51 % | 0.0323 | 0.0238 | 1350 / 1350 | 65 / 110 / 114, 0.087 |
| 10^6 | 20.6 / 63.0 | 75.1 | 78 | 50 % | 0.1526 | 0.1107 | 1350 / 1350 | 55 / 110 / 115, 0.484 |

At every size the card draws about half its limit and only 12 to 18 W above the open
card at rest: the cores are underused, waiting on their own instruction submit (setup)
or on each other (the slowest core), not drawing near the limit.

Calibration points on the same sensor (a loop that is not in the repository, 5 to 7 s, 119 cores): the closed
card 21.1 W; the open card at rest 62.6 to 63.0 W; a spin-only handler, every compute
core spinning on the clock, 63.4 W; the arithmetic check, 1,000,448 inputs per command
at 286.6 us, 70.5 W mean and 72 W peak. Neither loop loads the vector unit fully (the
check moves its data through the register window by the cores), nor the matrix unit or
the NoC; the fill draws more than both. The spin handler writes no output, and its
commands are acknowledged about 5 us after submit although compute spins for 1 ms; its
power counts the stretch from the first submit to the last, and no time in this README
comes from the acknowledgement of a command without output.

The read-back runs at about 1 MB/s through slow dispatch. The frame is 20 MiB: five
float32 planes (depth, three colours, winner) per pixel. Depth and colour must stay
float32 to be compared bit for bit; dropping the winner plane saves 4 MiB, half-float
colour another 6 MiB but no longer agrees exactly. The winner plane alone (4 MiB, or
int32) would be enough: the host can recompute colour and depth exactly from the
winner's record with the reference's own float32 arithmetic. Not done; read-back only
lengthens correctness runs.

**Profile.** Measured on the kernels before the masked fields and timed waits (1 to 2
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
- 10^6 (1.64): triangle setup, bound by the core submitting immediates: 55 % of all busy
  time, cores within 2 % (2.69 M against 2.64 M). Without bands it was one tile of
  8,780 triangles on one core (3.82 M cycles, 2.83 ms).

## Not on the card yet

Triangle setup, the per-tile counts and the scatter into the tile lists run on the
Tensix cores (COUNT and SETUP, above), the middle pass between them on the X280 thread. The
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

- Small triangles as lanes: an entry whose box, clipped to its tile, is at most 4 x 4
  pixels (73 % of the entries at 10^6) filled 32 at a time, one per lane, took a tile's
  fill to 0.49 to 0.78 of row by row on one core (`experiments/programs/raster_passes`, probe 4).
  It is not in the frame: the lists would have to carry those entries as runs of their
  own per item (COUNT and SUM counting them apart, PLACE's directory naming them, SETUP's
  writer sending them there), and the fill would take each run of 32 into Dst in lane
  order. FILL is a third of the settled 10^6 frame and two fifths of Sponza's.
- The per-triangle numbers spread by a mover (the option `--spread` on `run` and
  `bench_thread`, RASTER_SPREAD; off by default): for a record covering 4 rows or more
  the reader (every second one the writer) writes one row group (8 lanes) of each of the
  21 numbers into a tile, 192 words, the unpacker puts it in Dst, and the pack thread
  loads each slot of four into L0 to L3 and SFPTRANSP gives every lane its number. Settled
  frames, with and without it back to back in one hold of the card (2026-11-18, card 1,
  AICLK 1,350 MHz read with the card open, host load 59 to 72, `bench_thread --scene <s>
  --runs 5 --frames 0`, images exact): 10^6 6,200.5 us (FILL 2,103.7), Sponza 3,671.6
  (1,551.6), Bistro interior 6,315.7 (2,075.2), against 6,201.4 / 3,711.2 / 6,335.1
  without it: level at 10^6, 1.1 % less on Sponza, 0.3 % less on Bistro.

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
  profile item from the second command on. Worse, the two sides wrap differently:
  compute's push wraps its pointer when it reaches or passes the end, the writer's pop
  only when it lands exactly on it, so after one straddling item the writer's read
  pointer walked on through L1 and a 4,000-command run did not return. The kernels now
  check every item against the buffer's end before touching it (give-up word and stop flag on failure).
- **The destination register in 32-bit mode has its own lane order** in the memory
  window: vector v, lane l at word 64 (v / 2) + 16 (l / 8) + 2 (l mod 8) + v mod 2. The
  frame goes out in that order and the host permutes it.
- **Waits in the compute kernel have no timeout** in the runtime's calls, and a run that
  goes wrong this way did not end through the stop flag: the feeder's deadline set it,
  but no thread blocked inside a command looks at it. The kernels now poll every wait of
  their own with a wall-clock timeout (0.8 s) and set a give-up word: list pages, the
  handshake between the compute threads, output space, the NoC barriers. Masking the
  record fields and the wait timeouts cost 1 to 2 % of the frame (10^6: 1.994 to 2.031
  ms).
- **Waits software cannot time out.** The vector unit's drain (`tensix_sync`), the
  hardware semaphore waits inside the destination-register handshake
  (`tile_regs_acquire`, `copy_tile`) and a `SEMWAIT` block the thread in hardware; no
  wall clock is looked at. An instruction word the core should never have submitted
  (a data value spilled into the opcode) can wedge a compute thread there for good. The
  only defences are before the push: every record field masked to its width where it
  becomes part of an instruction, host checks of every word written ahead of time, and
  a ladder of runs from one command and one core up.
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
  - a timed wait for compute;
  - a contiguous-reserve call on a circular buffer that rejects a wrapping item;
  - a per-command cycle counter that compute can write, so kernels need no profile page
    of their own.
