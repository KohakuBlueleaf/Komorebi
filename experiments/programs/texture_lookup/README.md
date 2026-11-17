# texture_lookup

What a texture lookup costs on one Tensix core, to size a shading pass after the
rasterizer's FILL. A texel is addressed as (page, offset): a page of 32 x 32 texels
(4 KB of RGBA8 words), its key the upper bits of a 32-bit word and its offset the low
10. Pages are held in the core's L1, either all of them (the page is the address) or
in a small table of 8, 16 or 32 slots found by a hash, filled from DRAM on a miss.
Four probes, then a projection of a whole frame over 120 cores:

1. **Loop**: 1,024 pixels, each a texel address, the texels written out as a plane. The
   cycles a pixel on each of the four CPUs of a core that run C++ (the reader, the
   writer, the unpack and the pack thread), all pages held or through the table.
2. **Miss**: one page from DRAM into L1, alone and with 2, 4 or 8 reads in flight,
   from the reader (NoC 0) and from the writer (NoC 1).
3. **Colour**: texel x light term (r, g and b times a float in [0, 1], rounded), in
   integers on a CPU against the vector unit (unpack, multiply, pack). Cycles a tile.
4. **Real tiles**: the texel reads of shaded views of Sponza and Bistro, cut into the
   rasterizer's items per core, replayed through the table with every miss a real
   DRAM read.

```
python experiments/programs/texture_lookup/probe.py loop --cores 1 --json loop.json
python experiments/programs/texture_lookup/probe.py miss --cores 12 --json miss.json
python experiments/programs/texture_lookup/probe.py colour --cores 1 --json colour.json
python -m reference.public fetch sponza    # once: the scene and its materials
python -m reference.public materials sponza
PYTHONPATH=src:. python experiments/programs/texture_lookup/streams.py --out streams
python experiments/programs/texture_lookup/frames.py inputs --streams streams --out inputs
python experiments/programs/texture_lookup/probe.py replay --cores 12 --pool-pages 2048 \
    --streams inputs/samples.npz --passes 2 --json replay-samples.json
python experiments/programs/texture_lookup/frames.py project --streams streams \
    --inputs inputs --results . --out projection.json
```

- `probe.py` runs one of the four probes on 1 to 12 cores of grid row 0 and prints one
  JSON line: per row the cycles (minimum, median, maximum over the cores' runs) and
  whether every texel, sum and miss count equals the host's. It reads the results
  while the program runs through a REPORT command (the writer copies them into a DRAM
  tensor), since reading an L1 tensor while the cores are busy waits for them to stop.
- `run.sh <tag> <probe.py args>` is the same job under a hang watcher.
- `kernels/`: `table.hpp` (the table, the loop and the integer colour, the same code on
  every CPU), `fetch.hpp` (the DRAM reads, data-movement CPUs only), one kernel a
  CPU kind, `lookup.h` (the commands and layouts the host reads too).
- `streams.py` writes a view's texel reads (host only), `frames.py` the replay inputs
  and the projection. `streams.py` needs the rasterizer reference's SHADE stage with
  materials and textures (`projects.raster.reference.shade` and the scene loaders);
  it does not run without it.

Conditions of every number: Blackhole p150a (card 1), 2026-11-14, AICLK 1350 MHz
throughout (min = mean in every run), host load 4.3 to 4.6 (4.9 to 9.4 for the
repeat runs). Every run was exact: 0 wrong texels, every miss sum equal to the pool's
words, every colour plane equal to its host reference, every replayed item's misses
equal to the host's walk of the same table. The loop, colour and sample replay were run
again after the kernels' last change: within 0.5 % of the numbers below.

## 1. The loop

Cycles a pixel, median over 16 runs, one core (12 cores: the same within 0.02 cycle).
The table holds every page the stream uses; "coherent" is a tile mapped one texel a
pixel onto a 2 x 2 block of pages, "scattered" a random page and texel each pixel.
"Last page" keeps the previous pixel's key and slot and skips the hash when it repeats.

| Form | reader | writer | unpack | pack |
|---|---|---|---|---|
| All pages held (address = page << 12 \| offset << 2) | 10.0 | 10.0 | 10.1 | 9.8 |
| Table 8 / 16 / 32, coherent | 33.5 / 28.2 / 32.1 | 33.2 / 28.1 / 31.9 | 33.4 / 28.3 / 32.1 | 32.9 / 27.9 / 31.6 |
| Table, scattered | 36.6 to 36.7 | 36.4 to 36.6 | 36.6 to 36.7 | 35.9 to 36.0 |
| Table + last page, coherent | 17.3 to 17.8 | 17.3 to 17.7 | 17.3 to 17.8 | 17.0 to 17.4 |
| Table + last page, scattered | 38.2 to 40.3 | 37.9 to 40.0 | 38.0 to 40.0 | 37.5 to 39.6 |

- The four CPUs are one design: within 3 % of each other in every form. The lookup does
  not prefer a CPU; the CPU is chosen by what else it has to do.
- A held page costs 10 cycles a pixel (a load, a store, the address); the table adds
  18 to 26 (the hash and up to four tag compares). Keeping the last page brings a
  coherent tile to 17 and costs 2 to 4 cycles on a scattered one.

## 2. A miss

Cycles from the first read issued until all landed, median over 16 runs (12 cores:
the slowest core's median). 4 KB pages unless stated.

| Reads in flight | 1 core, reader | 1 core, writer | 12 cores, reader | 12 cores, writer |
|---|---|---|---|---|
| 1 | 581 | 569 | 589 | 664 |
| 2 | 642 | 626 | 663 | 1,090 |
| 4 | 766 (192 a page) | 765 | 805 | 2,344 |
| 8 | 1,017 (127 a page) | 1,030 | 1,169 (146 a page) | 4,712 |
| 1, 1 KB / 2 KB | 515 / 540 | 502 / 524 | 517 / 540 | 508 / 545 |

- A miss is about 580 cycles, most of it latency: a page of 1 KB is 515. Eight reads
  in flight cost 1.75 times one, so a batch of misses is 4.6 times cheaper a page.
- From 12 cores in one row the writer's reads (NoC 1) slow down with every read in
  flight (4,712 for 8, 4.0 times the reader); the reader's (NoC 0) barely move. Misses
  belong on the reader.

## 3. Colour

Cycles for a tile of 1,024 pixels (texel plane RGBA8, light plane float32), median over
16 runs.

| Path | Cycles a tile | A pixel |
|---|---|---|
| Integers on a CPU (0.16 fixed point; reader / writer / unpack / pack) | 44,095 / 44,160 / 44,140 / 44,054 | 43 |
| Vector unit: unpack both planes, multiply, pack (16 tiles in a row) | 1,533 | 1.5 |
| Vector unit, one tile alone | 1,586 | |
| Its copy path alone (unpack and pack, no multiply) | 258 | |

- The vector unit is 29 times faster. Its product is exact: each channel becomes a
  float without rounding (`2^23 + value - 2^23`), then one multiply-add rounds once to
  the nearest integer. The integer path truncates the light to 16 bits, so a few
  products round the other way: 3 words of 1,024 differ between the two, each equal
  to its own host reference.

## 4. Real tiles

The views are the material reference's Sponza and Bistro frames at 1024 x 1024: every
texel read of the frame (albedo, mask, normal, specular, as the SHADE stage reads them,
the mip level chosen per triangle), cut into the rasterizer's items per core and read
row by row. Sponza's overview reads 1,825,013 texels from 6,858 pages; Bistro's
interior 1,863,963 from 3,502, its exterior 2,820,613 from 3,903. The DRAM pool holds
2,048 pages of synthetic texels (key modulo 2,048): the misses are the real ones, the
texel values are not.

Sample tiles, 12 at once (one a core), cycles a read with a cold table, table of 8 /
32 slots:

| Tile | reads, pages | table | table + last page |
|---|---|---|---|
| Sponza, few pages | 1,024, 2 | 58.5 / 58.6 | 34.0 / 34.1 |
| Sponza, median / many pages | 3,072, 18 / 2,048, 470 | 137 / 68.5 ; 263 / 214 | 137 / 68.5 ; 263 / 213 |
| Bistro interior, material edge (17 materials) | 1,051, 75 | 258 / 118 | 257 / 118 |
| Bistro exterior, many pages | 2,326, 163 | 310 / 208 | 310 / 208 |

A cost model fitted to all 1,107 replayed items (samples and each view's 12 busiest
cores, whole item lists), cycles an item = a x reads + b x page changes + c x misses +
d: table, a = 64.1, b = 4.4, c = 637, d = -1,150; with the last page, a = 34.6, b =
33.9, c = 637, d = -1,218 (median error 1.0 %, largest 14 %). On the busiest cores the
model and the card agree within 1 % (Sponza, 16 slots: 2.108 ms measured, 2.099
model). The projection walks every core's items through the table on the host and
applies the model. A frame, table + last page, 4 KB pages, cold table, slowest core /
mean core, ms:

| View | 8 slots | 16 slots | 32 slots (hit rate, DRAM read) | FILL a frame |
|---|---|---|---|---|
| Sponza overview | 2.44 / 1.45 | 2.05 / 1.12 | 1.53 / 0.90 (0.970, 225 MB) | 1.59 |
| Bistro interior overview | 1.48 / 1.03 | 1.21 / 0.87 | 1.08 / 0.83 (0.989, 83 MB) | 2.10 |
| Bistro exterior overview | 4.54 / 2.82 | 2.96 / 2.05 | 2.10 / 1.51 (0.969, 355 MB) | |

- At 8 slots the misses are 60 % of Sponza's slowest core and 70 % of Bistro
  exterior's; at 32 slots they are 0.59 ms of Sponza's 1.53 and 0.71 of Bistro
  exterior's 2.10. A table hit with the replay's bookkeeping (the use count of each
  slot) costs 34 to 65 cycles a read.
- Pages of 16 x 16 texels (1 KB) are worse in every view: Sponza at 32 slots 2.53 ms,
  hit rate 0.906 (projected with the 4 KB miss cost; a 1 KB miss is only 11 % cheaper).
- The close views: Sponza 1.19 ms, Bistro interior 1.27, Bistro exterior 2.24 at 32
  slots. FILL a frame is `projects/raster`'s settled frame on 120 cores.

## What this says for the shading pass

- **Which CPU looks up**: the reader. Every CPU runs the table at the same speed, but
  only the data-movement CPUs can read a missing page, and the reader's NoC 0 holds its
  rate with 12 cores reading at once where the writer's NoC 1 falls by 4 times. The
  colour goes to the vector unit (1,533 cycles a tile against 44,100 in integers:
  about 0.01 ms a core a frame for 8.7 tiles, against 0.28).
- **Page and table**: pages of 32 x 32 texels (4 KB), a table of 32 slots in sets of
  4 (128 KB of L1), keeping the last page. Smaller pages miss more than they save;
  each doubling of the table saves 11 to 35 % of the frame.
- **A frame**: 1.53 ms on Sponza's overview (FILL 1.59), 1.08 on Bistro's interior
  (FILL 2.10), 2.10 on Bistro's exterior, misses read one at a time. Not measured:
  issuing a core's misses 8 at a time (146 cycles a page instead of 589) would cut the
  miss part about 4 times, Sponza to about 1.1 ms.

Not measured either: more than 12 cores reading at once (the projection assumes the
miss cost of 12 cores in one row holds on 120), and a replay loop unrolled like the
synthetic one (the replay pays a store a read for the table's use counts).
