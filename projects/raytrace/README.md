# Ray tracing on the card

The ray tracer of [`reference/`](reference/README.md) on the card: the same saved box
tree, the same rays, the same float32 operations in the same order, run by the 120
Tensix cores of a Blackhole p150a under the dispatch kernel. The whole frame (ray
generation, the four traces, the shading) runs on the cores; every stage is checked
against numpy lane by lane and the image against numpy's image, pixel by pixel.

## The thread program

A pass (primary, shadow, bounce or shadow at the bounce hit) is one record of the
segment `TRACE` sent to every core of the grid. Each core walks the packets its
directory entry names, 32 rays to a packet, one ray per lane of the vector unit:

```
thread trace(core, mode):                       # one per core of the grid
    first, count = directory[core]
    for page in first .. first + count - 1:     # a packet: 8 x 4 pixels of one tile
        rays = packets[page]                    # origin, direction, 1 / direction, tmax
        stack = [root]
        while stack:
            node = pop(stack)
            alive = lanes whose entry distance into node <= their best   # 32 lanes at once
            if no lane alive: continue
            if node is a leaf:
                hit = triangle test of node's triangle on the alive lanes
                best, tri, u, v = the closer hit (equal t: the lower triangle index)
                if mode == any: a lane with a hit leaves the walk
            else:
                push both children that some alive lane enters, nearer on top
                (by the packet's direction signs and the node's order word)
        hits[page] = t, tri, u, v, boxes and triangles tested, cycles   # per lane
```

`LOAD` before it copies the tree image (nodes, an order word per node, triangle
records) from DRAM into every core's L1 once. A scene whose tree does not fit in L1
runs `TREELETS` instead: each packet first walks the top tree (in L1) and lists the
treelets it reaches, nearer first; the reader then reads treelets from DRAM into a few
L1 slots and hands compute the packets that listed the treelet in the slot, and the
packet states stay in L1 between visits.

| Kernel | What it does |
|---|---|
| `reader.cpp` | LOAD: the tree image into the tree buffer. TRACE: this core's packet pages from DRAM to compute. TREELETS: also bins packets by treelet, reads treelets into slots (least recently used replaced), writes work pages |
| `compute.cpp` | the pack thread walks each packet on the vector unit (`walk.hpp`): slab test and triangle test in the reference's operation order, with an exact reciprocal (`sfpu.hpp`); the stages' arithmetic (`shade.hpp`); the unpack thread hands it the pages |
| `writer.cpp` | 64 packets' output pages of each stream to one DRAM page (the host reads per page); the profile page |

| Record word | Meaning |
|---|---|
| a0 | directory: per core, first packet page and page count |
| a1, a2 | packet pages, hit pages |
| a3 | mode: nearest hit, or any hit (shadow rays) |
| a4 | profile pages: per core, the cycles of every step and the counts |
| a5 to a9 | TREELETS: treelet images, list entries, slot bytes, slots, schedule |

The frame is eight records in one load (`frame.py`), each core keeping the packets
of its tiles from the first record to the last. The other stages are threads of the
same form: per packet, their input pages into the vector unit, the stage, their
output pages; a packet is 8 x 4 pixels, lane `l` at `(l % 8, l / 8)` from the pixel in
its header (the lane offsets are a constant page, not computed per lane).

```
GEN      desc                      -> primary packet        camera rays, 1 / direction
TRACE    primary packet            -> primary hits          t, triangle, u, v, normal, vertex colours
SHADE1   primary packet, hits      -> shadow packet, bounce packet, state 1
                                      sky or albedo, light cosine, the bounce direction (hash)
TRACE    shadow packet (any hit)   -> shadow hits
TRACE    bounce packet             -> bounce hits
SHADE2   state 1, shadow hits, bounce packet, bounce hits -> second shadow packet, state 2
TRACE    second shadow packet      -> second shadow hits
IMAGE    state 2, second shadow hits -> colour
```

A nearest-hit record carries the hit triangle's normal and vertex colours out of the
walk (they are in the triangle record in L1), so shading reads no scene data. Stage
records give the constant page (camera, light, sky, lane offsets) in a3, the profile
pages in a4, and the streams' pages in a1, a2, a5 to a9.

Every wait in the kernels is bounded by the wall clock and the run's stop word; a core
that gives up says why in its profile, and the driver counts it as a failure. `ARITH`
and `RECIP` run the walk's operations on given inputs for the arithmetic check;
`PROBE` times one node visit in several forms.

The same eight records also run as one graph (`framegraph.py`): uploaded once into
the dispatch kernel's upload region and kept, each record a node after the records
that write its inputs (the shadow and bounce traces may run together). The camera is
bound as the address of its constant page (camera, light, sky, lane offsets), the
colour as its address: a new camera is a new constant page and a `restart` with its
address, with no new upload and no host step inside the frame. The kernel's X280 decides what to issue next; host-fed, the same executor runs
on the host and issues through the host ring, and both give the same image.

`run.py` runs the traces alone, with ray generation and shading on the host (the
reference's functions): it checks every ray's hit and times each pass by itself.

## Running it

Setup is that of [01_mapcollect](../../examples/01_mapcollect/README.md#what-it-needs).
The drivers make a generated scene, its tree and the numpy reference image when they
are missing (in `KR_DATA`, see [`reference/`](reference/README.md#running)); a public
scene is fetched first with `python -m ref.public fetch`. From the repository root:

```bash
python -m projects.raytrace.frame --scene tiny --reps 16 --json frame.json
python -m projects.raytrace.frame --tiles 0 300 528 1000 --stages
python -m projects.raytrace.frame --scene sponza_close --treelets 128 --pictures DIR
python -m projects.raytrace.framegraph --cameras 2
python -m projects.raytrace.run --scene tiny --reps 5 --json tiny.json
python -m projects.raytrace.run --scene s10k --treelets 128 --tiles 0 300 528 1000
python -m projects.raytrace.arith --n 4096 --scene tiny
KMR_CARD=1 python -m pytest tests/hardware/test_raytrace_card.py
```

A tree larger than L1 runs with `--treelets KIB`: each core keeps the top of the tree,
the four trace records become TREELETS records (one per chunk of a core's packets that
fits the treelet buffers), and treelets are read from DRAM as packets need them; the
stage records are unchanged. `--pictures DIR` writes the card's image, numpy's and
their difference as PNGs, and the JSON line, as `DIR/<scene>_*`.

`frame.py` prints one JSON line: per record its median time and the cycles per packet
of loading, computing and writing, the clock, the card's idle and mean power, the image
check and `ok`; `--stages` also reads every stage's pages back and checks them (use it
on a few tiles: the whole frame's pages are 490 MB). `framegraph.py` runs the graph
for each camera in the kernel and host-fed, and checks every image against numpy's
and the two against each other. `run.py` prints a JSON line per pass (card time per
record, rate, profile sums, the check) and a last line with `ok`: every record done, every core ended, nothing given
up, no walk error, no ray differs from numpy, and no pixel of the image differs. It
exits 1 without it. `--reps` reruns each pass; the time of a pass is the record's
event time (issue to completion, the X280's timer), and records run one at a time.
`--save DIR` writes the result files in the reference's layout to `DIR/card/`, so
`reference/agree.py --external DIR` compares them.

## What is exact

Everything the card computes: the camera rays, the nearest hit (`t`, triangle, `u`,
`v`) of every primary and bounce ray, the occlusion of every shadow ray, the shading
and the bounce directions are bit for bit those of numpy, so the image is too. The vector unit has no IEEE division and its multiply-add
is partly fused, so the walk avoids both: products and sums are separate operations in
the reference's order, and the reciprocal of the triangle test is corrected to the
correctly rounded result with an exact remainder (`sfpu.hpp`). `arith.py` checks every
operation of the walk on scene rays, random rays and edge cases against numpy, and the
reciprocal over whole exponent ranges. The host refuses a scene whose triangle
determinant could leave the reciprocal's exact range. Shading divides and takes square
roots: both are correctly rounded with an exact remainder (`divsqrt.hpp`, checked by
`arith.py --math` over a million operands each), and the reference's 32-bit hash is
built from 16-bit halves on the 23-bit multiplier. The number of boxes and triangles
tested differs from the reference (packets walk together), the hits do not.

With treelets the walk visits the tree in another order than the reference's, and a
ray whose two nearest triangles lie at equal distances (rounded to float32) may keep a
different one. The image check counts such pixels apart as ties: a pixel whose colour
is the one the reference computes with a triangle within 1e-5 of the nearest
distance, for the primary or the bounce ray (`ties.py`, every triangle tested against
those rays). Ties pass the check; every other differing pixel fails it. Per scene,
one frame with treelets of 128 KiB: 10^4 and 10^5 none, 10^6 1, Sponza 192 (close)
and 221 (overview), Bistro interior 1 and 1. Bistro interior also has 1 (close) and
2 (overview) pixels that differ otherwise: a bounce ray keeps a hit 3e-5 to 1.3e-4
farther than the nearest, which it finds when traced in another packet; the check
fails there (`run.py --scene bistro_interior_overview --tiles 811 --treelets 128`).

## Limits

- Kernel text: a program's kernels must fit tt-metal's kernel configuration buffer
  on a Tensix core, 70,656 bytes with the tt-metal this tree builds on; a program
  that does not fit fails at launch with `Program size (...) too large for kernel
  config buffer (70656) on TENSIX` and a size per kernel. The ray tracer's kernels
  fit; with the division, square root, reciprocal and clamped inverse inlined at
  every use they did not, so these are calls.
- Readback: reading back is slow ([host API](../../docs/reference/host.md#session));
  large results are read in 2 MB chunks with a line per chunk, and the whole frame's
  stage pages (490 MB) are checked on tiles only.

## Measured

### The whole frame on the card

Tiny scene, 1024 x 1024 pixels, 32,768 packets on the 120 cores, tree in L1;
2026-11-13 11:35, AICLK 1350 MHz, X280 1000 MHz, host load average 3.7;
`python -m projects.raytrace.frame --scene tiny --reps 200 --json frame.json`. Median
of 200 frames in one load; image 0 of 1,048,576 pixels differing. Card power 72.3 W
mean over the 4.6 s of records (66.2 W idle before), 1.68 J per frame.

| record | card ms | cycles per packet: load, compute, write |
|---|---|---|
| GEN | 0.542 | 130, 966, 1194 |
| TRACE primary | 3.399 | 1140, 6676, 1539 |
| SHADE1 | 2.232 | 2006, 4988, 2907 |
| TRACE shadow | 2.225 | 1139, 4762, 564 |
| TRACE bounce | 8.791 | 1140, 16213, 1539 |
| SHADE2 | 1.565 | 2640, 1551, 2399 |
| TRACE second shadow | 3.882 | 1139, 7501, 564 |
| IMAGE | 0.566 | 1295, 181, 340 |
| frame | 23.20 | |

| | card | C++ 96 threads | V100 | card / V100 |
|---|---|---|---|---|
| frame ms | 23.20 | 22.22 | 2.59 | 0.11 |
| Mray/s, all rays over the frame | 108.8 | 113.54 | 972.85 | 0.11 |
| primary Mray/s (GEN + trace) | 266.1 | 154.19 | 1258.04 | 0.21 |
| bounce Mray/s (trace) | 74.9 | 95.28 | 583.14 | 0.13 |
| J per frame | 1.68 (card) | n/a | 0.51 (GPU board) | |

The C++ and V100 frame is the reference's five stages, ray generation and shading
inside them; neither counts loading or building the scene and tree, copying them to
the GPU or copying the image back ([reference](reference/README.md#timings-and-power)).
The card's frame is its eight records likewise: the tree is loaded before them and
the colour is read back after, neither counted.

Reading it against the V100: below half everywhere, room left, in three places.
- Moving pages between L1 and the vector unit: the cores copy them word by word
  through the Dst window, about 90 cycles per 32-word vector in and 115 out. In the
  stage records that is 47 to 69 % of the time (SHADE1: 4.9k of 10.4k cycles per
  packet). The unpacker and packer move tiles between L1 and Dst without the core;
  pages laid out as Dst tiles would take that cost off every record, traces included.
- The traces: as below (box tests, bounce coherence). Here the shadow and bounce
  records also walk every packet of the frame, with the lanes of missed pixels empty
  (rays stay on their core): the second shadow trace takes 3.88 ms for 158,779 rays,
  against 1.60 ms when `run.py` packs them; a later version re-bins them into full
  packets.
- Shading arithmetic: SHADE1 computes for 4,988 cycles per packet (the table); its
  bounce direction alone is twelve 32-bit hashes, two square roots and six divisions.

### The whole frame on larger scenes

Treelets of at most 128 KiB from DRAM (the tiny row: the whole tree in L1); 2026-11-13
14:27 to 14:50, card 1, AICLK 1350 MHz, X280 1000 MHz, one frame each;
`python -m projects.raytrace.frame --scene SCENE --treelets 128` (tiny: no
`--treelets`). Card ms per record; Mray/s counts the reference's rays of the four
traces over the card's frame; C++ (all threads) and V100 are the stored reference
results of the same frame.

| scene | treelets | GEN | primary | SHADE1 | shadow | bounce | SHADE2 | shadow 2 | IMAGE | frame ms | Mray/s | C++ ms | V100 ms | W | pixels differing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tiny | (L1) | 0.54 | 3.40 | 2.23 | 2.22 | 8.79 | 1.57 | 3.88 | 0.57 | 23.21 | 108.7 | 36.52 | 2.59 | 61.0 | 0 |
| s10k | 16 | 0.54 | 9.14 | 2.23 | 5.62 | 33.68 | 1.56 | 13.61 | 0.57 | 66.95 | 38.5 | 36.92 | 4.19 | 61.0 | 0 |
| s100k | 128 | 0.54 | 13.33 | 2.23 | 8.33 | 66.70 | 1.57 | 23.70 | 0.56 | 116.96 | 22.9 | 39.91 | 6.59 | 60.0 | 0 |
| s1m | 1024 | 0.54 | 23.37 | 2.23 | 15.07 | 157.31 | 1.57 | 55.67 | 0.57 | 256.32 | 11.1 | 48.04 | 11.57 | 61.0 | 1 |
| Sponza, close | 256 | 0.54 | 57.42 | 2.23 | 40.13 | 367.25 | 1.57 | 211.69 | 0.57 | 681.40 | 6.1 | 262.31 | 101.34 | 70.7 | 192 |
| Sponza, overview | 256 | 0.54 | 67.81 | 2.23 | 37.66 | 359.48 | 1.57 | 208.68 | 0.56 | 678.53 | 6.0 | 252.45 | 94.91 | 61.0 | 221 |
| Bistro interior, close | 1024 | 0.54 | 97.66 | 2.23 | 73.80 | 678.77 | 1.57 | 411.59 | 0.56 | 1266.72 | 3.3 | 395.61 | 211.44 | 73.6 | 2 |
| Bistro interior, overview | 1024 | 0.54 | 74.68 | 2.23 | 54.16 | 631.55 | 1.57 | 345.49 | 0.57 | 1110.77 | 3.7 | 344.46 | 154.04 | 70.9 | 3 |

Ties and the Bistro misses: "What is exact". The four traces take 93 % (s10k) to over 99 % (Sponza, Bistro) of the
frame, the bounce trace alone half or more: rays stay with their core and every core
reads the treelets its rays need from DRAM.

### The frame as a graph

`python -m projects.raytrace.framegraph --cameras 2` (2026-11-13 11:36, AICLK 1350
MHz, X280 1000 MHz, host load average 3.7): the whole tiny frame for two cameras (the second a step
forward), host wall time from the graph's start to its end event. The upload is 896
bytes.

| camera | in the kernel ms | host-fed ms | pixels differing from numpy (both) | kernel against host-fed |
|---|---|---|---|---|
| 0 | 23.52 | 26.08 | 0 | 0 |
| 1 | 23.62 | 25.43 | 0 | 0 |

The frame as a graph takes the time of its records (23.20 ms above) plus about 0.3
ms; host-fed adds about 2 ms of host round trips over the eight records.

### The four traces alone

Tiny scene (398 triangles), 1024 x 1024 pixels, the whole grid (120 cores), tree in
L1; 2026-11-13 09:14, AICLK 1350 MHz, host load average about 10 (shared host);
`python -m projects.raytrace.run --scene tiny --reps 5 --json tiny.json`. Median of 5
records per pass (the spread was under 2 us). Every pass exact, image 0 pixels
differing. Power was not sampled in this run.

| pass | rays | card ms | card Mray/s | C++ 96 threads Mray/s | V100 Mray/s | card / V100 |
|---|---|---|---|---|---|---|
| primary | 1,048,576 | 3.007 | 348.7 | 154.19 | 1258.04 | 0.28 |
| shadow | 658,018 | 2.111 | 311.8 | 119.45 | 1488.33 | 0.21 |
| bounce | 658,018 | 7.683 | 85.6 | 95.28 | 583.14 | 0.15 |
| shadow at bounce hits | 158,779 | 1.602 | 99.1 | - | - | - |
| four traces | 2,523,391 | 14.403 | | | | |

C++ and V100 rates are the medians of the reference's table
([reference](reference/README.md#timings-and-power)); the reference gives no separate
rate for the second shadow rays. Its whole frame, shading included, takes 22.22 ms on
96 CPU threads and 2.59 ms on the V100.

Reading it: every pass is below half of the V100's rate, so there is room left.
- Where the cycles go (primary, summed over cores): box tests 56 %, the entry test
  and lane check of a pop 13 %, loading packets 9 %, triangle tests 8 %, writing hits
  5 %. The slowest core is 1.24 times the mean: tiles are split by packet count, and
  `--balance` splits them by measured cycles instead.
- Bounce rays go in random directions: a packet's lanes leave the walk at different
  nodes, so most box tests serve few lanes. `--sort` packs them by direction octant
  and origin.
- Not counted above: the host steps (ray generation, shading) and the readback of the
  hit pages; reading back is slow ([host API](../../docs/reference/host.md#session)).
