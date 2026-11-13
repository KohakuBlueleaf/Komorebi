# Ray tracing on the card

The ray tracer of [`reference/`](reference/README.md) with the card as the intersector:
the same saved box tree, the same rays, the same float32 operations in the same order,
walked by the 120 Tensix cores of a Blackhole p150a under the dispatch kernel. Every
trace is checked against numpy ray by ray and the frame image against numpy's image,
pixel by pixel.

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
| `compute.cpp` | the pack thread walks each packet on the vector unit (`walk.hpp`): slab test and triangle test in the reference's operation order, with an exact reciprocal (`sfpu.hpp`); the unpack thread hands it the pages |
| `writer.cpp` | 64 packets' hits to one DRAM page (the host reads per page); the profile page |

| Record word | Meaning |
|---|---|
| a0 | directory: per core, first packet page and page count |
| a1, a2 | packet pages, hit pages |
| a3 | mode: nearest hit, or any hit (shadow rays) |
| a4 | profile pages: per core, the cycles of every step and the counts |
| a5 to a9 | TREELETS: treelet images, list entries, slot bytes, slots, schedule |

Every wait in the kernels is bounded by the wall clock and the run's stop word; a core
that gives up says why in its profile, and the driver counts it as a failure. `ARITH`
and `RECIP` run the walk's operations on given inputs for the arithmetic check;
`PROBE` times one node visit in several forms.

On the host, `run.py` makes the frame with the reference's functions: primary rays per
32 x 32 tile in packets of 8 x 4 pixels; from the card's hits, the shadow rays, the
bounce directions (the reference's hash) and the shading. Ray generation and shading
are host steps in this version; only the four traces run on the card.

## Running it

Setup is that of [01_mapcollect](../../examples/01_mapcollect/README.md#what-it-needs),
plus the scenes and trees of the reference (`KR_DATA`, see
[`reference/`](reference/README.md#running)). From the repository root:

```bash
python -m projects.raytrace.run --scene tiny --reps 5 --json tiny.json
python -m projects.raytrace.run --scene s10k --treelets 128 --tiles 0 300 528 1000
python -m projects.raytrace.arith --n 4096 --scene tiny
KMR_CARD=1 python -m pytest tests/hardware/test_raytrace_card.py
```

`run.py` prints a JSON line per pass (card time per record, rate, profile sums, the
check) and a last line with `ok`: every record done, every core ended, nothing given
up, no walk error, no ray differs from numpy, and no pixel of the image differs. It
exits 1 without it. `--reps` reruns each pass; the time of a pass is the record's
event time (issue to completion, the X280's timer), and records run one at a time.
`--save DIR` writes the result files in the reference's layout to `DIR/card/`, so
`reference/agree.py --external DIR` compares them.

## What is exact

Everything the card computes: the nearest hit (`t`, triangle, `u`, `v`) of every
primary and bounce ray and the occlusion of every shadow ray are bit for bit those of
numpy, so the image is too. The vector unit has no IEEE division and its multiply-add
is partly fused, so the walk avoids both: products and sums are separate operations in
the reference's order, and the reciprocal of the triangle test is corrected to the
correctly rounded result with an exact remainder (`sfpu.hpp`). `arith.py` checks every
operation of the walk on scene rays, random rays and edge cases against numpy, and the
reciprocal over whole exponent ranges. The host refuses a scene whose triangle
determinant could leave the reciprocal's exact range. The number of boxes and
triangles tested differs from the reference (packets walk together), the hits do not.

## Measured

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
  hit pages, about 0.7 MB/s through the host package (45 s for the frame's four
  passes).
