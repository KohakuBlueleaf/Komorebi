# Ray tracing on the card

The ray tracer of [`reference/`](reference/README.md) on the card: the same saved box
tree, the same rays, the same float32 operations in the same order, run by the 120
Tensix cores of a Blackhole p150a under the dispatcher. The whole frame (ray
generation, the four traces, the shading) runs on the cores; every stage is checked
against numpy lane by lane and the image against numpy's image, pixel by pixel.

## The thread program

A pass (primary, shadow, bounce or shadow at the bounce hit) is one command of the
handler `TRACE` sent to every core of the grid. Each core walks the packets its
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
L1 slots and hands compute the packets that listed the treelet in the slot. Packet
states stay in L1 between visits as Dst tile images: at each visit the unpacker loads
one into Dst and the packer stores it back, about 270 cycles where copying it word by
word took 3,400.

| Kernel | What it does |
|---|---|
| `reader.cpp` | LOAD: the tree image into the tree buffer. TRACE: this core's packet pages from DRAM to compute. TREELETS: also bins packets by treelet, reads treelets into slots (least recently used replaced), writes work pages. SORT and UNSORT: this core's rays into a new order and their hits back (below) |
| `compute.cpp` | the pack thread walks each packet on the vector unit (`walk.hpp`): slab test and triangle test in the reference's operation order, with an exact reciprocal (`sfpu.hpp`); the stages' arithmetic (`shade.hpp`); the unpack thread hands it the pages and, at treelet visits, loads each packet's state into Dst with the math thread; the packer stores it back |
| `writer.cpp` | 64 packets' output pages of each stream to one DRAM page (the host reads per page); the profile page |

| Command word | Meaning |
|---|---|
| a0 | directory: per core, first packet page and page count |
| a1, a2 | packet pages, hit pages |
| a3 | mode: nearest hit, or any hit (shadow rays) |
| a4 | profile pages: per core, the cycles of every step and the counts |
| a5 to a9 | TREELETS: treelet images, list entries, slot bytes, slots, schedule |
| a1 to a9 | SORT, UNSORT: stream in, stream out, order pages, profile, vectors to put back (0: SORT), the key's low corner and scale |

The frame is eight commands in one load (`frame.py`), each core keeping the packets
of its tiles from the first command to the last. The other stages are threads of the
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
commands give the constant page (camera, light, sky, lane offsets) in a3, the profile
pages in a4, and the streams' pages in a1, a2, a5 to a9.

Every wait in the kernels has a wall-clock timeout and the run's stop flag; a core
that gives up says why in its profile, and the driver counts it as a failure. `ARITH`
and `RECIP` run the walk's operations on given inputs for the arithmetic check;
`PROBE` times one node visit in several forms.

The same eight commands also run as one graph (`framegraph.py`): uploaded once into
the dispatcher's upload region and kept, each command a node after the commands
that write its inputs (the shadow and bounce traces may run together). The camera is
bound as the address of its constant page (camera, light, sky, lane offsets), the
colour as its address: a new camera is a new constant page and a `restart` with its
address, with no new upload and no host step inside the frame. The kernel's X280 decides what to submit next; host-fed, the same executor runs
on the host and submits through the host ring, and both give the same image.

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
the four trace commands become TREELETS commands (one per chunk of a core's packets that
fits the treelet buffers), and treelets are read from DRAM as packets need them; the
stage commands are unchanged. `--pictures DIR` writes the card's image, numpy's and
their difference as PNGs, and the JSON line, as `DIR/<scene>_*`.

`--sort` (with `--treelets`) puts a SORT command before the bounce and second-shadow
traces and an UNSORT command after each. SORT runs on each core's data-movement core:
it reads one vector of all the core's packets at a time into L1 (the slot buffer,
free between treelet commands), keys each ray by its direction's octant and the Morton
cell (2^9 per axis over the root box) of its origin, orders the keys (radix sort,
three passes of 11 bits, rays without a ray last) and writes the packets in that
order with their headers' direction signs, and the order. UNSORT writes the hits back
to their pixels' lanes. Rays never leave their core; the stages are unchanged.

`frame.py` prints one JSON line: per command its median time and the cycles per packet
of loading, computing and writing, the clock, the card's idle and mean power, the image
check and `ok`; `--stages` also reads every stage's pages back and checks them (use it
on a few tiles: the whole frame's pages are 490 MB). `framegraph.py` runs the graph
for each camera in the kernel and host-fed, and checks every image against numpy's
and the two against each other. `run.py` prints a JSON line per pass (card time per
command, rate, profile sums, the check) and a last line with `ok`: every command done, every core ended, nothing given
up, no walk error, no ray differs from numpy, and no pixel of the image differs. It
exits 1 without it. `--reps` reruns each pass; the time of a pass is the command's
event time (submit to completion, the X280's timer), and commands run one at a time.
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
reciprocal over whole exponent ranges. The host rejects a scene whose triangle
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
one frame with treelets of 128 KiB and sorted rays: 10^4 and 10^5 none, 10^6 1, Sponza
244 (close) and 234 (overview), Bistro interior 1 and 1. Bistro interior also has 1
(close) and 1 (overview) pixels that differ otherwise (2 before sorting): a bounce ray keeps a hit 3e-5 to 1.3e-4
farther than the nearest, which it finds when traced in another packet; the check
fails there (`run.py --scene bistro_interior_overview --tiles 811 --treelets 128`).

## Limits

- Kernel text: a program's kernels must fit the session's kernel buffer on a Tensix
  core, 70,656 bytes unless the session sets `kernel_bytes`; a program that does not
  fit is rejected before launch with its sizes per thread (`km.KernelSizeError`,
  [host API](../../docs/reference/host.md#program)). Two remedies: a larger buffer
  (`Session(kernel_bytes=...)`, taken from the L1 left for circular buffers) or a
  smaller build of one kernel (`km.Program(..., opt={...})`). The ray tracer's kernels
  take 73,728 bytes since SORT, and `card.py` opens its session with 80 KiB; with the
  division, square root, reciprocal and clamped inverse inlined at every use they
  took more, so these are calls.
- Readback: reading back is slow ([host API](../../docs/reference/host.md#session));
  large results are read in 2 MB chunks with a line per chunk, and the whole frame's
  stage pages (490 MB) are checked on tiles only.

## Measured

### The whole frame on the card

Tiny scene, 1024 x 1024 pixels, 32,768 packets on the 120 cores, tree in L1;
2026-11-13 11:35, AICLK 1350 MHz, X280 1000 MHz, host load average 3.7;
`python -m projects.raytrace.frame --scene tiny --reps 200 --json frame.json`. Median
of 200 frames in one load; image 0 of 1,048,576 pixels differing. Card power 72.3 W
mean over the 4.6 s of commands (66.2 W idle before), 1.68 J per frame.

| command | card ms | cycles per packet: load, compute, write |
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
| frame ms | 23.20 | 20.59 | 2.59 | 0.11 |
| Mray/s, all rays over the frame | 108.8 | 122.57 | 972.85 | 0.11 |
| primary Mray/s (GEN + trace) | 266.1 | 166.90 | 1251.88 | 0.21 |
| bounce Mray/s (trace) | 74.9 | 104.47 | 582.57 | 0.13 |
| J per frame | 1.68 (card) | n/a | 0.50 (GPU board) | |

C++ and V100: the benchmark run of the next table ([`results/reference-bench.json`](results/reference-bench.json)).
The C++ and V100 frame is the reference's five stages, ray generation and shading
inside them; neither counts loading or building the scene and tree, copying them to
the GPU or copying the image back ([reference](reference/README.md#timings-and-power)).
The card's frame is its eight commands likewise: the tree is loaded before them and
the colour is read back after, neither counted.

Reading it against the V100: below half everywhere, room left, in three places.
- Moving pages between L1 and the vector unit: the cores copy them word by word
  through the Dst window, about 90 cycles per 32-word vector in and 115 out. In the
  stage commands that is 47 to 69 % of the time (SHADE1: 4.9k of 10.4k cycles per
  packet). The unpacker and packer move tiles between L1 and Dst without the core;
  pages laid out as Dst tiles would take that cost off every command, traces included.
- The traces: as below (box tests, bounce coherence). Here the shadow and bounce
  commands also walk every packet of the frame, with the lanes of missed pixels empty
  (rays stay on their core): the second shadow trace takes 3.88 ms for 158,779 rays,
  against 1.60 ms when `run.py` packs them; a later version re-bins them into full
  packets.
- Shading arithmetic: SHADE1 computes for 4,988 cycles per packet (the table); its
  bounce direction alone is twelve 32-bit hashes, two square roots and six divisions.

### The whole frame on larger scenes

Treelets of at most 128 KiB from DRAM with each core's bounce and second-shadow rays
sorted (the tiny row: the whole tree in L1, not sorted, the run above); 2026-11-13 16:20 to 16:28,
card 1, AICLK 1350 MHz, X280 1000 MHz, one frame each;
`python -m projects.raytrace.frame --scene SCENE --treelets 128 --sort --json
frame.json` (tiny: neither flag), the JSON of every row in [`results/`](results/).
Card ms per command; Mray/s counts the reference's rays of the four traces over the
card's frame. C++ (96 threads of the host's 48-core EPYC 7K62) and V100 are the
reference's five stages, median of 3 processes x 3 frames, from one run of `python -m
projects.raytrace.reference.bench` on 2026-11-13 14:57 to 15:20
([`results/reference-bench.json`](results/reference-bench.json)), the source of every
C++ and V100 number in this README.

| scene | treelets | GEN | primary | SHADE1 | shadow | SORT bounce | bounce | UNSORT bounce | SHADE2 | SORT shadow 2 | shadow 2 | UNSORT shadow 2 | IMAGE | frame ms | Mray/s | C++ ms | V100 ms | V100 / card | W | ties | pixels differing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tiny | (L1) | 0.54 | 3.40 | 2.23 | 2.23 | - | 8.79 | - | 1.57 | - | 3.88 | - | 0.57 | 23.20 | 108.7 | 20.59 | 2.59 | 0.112 | 63.0 | 0 | 0 |
| 10^4 | 16 | 0.55 | 8.19 | 2.22 | 4.83 | 4.91 | 20.99 | 3.39 | 1.57 | 4.85 | 4.57 | 1.27 | 0.57 | 57.91 | 44.5 | 26.44 | 4.16 | 0.072 | 63.0 | 0 | 0 |
| 10^5 | 128 | 0.55 | 11.57 | 2.22 | 7.34 | 4.94 | 34.48 | 3.45 | 1.57 | 4.85 | 8.72 | 1.26 | 0.56 | 81.51 | 32.9 | 35.47 | 6.57 | 0.081 | 63.0 | 0 | 0 |
| 10^6 | 1024 | 0.55 | 17.42 | 2.23 | 11.44 | 4.95 | 83.44 | 3.34 | 1.57 | 4.85 | 22.58 | 1.28 | 0.57 | 154.21 | 18.5 | 53.66 | 11.61 | 0.075 | 63.0 | 1 | 0 |
| Sponza, close | 256 | 0.55 | 42.07 | 2.22 | 27.07 | 4.98 | 213.34 | 3.40 | 1.57 | 5.01 | 57.82 | 1.26 | 0.57 | 359.86 | 11.5 | 264.78 | 102.00 | 0.283 | 63.0 | 244 | 0 |
| Sponza, overview | 256 | 0.55 | 50.10 | 2.23 | 27.86 | 4.96 | 194.64 | 3.48 | 1.57 | 4.96 | 56.38 | 1.25 | 0.57 | 348.53 | 11.8 | 230.36 | 95.52 | 0.274 | 69.2 | 234 | 0 |
| Bistro interior, close | 1024 | 0.55 | 66.01 | 2.23 | 52.68 | 4.95 | 347.66 | 3.39 | 1.57 | 4.95 | 111.15 | 1.27 | 0.57 | 596.97 | 7.0 | 401.87 | 211.42 | 0.354 | 61.0 | 1 | 1 |
| Bistro interior, overview | 1024 | 0.55 | 56.76 | 2.23 | 41.36 | 4.95 | 307.96 | 3.30 | 1.57 | 4.95 | 98.34 | 1.19 | 0.56 | 523.72 | 7.8 | 336.80 | 153.94 | 0.294 | 73.1 | 1 | 1 |

Ties and the Bistro misses: "What is exact". Sorting took the bounce trace from 359 to
251 ms and the second shadow from 209 to 75 ms on Sponza's overview (Bistro's
overview: 632 to 366 and 346 to 118), for 5 ms per SORT and 1 to 3.4 ms per UNSORT on
every scene: a packet of sorted rays shares more of the tree (live lanes per inner
node visit of the bounce trace, Sponza overview: 4.2 of 32 before, 6.0 after). Moving
the packet states through the unpacker and packer took Sponza's overview from 451 to
349 ms and Bistro's from 633 to 524. What is left in the traces (Sponza overview, mean
core): box tests 61 %, stack pops 13 %, triangle tests 11 %, state moves 3 %; and the
slowest core of each command against the mean: 71 of the 349 ms (Bistro: 147 of 524). The
host submits each command as one 64-byte line and waits for one event each; nothing is
read back inside the frame. The X280 work cluster is not used here; the next section
runs the frame from it.

### The frame planned on the X280

`python -m projects.raytrace.frame --scene SCENE --treelets 128 --sort --plan frame
--reps 3 --json plan.json` (2026-11-13 17:31 to 17:38, card 1, AICLK 1350 MHz, X280 1000
MHz; JSON: `results/plan-*.json`). A thread on X280 work cluster 1 (`thread/plan.c`,
`planner.py`) submits the commands of three frames and waits for each command's event.
The host only loads the program and reads back the image and the logs. In every chunk
command of a trace, each core writes a cost block to DRAM: its packets' busy cycles and
hit lanes. Before each trace, the four X280 cores of the cluster read the blocks and cut
the tile order into ranges of equal predicted cost, at most one command's packets (82 to
135) per range. They then write each core's directory entry for the trace's chunk
commands. No ray moves in DRAM; only the cores' ranges change.

The cold frame (the first, without history): the shadow trace is cut by the primary
trace's costs, the other traces as the host cuts them. A warm frame (a later one of the
same camera) cuts each trace by the same trace's costs in the frame before. Commands ms
is the sum of the commands. Card frame ms is the thread's clock over the frame's
commands and plans (the cold frame: less the scene load).

| scene | frame | primary | shadow | bounce | shadow 2 | commands per trace | commands ms | X280 plan ms | card frame ms |
|---|---|---|---|---|---|---|---|---|---|
| 10^4 | cold | 8.28 | 7.01 | 21.08 | 4.66 | 3 / 8 / 3 / 3 | 60.25 | 2.16 | 62.46 |
| 10^4 | warm | 12.11 | 5.99 | 21.47 | 6.43 | 7 / 8 / 8 / 8 | 65.28 | 5.27 | 70.63 |
| 10^5 | cold | 11.65 | 8.04 | 34.58 | 8.81 | 3 / 8 / 3 / 3 | 82.30 | 2.17 | 84.53 |
| 10^5 | warm | 14.01 | 6.70 | 36.46 | 11.11 | 8 / 8 / 8 / 8 | 87.58 | 5.34 | 93.01 |
| 10^6 | cold | 17.51 | 12.63 | 83.58 | 22.63 | 4 / 8 / 4 / 4 | 155.66 | 2.53 | 158.26 |
| 10^6 | warm | 15.93 | 9.91 | 109.75 | 32.18 | 8 / 8 / 8 / 8 | 187.09 | 5.32 | 192.50 |
| Sponza, close | cold | 42.18 | 30.77 | 213.49 | 57.89 | 4 / 6 / 4 / 4 | 363.93 | 2.31 | 366.30 |
| Sponza, close | warm | 37.34 | 22.24 | 189.56 | 51.00 | 6 / 5 / 5 / 5 | 319.68 | 3.64 | 323.38 |
| Sponza, overview | cold | 50.19 | 26.18 | 194.80 | 56.45 | 4 / 6 / 4 / 4 | 347.06 | 2.30 | 349.42 |
| Sponza, overview | warm | 39.36 | 20.20 | 179.83 | 50.59 | 6 / 7 / 6 / 5 | 309.43 | 4.14 | 313.63 |
| Bistro interior, close | cold | 66.12 | 54.73 | 347.98 | 111.09 | 4 / 7 / 4 / 4 | 599.33 | 2.38 | 601.77 |
| Bistro interior, close | warm | 60.61 | 44.55 | 345.49 | 122.36 | 7 / 7 / 5 / 8 | 592.44 | 4.35 | 596.86 |
| Bistro interior, overview | cold | 56.86 | 39.44 | 307.98 | 98.45 | 4 / 7 / 4 / 4 | 522.23 | 2.37 | 524.66 |
| Bistro interior, overview | warm | 44.25 | 29.14 | 278.87 | 100.98 | 7 / 8 / 7 / 7 | 472.83 | 4.57 | 477.48 |

The image of the last frame is checked: every scene as in the table above. Bistro
interior's close view has one more bounce miss (pixel 484,724): a miss of the kind in
"What is exact" depends on which packets share a range. The plan of a trace takes 0.2 to
1.4 ms on the four X280 cores and is not hidden: the Tensix cores wait for it.

What the plan gains is the time lost to each command's slowest core. Per trace, that is
the sum over its commands of the slowest core's busy time less the mean core's, from the
last frame's profile (`--profile`). Commands = the trace's commands in that frame.
Sponza and Bistro interior overviews: four ways to cut, three frames each
(`results/predictor-*.json`, `results/predictor-analysis.json`):

| cut | Sponza frame ms | lost ms | commands | Bistro frame ms | lost ms | commands |
|---|---|---|---|---|---|---|
| as the host (`--plan host`) | 348.9 | 70.5 | 4 / 4 / 4 / 4 | 524.6 | 147.2 | 4 / 4 / 4 / 4 |
| counts: equal hits of the trace before, one frame (`counts`) | 365.5 | 93.6 | 4 / 5 / 4 / 4 | 585.1 | 215.1 | 4 / 8 / 4 / 4 |
| the trace before's cycles (`trace`) | 376.4 | 104.1 | 4 / 6 / 4 / 4 | 562.5 | 201.7 | 4 / 7 / 5 / 5 |
| the frame before's cycles (`frame`), warm frame | 310.2 | 39.8 | 6 / 7 / 6 / 5 | 471.4 | 84.4 | 7 / 8 / 7 / 7 |

The counts and the trace before cut the primary trace into equal packets. The bounce and
the second shadow are cut by the trace before's mean per owner core. Both predict worse
than the host's even ranges: the primary trace alone loses 24.4 against 18.1 ms on
Sponza. A core's cost is set by the few heavy packets in its range, and the trace before
does not know which packets of the next trace are heavy. The frame before does. It
leaves about 40 of Sponza's 70.5 ms and 84 of Bistro's 147, most of it in the bounce
trace (24 and 49 ms). There, a range holds at most one command's packets: the slowest
core of each command still carries whole heavy packets.

The last command of each trace is not the cause. Cutting each core's packets into
commands of equal size instead of full commands then the rest left Sponza at 348.5 ms
and Bistro at 526.9 (523.7 before). The last command loses 8 of the 70.5 ms on Sponza
and 33 of the 147 on Bistro.

A core that pulls ranges of g packets from a shared counter instead (estimate, not
built): greedy pulls in tile order over 120 cores, with the cost blocks' per-packet
cycles of the host-cut frame and 1 us per pull. The table gives ms lost to the slowest
core per trace, summed over the four traces. Pulls per trace per core = 32,768 packets /
(120 g).

| g, packets per pull | pulls per trace per core | Sponza lost ms | Bistro lost ms |
|---|---|---|---|
| 1 | 273 | 1.9 | 2.9 |
| 4 | 68 | 3.3 | 6.4 |
| 8 | 34 | 6.5 | 12.3 |
| 16 | 17 | 12.0 | 24.4 |
| 89 (a full command) | 3 | 59.0 | 83.9 |

Per-packet cycles cover 62 to 83 % of a core's busy time (10^4), so these are lower
bounds. A pull of 4 packets (68 pulls per trace per core) leaves 3 to 6 ms of the 70.5
and 147.

### The frame as a graph

`python -m projects.raytrace.framegraph --cameras 2` (2026-11-13 11:36, AICLK 1350
MHz, X280 1000 MHz, host load average 3.7): the whole tiny frame for two cameras (the second a step
forward), host wall time from the graph's start to its end event. The upload is 896
bytes.

| camera | in the kernel ms | host-fed ms | pixels differing from numpy (both) | kernel against host-fed |
|---|---|---|---|---|
| 0 | 23.52 | 26.08 | 0 | 0 |
| 1 | 23.62 | 25.43 | 0 | 0 |

The frame as a graph takes the time of its commands (23.20 ms above) plus about 0.3
ms; host-fed adds about 2 ms of host round trips over the eight commands.

### The four traces alone

Tiny scene (398 triangles), 1024 x 1024 pixels, the whole grid (120 cores), tree in
L1; 2026-11-13 09:14, AICLK 1350 MHz, host load average about 10 (shared host);
`python -m projects.raytrace.run --scene tiny --reps 5 --json tiny.json`. Median of 5
commands per pass (the spread was under 2 us). Every pass exact, image 0 pixels
differing. Power was not sampled in this run.

| pass | rays | card ms | card Mray/s | C++ 96 threads Mray/s | V100 Mray/s | card / V100 |
|---|---|---|---|---|---|---|
| primary | 1,048,576 | 3.007 | 348.7 | 166.90 | 1251.88 | 0.28 |
| shadow | 658,018 | 2.111 | 311.8 | 125.57 | 1482.66 | 0.21 |
| bounce | 658,018 | 7.683 | 85.6 | 104.47 | 582.57 | 0.15 |
| shadow at bounce hits | 158,779 | 1.602 | 99.1 | - | - | - |
| four traces | 2,523,391 | 14.403 | | | | |

C++ and V100 rates are the medians of the benchmark run above
([`results/reference-bench.json`](results/reference-bench.json)); the reference gives
no separate rate for the second shadow rays. Its whole frame, shading included, takes
20.59 ms on 96 CPU threads and 2.59 ms on the V100.

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
