# ray_wide

Would the bounce and second-shadow traces be faster if one vector operation tested one
ray (or a few) against the child boxes of a wide node, instead of 32 rays against one
box? Today (`projects/raytrace`, main) a packet of 32 rays walks a binary tree node
first: two box tests of 32 lanes for every inner node. On Sponza's bounce rays only 6.0
of the 32 lanes are alive at an inner node visit, and grouping rays by treelet
([ray_moves](../ray_moves/README.md)) raised that only to 7.0: bounce rays diverge.
One ray against many boxes fills the lanes whatever the rays do. This experiment costs
that design: the counts on the host (part A), the cycles of one node visit on a card
(part B), and the predicted traces and frame (part C). It builds no tracer.

```
python experiments/ray_wide/wide.py --data DATA --cache CACHE --json wide.json      # A
python experiments/ray_wide/wide.py --tables wide.json --today results/today.json
python experiments/ray_wide/probe.py --data DATA --cache CACHE --cores 120 \
    --json probe.json                                                               # B
python experiments/ray_wide/predict.py results/wide.json results/today.json \
    results/probe-120-cores.json experiments/ray_moves/results/paths.json          # C
```

`DATA` is the folder of the repository's scene downloader; `CACHE` holds the rays of
the four traces as [ray_moves](../ray_moves/README.md)' `paths.py` makes them (made
there when missing). The results of the runs below are in `results/`; `results/today.json` holds the
counters and cycles of today's frame profile on main (Sponza and Bistro overviews and
the 10^6 scene, 128 KiB treelets, sorted bounce rays) and the V100's rates of the
reference benchmark.

## Answer

**No layout beats today's traces.** The best, one ray against 32 children with the
child boxes brought into Dst by the unpacker, predicts Sponza's bounce trace at 264 ms
against 194.6 today, its second shadow at 124 ms against 56.4, and the frame at 471 ms
against 348.5 (V100: 95.5). Bistro: 366 ms against 308.0, 212 against 98.3, frame 680
against 523.7. The term that sinks it is the node visit itself: 729 cycles for one ray
at its best (218 to load the boxes, 211 for the slab test with the ray's words, 98 to
bring the distances to the core, 202 for the core's stack step), against today's 400
cycles for a two-box step shared by about 6 live rays. Even with the boxes in the lanes
for free, Sponza's bounce would be 196 ms: no gain. Bistro's bounce would then be 279
ms against 308, but its second shadow 156 against 98 and its frame 539 against 524.

## Part A: counts (host)

`wide.py` collapses the reference's binary tree into wide nodes of up to W children
(W = 8, 16, 32): from each wide node's root, the child with the largest box surface is
replaced by its two children until there are W, never across a treelet's edge (the
treelets of 128 KiB as the frame cuts them; the top tree is its own region). Leaves and
triangles are the reference's (one triangle a leaf), so every hit is the same hit.
Every ray of the four traces then walks the wide tree alone: at a node it tests every
child box; leaves it enters are tested there, in one batch; inner children it enters
are pushed, nearest on top ("stored order": in the order the node holds them). Each
trace's hits are checked against `reference.raytrace` on 4096 rays: none differs, in
any scene, trace, width or order.

Per ray, means over every ray of the trace:

- **pops**: stack entries taken; **node visits**: those within the ray's best;
- **boxes tested**: children of the nodes visited; **children entered**: boxes hit
  within the best; **pushes**: inner children pushed;
- **leaf visits**: node visits with at least one leaf entered; **triangle tests**;
- **operations a ray**: vector operations of 32 lanes for box tests and triangle tests
  in each layout: 1 ray x 32 children, 2 rays x 16 or 4 rays x 8 (the lanes are 4 rows
  of 8; rays sharing an operation are each at their own node). A node with more
  children than a ray's lanes takes several operations; an operation of 2 or 4 rays
  counts a half or a quarter;
- **lanes full**: boxes tested over the lanes of the operations that tested them;
- **today (2)**: today's counters on the card over its packets' lanes (a packet's counts
  over 32): pops, inner nodes (two box tests of 32 lanes each), leaves; its "operations"
  are box tests plus leaves.

### Sponza overview

| W | trace | pops | node visits | boxes tested | children entered | pushes | leaf visits | triangle tests | operations a ray, 1 x 32 / 2 x 16 / 4 x 8 (boxes + triangles) | lanes full, 1 x 32 / 2 x 16 / 4 x 8 | stored order: visits, triangles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| today (2) | primary | 15.91 | 7.60 | 15.20 | | | | 0.46 | 15.66 | | |
| today (2) | shadow | 9.47 | 4.52 | 9.03 | | | | 0.21 | 9.24 | | |
| today (2) | bounce | 85.63 | 41.83 | 83.65 | | | | 4.30 | 87.95 | | |
| today (2) | shadow2 | 25.87 | 12.56 | 25.11 | | | | 1.19 | 26.30 | | |
| 8 | primary | 60.96 | 56.72 | 401.46 | 67.37 | 59.96 | 5.15 | 7.37 | 61.9 / 30.9 / 15.5 | 0.22 / 0.44 / 0.89 | 59.71, 8.03 |
| 8 | shadow | 32.95 | 32.95 | 232.27 | 37.39 | 35.32 | 1.41 | 1.99 | 34.4 / 17.2 / 8.6 | 0.22 / 0.44 / 0.88 | 33.15, 1.97 |
| 8 | bounce | 64.35 | 61.35 | 439.90 | 70.61 | 63.35 | 5.30 | 7.22 | 66.7 / 33.3 / 16.7 | 0.22 / 0.45 / 0.90 | 67.12, 8.89 |
| 8 | shadow2 | 35.68 | 35.68 | 253.00 | 40.89 | 38.60 | 1.54 | 2.20 | 37.2 / 18.6 / 9.3 | 0.22 / 0.44 / 0.89 | 34.99, 2.17 |
| 16 | primary | 39.73 | 34.54 | 461.16 | 44.05 | 38.73 | 3.75 | 5.25 | 38.3 / 19.1 / 15.9 | 0.42 / 0.83 / 0.96 | 41.78, 8.13 |
| 16 | shadow | 25.10 | 25.10 | 322.35 | 30.15 | 28.00 | 1.31 | 2.04 | 26.4 / 13.2 / 11.0 | 0.40 / 0.80 / 0.95 | 25.32, 2.06 |
| 16 | bounce | 46.09 | 42.52 | 568.05 | 51.91 | 45.09 | 4.83 | 6.72 | 47.4 / 23.7 / 19.7 | 0.42 / 0.83 / 0.96 | 46.97, 8.77 |
| 16 | shadow2 | 26.94 | 26.94 | 349.71 | 32.97 | 30.52 | 1.45 | 2.29 | 28.4 / 14.2 / 11.8 | 0.41 / 0.81 / 0.95 | 26.75, 2.23 |
| 32 | primary | 37.17 | 30.34 | 676.18 | 42.07 | 36.17 | 3.73 | 5.74 | 34.1 / 25.7 / 22.7 | 0.70 / 0.89 / 0.97 | 39.50, 9.84 |
| 32 | shadow | 19.88 | 19.88 | 440.56 | 25.49 | 23.17 | 1.23 | 2.15 | 21.1 / 16.1 / 14.6 | 0.69 / 0.89 / 0.97 | 21.58, 2.16 |
| 32 | bounce | 40.33 | 35.75 | 814.71 | 46.10 | 39.33 | 4.33 | 6.58 | 40.1 / 30.6 / 27.3 | 0.71 / 0.90 / 0.97 | 41.90, 9.58 |
| 32 | shadow2 | 21.19 | 21.19 | 479.08 | 27.95 | 25.39 | 1.32 | 2.31 | 22.5 / 17.4 / 15.8 | 0.71 / 0.90 / 0.97 | 22.95, 2.38 |

| W | wide nodes | children a node | node MiB, wide / today | treelet images MiB, wide / today | largest treelet KiB, wide / today |
| --- | --- | --- | --- | --- | --- |
| 8 | 79,358 | 4.30 | 17.0 / 18.0 | 39.5 / 40.5 | 196 / 164 |
| 16 | 55,682 | 5.71 | 23.8 / 18.0 | 46.2 / 40.5 | 232 / 164 |
| 32 | 47,080 | 6.57 | 40.2 / 18.0 | 62.7 / 40.5 | 356 / 164 |

### Bistro interior overview

| W | trace | pops | node visits | boxes tested | children entered | pushes | leaf visits | triangle tests | operations a ray, 1 x 32 / 2 x 16 / 4 x 8 (boxes + triangles) | lanes full, 1 x 32 / 2 x 16 / 4 x 8 | stored order: visits, triangles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| today (2) | primary | 17.99 | 8.67 | 17.33 | | | | 1.01 | 18.34 | | |
| today (2) | shadow | 13.14 | 6.31 | 12.63 | | | | 0.66 | 13.29 | | |
| today (2) | bounce | 97.57 | 47.45 | 94.90 | | | | 11.26 | 106.16 | | |
| today (2) | shadow2 | 33.62 | 16.32 | 32.65 | | | | 2.95 | 35.59 | | |
| 8 | primary | 58.63 | 57.67 | 414.61 | 81.02 | 57.63 | 11.20 | 23.27 | 68.9 / 34.4 / 17.2 | 0.23 / 0.45 / 0.90 | 74.94, 34.34 |
| 8 | shadow | 38.54 | 38.54 | 277.87 | 54.94 | 40.85 | 5.26 | 14.05 | 43.8 / 21.9 / 10.9 | 0.23 / 0.45 / 0.90 | 48.85, 16.30 |
| 8 | bounce | 67.49 | 65.78 | 447.07 | 117.58 | 66.49 | 18.79 | 50.93 | 84.6 / 42.3 / 21.1 | 0.21 / 0.42 / 0.85 | 73.07, 56.94 |
| 8 | shadow2 | 49.42 | 49.42 | 350.95 | 75.39 | 51.35 | 8.69 | 23.98 | 58.1 / 29.1 / 14.5 | 0.22 / 0.44 / 0.89 | 54.73, 25.10 |
| 16 | primary | 46.64 | 44.95 | 567.61 | 71.58 | 45.64 | 12.00 | 25.62 | 57.0 / 28.5 / 22.2 | 0.40 / 0.79 / 0.92 | 54.89, 32.70 |
| 16 | shadow | 29.02 | 29.02 | 389.25 | 47.27 | 32.28 | 5.39 | 14.30 | 34.4 / 17.2 / 14.2 | 0.42 / 0.84 / 0.94 | 36.63, 15.92 |
| 16 | bounce | 49.88 | 47.62 | 567.07 | 100.75 | 48.88 | 18.13 | 51.68 | 65.7 / 32.9 / 24.4 | 0.37 / 0.74 / 0.90 | 52.74, 56.31 |
| 16 | shadow2 | 37.14 | 37.14 | 479.24 | 64.34 | 39.60 | 8.62 | 24.52 | 45.8 / 22.9 / 18.3 | 0.40 / 0.81 / 0.93 | 39.93, 24.36 |
| 32 | primary | 34.36 | 32.62 | 762.57 | 59.57 | 33.36 | 7.68 | 25.92 | 40.3 / 31.1 / 27.0 | 0.73 / 0.88 / 0.96 | 41.73, 33.41 |
| 32 | shadow | 23.08 | 23.08 | 556.68 | 41.05 | 25.58 | 3.07 | 14.47 | 26.2 / 21.0 / 19.0 | 0.75 / 0.90 / 0.97 | 26.27, 15.69 |
| 32 | bounce | 36.80 | 34.62 | 777.65 | 89.67 | 35.80 | 11.20 | 53.59 | 45.8 / 34.3 / 29.1 | 0.70 / 0.86 / 0.95 | 38.58, 56.31 |
| 32 | shadow2 | 28.66 | 28.66 | 669.97 | 56.31 | 31.21 | 5.08 | 24.80 | 33.7 / 26.5 / 23.4 | 0.73 / 0.88 / 0.96 | 30.58, 25.12 |

| W | wide nodes | children a node | node MiB, wide / today | treelet images MiB, wide / today | largest treelet KiB, wide / today |
| --- | --- | --- | --- | --- | --- |
| 8 | 301,460 | 4.39 | 64.4 / 70.1 | 152.0 / 156.0 | 168 / 156 |
| 16 | 212,476 | 5.80 | 90.8 / 70.1 | 178.4 / 156.0 | 220 / 156 |
| 32 | 171,439 | 6.95 | 146.5 / 70.1 | 234.0 / 156.0 | 320 / 156 |

### The 10^6 scene

| W | trace | pops | node visits | boxes tested | children entered | pushes | leaf visits | triangle tests | operations a ray, 1 x 32 / 2 x 16 / 4 x 8 (boxes + triangles) | lanes full, 1 x 32 / 2 x 16 / 4 x 8 | stored order: visits, triangles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| today (2) | primary | 4.10 | 1.94 | 3.89 | | | | 0.21 | 4.10 | | |
| today (2) | shadow | 2.43 | 1.16 | 2.33 | | | | 0.20 | 2.53 | | |
| today (2) | bounce | 19.82 | 9.57 | 19.13 | | | | 1.29 | 20.42 | | |
| today (2) | shadow2 | 4.99 | 2.43 | 4.86 | | | | 0.45 | 5.32 | | |
| 8 | primary | 14.12 | 12.26 | 86.82 | 14.98 | 13.12 | 1.42 | 1.85 | 13.7 / 6.8 / 3.4 | 0.22 / 0.44 / 0.89 | 20.27, 3.81 |
| 8 | shadow | 10.95 | 10.95 | 76.54 | 13.78 | 11.61 | 1.51 | 2.15 | 12.5 / 6.2 / 3.1 | 0.22 / 0.44 / 0.87 | 11.82, 2.39 |
| 8 | bounce | 16.27 | 15.12 | 105.00 | 17.94 | 15.27 | 2.01 | 2.66 | 17.1 / 8.6 / 4.3 | 0.22 / 0.43 / 0.87 | 18.80, 3.66 |
| 8 | shadow2 | 10.21 | 10.21 | 71.06 | 13.75 | 11.71 | 1.39 | 2.01 | 11.6 / 5.8 / 2.9 | 0.22 / 0.43 / 0.87 | 11.61, 2.34 |
| 16 | primary | 11.43 | 9.40 | 119.48 | 12.32 | 10.43 | 1.28 | 1.89 | 10.7 / 5.3 / 4.3 | 0.40 / 0.79 / 0.94 | 13.64, 3.32 |
| 16 | shadow | 8.34 | 8.34 | 104.56 | 11.30 | 9.08 | 1.34 | 2.22 | 9.7 / 4.8 / 3.8 | 0.39 / 0.78 / 0.94 | 9.00, 2.49 |
| 16 | bounce | 12.72 | 11.47 | 140.25 | 14.40 | 11.72 | 1.81 | 2.68 | 13.3 / 6.6 / 5.2 | 0.38 / 0.77 / 0.92 | 13.32, 3.40 |
| 16 | shadow2 | 7.85 | 7.85 | 98.10 | 11.56 | 9.45 | 1.26 | 2.09 | 9.1 / 4.6 / 3.6 | 0.39 / 0.78 / 0.93 | 8.75, 2.40 |
| 32 | primary | 9.94 | 7.86 | 200.56 | 10.90 | 8.94 | 1.17 | 1.95 | 9.0 / 7.4 / 6.8 | 0.80 / 0.92 / 0.96 | 11.13, 3.48 |
| 32 | shadow | 6.71 | 6.71 | 168.13 | 9.64 | 7.28 | 1.27 | 2.30 | 8.0 / 6.5 / 5.8 | 0.78 / 0.90 / 0.95 | 7.08, 2.47 |
| 32 | bounce | 10.27 | 9.03 | 222.37 | 12.01 | 9.27 | 1.67 | 2.73 | 10.7 / 8.6 / 7.7 | 0.77 / 0.90 / 0.95 | 10.51, 3.46 |
| 32 | shadow2 | 6.29 | 6.29 | 157.11 | 9.94 | 7.67 | 1.19 | 2.20 | 7.5 / 6.0 / 5.4 | 0.78 / 0.90 / 0.95 | 6.98, 2.44 |

| W | wide nodes | children a node | node MiB, wide / today | treelet images MiB, wide / today | largest treelet KiB, wide / today |
| --- | --- | --- | --- | --- | --- |
| 8 | 389,787 | 3.56 | 83.3 / 68.5 | 169.0 / 156.0 | 196 / 156 |
| 16 | 238,932 | 5.18 | 102.1 / 68.5 | 187.5 / 156.0 | 224 / 156 |
| 32 | 154,253 | 7.47 | 131.8 / 68.5 | 217.7 / 156.0 | 392 / 156 |

Reading it:

1. Node visits a ray on Sponza's bounce: 61.4 (W = 8), 42.5 (16), 35.8 (32); today's
   packet walk takes 41.8 inner node operations a ray (its share of a packet's 1338).
   Operations a ray: 40.1 at W = 32, 1 x 32 against today's 88 lane-shares of two-box
   and leaf operations; the wide walk does fewer operations, each for one ray.
2. The second shadow: 21.2 node visits a ray at W = 32 against today's 12.6 inner node
   operations: today's second-shadow packets are more coherent (sorted) than its bounce.
3. Lanes full: the collapse gives 3.6 to 7.5 children a node on average (one triangle a
   leaf: the subtrees near the leaves are too small to fill W), so 1 x 32 fills 22 % of
   its lanes at W = 8 and 70 to 80 % at W = 32; 4 x 8 fills 85 to 97 % at every W.
4. Node bytes: wide nodes keep W slots of 7 words (a box and a child word). At W = 8
   the treelet images are about today's (39.5 MiB against 40.5 on Sponza, 152 against
   156 on Bistro, 169 against 156 on the 10^6 scene); at W = 32 they are 1.4 to 1.55
   times larger and Sponza's largest treelet is 356 KiB against 164.
5. Nearest first: taking entered children by entry distance saves 15 % of node visits
   and 31 % of triangle tests on Sponza's bounce at W = 32 (stored order: 41.9 and 9.58,
   against 35.8 and 6.58), 10 % and 5 % on Bistro's; on the any-hit traces it changes little (the second shadow at W = 8 is 2 % better in stored order).

## Part B: one node visit on a card

`probe.py` loads one program on the grid whose pack thread drives the vector unit as
the ray tracer's walk does (`kernels/compute.cpp`, a copy of the `PROBE` handler that
includes `projects/raytrace/kernels/sfpu.hpp` unchanged). Its data are real visits: 64
random rays of Sponza's bounce trace, each walked through the wide tree with its final
best (from the reference), give the node sequences; an operation takes the next visit
of each of its rays. The node records (six box words of each child, structure of
arrays) are in the core's L1. One operation, in five forms, each adding a step:

- **load**: the child boxes into the lanes of Dst slots, by one of three ways:
  - *window*: the core copies each ray's record into its lanes, one strided vector
    store a box word per ray (`vsse32` into the Dst window, 6 x rays a visit);
  - *gather*: one indexed vector load gathers every lane's word from the rays' records,
    one store a box word (2 x 16 and 4 x 8);
  - *unpack*: the node is a float32 tile in a circular buffer, laid out so that its six
    box words land in slots 0 to 5; the unpack and math threads copy it into Dst tile 0
    as the walk's packet states are copied today, and the pack thread waits for it (1 x
    32 only: rays of 2 x 16 and 4 x 8 sit in different nodes, so no one tile holds an
    operation's boxes);
- **test**: and the slab test (`rt::box_test`, the walk's exact test) with the ray's
  origin, reciprocal direction and best from immediates (1 x 32) or from Dst slots that
  the core filled when the ray took its lanes (2 x 16, 4 x 8), waited for;
- **out**: and the entry distances (+inf where missed) to the core;
- **stack**: and the core's stack step per ray: the entered children sorted by distance
  and pushed, one popped;
- **check**: as out, the distances kept: every lane checked against numpy's slab test,
  bit for bit; **ray**: a ray's seven words into its lanes, alone.

Blackhole card, AICLK 1350 MHz before and after every run, host load 5 to 9. One core
first, then all 120 with one command: the same cycles to the cycle (the probe touches
nothing outside its core). Cycles an operation on the slowest core, best of 3, 20
passes over 256 operations (the unpacker: 160 passes over 32 tiles):

| layout, W | load by | load | + test | + out | + stack | a visit | entered a visit | check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 x 32, 32 | window | 403 | 617 | 716 | 913 | 913 | 1.25 | 5726 lanes exact |
| 1 x 32, 32 | unpack | 218 | 429 | 527 | **729** | **729** | 1.28 | 744 lanes exact |
| 2 x 16, 16 | window | 829 | 922 | 1020 | 1287 | 644 | 1.15 | 6796 lanes exact |
| 2 x 16, 16 | gather | 1563 | 1655 | 1753 | 2020 | 1010 | 1.15 | 6796 lanes exact |
| 4 x 8, 8 | window | 1589 | 1682 | 1780 | 2132 | 533 | 1.10 | 7104 lanes exact |
| 4 x 8, 8 | gather | 1601 | 1693 | 1791 | 2143 | 536 | 1.10 | 7104 lanes exact |

A ray into its lanes (2 x 16, 4 x 8): 639 and 605 cycles a ray.

- **Boxes into lanes** is the largest step. A strided store into the Dst window costs
  about 66 cycles whatever its length (8 or 32 words), so 4 x 8 pays 24 of them a
  visit; the gathered load costs as much as the stores it saves. The unpacker brings a
  1 x 32 node in 218 cycles, the pack thread waiting.
- **The test** is 92 cycles with the ray in Dst and 211 with it as immediates (14
  immediate loads and 7 stores a visit).
- **Out** is 98 to 100 cycles (the wait for the vector unit and the read of one slot).
- **The stack step** is 196 to 352 cycles an operation: 1.1 to 1.3 children entered a
  visit, sorted and pushed by the core.

## Part C: the prediction

`predict.py`, per trace and per ray, with the cheapest measured load of each layout:

- **boxes**: Part A's box operations a ray times part B's cycles an operation (the
  stack form);
- **triangles**: Part A's triangle operations a ray times today's triangle test (737
  cycles nearest, 485 any hit) plus 9/6 of the layout's box loading (a triangle's nine
  words into the lanes; not measured);
- **pops**: entries popped beyond the best, 36 cycles each (today's pop);
- **ray loads**: 2 x 16 and 4 x 8 load a ray into its lanes and back at each treelet it
  visits (15.7 treelets a ray on Sponza's bounce, from ray_moves);

then the walk's milliseconds on the slowest core: cycles a ray, times rays over the
cores in use, times today's ratio of the slowest core's walk to the mean core's. The
rest of the trace (treelet reads, packet states, output, dispatch) stays as today: the
trace's measured time less today's walk on its slowest core. The frame keeps today's
primary and shadow traces (their packets stay coherent) and drops the SORT and
UNSORT commands (no packets to sort). Milliseconds:

| scene | layout, W | load by | bounce | second shadow | frame |
| --- | --- | --- | --- | --- | --- |
| Sponza overview | today | | 194.64 | 56.38 | 348.53 |
| Sponza overview | V100 | | 59.0 | | 95.52 |
| Sponza overview | 1 x 32, 8 | unpack | 409.9 | 195.5 | 688.2 |
| Sponza overview | 1 x 32, 16 | unpack | 304.4 | 152.6 | 539.9 |
| Sponza overview | 1 x 32, 32 | unpack | **264.0** | **124.0** | **470.9** |
| Sponza overview | 2 x 16, 8 / 16 / 32 | window | 516.8 / 423.6 / 488.2 | 259.3 / 221.3 / 248.4 | 859.0 / 727.8 / 819.5 |
| Sponza overview | 4 x 8, 8 / 16 / 32 | window | 450.1 / 498.3 / 617.7 | 227.0 / 262.8 / 318.5 | 760.0 / 844.0 / 1019.1 |
| Sponza overview | 1 x 32, 32, boxes free | unpack | 195.8 | 90.5 | 369.1 |
| Sponza overview | 4 x 8, 8, boxes free | window | 245.6 | 126.6 | 455.1 |
| Bistro overview | today | | 307.96 | 98.34 | 523.72 |
| Bistro overview | V100 | | 97.2 | | 153.94 |
| Bistro overview | 1 x 32, 8 / 16 / 32 | unpack | 602.4 / 495.2 / **365.5** | 338.6 / 275.3 / **211.8** | 1044.0 / 873.6 / **680.4** |
| Bistro overview | 2 x 16, 8 / 16 / 32 | window | 683.5 / 588.7 / 587.2 | 431.6 / 375.7 / 403.4 | 1218.2 / 1067.5 / 1093.6 |
| Bistro overview | 4 x 8, 8 / 16 / 32 | window | 589.9 / 643.0 / 712.1 | 377.9 / 434.2 / 507.7 | 1070.8 / 1180.3 / 1322.8 |
| Bistro overview | 1 x 32, 32, boxes free | unpack | 279.4 | 156.5 | 538.9 |

"Boxes free": the same with the load step at zero cycles, an upper bound on what any
way of bringing boxes into the lanes could give. The V100 gives no separate rate for
the second shadow rays. Today's walk takes 21,500 cycles a ray on Sponza's bounce and
6,300 on its second shadow (the mean core); the best wide row takes 30,800 and 16,500.

The three largest terms (Sponza, 1 x 32, W = 32, unpacker):

1. **Box operations: 193.6 ms (bounce) and 102.8 ms (second shadow).** Rest on part B's
   729 cycles a visit, measured on Sponza's bounce rays with their final best (a real
   walk enters slightly more children before its best shrinks), and on part A's 35.8
   and 21.2 node visits a ray, from an exact walk.
2. **The rest of each trace, 35.0 and 14.2 ms**, kept as today: rests on treelet reads,
   packet states and output costing what they cost today with rays walked one at a
   time.
3. **Triangles: 34.2 and 7.1 ms.** Rest on today's triangle test cycles and on loading
   a triangle's words into the lanes at 9/6 of a node's box loading; not measured.

On Bistro the same three; its triangles are larger (91.7 ms in the bounce: 11.2 leaf
visits and 53.6 triangle tests a ray).

## What a tracer built this way should use

None of these layouts: on this vector unit, driven by the pack thread, a node visit of
one ray costs about twice a binary packet step of 32 lanes, and the bounce rays' 6 live
lanes are enough for the packet walk to stay ahead. If wide nodes are tried again, the
row to start from is 1 x 32 at W = 32 with the unpacker (the only one near today), and
the steps to remove are not the box tests but the fixed costs around them: the wait for
the vector unit and the distances' read-back (98 cycles), the core's sort and push (202)
and the ray's words as immediates (about 110 of the test's 211). Nearest-first order
is worth keeping in any walk (15 % fewer visits, 31 % fewer triangle tests).
