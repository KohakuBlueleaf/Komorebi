# Dense: the matrix product

`Y = A @ W` with A `[M, K]` and W `[K, N]` in bf16, fp32 accumulation, bf16 output,
HiFi2 (the fidelity Kohaku-TT uses by default for bf16 operands), on the resident
workers of one Blackhole p150a. Records reach the grid through the dispatch kernel on
X280 cluster 0 (`komorebi.Session`, `Run.issue`).

## The thread program

```
thread product(i):                             # one thread per core of the team
    rows = share(M tiles, team, i)             # space: the output rows this core owns
    for m in rows:                             # time: row tiles
        a = read_dram(A[m, :])                 # the whole row of A, one item
        for j in blocks(N tiles, ct):          # time: column blocks of ct tiles
            acc = 0                            # fp32, in DST
            for k in chunks(K tiles, kc):      # time: K chunks
                w = W_resident[k, j]           # resident form: W read once by LOAD
                  or read_dram(W[k, j])        # streamed form: every chunk from DRAM
                acc += a[k] @ w
            write Y[m, j] = acc
```

There are no breakpoints: no thread waits for another. The resident form keeps W in
every core's L1 from one `LOAD` record on; it is chosen when W and the other buffers
fit L1. Otherwise W is streamed: every core reads every chunk of W it needs from DRAM,
once per row tile it owns, which costs DRAM bandwidth whenever W is large.

## How it maps onto the runtime

| Thread program | Kernel | What it does |
|---|---|---|
| `read_dram(A[m, :])`, `read_dram(W[k, j])` | `kernels/reader.cpp` (NoC0) | A's row (or K chunk) into `kCbA`, streamed W chunks into `kCbWs`; `LOAD` reads all of W into `kCbW` |
| `acc += a[k] @ w` | `kernels/compute.cpp` | `matmul_block` over the chunk's K tiles, ct output tiles in DST |
| `write Y[m, j]` | `kernels/writer.cpp` (NoC1) | the packed column block to Y's pages |

| Segment | Arguments |
|---|---|
| `LOAD` | a0 W, a1 the W buffer, a3 K tiles, a4 N tiles |
| `PRODUCT` | a0 A, a1 Y, a2 to a4 M, K, N tiles, a5 K chunk, a6 column block, a7 mode, a8 form, a9 W (streamed form) |
| `PROF` | a0 the profile tensor: each core's profile page (cycles per unit, summed over the records) |

`PRODUCT` modes 1 and 2 are the cut-down runs: reads only (A read and popped, no
product, no writes), and compute only (A taken from L1 as it is, no reads, packed but
not written).

`plan.py` is the plan as pure functions (form, K chunk, column block, buffer pages, the
L1 budget). `card.py` loads the program and issues the records; `run.py` runs one shape
and checks it; `bench.py` measures one shape. The references in `reference/` run the
same product with Kohaku-TT on the card and with cuBLAS on an idle GPU (by hand, not
tests).

## The lane form

When neither W nor A fits one core, the product runs on a rectangle of cores, and
the operands travel between cores instead of each core reading them from DRAM:

```
thread product(y, x):                          # one thread per core of the rectangle
    band = output tiles [rows of band y, columns of band x]
    for (r, c) in blocks(band, bm x bn):       # time: output blocks
        acc = 0                                # fp32 partial sums in L1 (kCbPart)
        for k in blocks(K tiles, kb):          # time: K blocks
            a = lane_row(y).take(A[r, k])      # the row's first core reads it from
                                               #   DRAM and hands it along the row
            w = lane_col(x).take(W[k, c])      # the column's bottom core reads it and
                                               #   hands it up the column
            acc += a @ w                       # breakpoint: a block waits until every
                                               #   taker freed the slot it reuses
        write Y[r, c] = acc
```

The breakpoints are the lanes' (`km::Lane`, `komorebi/device/km_lane.hpp`): the
sender of block k waits until every receiver's compute has freed block k minus the window
(two blocks here), and a receiver waits for block k's VALID word. Every wait between
cores is bounded; compute's waits on its own threads' registers (`tile_regs_wait`) are not.
The records carry `KM_F_EXCL` (no two records with lanes in flight together).

| Layout | Cores | A | W | Chosen when (`plan.layout_of`) |
|---|---|---|---|---|
| `2d` | bands of rows down the grid, bands of columns across it | along each row (row lane, NoC0, in the reader) | up each column (column lane, NoC1, in the writer) | otherwise |
| `rows` | one band of row tiles per core, whole columns of cores | each core reads its own | up each column | at least 3 row tiles per core, more than 2d reuses |
| `cols` | one band of column tiles per core, whole rows of cores | along each row | each core reads its own | fewer row tiles than grid rows |

A core that reads its own operand runs a lane of one core (no receivers), so the three
layouts share one set of kernels (`kernels/grid/`). Blocks are the largest that fit L1
(the rule halves rows first, columns first for `rows`); K blocks of 1 to 8 tiles;
the compute is that of Kohaku-TT's gemm (packer accumulation into fp32 partial sums,
reloaded for the last K block; subblocks of up to 4 tiles).

| Segment | Arguments |
|---|---|
| `GRID` | a0 A, a1 W, a2 Y, a3 mode: 0 full, 1 lanes only (blocks handed over and taken, no product, no writes), 2 product only (blocks from L1 as they are, no reads, no lanes, packed but not written) |
| `GRID_PROF` | a0 the profile tensor |

## The K split

When the output has few tiles (M of one or a few row tiles), the lane form leaves
most cores idle or each reading a long column of W. The K split puts a group of S
cores on each output block and gives each core 1/S of K:

```
thread part(g, d):                             # core d of group g, S cores to a group
    block = output tiles of group g (oh x ow)
    acc = 0                                    # fp32 in L1 (kCbSum)
    for k in chunks(share(K tiles, d, S), kc): # time: K chunks of this core's share
        acc += read_dram(A[block rows, k]) @ read_dram(W[k, block columns])
    acc = reduce(acc, group g)                 # owner: every core sends its part to
                                               #   core 0; tree: halves in log2 S rounds
    if d == 0: write Y[block] = acc            # breakpoint: a part is sent when its
                                               #   receiver has a free arrival slot
```

The parts travel over the cores' ports (`km::reduce`, `komorebi/device/km_port.hpp`):
the reader takes arrivals into `kCbArrive` (fp32, a slot per sender), compute adds them
in rank order into `kCbSum` by packer accumulation, the writer sends a member's part
to its receiver or writes the root's sum to Y. Records carry `KM_F_EXCL` (the ports).
Plans are named by hand: `--split s,oh,ow[,kc]` (`plan.split_plan`; K chunk default:
the whole share, at most 32 tiles) and `--reduce owner|tree`.

| Segment | Arguments |
|---|---|
| `SPLIT` | a0 A, a1 W, a2 Y, a3 reduce (0 owner, 1 tree), a4 mode: 0 full, 1 parts only (each core's part packed and dropped: no reduce, no writes) |

Measured on 2026-11-13 at 13:00 to 13:06 (X280 at 1,000 MHz, card clock 1,350 MHz
throughout), each against Kohaku-TT's best route in the same session, every check
passed (`results/<preset>-split-*.json`, Kohaku-TT's as `-split-ktt-<route>`):

| Preset | Ours: K split, us per call (single call) | Parts only | Kohaku-TT best, us | Lane or resident form, us |
|---|---|---|---|---|
| `kvproj` | 7,1,4 tree: 25.5 (9.3) | 20.1 | auto, 28.6 | resident, 6.3 |
| `dec4096` | 14,1,16,5 tree: 121.7 (101.9); owner 131.2 | 107.0 | own, 81.0 (gemm 81.0, auto 81.0) | `cols`, 197.8 |

Bounds: each record costs about 16 us more back to back when exclusive. A probe of
parts-only records (no ports, so allowed without the flag) on the 7,1,8 plan:
98.8 us single either way, back to back 115 to 116 us exclusive against 95.4 us not;
on `kvproj` 21 to 23 us against 5.0 to 5.1 us. Without that cost `dec4096` is bound by
W read from DRAM: 32 MiB in 95.4 us is 352 GB/s, against Kohaku-TT's 414 GB/s; each
core reads its chunks on NoC0 only.

## Run

```
python projects/dense/run.py --shape lin384 --calls 100
python projects/dense/run.py --m 256 --k 64 --n 128 --inputs int --form stream
python projects/dense/bench.py --shape lin384 --json lin384.json
python projects/dense/run.py --m 512 --k 64 --n 384 --inputs int --form grid --grid rows
python projects/dense/bench.py --shape sq4096 --form grid
python projects/dense/bench.py --shape dec4096 --split 14,1,16,5 --reduce tree
KTT_GEMM=1 python projects/dense/reference/ktt_bench.py --shape lin384
python -m projects.dense.reference.gpu_bench --shape lin384
```

`--shape` names a preset of `plan.SHAPES` (M x K x N): `kvproj` 128 x 896 x 128,
`lin384` 16,384 x 384 x 384, `lin1024` 16,384 x 1024 x 1024, `sq4096` 4096 cubed,
`sq8192` and `sq16384` cubed, `dec4096` 32 x 4096 x 4096. Setup is that of
[01_mapcollect](../../examples/01_mapcollect/README.md#what-it-needs). Inputs are
uniform in [-1, 1) rounded to bf16, from a seed (`ref/dense.py`, `inputs`).

`--form grid` takes the lane form with the rule's plan; `--grid` names a layout or a
whole plan (`[layout:]rows,cols,pcm,pcn,bm,bn,kb`). M, K and N are whole tiles of 32.

`run.py` and `bench.py` compare Y with the float64 product of the same bf16 operands
on 256 sampled rows: all rows when there are at most 256, else 8 whole row tiles drawn
from a seed (`ref.sampled_rows`). They read only those row tiles, one at a time (at
most 1 MB each, a progress line each). They exit 1 when the check fails.
`bench.py` times one run of records back to back (or runs of `--batch` records, each
waited for): the time per call is the runs' time (first issue to last event) per call.
`reference/ktt_bench.py` runs Kohaku-TT's product on the same inputs in Kohaku-TT's
own environment (fast dispatch), with the route its caller selects (`KTT_GEMM=1`: its
gemm; unset: its automatic choice), and checks the same rows the same way. Kohaku-TT
is the public repository https://github.com/KohakuBlueleaf/Kohaku-TT; the bench runs
in a Python environment where it is installed, and records which version it ran.

**The check's bound.** Relative RMS error at most 1e-2 and largest error at most 2e-2 of
the largest output. The output's rounding to bf16 alone is up to 2^-9 relative per
element; the runs in the table below give 3.96e-3 to 3.98e-3 RMS and 5.1e-3 to 6.2e-3
largest; Kohaku-TT's routes give the same RMS on the same rows (each run's JSON
carries both figures). With `--inputs int` (integers in [-2, 2], K at most 64) every product
and sum is exact, and the check is bit for bit.

## Results

Measured on 2026-11-13 between 11:45 and 12:50 on one p150a (card clock nominal 1,350
MHz, X280 at the session default of 1,000 MHz), ours at the commits of this directory
of that day, Kohaku-TT at ac81ff2; for each preset ours first, then Kohaku-TT's three
routes, one after the other in the same session. Host load average (1 min) 3.0 to 12.7
on a shared 96-thread host. HiFi2 on bf16 with fp32 accumulation; every row's check
passed (relative RMS 3.96e-3 to 3.98e-3, the same on both sides).

- Ours: `python projects/dense/bench.py --shape <preset> [--form grid]`: records back
  to back for about four seconds in one run, time per call from the first issue to
  the last event.
- Kohaku-TT: `reference/ktt_bench.py --shape <preset>` with `KTT_GEMM=1` (gemm),
  unset (auto) and `KTT_GEMM=0 KTT_OWN_MATMUL_MIN_ROWS=1` (own); calls back to back for
  about four seconds and one synchronize. The fastest route is shown, the others in
  brackets.
- V100: `python -m projects.dense.reference.gpu_bench --shape <preset>`, cuBLAS fp16
  with fp32 accumulation on one idle Tesla V100 (PG503-216), with its mean board power.
- Power: mean whole-card power over the loop; AICLK sampled every 20 ms (min / mean).
  Only the product is timed: uploads and read-back are not part of a call.
- The runs' JSON lines are in `results/` (`<preset>-ours.json`, `<preset>-ktt-<route>.json`,
  `<preset>-gpu.json`); 16,384 cubed checks 8 sampled row tiles (256 rows).

| Preset | M x K x N | Ours: form, us per call | W, MHz | Kohaku-TT best: route, us | W, MHz | V100 fp16, us |
|---|---|---|---|---|---|---|
| `kvproj` | 128 x 896 x 128 | resident, 6.3 | 67 W, 1350 / 1350 | auto, 28.7 (gemm 28.9, own 63.1) | 63 W, 1350 / 1350 | 10.3 (112 W) |
| `lin384` | 16,384 x 384 x 384 | resident, 61.2 | 130 W, 1225 / 1296 | gemm, 94.1 (auto 101.7, own 138.4) | 116 W, 1343 / 1348 | 83.3 (291 W) |
| `lin1024` | 16,384 x 1024 x 1024 | rows, 393.5 | 128 W, 1168 / 1255 | gemm, 389.9 (auto 778.0, own 778.0) | 136 W, 1200 / 1264 | 469.0 (293 W) |
| `sq4096` | 4096 cubed | 2d, 958.3 | 138 W, 850 / 1008 | gemm, 957.0 (auto 974.1, own 1119.2) | 151 W, 850 / 959 | 1740.6 (293 W) |
| `sq8192` | 8192 cubed | 2d, 7985.9 | 147 W, 881 / 1034 | auto, 11797.6 (gemm 12310.9, own 11836.2) | 139 W, 800 / 1088 | 12997.7 (295 W) |
| `sq16384` | 16,384 cubed | 2d, 84018.0 | 133 W, 1125 / 1220 | gemm, 75522.5 (auto 90785.6, own 92389.3) | 142 W, 875 / 972 | 105839.6 (295 W) |
| `dec4096` | 32 x 4096 x 4096 | cols, 197.8 | 75 W, 1343 / 1350 | own, 81.0 (gemm 105.2, auto 81.0) | 91 W, 1350 / 1350 | 77.4 (183 W) |

Behind: `sq16384` (11%) and `dec4096` (2.4 times). Single calls, waited for, against
the loop, from the same JSON (Kohaku-TT's `synced_us_median`), in us:

| Preset | Ours single / loop | Kohaku-TT best route single / loop |
|---|---|---|
| `sq4096` | 660.0 / 958.3 | gemm 781.3 / 957.0 |
| `sq8192` | 5765.8 / 7985.9 | auto 4568.4 / 11797.6 |
| `sq16384` | 75877.0 / 84018.0 | gemm 56356.7 / 75522.5 (auto 32935.1 / 90785.6) |

From 4096 cubed up both sides are bound in a loop by the card's power limit (133 to
151 W mean; the firmware lowers the clock and switches the tile product to a throttled
form under its current limit), so time per call follows energy per call: 16,384 cubed
takes 11.2 J per call here and 10.7 J in Kohaku-TT's gemm.

Bounds of our rows, from the cut-down runs (`cut_us_per_call` in each JSON) and the
profile of one call (`profile`: shares of the slowest core's cycles; cycles per tile
product from its matrix-unit share, at the 1,350 MHz read at load). Cut-down runs that
follow a long loop run at a throttled clock.
- `kvproj` (resident): record issue. A call issues every 6.3 us; the slowest core's
  body takes 4.6 us; reads only 5.2 us, compute only 5.5 us; a single call, waited
  for, takes 6.56 us.
- `lin384` (resident): DRAM writes. A in and Y out are 25 MB per call (410 GB/s);
  the slowest core spends 82% of its cycles writing Y and its packer waits for space
  77%; reads only 32.9 us, compute only 21.4 us.
- `lin1024` (`rows`, on par): the W lane. W is handed along to all 110 cores: lanes only
  284.9 us, product only 173.4 us; the slowest core spends 77% on the lane and its
  unpacker waits for W 45%; the matrix unit takes 46.2 cycles per tile product
  against the probe's 34.4. A single call takes 355.7 us at 1,350 MHz, 382.6 at the
  loop's mean 1,255 MHz; the other 11 us of the loop's 393.5 are between records.
- `sq4096` (`2d`, on par): the matrix unit, then power. The matrix unit is 81% of the
  slowest core at 36.2 cycles per tile product (19,968 on that core, 4.7% above an
  even share of 110 cores); a single call takes 660.0 us at 1,350 MHz. In the loop the
  clock falls to a mean of 1,008 MHz (884 us at that clock); both sides end at 957
  to 958 us.
- `sq8192` (`2d`, ahead): the row lane in a single call (98.5% of the slowest core;
  W read from DRAM 70.9%, four times per call since blocks of 7 of 26 rows), the
  matrix unit 75% at 36.4 cycles per tile product; in the loop, power.
- `sq16384` (`2d`, behind): the lanes with the rule's blocks. Each core's 52 x 44
  tiles are blocks of 4 x 44 with K blocks of 2, so W passes each column 13 times:
  the row lane is 99.8% of the slowest core, its unpacker waits for W 52%, the matrix
  unit 35%. Blocks of 13 x 12 with K blocks of 4 (`--grid 2d:10,12,52,44,13,12,4`)
  take a single call to 36.3 ms (matrix unit 87%, 35.8 cycles per tile product) but
  the loop to 99.2 ms at 143 W (`sq16384-ours-blocks-13x12.json`): the denser call
  is throttled harder.
- `dec4096` (`cols`, behind): W from DRAM. 32 MB of W are read by 72 cores, 512 KB
  each with one block in flight per lane sender: the slowest core spends 97% reading
  W, 2.9 GB/s per core, 170 GB/s in all against Kohaku-TT's 414 GB/s; lanes only
  195.8 us, product only 26.1 us. The K split (above) takes it to 121.7 us.

## Files

| File | What it is |
|---|---|
| `kernels/dense.h` | segments, record words, buffers, profile words, the bounded NoC barrier |
| `kernels/reader.cpp`, `compute.cpp`, `writer.cpp` | the three kernels |
| `plan.py` | the plan as pure functions |
| `card.py` | the program and its records on a session |
| `kernels/grid/grid.h`, `reader.cpp`, `compute.cpp`, `writer.cpp` | the lane form: arguments, row lane and A, the product, column lane and Y |
| `kernels/split/split.h`, `reader.cpp`, `compute.cpp`, `writer.cpp` | the K split: arguments, A and W and arrivals, the part and the sum, sending and Y |
| `run.py`, `bench.py` | one shape run and checked; one shape measured |
| `results/` | the runs behind the results table |
| `reference/ktt_bench.py` | Kohaku-TT's product on the same inputs, timed and checked the same way |
| `bench_ttnn.py` | TT-NN's product and SwiGLU on the same inputs; it imports `ttnn` and `torch` from the tt-metal tree `TT_METAL_HOME` names (Komorebi needs neither) |
| `reference/gpu_bench.py`, `reference/gpu/gemm_gpu.cu` | cuBLAS on one idle GPU (fp32 and fp16) |
