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
(two blocks here), and a receiver waits for block k's VALID word. Every wait is bounded.
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

## Run

```
python projects/dense/run.py --shape lin384 --calls 100
python projects/dense/run.py --m 256 --k 64 --n 128 --inputs int --form stream
python projects/dense/bench.py --shape lin384 --json lin384.json
python projects/dense/run.py --m 512 --k 64 --n 384 --inputs int --form grid --grid rows
python projects/dense/bench.py --shape sq4096 --form grid
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

Measured on 2026-11-13 between 11:45 and 12:30 on one p150a (card clock nominal 1,350
MHz, X280 at the session default of 1,000 MHz), ours at commit "Projects: the matrix
product's lane form", Kohaku-TT at ac81ff2; for each preset ours first, then
Kohaku-TT's three routes, one after the other in the same session. Host load average
(1 min) 3.0 to 10.6 on a shared 96-thread host. HiFi2 on bf16 with fp32 accumulation;
every row's check passed (relative RMS 3.96e-3 to 3.98e-3, the same on both sides).

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
  `<preset>-gpu.json`).

| Preset | M x K x N | Ours: form, us per call | W, MHz | Kohaku-TT best: route, us | W, MHz | V100 fp16, us |
|---|---|---|---|---|---|---|
| `kvproj` | 128 x 896 x 128 | resident, 6.3 | 67 W, 1350 / 1350 | auto, 28.7 (gemm 28.9, own 63.1) | 63 W, 1350 / 1350 | 10.3 (112 W) |
| `lin384` | 16,384 x 384 x 384 | resident, 61.2 | 130 W, 1225 / 1296 | gemm, 94.1 (auto 101.7, own 138.4) | 116 W, 1343 / 1348 | 83.3 (291 W) |
| `lin1024` | 16,384 x 1024 x 1024 | rows, 393.5 | 128 W, 1168 / 1255 | gemm, 389.9 (auto 778.0, own 778.0) | 136 W, 1200 / 1264 | 469.0 (293 W) |
| `sq4096` | 4096 cubed | 2d, 958.3 | 138 W, 850 / 1008 | gemm, 957.0 (auto 974.1, own 1119.2) | 151 W, 850 / 959 | 1740.6 (293 W) |

Not measured yet at the default X280 clock: `sq8192` (8192 cubed), `sq16384`
(16,384 cubed) and `dec4096` (32 x 4096 x 4096).

Bounds of our rows, from the cut-down runs (`cut_us_per_call` in each JSON) and the
profile of one call:
- `kvproj`: dispatch. A call issues every 6.3 us; reads only 5.2 us, compute only 5.5
  us. At the X280's 1,000 MHz the single call takes 6.56 us (6.32 us at 1,750 MHz)
  while the back-to-back rate is the same (6.28 against 6.29 us).
- `lin384`: DRAM. A in and Y out are 25 MB per call; reads only 32.9 us, compute only
  21.4 us.
- `lin1024` (`rows`, on par: 393.5 against 389.9 us): the lanes. Lanes only 284.9
  us, product only 173.4 us.
- `sq4096` (`2d`, on par: 958.3 against 957.0 us, at a mean clock of 1,008 against
  959 MHz: both loops throttle under full load): the matrix unit. Product only 668.5
  us, lanes only 563.0 us.

## Files

| File | What it is |
|---|---|
| `kernels/dense.h` | segments, record words, buffers, profile words, the bounded NoC barrier |
| `kernels/reader.cpp`, `compute.cpp`, `writer.cpp` | the three kernels |
| `plan.py` | the plan as pure functions |
| `card.py` | the program and its records on a session |
| `kernels/grid/grid.h`, `reader.cpp`, `compute.cpp`, `writer.cpp` | the lane form: arguments, row lane and A, the product, column lane and Y |
| `run.py`, `bench.py` | one shape run and checked; one shape measured |
| `results/` | the runs behind the results table |
| `reference/ktt_bench.py` | Kohaku-TT's product on the same inputs, timed and checked the same way |
| `reference/gpu_bench.py`, `reference/gpu/gemm_gpu.cu` | cuBLAS on one idle GPU (fp32 and fp16) |
