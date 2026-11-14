# Dense: the matrix product

`Y = A @ W` with A `[M, K]` and W `[K, N]` in bf16, fp32 accumulation, bf16 output,
HiFi2 (the fidelity Kohaku-TT uses by default for bf16 operands), on the persistent kernels of one Blackhole p150a. Commands reach the grid through the dispatcher on
X280 cluster 0 (`komorebi.Session`, `Run.submit`).

## What each core does

Every core of the group works on its own rows of A: `km::partition` of the M row tiles
over the cores the command names. Its three kernels, per row tile it owns:

1. `kernels/reader.cpp` (NoC0): reads A's row of K tiles from DRAM into `kCbA`, as one
   item when the row is one K chunk or W is streamed, otherwise one K chunk at a time.
   In the streamed form it also reads, for every column block and K chunk, that chunk
   of W (`chunk_tiles` rows of `block_width` tiles) from DRAM into `kCbWs`. `LOAD` reads
   all of W from DRAM into the W buffer (`kCbW`) once.
2. `kernels/compute.cpp`: for each column block of `block_width` tiles, waits for A
   (and for the streamed W chunk), runs `matmul_block` over the K tiles of each chunk
   into the same fp32 tiles of DST (the resident form takes W from the buffer `LOAD`
   filled), and packs the finished block into `kCbOut`.
3. `kernels/writer.cpp` (NoC1): waits for the packed column block in `kCbOut`, writes
   its tiles to Y's pages in DRAM and frees the buffer.

There are no breakpoints: no thread waits for another. The resident form keeps W in
every core's L1 from one `LOAD` command on; it is chosen when W and the other buffers
fit L1. Otherwise W is streamed: every core reads every chunk of W it needs from DRAM,
once per row tile it owns, which costs DRAM bandwidth whenever W is large.

| Handler | Arguments |
|---|---|
| `LOAD` | a0 w_address (W), a1 w_buffer (the W buffer), a3 k_tiles, a4 n_tiles |
| `PRODUCT` | a0 a_address (A), a1 y_address (Y), a2 to a4 m_tiles, k_tiles, n_tiles, a5 chunk_tiles (K chunk), a6 block_width (column block), a7 mode, a8 form, a9 w_address (W, streamed form) |
| `PROF` | a0 profile_address (the profile tensor): each core's profile page (cycles per unit, summed over the commands) |

`PRODUCT` modes 1 and 2 are the cut-down runs: reads only (A read and popped, no
product, no writes), and compute only (A taken from L1 as it is, no reads, packed but
not written).

`plan.py` is the plan as pure functions (form, K chunk, column block, buffer pages, the
L1 budget). `card.py` loads the program and submits the commands; `run.py` runs one shape
and checks it; `bench.py` measures one shape. The references in `reference/` run the
same product with Kohaku-TT on the card and with cuBLAS on an idle GPU (by hand, not
tests).

## The broadcast form

When neither W nor A fits one core, the product runs on a rectangle of cores, and
the operands travel between cores instead of each core reading them from DRAM. Each core of the rectangle owns a band of output tiles (the rows of its row band, the
columns of its column band) and walks it in output blocks of `block_height` x
`block_width` tiles, each over K blocks of `k_block_tiles` tiles:

1. `kernels/grid/reader.cpp` (NoC0, the row broadcast): the row's first core reserves
   the slot, reads the A block of that K block from DRAM into it and sends it along
   the row with `km::Broadcast` (`cast`); the other cores of the row `arrive`.
2. `kernels/grid/writer.cpp` (NoC1, the column broadcast): the column's bottom core
   reads the W block of that K block from DRAM and hands it up the column the same
   way; the others `arrive`.
3. `kernels/grid/compute.cpp`: takes the A block and the W block, runs `matmul_block`
   over the block's subblocks, and sums the K blocks into fp32 partial sums in L1
   (`kCbPart`) by packer accumulation; the last K block is reloaded from `kCbPart` and
   packed in bf16 into `kCbGridOut`, and each block is handed back to the broadcast
   (`km::broadcast_done`) so its slot can be reused.
4. `kernels/grid/writer.cpp`: writes a finished output block to Y's pages once the
   first W block of the next output block has been delivered (the last block when
   the loop ends).

The breakpoints are the broadcasts' (`km::Broadcast`, `komorebi/device/km_broadcast.hpp`): the
sender of block k waits until every receiver's compute has freed block k minus the window
(two blocks here), and a receiver waits for block k's VALID word. Every wait between
cores is timed; compute's waits on its own threads' registers (`tile_regs_wait`) are not.
The commands carry `KM_FLAG_EXCLUSIVE` (no two commands with broadcasts in flight together).

| Layout | Cores | A | W | Chosen when (`plan.layout_of`) |
|---|---|---|---|---|
| `2d` | bands of rows down the grid, bands of columns across it | along each row (row broadcast, NoC0, in the reader) | up each column (column broadcast, NoC1, in the writer) | otherwise |
| `rows` | one band of row tiles per core, whole columns of cores | each core reads its own | up each column | at least 3 row tiles per core, more than 2d reuses |
| `cols` | one band of column tiles per core, whole rows of cores | along each row | each core reads its own | fewer row tiles than grid rows |

A core that reads its own operand runs a broadcast of one core (no receivers), so the three
layouts share one set of kernels (`kernels/grid/`). Blocks are the largest that fit L1
(the rule halves rows first, columns first for `rows`); K blocks of 1 to 8 tiles;
the compute is that of Kohaku-TT's gemm (packer accumulation into fp32 partial sums,
reloaded for the last K block; subblocks of up to 4 tiles).

| Handler | Arguments |
|---|---|
| `GRID` | a0 a_address (A), a1 w_address (W), a2 y_address (Y), a3 mode: 0 full, 1 broadcasts only (blocks delivered and taken, no product, no writes), 2 product only (blocks from L1 as they are, no reads, no broadcasts, packed but not written) |
| `GRID_PROF` | a0 profile_address (the profile tensor) |

## The K split

When the output has few tiles (M of one or a few row tiles), the broadcast form leaves
most cores idle or each reading a long column of W. The K split puts a group of S
cores on each output block and gives each core 1/S of K:

Each group of S cores owns one output block of `block_height` x `block_width` tiles;
core d of the group takes the d-th share of K (`km::partition` of the K tiles over the
group) in chunks of `chunk_tiles`:

1. `kernels/split/reader.cpp` and `kernels/split/writer.cpp`: each reads its half of
   the core's K chunks from DRAM, A's `block_height` x `chunk_tiles` tiles and W's
   `chunk_tiles` x `block_width` tiles per chunk (`move_chunks`, below).
2. `kernels/split/compute.cpp`: per chunk, runs `matmul_block` and packs the result
   into `kCbSum` (fp32 in L1) with packer accumulation, so `kCbSum` ends as this
   core's part of the sum over its share of K.
3. The parts are summed over the cores' ports. With `owner`, every core but core 0 of
   the group sends its part to core 0, each into its own arrival slot; with `tree`,
   the cores halve in log2 S rounds. A sender's writer writes the part into the
   receiver's slot and adds one to the receiver's port (`km::reduce_emit`); the
   receiver's reader waits for the arrivals (`km::reduce_arrive`) and its compute adds
   them into `kCbSum` (`km::reduce`).
4. Core 0 of the group packs the sum in bf16 into `kCbSplitOut`, and its writer
   (`kernels/split/writer.cpp`) writes the block to Y's pages.

A core's K chunks are read by both of its movers in turn (`km::move`, pattern `AB`,
`kernels/split/chunks.hpp`): the reader on NoC0 the even chunks, the writer on NoC1 the
odd ones, each into its own half of the A pair and the W pair (one chunk a half).
The parts travel over the cores' ports (`km::reduce`, `komorebi/device/km_port.hpp`):
the reader takes arrivals into `kCbArrive` (fp32, a slot per sender), compute adds them
in rank order into `kCbSum` by packer accumulation, the writer sends a core's part
to its receiver or writes the root's sum to Y. Commands carry `KM_FLAG_EXCLUSIVE` (the ports).
Plans are named by hand: `--split s,oh,ow[,kc]` (`plan.split_plan`; K chunk default:
the whole partition, at most 32 tiles) and `--reduce owner|tree`.

| Handler | Arguments |
|---|---|
| `SPLIT` | a0 a_address (A), a1 w_address (W), a2 y_address (Y), a3 reduce_form (0 owner, 1 tree), a4 mode: 0 full, 1 parts only (each core's part packed and dropped: no reduce, no writes) |

Measured on 2026-11-13 at 13:00 to 13:06 (X280 at 1,000 MHz, card clock 1,350 MHz
throughout), each against Kohaku-TT's best route in the same session, every check
passed (`results/<preset>-split-*.json`, Kohaku-TT's as `-split-ktt-<route>`):

| Preset | Ours: K split, us per call (single call) | Parts only | Kohaku-TT best, us | Broadcast or resident form, us |
|---|---|---|---|---|
| `kvproj` | 7,1,4 tree: 12.0 (9.4) | 7.9 | auto, 28.6 | resident, 6.0 |
| `dec4096` | 14,1,16,5 tree: 92.4 (90.2); owner 109.0 | 81.7 | own, 81.0 (gemm 81.0, auto 81.0) | `cols`, 183.9 |

The `kvproj` K split call, the resident and `cols` cells were measured again on
2026-11-14 (best of three, the same commands): the loop took 25.5, 6.3 and 197.8 us
on 2026-11-13, a single `kvproj` K split call 9.3, so the gap between a single call and a
call in the loop was 16.2 us, and is 2.6 us now (`cols`: 15.7 us then, 0.9 us now). The `owner`
form of `dec4096` took 109.0 and now takes 108.8.

The `dec4096` K split was measured again on 2026-11-14 (card clock 1,343 to 1,350 MHz,
best of three) with the chunks on both movers; the same day and card, with every chunk
on the reader as before, it took 103.6 (101.6) tree, 117.1 owner, parts only 92.7
(`-reader.json`).
`kvproj` (one chunk a core: the reader alone) is not changed by it, 12.0 against 11.9.

Bounds: each command costs about 16 us more back to back when exclusive. A probe of
parts-only commands (no ports, so allowed without the flag) on the 7,1,8 plan:
98.8 us single either way, back to back 115 to 116 us exclusive against 95.4 us not;
on `kvproj` 21 to 23 us against 5.0 to 5.1 us. Without that cost `dec4096` is bound by
W read from DRAM: 32 MiB in 95.4 us is 352 GB/s, against Kohaku-TT's 414 GB/s, with
each core's chunks on NoC0 only. On both NoCs the parts-only call takes 81.7 us: at
least 410 GB/s.

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
uniform in [-1, 1) rounded to bf16, from a seed (`reference/dense.py`, `inputs`).

`--form grid` takes the broadcast form with the rule's plan; `--grid` names a layout or a
whole plan (`[layout:]rows,cols,pcm,pcn,bm,bn,kb`). M, K and N are whole tiles of 32.

`run.py` and `bench.py` compare Y with the float64 product of the same bf16 operands
on 256 sampled rows: all rows when there are at most 256, else 8 whole row tiles drawn
from a seed (`sampled_rows` in `reference/dense.py`). They read only those row tiles, one at a time (at
most 1 MB each, a progress line each). They exit 1 when the check fails.
`bench.py` times one run of commands back to back (or runs of `--batch` commands, each
waited for): the time per call is the runs' time (first submit to last event) per call.
`reference/ktt_bench.py` runs Kohaku-TT's product on the same inputs in Kohaku-TT's
own environment (fast dispatch), with the route its caller selects (`KTT_GEMM=1`: its
gemm; unset: its automatic choice), and checks the same rows the same way. Kohaku-TT
is the public repository https://github.com/KohakuBlueleaf/Kohaku-TT; the bench runs
in a Python environment where it is installed, and commands which version it ran.

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

- Ours: `python projects/dense/bench.py --shape <preset> [--form grid]`: commands back
  to back for about four seconds in one run, time per call from the first submit to
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

Ours was measured again on 2026-11-14 between 02:55 and 03:34 (host load average 3 to 12; the
clocks of each run are in its JSON), with the same commands, best of three
runs; Kohaku-TT's and the V100's columns were not run again. A row that moved by less than
3 % keeps its value of 2026-11-13: `lin384` (+1.6 %), `lin1024` (-0.6 %), `sq4096`
(+2.7 %, see below) and the `dec4096` K split (+0.1 %). Four rows are today's: `kvproj`
(-3.8 %), `sq8192` (+7.8 %), `sq16384` (+4.7 %) and `dec4096`, whose row is now the
K split (`cols` took 183.9 us, -7.0 %).

| Preset | M x K x N | Ours: form, us per call | W, MHz | Kohaku-TT best: route, us | W, MHz | V100 fp16, us |
|---|---|---|---|---|---|---|
| `kvproj` | 128 x 896 x 128 | resident, 6.0 | 71 W, 1350 / 1350 | auto, 28.7 (gemm 28.9, own 63.1) | 63 W, 1350 / 1350 | 10.3 (112 W) |
| `lin384` | 16,384 x 384 x 384 | resident, 61.2 | 130 W, 1225 / 1296 | gemm, 94.1 (auto 101.7, own 138.4) | 116 W, 1343 / 1348 | 83.3 (291 W) |
| `lin1024` | 16,384 x 1024 x 1024 | rows, 393.5 | 128 W, 1168 / 1255 | gemm, 389.9 (auto 778.0, own 778.0) | 136 W, 1200 / 1264 | 469.0 (293 W) |
| `sq4096` | 4096 cubed | 2d, 958.3 | 138 W, 850 / 1008 | gemm, 957.0 (auto 974.1, own 1119.2) | 151 W, 850 / 959 | 1740.6 (293 W) |
| `sq8192` | 8192 cubed | 2d, 8608.3 | 141 W, 800 / 969 | auto, 11797.6 (gemm 12310.9, own 11836.2) | 139 W, 800 / 1088 | 12997.7 (295 W) |
| `sq16384` | 16,384 cubed | 2d, 87944.5 | 137 W, 1056 / 1175 | gemm, 75522.5 (auto 90785.6, own 92389.3) | 142 W, 875 / 972 | 105839.6 (295 W) |
| `dec4096` | 32 x 4096 x 4096 | K split 14,1,16,5 tree, 92.4 | 83 W, 1343 / 1350 | own, 81.0 (gemm 105.2, auto 81.0) | 91 W, 1350 / 1350 | 77.4 (183 W) |

Behind: `sq16384` (16%) and `dec4096` (14%). Single calls, waited for, against
the loop, from the same JSON (Kohaku-TT's `synced_microseconds_median`), in us:

| Preset | Ours single / loop | Kohaku-TT best route single / loop |
|---|---|---|
| `sq4096` | 660.0 / 958.3 | gemm 781.3 / 957.0 |
| `sq8192` | 5771.5 / 8608.3 | auto 4568.4 / 11797.6 |
| `sq16384` | 76299.6 / 87944.5 | gemm 56356.7 / 75522.5 (auto 32935.1 / 90785.6) |

From 4096 cubed up both sides are bound in a loop by the card's power limit (137 to
151 W mean; the firmware lowers the clock and switches the tile product to a throttled
form under its current limit), so time per call follows energy per call: 16,384 cubed
takes 12.1 J per call here and 10.7 J in Kohaku-TT's gemm.

On 2026-11-14 the 4096 and 8192 cubed loops ran in one of two modes, run after run with
the same command. 4096 cubed took 983.8, 991.2 and 993.1 us in three runs (mean clock 965
to 974 MHz) and 1,550 to 1,605 us in six (mean clock 1,085 to 1,089 MHz, 135 to 145 W); the product-only cut took 689 to 698 us in the first mode and 1,703 to 1,717 us in
the second, single calls 660 us in both. 8192 cubed took 8,608.3 us in one run and 13,336.3
and 13,490.2 us in two. The table shows the fastest run, and the 4096 cubed row keeps its
value of 2026-11-13, which the fastest of the nine runs is within 3 % of. What selects the
mode is not known (the firmware's current limit is our reading, not a measurement).
Kohaku-TT's columns were measured on 2026-11-13 and not run again.

Bounds of our rows, from the cut-down runs (`cut_microseconds_per_call` in each JSON) and the
profile of one call (`profile`: shares of the slowest core's cycles; cycles per tile
product from its matrix-unit share, at the 1,350 MHz read at load). Cut-down runs that
follow a long loop run at a throttled clock.
- `kvproj` (resident): command submit. A call submits every 6.0 us; the slowest core's
  body takes 4.7 us; reads only 4.7 us, compute only 5.5 us; a single call, waited
  for, takes 6.50 us.
- `lin384` (resident): DRAM writes. A in and Y out are 25 MB per call (410 GB/s);
  the slowest core spends 82% of its cycles writing Y and its packer waits for space
  77%; reads only 32.9 us, compute only 21.4 us.
- `lin1024` (`rows`, on parallel): the W broadcast. W is handed along to all 110 cores: broadcasts only
  284.9 us, product only 173.4 us; the slowest core spends 77% on the broadcast and its
  unpacker waits for W 45%; the matrix unit takes 46.2 cycles per tile product
  against the probe's 34.4. A single call takes 355.7 us at 1,350 MHz, 382.6 at the
  loop's mean 1,255 MHz; the other 11 us of the loop's 393.5 are between commands.
- `sq4096` (`2d`, on parallel): the matrix unit, then power. The matrix unit is 81% of the
  slowest core at 36.2 cycles per tile product (19,968 on that core, 4.7% above an
  even partition over 110 cores); a single call takes 660.0 us at 1,350 MHz. In the loop the
  clock falls to a mean of 1,008 MHz (884 us at that clock); both sides end at 957
  to 958 us.
- `sq8192` (`2d`, ahead): the row broadcast in a single call (98.3% of the slowest core;
  W read from DRAM 71.3%, four times per call since blocks of 7 of 26 rows), the
  matrix unit 75% at 36.4 cycles per tile product; in the loop, power.
- `sq16384` (`2d`, behind): the broadcasts with the rule's blocks. Each core's 52 x 44
  tiles are blocks of 4 x 44 with K blocks of 2, so W passes each column 13 times:
  the row broadcast is 99.8% of the slowest core, its unpacker waits for W 54%, the matrix
  unit 35%. Blocks of 13 x 12 with K blocks of 4 (`--grid 2d:10,12,52,44,13,12,4`)
  take a single call to 36.3 ms (matrix unit 87%, 35.8 cycles per tile product) but
  the loop to 99.2 ms at 143 W (`sq16384-ours-blocks-13x12.json`): the denser call
  is throttled harder.
- `dec4096` (K split, behind): W from DRAM. 32 MiB of W in 92.4 us is 363 GB/s against
  Kohaku-TT's 414 GB/s (81.0 us); the parts-only call takes 81.7 us (410 GB/s). The `cols`
  form (183.9 us) is bound the same way on 72 cores: 512 KB each with one block in flight
  per broadcast sender, the slowest core spends 97% reading W, 2.9 GB/s per core, 184 GB/s
  in all; broadcasts only 182.2 us, product only 13.1 us.

## On two islands

`islands.py` runs the grid form over the islands of a session (two p150a cards with
`--cards 0,1`, see [Islands](../../docs/reference/islands.md)). Each island takes its part
of A's row tiles (`island_rows` in `islands.py`), makes the grid plan of the broadcast
form for those rows alone (`plan.grid_plan`, the project's rule) and loads the grid
program of that plan (`card.grid_program`) on its own rectangle. The kernels are those
of `kernels/grid/`.

A and Y are split along their row tiles over every rank (`split=0`, the session's
hint; the rows below were measured when they were placed by island, even parts of the
row tiles, which two islands of equal ranks split alike); W is a copy on every island. Each island's program is the grid form of
its own plan, with its rows' first tile and end in its words, and each call is one
GRID command an island on its rectangle's flat ranks (the rows below were measured
with one command over a flat group of both rectangles; a group that reaches another
island now crosses the link, and a crossing command is never exclusive). `--cards 0` is the same
session over one island: it holds every row, and its program is the one `bench.py`
runs. The script has no branch for one card.

```
python projects/dense/islands.py --shape sq8192 --cards 0,1
python projects/dense/islands.py --shape sq8192 --cards 0
python projects/dense/islands.py --m 7680 --k 64 --n 128 --inputs int --cards 0,1
```

Measured on 2026-11-14 between 21:39 and 21:48, X280 at the session default of 1,350
MHz, host load average 2.7 to 8.2, one run each (three single calls, then calls back
to back for about four seconds or more); the check is that of `run.py` on 256 sampled
rows, which fall on both islands. Time per call is the loop's host time per call;
uploads and the read-back are host transfers, not part of a call. The first two rows
are this script; the rows marked "by rank" are its first version, which placed A and
Y as the session does by default (partition over the ranks in island order) and ran
one card through the card's own session. The runs' JSON lines are in `results/`
(`<preset>-islands-0x1-even.json`, `sq8192-islands-0-one.json`; by rank:
`-islands-0x1.json`, `-islands-0.json`).

| Preset | Cards | Row tiles per island | Loop, us per call | TFLOP/s | Single call, us | AICLK mean, MHz | Upload, s | Read-back, s |
|---|---|---|---|---|---|---|---|---|
| `sq8192` | 0 | 256 | 13,562.5 | 81.1 | 5,788 | 1039 | 2.94 | 0.51 |
| `sq8192`, W copied on both islands: no crossing | 0, 1 | 128 / 128 | 4,269.2 | 257.5 | 2,931 | 925 / 980 | 1.23 | 0.27 |
| `sq8192`, B moved once a call | 0, 1 | 128 / 128 | 27,196.9 | 40.4 | 2,942 | 1350 / 1350 | 1.12 | 0.27 |
| `sq8192`, by rank | 0 | 256 | 13,192.0 | 83.3 | 5,717 | 1034 | 0.85 | 0.28 |
| `sq8192`, by rank | 0, 1 | 136 / 120 | 5,242.3 | 209.7 | 4,686 | 970 / 1208 | 1.12 | 0.27 |
| `sq16384`, by rank | 0 | 512 | 91,763.8 | 95.9 | 77,693 | 1138 | 10.78 | 0.29 |
| `sq16384`, by rank | 0, 1 | 272 / 240 | 48,306.7 | 182.1 | 42,259 | 1170 / 1173 | 3.99 | 0.28 |

Every check passed (relative RMS 3.96e-3 to 3.97e-3, largest 5.6e-3 to 6.5e-3).
Placed by island, with W copied on both islands (nothing crosses; not the
linear layer of the next section, which holds W once), both islands of `sq8192` get
128 row tiles and the same plan (bands of 13 x 24, blocks of 7 x 24, K blocks of 4):
4,269.2 us per call, 0.50 of the best one-card run (8,608.3 above) and 0.81 of the by-rank split, whose island 1 got blocks
of 12 x 24 with K blocks of 1 tile and was the slower part. Both one-card `sq8192` loops
ran in the slower of the two modes described above (13,192 and 13,562 us): the same
plan and kernels as `bench.py`, which runs 21 single calls, then waits 1 s and runs
its loop under the power sampler, where this script runs 3 single calls and its loop
at once; `bench.py` itself ran in that mode in two of its three runs. `sq16384` on two
cards takes 0.53 of one card's time per call.

"B moved once a call": before each call island 0's link cores send B (the weight, 128
MiB) from island 0's copy into the same pages of island 1's (measured when B was held
by island 0 alone and island 1 had room for every page) (`send_pages`, waited for on island 1's link core), then one GRID
command an island runs as above, waited for
(`--links 2 --kernel-bytes 90112 --b-moved`; the page links' code needs the larger
kernel buffer). Of the 27,196.9 us a call the crossing takes 23,709.5 us, 5.66 GB/s:
with two links, link 0 carries the commands across and link 1 alone the pages (the
32 GB/s of [Islands](../../docs/reference/islands.md) is four page links), and the
product 3,460.8 us, waited for each call (2,927 us an island). Island 1's sampled rows
are computed from the crossed B.

The integer case above (7680 x 64 x 128, 120 row tiles each) is bit for bit on two
cards and on one (by rank, the first version).

"B read across": with `--b-far` B is held by island 0 alone (split over island 0's
ranks), and island 1's grid reads its W pages from island 0's DRAM in the kernels
themselves (the grid writer reads W through `km::read_pages`, which asks island 0 for
the pages it does not hold); no copy is made before the call. Measured on 2026-11-15
at 00:22, the integer case above (7680 x 64 x 128, 121 and 119 row tiles), one page
link (`--m 7680 --k 64 --n 128 --inputs int --b-far --links 1 --kernel-bytes 131072`:
the grid's kernels take 122,752 bytes of kernel buffer with the page links built in,
more than the two-card default of 112 KiB): bit for bit on the 256 sampled rows of
both islands, 36.7 us a call in the loop; island 0 15.2 us, island 1 31.8 us.

## One flat group over two cards (naive)

`flat.py` runs the grid form's kernels, unchanged, on ONE flat group of both cards'
ranks, every tensor held once. The project's rule makes one plan for the islands' grids
stood one below the other (each less the column of its link core: 11 x 20 on two
p150a), so the rule sees one tall grid. "Below", not "beside": rows then stay inside
an island, so A (read by each row's first core) and Y (written by each core) split along
their rows never cross, and W is read only by each column's bottom core, all on the
bottom island; side by side, W and Y would have to be split along their columns and
every column's sender would read W on its own island. A column that spans the islands
is a broadcast an island part: its bottom part's first core sends each block through
the relay to the first core of the part above (`layout.island_broadcast_args`). The
plan sits so island 0's bands end where its half of A's rows ends. With `--hints` the
tensors are placed so no page crosses: A and Y split along their rows over the two
ranks at the islands' edge (half an island), W over the bottom island's ranks; a rect
on one island holds all three there. On one card the same script makes the card's own
plan.

```
python projects/dense/flat.py --m 1024 --k 1024 --n 1024 --cards 0,1 --hints
python projects/dense/flat.py --m 1024 --k 1024 --n 1024 --cards 0
```

Cubes, the loop's host time per call (about four seconds of calls back to back), the
check of `run.py` on 256 sampled rows passed on every row. One card: `bench.py --form
grid` on card 0 (128 to 2048 cubed measured on 2026-11-14 between 23:48 and 23:56; 4096 to
16,384 the results table above). Two cards: `flat.py --cards 0,1`, measured on 2026-11-15
(the 1024 cubed hinted row on 2026-11-14), X280 and card clocks 1,350 MHz, "hints" with
`--hints`, "none" without (the session's default placement). "Crossed" is what
`run.crossings()` counted a call: bytes, page asks, pages and puts of the page calls,
parts relayed by broadcasts; "rate" is those bytes over the call. The hinted 1024 and
2048 rows and the unhinted 4096 to 16,384 rows were measured again on 2026-11-15 between
01:44 and 01:46, once each, after the relay began to write a relayed part straight into
the far rank's L1 (they were 811.3, 3,081.0, 33,930.6, 115,029.1 and 1,607,016.8 us);
the other rows are of the earlier relay. Results: `results/sq*-flat-0x1.json` (hints)
and `results/sq*-flat-0x1-nohint.json`.

| Cube | One card: layout, us, TFLOP/s | Two cards: placement, layout, us, TFLOP/s | Crossed a call, rate | Two / one |
|---|---|---|---|---|
| 128 | cols, 9.3, 0.45 | hints: cols on island 0 (11 x 1), 17.2, 0.24 | nothing | 1.85 |
| 128 | | none: the same plan, 17.4, 0.24 | nothing (all three held by island 0) | 1.87 |
| 256 | cols, 13.4, 2.51 | hints: cols on island 0 (11 x 1), 20.4, 1.64 | nothing | 1.52 |
| 256 | | none: the same plan, 20.4, 1.64 | nothing (all three held by island 0) | 1.53 |
| 512 | 2d, 17.8, 15.07 | hints: cols on island 0 (11 x 2), 53.5, 5.02 | nothing | 3.01 |
| 512 | | none: the same plan, 53.5, 5.02 | nothing (all three held by island 0) | 3.00 |
| 1024 | 2d, 43.0, 49.96 | hints: 2d, 16 x 8 bands of 2 x 4 tiles, 226.9, 9.46 | 2.10 MB: 48 relayed parts; 9.24 GB/s | 5.28 |
| 1024 | | none: the same plan, 3,234.8, 0.66 | 7.37 MB: 320 asks, 2,560 pages, 160 blocks; 2.28 GB/s | 75.2 |
| 2048 | 2d, 172.2, 99.77 | hints: 2d, 16 x 8 bands of 4 x 8 tiles, 738.4, 23.27 | 8.39 MB: 96 relayed parts; 11.36 GB/s | 4.29 |
| 2048 | | none: the same plan, 10,801.7, 1.59 | 29.4 MB: 768 asks, 10,240 pages, 576 blocks; 2.73 GB/s | 62.7 |
| 4096 | 2d, 958.3, 143.4 | hints: none keep A's and Y's pages home (bands of 7 tile rows, the islands' edge at row 64) | | |
| 4096 | | none: 2d, 19 x 11 bands of 7 x 12 tiles, 33,126.4, 4.15 | 105.0 MB: 2,877 asks, 34,048 pages, 12,544 puts, 528 relayed parts; 3.17 GB/s | 34.6 |
| 8192 | 2d, 8,608.3, 127.7 | hints: none keep the pages home (bands of 13 tile rows, edge at 128) | | |
| 8192 | | none: 2d, 20 x 11 bands of 13 x 24 tiles, 51,750.4, 21.25 | 431.9 MB: 3,910 asks, 75,520 pages, 3,584 puts, 4,223 relayed parts; 8.35 GB/s | 6.01 |
| 16,384 | 2d, 87,944.5, 100.0 | hints: none keep the pages home (bands of 26 tile rows, edge at 256) | | |
| 16,384 | | none: 2d, 20 x 11 bands of 26 x 48 tiles, 592,515.7, 14.85 | 5.93 GB: 28,266 asks, 1,003,520 pages, 14,336 puts, 59,133 relayed parts; 10.01 GB/s | 6.74 |

From 1024 cubed up every two-card point is bound by the links: what crosses, over the
call, comes to 3.2 to 11.4 GB/s through one link and its link cores (2.3 to 3.8 GB/s on
the earlier relay). With hints only W's blocks cross (each once, through the column broadcasts'
relay); without, A's pages that island 1's rows read from island 0 cross too, three to
four times the bytes and three to four times the time. 128 to 512 cubed fit on island 0's
cores: nothing crosses, and the default placement holds the small tensors on island 0 as
the hints do. Their two-card time is the eleven columns (512 cubed gets a cols plan of
11 x 2 cores where the twelve-column card takes 2d) and the two-card host loop: one
island's part takes 8.4, 16.9 and 50.5 us of the 17.2, 20.4 and 53.5. The same script on
card 0 alone takes 43.9 us a call at 1024 cubed (48.9 TFLOP/s; single calls 41.0 us
against `bench.py`'s 40.8): the flat plan costs one card nothing but the script's loop.
The 1024 cubed hinted row ran before the host's placement hints reached the tree, with A
and Y in even halves by island and W held by island 1: the placement `--hints` now
gives. From 4096 cubed the rule's bands do not end at the islands' edge, so no split
hint keeps A's rows home and those cubes have only the unhinted row. Over islands the
script's command deadline is three times the call at 2 TFLOP/s (the project's 20 TFLOP/s
gave the 16,384 cubed call 1.37 s, short of its 1.61 s).

## A linear layer written for islands

`linear.py` and `kernels/linear/` are one set of kernels for Y = X x W that runs
unchanged on one island or several: one program, one LINEAR command a call on every
rank of the session. The script places the tensors by hints and submits; it never
names an island, and the kernels hold no island count. X is split along K (its
columns) and W along K (its rows) alike, Y along N (its columns: what the next layer
takes as its X); W is held once. With two islands the hints' group is the last rank of
island 0 and the first of island 1, so each island holds half of each; with one island
every tensor is whole. Every decision in the kernels comes from the tensors' reference
words (the tensor table: which K rows of W and which columns of Y an island holds) and
from `km::island()`, `km::islands()` and the command's group:

- **Each island computes its partial product X_i x W_i (M x N) from what it holds.**
  Its ranks run the grid form above (M x the island's K x one island's columns, its two
  broadcasts inside the island). Nothing of X or W crosses and no page is asked across:
  a row of X is read only where it is held. A broadcast reaches a straight run of
  cores that make the same rows (X) or columns (W), so each group of cores sharing
  broadcasts is a rectangle, but the island need not be one: where the link cores take
  the end of the last row, the plan may cover the ranks with two rectangles sharing one
  block and K block (beside: columns split; below: rows split), each in bands of whole
  blocks, chosen by a cost of blocks a core and the block's shape. The K block is the
  one-card plan's at that size.
- **The columns other islands keep come first, the island's own last.** For each
  other island's columns each core packs its output blocks in bfloat16 (the partial
  rounded once) into a buffer of two chunks (a chunk: the fewest whole subblocks of at
  least 64 KiB) and puts each chunk to that island as a run (`km::put_pages` with a
  `km::PutRun`), writing its tiles into the link cores' rings (`km::put_piece`): the
  chunk is free on return, while compute fills the other. Then one sync: the puts
  counted (`km::put_wait`, once), a barrier (a binary tree of tokens inside the island, its leader to the
  other leaders through `km::to_island(i).put`, the tree back down). Then each core
  makes its blocks of its own island's columns, and for each the tiles the other
  islands put for it join DST before the K block they are taken at (the reader reads
  them from the island's own DRAM); the sum is written to Y here.
- **Where the put tiles land: S**, a scratch tensor on every island (`copy=True`),
  laid out block by block: a set a sending island, in a set each core's output blocks
  in the order it makes them (its first page a core argument), a block's tiles in the
  order compute packs them. The plan is the same on every island, so both sides name
  the same pages, and a core's chunks are consecutive pages; a block wholly past the
  island's columns is not put.

So two islands cross M x N / 2 tiles each way, one M x N bf16 matrix in all, once a
call (whole blocks: a block past Y's edge crosses its padding too).

On one island every shard is the whole: the other islands' columns are an empty range,
there is no sync, and the kernels run the grid form and write Y. The first version of
the kernels beside the project's own grid kernels (`kernels/grid`), the same plan in
the same session (card 0, `--beside`):

| Cube | Kernels | Text, B (BRISC / NCRISC / TRISC 0, 1, 2) | Kernel buffer, B | Single call, us | Loop, us per call | TFLOP/s | AICLK min / mean, MHz |
|---|---|---|---|---|---|---|---|
| 1024 | linear | 26,348 / 26,340 / 13,288, 5,588, 6,264 | 78,848 | 40.4 to 41.6 | 43.76 | 49.07 | 1343 / 1349 |
| 8192 | linear | 26,348 / 26,340 / 13,288, 5,588, 6,264 | 78,848 | 5,757 to 5,805 | 12,506.7 | 87.91 | 800 / 1048 |
| 8192 | grid (the project's) | 20,168 / 18,796 / 11,976, 4,436, 5,540 | 61,936 | 5,780 to 5,807 | 12,802.5 | 85.88 | 800 / 1020 |

```
python projects/dense/linear.py --cube 8192 --cards 0 --kernel-bytes 81920 --beside
```

Measured on 2026-11-14 at 23:30 and 23:34, one run each (three single calls, then
calls back to back for about three seconds; at 8192 the grid kernels after a
three-second cool-down); X280 at 1,350 MHz. Both checks passed (relative RMS 3.96e-3 to
3.97e-3, largest 5.3e-3 to 6.5e-3). The 8192 loops ran with the AI clock falling to
800 MHz (the slow mode of the results above; the best loop there is 8,608.3 us); in
the single calls the two sets of kernels take the same time. The linear kernels need
`--kernel-bytes 81920` (their text is 78,848 B against the default 70,656). JSON lines:
`results/linear-<cube>-0.json`. These two runs are of the first version of the
kernels, whose form came from the island's own part of W alone; the curve below is of
the version described above, whose text on one card is 87,888 B (`--kernel-bytes
90112`).

### One card and two, 128 to 16384 cubed

The same script and kernels on card 0 and on cards 0 and 1, one run a point (three
single calls, then calls back to back for about three seconds), on 2026-11-15 between
05:15 and 05:38 (the K form's rows after 05:33, after the link cores' chunks stopped
counting the ranks' buffers), host load 5 to 8; every check passed (relative RMS 3.94e-3 to 3.96e-3
on one card and in the rows form, 4.12e-3 to 4.14e-3 in the K form: the partial
rounded to bf16 once). Two cards, K form: X and W split along K, Y along N, W held once
(`--place k`, the default). Two cards, rows form: X and Y split along M, W a copy on
each card (`--place rows`): the replicated weight, nothing crosses. Microseconds, the
median single call / the loop's time a call; "x" is one card's single over two cards'
(above 1: two cards are faster).

| Cube | One card | Two cards, K form (W sharded) | x | Two cards, rows form (W copied) | x | K form: puts / sync / own, us | K form: crossed a call, rate | Kohaku-TT: one card / two cards (W copied) / two cards (K split) |
|---|---|---|---|---|---|---|---|---|
| 128 | 7.2 / 16.1 | 72.7 / 39.5 | 0.10 | 7.6 / 41.3 | 0.95 | 10 / 61 / 0 | 33 KB | |
| 256 | 11.6 / 49.3 | 79.0 / 47.1 | 0.15 | 10.0 / 24.3 | 1.16 | 13 / 63 / 2 | 132 KB | |
| 512 | 15.9 / 20.3 | 179.8 / 171.1 | 0.09 | 19.8 / 25.6 | 0.80 | 15 / 158 / 5 | 0.53 MB | |
| 1024 | 43.8 / 47.2 | 154.8 / 143.3 | 0.28 | 40.6 / 46.6 | 1.08 | 48 / 85 / 19 | 2.1 MB, 14.7 GB/s | |
| 2048 | 167.1 / 195.6 | 356.0 / 405.4 | 0.47 | 125.4 / 134.3 | 1.33 | 131 / 150 / 75 | 9.2 MB, 22.7 GB/s | |
| 4096 | 655.0 (a) | 1,312.9 / 1,310.9 | 0.50 | 420.7 / 543.9 | 1.56 | 743 / 337 / 310 | 34.1 MB, 26.0 GB/s | 819.8 / 487.3 / 868.8 |
| 8192 | 4,488.1 (a) | 4,992.6 / 5,474.4 | 0.90 | 2,357.6 (b) | 1.90 | 3,270 / 331 / 1,483 | 144.0 MB, 26.3 GB/s | 4,549.5 / 2,482.8 / 3,930.1 |
| 16384 | 34,544.8 (a) | 27,151.6 / 35,610.2 | 1.27 | 17,731.5 (b) | 1.95 | 16,463 / 248 / 10,542 | 565.1 MB, 15.9 GB/s | 32,489.3 / 17,067.8 / 32,970.6 |

(a) single calls of the one-card table above (`results/one-card-*.json`), not run
again; (b) single calls of the rows form below (`results/linear-rows-*.json`, an earlier
head; the rows form puts nothing), not run again. Kohaku-TT: the table "Shapes, one card and two" below, its default two-card form
(rows split, W all-gathered once to a kept copy on each chip) and its K split. The
K form's passes are those of the second single call, the slower island's slowest core,
at 1,350 MHz: the other island's columns made and put, the sync (the puts counted, the
barrier), the own columns. Crossed is `run.crossings()` over the loop, both ways: the
pages are exactly the count from the placements and the plan (`Linear.crossed`, whole
blocks) plus the command's two relayed parts; the rate is over the loop's call. The
loops from 4096 up lowered the AI clock (one card 1,162 MHz at 2048; two cards 950 to
1,025 at 4096 rows, 1,137 to 1,187 at 8192 K, 806 to 925 at 16384 K).

Where two cards win: the rows form from 2048 up (1.33 to 1.95; Kohaku-TT's default
form 1.68 to 1.90 at 4096 to 16384); the K form only at 16384 (1.27; Kohaku-TT's K
split 0.94 to 1.16 at 4096 to 16384), crossing over between 8192 (0.90) and 16384. Below 2048 a two-card call is the command over
two islands and the sync (21 to 160 us). The K form's loss, from the passes and the
writers' counters (T's words 7 to 12, `Linear.passes()`):

- **The crossing.** The puts pass makes the same amount of work as the own pass but
  takes 3,270 against 1,483 us at 8192 (16,463 against 10,542 at 16384). A writer
  spends 340 to 470 us of it in `km::put_piece` (1,070 to 1,480 at 16384), most of
  that waiting for its ring's slots. The rest is not counted: the writer also sends W's
  column broadcast, so its puts may delay the cores it feeds (inferred).
- **The plan** is the whole budget's (8192: 14 x 12, K block 4, on 105 of the island's
  118 ranks): the own pass takes 1,483 us, where a quarter of a one-card call is 1,122.
- **The sync**: 40 to 1,290 us from 1024 up in single calls (mostly 90 to 400), of it
  each core's wait for its last confirmations 21 to 265 us.

The link cores' rings for the puts need a chunk of 192 KiB a link CPU or more (8
slots of an 8 KiB packet). The session now sizes it without the program's buffers,
which are on the ranks only: 512 KiB, 31 slots.
Before this, with line puts (one 64 KiB put in flight a core, no rings, the whole
budget), the K form took 10,486 / 45,753 us at 8192 / 16384, its puts at 8.8 GB/s each
way.
JSON lines: `results/linear-ring-<cube>-<0|0x1|0x1-rows>.json`.

```
python projects/dense/linear.py --cube 16384 --cards 0
python projects/dense/linear.py --cube 16384 --cards 0,1
python projects/dense/linear.py --cube 4096 --cards 0,1 --place rows
```

### The rows form: W replicated (Kohaku-TT's default form)

`--place rows` places X and Y split along M (each island half the rows) and W as a
copy on every island: the replicated weight, Kohaku-TT's default two-card form, not
this kernel's weight-sharded answer (`--place k`, above). The same kernels take it
from the placements alone: an island makes the rows of X it holds, no other island
keeps columns of those rows, so the puts' pass is empty and there is no sync; each
island runs the product of its rows with its own plan. Nothing crosses in a call (the
crossings count 0 bytes). Against it, the half product (M/2 x K x N) on card 0 with a
one-card session, the same kernels; single calls, medians of three, on 2026-11-15 at
02:45, every check passed. The island's and the call's times are each island's part
by its dispatcher's ticks (the call: the later island); the product is the kernels'
span (the first core taking the command to the last ending, the writers' clocks);
host is the host's time from submit to the event.

| Shape (M x K x N) | Half on one card: plan, part / product / host, us | Two cards, an island: plan; island 0, 1 part, us | Two-card call / host, us | Call / half |
|---|---|---|---|---|
| 4096 x 32768 x 4096 | 8 x 11, K block 8, 12 x 8 cores; 2,816.6 / 2,811.1 / 3,189.4 | the same block, 10 x 8 + 2 x 8 cores; 2,877.5, 2,890.2 | 2,916.5 / 3,381.0 | 1.035 |
| 8192 cubed | 13 x 12, K block 4, 11 x 10; 2,350.1 / 2,346.1 / 2,669.9 | the same block, 10 x 10 + 2 x 5; 2,357.6, 2,357.4 | 2,357.6 / 2,873.6 | 1.003 |
| 16384 cubed | 13 x 16, K block 4, 11 x 10; 17,725.5 / 17,696.4 / 18,067.2 | 13 x 16, K block 2, 10 x 10 + 2 x 7; 20,208.4, 20,309.6 | 20,674.5 / 21,267.5 | 1.166 |

The differences, named: at 16384 the plan: an island's buffers may use 1,312,176 B
(the two-card session keeps 49,728 B in every core, a host transfer buffer of 32,848
and a 96 KiB kernel buffer) where one card's use 1,380,000, and 13 x 16 blocks with
K block 4 need 1,369,216, so the island takes K block 2: its product is 2.4 to 3.3 ms
longer; at 4096 x 32768 x 4096 the island's two rectangles where one 12 x 8 would
fit (the picker's tie), 2 % longer products. The command's way in and its completion
on an island (part less product) are 3 to 35 us on either side, the same on one card
and two. The host's time from submit to the event is 320 to 340 us past the part on
one card and 470 to 520 on two. JSON lines: `results/linear-half-<shape>.json`,
`results/linear-rows-<shape>.json`.

```
python projects/dense/linear.py --dims 4096,32768,4096 --place rows --cards 0,1 --seconds 0
python projects/dense/linear.py --dims 2048,32768,4096 --cards 0 --seconds 0
```

### Measured before: W split along N

The earlier form with W split along N (Y along N, X held by island 0 and read across
by every row sender in small asks), measured on 2026-11-15 at about 01:00 and not run
again: 4096 cubed 23,342.2 us a call (67.6 MB crossed, 2.90 GB/s), 8192 110,614.0 us
(281.1 MB, 2.54 GB/s), 16384 473,437.4 us (1.210 GB, 2.56 GB/s); one card 1,475.2,
8,906.5 and 90,030.6 us then. JSON lines: `results/linear-curve-<cube>-0x1-n.json`.

## Shapes, one card and two, against Kohaku-TT

Y = X x W in bfloat16 over fifteen shapes, one card and two cards, ours and
[Kohaku-TT](https://github.com/KohakuBlueleaf/Kohaku-TT) (commit ac81ff2) on the same
inputs and the same check (relative RMS 3.96e-3 one card, 4.12e-3 to 4.14e-3 two cards;
Kohaku-TT 3.96e-3 and, split along K, 3.86e-3; every point exact to that bound). This is the first answer, on
the kernels and the put path of the one-exchange linear kernel above (main at 6b74b4a2);
the sweep is to be run once more when the plan and the put path change
(`compare/shapes.py` lists the shapes, `compare/ktt_shapes.py` runs Kohaku-TT's side,
`linear.py --m --k --n` ours).

**How Kohaku-TT's two cards work** (from its source and documentation, and the placement
log of the runs below):
- A tensor of 1 MiB or more is created split along its first dimension, so X is split by
  rows and W along K. For `x @ w` its rule engine (cheapest conversion by a cost model)
  chose, in every run here but one (below): gather W to every chip once (an all-gather on the first call;
  the copy is kept on the tensor and reused), each chip multiplies its half of the rows
  of X by the whole W, Y is split by rows. Nothing crosses the chips in a call after the
  first: no bytes, no sync, W held twice per chip. This is the column "Kohaku-TT 2 cards".
  The exception is 4096 x 16384 x 16384 (W of 512 MiB): there the engine split K, moved X
  by an all-to-all once, and left Y a partial sum on each chip, unreduced (its 4,936 us has
  no completion in it; the K-split column, 6,077 us, has).
- The form with our layout (X and W split along K, the partial completed to Y split along N;
  "K split" here, marked by `kohakutt.shard`): each chip's matmul gives a bf16 M x N partial; the
  completion is TT-NN's reduce-scatter over the Ethernet fabric (M x N / 2 x 2 bytes each way a
  call), a second program queued after the matmul on the same command queue: nothing overlaps,
  the program boundary is the sync. Its own collectives (4 links, one sender core under each
  Ethernet tile, 16 to 48 drain cores adding in fp32, two bf16 tiles a packet) serve fp32
  partials and the all-reduce of a replicated result, not this one.
- Each chip's matmul: `ktt_gemm` (10 x 11 = 110 cores, 2D multicast layout: A along core rows,
  W down columns on the other NoC, blocks of bm x bn tiles, K blocks of up to 16 tiles,
  fp32 accumulation, HiFi2) where its rule picks it (here 4096 cubed, 4096 x 16384 x 16384,
  16384 x 4096 x 16384 and the halves of those), TT-NN's matmul elsewhere.
- Its own split (documented, 2048 rows, 4096 to 16384 to 4096, per block): matmuls 713 and
  649 us, all-reduce and residual 288 us; four blocks 8.1 ms on two chips against 13.4 ms on one.
- The difference from ours: its default form moves nothing in a call (the weight is
  replicated once), ours crosses the whole M x N matrix a call by design (X and W stay split);
  in the K-split form it completes after the matmul, ours puts blocks while it computes
  the other island's columns and syncs once.

Times in microseconds, "single" the median of synced calls and "loop" calls back to back for
about three seconds (a cool-down of the same length between points), one run each; ratios are
one card over two cards on the single calls, the loops' in brackets (a one-card loop is
throttled by the card's power limit, so the loop ratio flatters two cards). Ours: three
single calls. Kohaku-TT: one card three calls; on two cards the steady median of nine calls
(the first calls after the first product are slow and the median of three had caught them:
2048 x 8192 x 2048 gave 747 / 1,021 / 315 us), and its loop of the first run. On two cards
its synced call has a floor of about 1.5 to 3 ms in some runs (2048 x 16384 x 2048: 3,000 us in a
run of nine calls, 508 us in a run of three, 851 us a call in the loop): below about 5 ms
the two-card single is the synchronization, not the product, and its loop is the better figure.

| M x K x N | Ours 1 card | Ours 2 cards | Ours ratio | Kohaku-TT 1 card | Kohaku-TT 2 cards | ratio | Kohaku-TT 2 cards, K split | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 8192x65536x8192 | 43,698.4 / 107,034.9 | 46,382.9 / 52,652.9 | 0.94 (2.03) | 34,544.0 / 95,826.2 | 20,452.8 / 58,286.5 | 1.69 (1.64) | 18,357.9 / 48,696.6 | 1.88 (1.97) |
| 4096x32768x4096 | 4,421.3 / 12,210.0 | 4,321.9 / 4,961.1 | 1.02 (2.46) | 5,393.5 / 14,906.9 | 2,794.7 / 7,427.0 | 1.93 (2.01) | 3,152.8 / 7,446.3 | 1.71 (2.00) |
| 2048x16384x2048 | 883.9 / 1,097.0 | 1,425.2 / 1,437.1 | 0.62 (0.76) | 792.2 / 1,823.0 | 2,999.7 / 851.3 | 0.26 (2.14) | 2,997.6 / 765.0 | 0.26 (2.38) |
| 8192x32768x8192 | 22,034.0 / 52,647.2 | 24,583.7 / 27,118.2 | 0.90 (1.94) | 17,331.4 / 47,951.6 | 10,324.4 / 29,266.7 | 1.68 (1.64) | 9,889.3 / 24,675.5 | 1.75 (1.94) |
| 4096x16384x4096 | 2,271.0 / 6,232.9 | 2,706.6 / 2,927.4 | 0.84 (2.13) | 2,824.0 / 7,365.9 | 1,503.3 / 1,435.8 | 1.88 (5.13) | 1,842.5 / 3,525.2 | 1.53 (2.09) |
| 2048x8192x2048 | 471.4 / 576.0 | 1,013.1 / 1,009.2 | 0.47 (0.57) | 472.7 / 872.0 | 317.0 / 197.5 | 1.49 (4.42) | 476.3 / 269.2 | 0.99 (3.24) |
| 4096x16384x16384 | 23,388.8 / 27,295.1 | 13,720.9 / 14,515.8 | 1.70 (1.88) | 8,955.3 / 23,754.6 | 4,936.4 / 12,588.7 | 1.81 (1.89) | 6,076.8 / 12,774.3 | 1.47 (1.86) |
| 16384x16384x16384 | 75,169.4 / 90,231.8 | 49,002.8 / 52,854.2 | 1.53 (1.71) | 32,489.3 / 95,676.9 | 17,067.8 / 47,303.6 | 1.90 (2.02) | 32,970.6 / 43,996.1 | 0.99 (2.17) |
| 8192x16384x8192 | 11,161.3 / 25,719.0 | 13,501.0 / 14,630.1 | 0.83 (1.76) | 8,777.9 / 24,630.6 | 5,257.1 / 14,485.0 | 1.67 (1.70) | 5,672.0 / 11,923.8 | 1.55 (2.07) |
| 4096x8192x4096 | 1,199.8 / 2,961.6 | 1,975.1 / 1,970.2 | 0.61 (1.50) | 1,551.6 / 3,551.8 | 865.1 / 1,745.2 | 1.79 (2.04) | 1,113.8 / 1,190.4 | 1.39 (2.98) |
| 2048x4096x2048 | 248.3 / 307.8 | 815.2 / 805.1 | 0.30 (0.38) | 311.0 / 387.8 | 255.9 / 109.2 | 1.22 (3.55) | 331.5 / 190.5 | 0.94 (2.04) |
| 8192x8192x8192 | 5,716.0 / 12,463.3 | 8,145.2 / 8,478.3 | 0.70 (1.47) | 4,549.5 / 12,094.0 | 2,482.8 / 5,905.9 | 1.83 (2.05) | 3,930.1 / 4,796.2 | 1.16 (2.52) |
| 4096x4096x4096 | 664.6 / 1,539.5 | 1,611.6 / 1,601.2 | 0.41 (0.96) | 819.8 / 1,531.3 | 487.3 / 345.1 | 1.68 (4.44) | 868.8 / 679.7 | 0.94 (2.25) |
| 8192x2048x8192 | 1,713.5 / 2,358.1 | 4,268.4 / 4,260.6 | 0.40 (0.55) | 1,725.8 / 3,010.0 | 879.0 / 778.1 | 1.96 (3.87) | 2,164.0 / 2,103.1 | 0.80 (1.43) |
| 16384x4096x16384 | 19,557.4 / 22,825.5 | 25,907.5 / 25,922.0 | 0.75 (0.88) | 14,782.2 / 20,692.6 | 7,784.0 / 10,721.3 | 1.90 (1.93) | 12,240.6 / 14,096.5 | 1.21 (1.47) |

Our two-card passes (the slower island's slowest core, ms at 1,350 MHz: the other island's
columns made and put; the sync; its own columns), the plan, the bytes crossed and the card
clock of each loop:

| M x K x N | Our plan (rows x cols, kb) | Crossed a call, MB | Put | Sync | Own | AICLK mean, MHz: ours 1, ours 2, Kohaku-TT 1, Kohaku-TT 2 |
|---|---|---|---|---|---|---|
| 8192x65536x8192 | 9 x 11, 1 | 146.0 | 23.07 | 6.24 | 21.82 | 1,023 / 1,165 / 1,020 / 1,030 |
| 4096x32768x4096 | 10 x 8, 4 | 34.1 | 2.12 | 1.13 | 1.59 | 1,051 / 1,143 / 1,010 / 1,010 |
| 2048x16384x2048 | 10 x 8, 16 | 9.2 | 0.73 | 0.29 | 0.44 | 1,081 / 1,349 / 1,008 / 997 |
| 8192x32768x8192 | 9 x 11, 1 | 146.0 | 12.55 | 2.29 | 11.07 | 1,030 / 1,208 / 1,021 / 1,026 |
| 4096x16384x4096 | 10 x 8, 4 | 34.1 | 1.37 | 0.79 | 0.84 | 1,017 / 1,240 / 1,014 / 1,299 |
| 2048x8192x2048 | 10 x 8, 16 | 9.2 | 0.52 | 0.25 | 0.22 | 1,079 / 1,350 / 1,010 / 1,298 |
| 4096x16384x16384 | 9 x 11, 4 | 147.1 | 7.09 | 1.55 | 6.13 | 1,150 / 1,273 / 1,037 / 1,016 |
| 16384x16384x16384 | 9 x 11, 4 | 579.6 | 27.00 | 2.92 | 23.11 | 1,115 / 1,254 / 1,012 / 1,016 |
| 8192x16384x8192 | 9 x 11, 1 | 146.0 | 7.18 | 1.69 | 5.68 | 1,011 / 1,239 / 1,003 / 1,032 |
| 4096x8192x4096 | 10 x 8, 4 | 34.1 | 1.03 | 0.47 | 0.47 | 1,048 / 1,344 / 1,025 / 1,015 |
| 2048x4096x2048 | 10 x 8, 16 | 9.2 | 0.43 | 0.24 | 0.12 | 1,093 / 1,349 / 1,003 / 1,350 |
| 8192x8192x8192 | 9 x 11, 1 | 146.0 | 4.54 | 1.16 | 3.01 | 1,023 / 1,289 / 1,013 / 1,022 |
| 4096x4096x4096 | 10 x 8, 4 | 34.1 | 0.85 | 0.45 | 0.29 | 1,037 / 1,349 / 1,018 / 1,344 |
| 8192x2048x8192 | 9 x 11, 1 | 146.0 | 2.76 | 0.48 | 1.01 | 981 / 1,350 / 999 / 1,314 |
| 16384x4096x16384 | 9 x 11, 4 | 579.6 | 19.15 | 0.42 | 6.32 | 1,154 / 1,350 / 951 / 958 |

Measured on 2026-11-15 between 02:11 and 03:08, host load average 4 to 11, X280 at 1,350 MHz; JSON in `results/shape-<M>x<K>x<N>-<ours|ktt>-<0|0x1>[-<variant>].json`
(the cubes of ours are the curve's above; the nine-call runs are in `results/singles9/`).
Ours: `python projects/dense/linear.py --m M --k K --n N --cards 0|0,1 --seconds 3`. Kohaku-TT:
`python projects/dense/compare/ktt_shapes.py --cards 0|0,1 --shapes MxKxN,... --variants default[,ksplit]`.

The best two-card ratio on single calls is 1.70 at 4096 x 16384 x 16384 (16384 cubed 1.53) for
ours and 1.96 at 8192 x 2048 x 8192 for Kohaku-TT's row split (1.93 at 4096 x 32768 x 4096; its
K split 1.88 at 8192 x 65536 x 8192), and our ratio grows with K at fixed M = N (M = N = 4096: 0.41, 0.61, 0.84,
1.02 for K = 4096 to 32768; M = N = 8192: 0.40 to 0.94 for K = 2048 to 65536) without
reaching 1.5 except where N is large. We do not beat Kohaku-TT's row split in ratio or in
two-card single time at any shape, except 2048 x 16384 x 2048 where its call sat on the
3 ms floor (and 4096 x 16384 x 16384, where its row of 1.81 leaves Y unreduced); against its K split (the same
exchange) we beat the ratio at 4096 x 16384 x 16384 (1.70 against 1.47), 16384 cubed (1.53
against 0.99) and 2048 x 16384 x 2048 (floor) only; in absolute
single time our two cards beat its one card at two shapes only (4096 x 32768 x 4096, 4096 x 16384 x 4096),
and in the loop our two cards beat its row split at 8192 x 65536 x 8192 (52.7 against 58.3 ms),
8192 x 32768 x 8192 and 4096 x 32768 x 4096 (5.0 against 7.4 ms), where our two-card clock averages 1,143 to
1,208 MHz against its 1,010 to 1,030.
Our rows below 1.0 are bound by what the passes show: with K small the product is too short
for the crossing (at 8192 x 2048 x 8192 an island puts 73 MB in 2.76 ms, 26 GB/s, against a
one-card call of 1.71 ms), at M = N = 2048 the put (0.43 to 0.73 ms) and a 0.24 to 0.29 ms sync are the call, and with large K at M = N = 8192
the plan has K blocks of one tile (the put buffer leaves little L1), so each pass takes
about 22 ms where a quarter of a one-card call is 11 ms; at 4096 x 32768 x 4096 the island's plan is
10 x 8 = 80 cores of 118 and its puts run at 8 GB/s.

## One card against Kohaku-TT's one card

The table above had Kohaku-TT's one card ahead at most shapes. Its source says which
kernel ran: a bf16 `x @ w` goes to its own block matmul `ktt_gemm` only for M up to 256,
N > K, or N = K >= 4096 with M from 1024 to 4096; everywhere else to TT-NN's
`minimal_matmul`. So of the five shapes below, 4096 cubed and 4096 x 16384 x 16384 are
`ktt_gemm`'s and the other three `minimal_matmul`'s.

`ktt_gemm` is the form the linear kernel already has on one island: 11 x 10 cores, X
multicast along core rows inside a panel and W down core columns, each on its own
data-movement core and NoC, blocks of bm x bn tiles, K blocks of up to 16 tiles, fp32
partials added by the packer in L1, the last K block reloading them, HiFi2, bf16 out,
a block's Y written after the next block's first K block. Every piece is said with
`km::Broadcast`, the grid plan and the tensor accessors, and so are the two it once
lacked, each an option of `linear.py`:

- which NoC each data-movement core uses (`--nocs 1,0`, `km.Program(noc=...)`): the
  reader (RISCV_0, X) on NoC 1 and the writer (RISCV_1, W and Y) on NoC 0, as Kohaku-TT
  places them, the broadcasts' senders and directions following. Exact, and slower:
  4,978.6 us at 8192 cubed, 36,489.1 at 16384 cubed;
- the VALID word after a block's multicast once it is sent, not acknowledged
  (`--valid sent`, `km::Broadcast::valid_when_sent`, see the broadcast reference for
  what it rests on). Exact, and no faster: 4,480.4 and 34,522.4 us. Both options at
  once: 4,959.4 and 36,430.2 us.

The plan picker keeps the defaults (NoC 0 for the reader, acknowledged VALIDs).

What set our time was the plan and the compute, measured one change at a time (single
calls, card 0, median of three):

| Change | 8192 cubed | 16384 cubed |
|---|---|---|
| before (the picker's plans at 3bc7f8a8): 26 x 8 blocks, K block 2 / 13 x 16, K block 4 | 5,974.5 | 35,401.8 |
| the plan's cost model: 13 x 12 blocks, K block 4 | 4,660.6 | |
| K blocks of any size: K block 6 | 4,586.0 | 35,229.5 |
| the partials added in L1 by every K block | **4,488.1** | **34,544.8** |
| then the reader on NoC 1, the writer on NoC 0 (`--nocs 1,0`) | 4,978.6 | 36,489.1 |
| then the VALID once sent (`--valid sent`), NoCs as before | 4,480.4 | 34,522.4 |

The last two rows: 2026-11-15 05:19 to 05:24, host load 4.8 to 6.9, each the median of
three single calls before a 3 s loop whose last output was checked (AICLK, sampled in
the loops, fell to 800 MHz there); `linear.py --cube N --cards 0 --nocs 1,0` and
`--valid sent`, each form in one session a cube.

- **The plan.** A K block now costs the larger of its compute (24.5 ns a tile product, 51
  ns a subblock's turn) and its slower feed (4.75 us a block, then 203 ns a tile of the
  larger of the X and W blocks, which one core reads and multicasts one after the other),
  fitted to the plans measured (within 1 % at 13 x 12 and 13 x 16, K block 4, at 26 x 8,
  K block 2 and at 4 x 44, K block 2; 9 % high at 7 x 24, K block 4). The old model gave
  26 x 8 at K block 2 and 13 x 12 at K block 4 the same cost at 8192 cubed; with the feed
  counted, 26 x 8 (26 tiles of X a K tile from one core) is fed, 13 x 12 is computed. On
  one island any K block up to 16 is tried (the last of a call shorter): 13 x 12 takes K
  block 6, the most L1 allows. Each core's broadcast wait counters (in the run report)
  show it: at 13 x 12 the row and column senders wait for their receivers 30 to 64 % of
  the call. On two islands the same model now keeps 4096 x 32768 x 4096's half on one
  rectangle where it fits (15 x 8 on 8 x 9 cores, computed for 114 ranks and 1,361,264 B)
  instead of two; the two-island plans are computed, not yet run.
- **The partials.** `ktt_gemm` and our kernel popped the fp32 partials at every K
  block's end so the next K block's packs could reserve them again: unpack waited there
  for pack's last subblock, every K block. As in TT-NN's `minimal_matmul`, the partials of
  a block are now reserved once, every K block's products are packed out of order onto
  them (added in L1 from the second K block on), and after the last a pass copies them to
  bf16 a subblock at a time. No file was copied from either project.
- **Not the cause:** the subblock's shape (1 x 4, 4 x 1 and 2 x 2 within 0.7 % on one
  plan) and the depth of Y's buffer (48 tiles against 2: 0.1 %).

The five shapes at this head against Kohaku-TT's one card (its column of the table
above), microseconds:

| M x K x N | Plan (11 x 10 cores) | Ours, single | Kohaku-TT, single | Ours / Kohaku-TT |
|---|---|---|---|---|
| 4096 cubed | 13 x 12, K block 6 | 655.0 | 819.8 (`ktt_gemm`) | 0.80 |
| 8192 cubed | 13 x 12, 2 x 2 blocks, K block 6 | 4,488.1 | 4,549.5 (`minimal_matmul`) | 0.99 |
| 16384 cubed | 13 x 12, 4 x 4 blocks, K block 6 | 34,544.8 | 32,489.3 (`minimal_matmul`) | 1.06 |
| 8192 x 65536 x 8192 | 13 x 12, 2 x 2 blocks, K block 6 | 33,411.5 | 34,544.0 (`minimal_matmul`) | 0.97 |
| 4096 x 16384 x 16384 | 13 x 12, 1 x 4 blocks, K block 6 | 8,666.4 | 8,955.3 (`ktt_gemm`) | 0.97 |

All exact to the same bound (relative RMS 3.96e-3 to 3.97e-3). 16384 cubed stays 6 %
slower: a core makes 52 x 48 tiles (the 11 x 10 bands of 512 tiles), 4.7 % more than an
even share, at 27.0 ns a product; 8192 x 65536 x 8192, the same tiles a core in 4 blocks
instead of 16, gives 26.1 ns, so about 74 us go at each block's end. `minimal_matmul`'s
8 x 8 blocks share the tiles more evenly; blocks that small are fed here, not computed
(the broadcast's 4.75 us a block).

**What only this model allows: the twelfth column.** Kohaku-TT's and TT-NN's grid on
this card is 11 x 10, a column of Tensix cores runs their dispatcher; here the X280
dispatches and all 12 x 10 compute. The powers of two above do not split into 12 column
bands with a 4-tile subblock, so the picker leaves them on 11 columns; a shape that does,
3840 x 32768 x 4608, with the same kernels and the picker on 12 columns (12 x 12 blocks,
K block 7) and on 11 (12 x 14, K block 6): 3,943.0 against 4,589.3 us, 1.16x from the
column (`--columns 11`).

Measured on 2026-11-15 between 03:59 and 04:26; `python projects/dense/linear.py --dims
M,K,N --cards 0 --seconds 0`; JSON in `results/one-card-*.json`.

## Files

| File | What it is |
|---|---|
| `kernels/dense.h` | handlers, command words, buffers, profile words, the timed NoC barrier |
| `kernels/reader.cpp`, `compute.cpp`, `writer.cpp` | the three kernels |
| `plan.py` | the plan as pure functions |
| `card.py` | the program and its commands on a session |
| `kernels/grid/grid.h`, `reader.cpp`, `compute.cpp`, `writer.cpp` | the broadcast form: arguments, row broadcast and A, the product, column broadcast and Y |
| `kernels/split/split.h`, `chunks.hpp`, `reader.cpp`, `compute.cpp`, `writer.cpp` | the K split: arguments, the K chunks as two streams on both movers, chunks and arrivals, the part and the sum, chunks and sending and Y |
| `run.py`, `bench.py` | one shape run and checked; one shape measured |
| `islands.py` | the grid form over the islands of a session, A and Y placed by island, W a copy on each |
| `linear.py`, `kernels/linear/linear.h`, `reader.cpp`, `compute.cpp`, `writer.cpp` | the linear layer written for islands: K split, the other islands' columns put whole, one sync, the own columns with theirs added |
| `flat.py` | the grid form, kernels unchanged, on one flat group over the islands: one plan for the islands' grids one below the other, every tensor held once |
| `results/` | the runs behind the results table |
| `compare/shapes.py`, `compare/ktt_shapes.py` | the shapes of the comparison with Kohaku-TT on one card and two, their table, and Kohaku-TT's side of it |
| `reference/ktt_bench.py` | Kohaku-TT's product on the same inputs, timed and checked the same way |
| `bench_ttnn.py` | TT-NN's product and SwiGLU on the same inputs; it imports `ttnn` and `torch` from the tt-metal tree `TT_METAL_HOME` names (Komorebi needs neither) |
| `reference/gpu_bench.py`, `reference/gpu/gemm_gpu.cu` | cuBLAS on one idle GPU (fp32 and fp16) |
