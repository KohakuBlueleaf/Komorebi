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

A and Y are placed by island (`by="island"`: even parts of the row tiles, island i
its part i); W is a copy on every island. Each island's program is the grid form of
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
| `sq8192` | 0, 1 | 128 / 128 | 4,269.2 | 257.5 | 2,931 | 925 / 980 | 1.23 | 0.27 |
| `sq8192`, B moved once a call | 0, 1 | 128 / 128 | 27,196.9 | 40.4 | 2,942 | 1350 / 1350 | 1.12 | 0.27 |
| `sq8192`, by rank | 0 | 256 | 13,192.0 | 83.3 | 5,717 | 1034 | 0.85 | 0.28 |
| `sq8192`, by rank | 0, 1 | 136 / 120 | 5,242.3 | 209.7 | 4,686 | 970 / 1208 | 1.12 | 0.27 |
| `sq16384`, by rank | 0 | 512 | 91,763.8 | 95.9 | 77,693 | 1138 | 10.78 | 0.29 |
| `sq16384`, by rank | 0, 1 | 272 / 240 | 48,306.7 | 182.1 | 42,259 | 1170 / 1173 | 3.99 | 0.28 |

Every check passed (relative RMS 3.96e-3 to 3.97e-3, largest 5.6e-3 to 6.5e-3).
Placed by island, both islands of `sq8192` get 128 row tiles and the same plan (bands
of 13 x 24, blocks of 7 x 24, K blocks of 4): 4,269.2 us per call, 0.50 of the best
one-card run (8,608.3 above) and 0.81 of the by-rank split, whose island 1 got blocks
of 12 x 24 with K blocks of 1 tile and was the slower part. Both one-card `sq8192` loops
ran in the slower of the two modes described above (13,192 and 13,562 us): the same
plan and kernels as `bench.py`, which runs 21 single calls, then waits 1 s and runs
its loop under the power sampler, where this script runs 3 single calls and its loop
at once; `bench.py` itself ran in that mode in two of its three runs. `sq16384` on two
cards takes 0.53 of one card's time per call.

"B moved once a call": B (the weight, 128 MiB) is held by island 0 alone (`like` its
ranks) and before each call island 0's link cores send it into the same pages of
island 1's DRAM (`send_pages`, waited for on island 1's link core), then one GRID
command an island runs as above, waited for
(`--links 2 --kernel-bytes 90112 --b-moved`; the page links' code needs the larger
kernel buffer). Of the 27,196.9 us a call the crossing takes 23,709.5 us, 5.66 GB/s:
with two links, link 0 carries the commands across and link 1 alone the pages (the
32 GB/s of [Islands](../../docs/reference/islands.md) is four page links), and the
product 3,460.8 us, waited for each call (2,927 us an island). Island 1's sampled rows
are computed from the crossed B.

The integer case above (7680 x 64 x 128, 120 row tiles each) is bit for bit on two
cards and on one (by rank, the first version).

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
| `results/` | the runs behind the results table |
| `reference/ktt_bench.py` | Kohaku-TT's product on the same inputs, timed and checked the same way |
| `bench_ttnn.py` | TT-NN's product and SwiGLU on the same inputs; it imports `ttnn` and `torch` from the tt-metal tree `TT_METAL_HOME` names (Komorebi needs neither) |
| `reference/gpu_bench.py`, `reference/gpu/gemm_gpu.cu` | cuBLAS on one idle GPU (fp32 and fp16) |
