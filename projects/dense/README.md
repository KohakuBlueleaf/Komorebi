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

The breakpoints are the broadcasts' (`km::Broadcast`, `komorebi/device/movement/km_broadcast.hpp`): the
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
The parts travel over the cores' ports (`km::reduce`, `komorebi/device/movement/km_port.hpp`):
the reader takes arrivals into `kCbArrive` (fp32, a slot per sender), compute adds them
in rank order into `kCbSum` by packer accumulation, the writer sends a core's part
to its receiver or writes the root's sum to Y. Commands carry `KM_FLAG_EXCLUSIVE` (the ports).
Plans are named by hand: `--split s,oh,ow[,kc]` (`plan.split_plan`; K chunk default:
the whole partition, at most 32 tiles) and `--reduce owner|tree`.

| Handler | Arguments |
|---|---|
| `SPLIT` | a0 a_address (A), a1 w_address (W), a2 y_address (Y), a3 reduce_form (0 owner, 1 tree), a4 mode: 0 full, 1 parts only (each core's part packed and dropped: no reduce, no writes), a5 the plan: its index in the program's table of plans (0, the first, when a command leaves it out) |

One loaded program can take several shapes: `card.split_program` takes a list of plans that
share their buffers (the same `oh`, `ow`, `kc`, subblock and arrival pages:
`split_plan(..., arrive=)`), and puts them in the kernels' arguments one after the other
(`plan.split_table`, at most 8); a command's a5 names the plan it runs
(`split_args(..., index=, plans=)`). The kernels read their plan words at a5 x 11 for each
command, so a command without a5 runs the first plan as before.

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

Run again on 2026-11-16 (card 1, one run each of the same `bench.py` commands, the grid
form for the cubes, the K split 14,1,16,5 with the tree for `dec4096`), on a tree whose
runtime and headers had changed since, all checks passing with the same relative RMS
(3.96e-3 to 3.98e-3): `kvproj` 6.3 us (+5 %), `lin384` 61.3 (+0.2 %), `lin1024` 377.4
(-4.1 %, mean clock 1,238 MHz against 1,255), `sq4096` 952.2 (-0.6 %), `sq8192` 8,180.2
(-5.0 %), `sq16384` 85,936.8 (-2.3 %), `dec4096` 93.9 (+1.6 %). The cubes move with the
clock the power limit leaves (843 to 1,131 MHz minimum). The table keeps its figures.

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

The clock read during the loop, 2026-11-18 (AICLK sampled while the session was open):
two cards at 4096 cubed fell to 831 MHz with `--place rows` (589.6 us a call over
201,181 calls, against 373 us single) and to 1,068 MHz with `--place out` (528.2 us a
call over 175,941 calls); `flat.py`'s 1024 x 4096 x 4096 loop on two cards held
1,343 to 1,350 MHz (15,710 us a call over 4,922 calls). A loop time below 1,350 MHz is a
throttled one, and is not set beside an unthrottled one.

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
of `kernels/grid/`. An island of a two-card session has 118 ranks (the session holds
two link cores of each card), so a plan whose group reaches past them is refused: a
2d plan is made again one column fewer at a time, then one row, before the next layout
(sq8192: 2d on 10 x 10 cores an island); its L1 is what the session leaves its
buffers.

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
| `sq8192`, W copied, 2d on 10 x 10 cores an island (2026-11-17, the session's split) | 0, 1 | 138 / 118 | 4,869.9 | 225.8 | 3,430 | 957 / 1115 | 0.83 | 0.04 |
| `sq8192`, B moved once a call | 0, 1 | 128 / 128 | 27,196.9 | 40.4 | 2,942 | 1350 / 1350 | 1.12 | 0.27 |
| `sq8192`, by rank | 0 | 256 | 13,192.0 | 83.3 | 5,717 | 1034 | 0.85 | 0.28 |
| `sq8192`, by rank | 0, 1 | 136 / 120 | 5,242.3 | 209.7 | 4,686 | 970 / 1208 | 1.12 | 0.27 |
| `sq16384`, by rank | 0 | 512 | 91,763.8 | 95.9 | 77,693 | 1138 | 10.78 | 0.29 |
| `sq16384`, by rank | 0, 1 | 272 / 240 | 48,306.7 | 182.1 | 42,259 | 1170 / 1173 | 3.99 | 0.28 |

Every check passed (relative RMS 3.96e-3 to 3.97e-3, largest 5.6e-3 to 6.5e-3). The
2026-11-17 row is today's command (host load 6 to 24): A and Y split over the ranks as
the session does (138 / 118 row tiles, the session warns the parts are not in
proportion), so island 0's part is the slower (median 5,314 us against 4,122).
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
the pages it does not hold); no copy is made before the call. Measured on 2026-11-15 at
00:22, the integer case above (7680 x 64 x 128, 121 and 119 row tiles), one page link
(`--m 7680 --k 64 --n 128 --inputs int --b-far --links 1 --kernel-bytes 131072`: the
grid's kernels take 122,752 bytes of kernel buffer with the page links built in, more
than the two-card default of 112 KiB): bit for bit on the 256 sampled rows of both
islands, 36.7 us a call in the loop (the clock during the loop was not read); island 0
15.2 us, island 1 31.8 us.

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
names an island, and the kernels hold no island count. Every decision in the kernels
comes from the tensors' reference words (the tensor table: which K rows of W, which
columns of W and Y, which rows of X and Y an island holds, and which tensors are
copies) and from `km::island()`, `km::islands()` and the command's group:

- **Each island computes from what it holds.** Its ranks run the grid form above (or,
  for small M, the K-split plan below) on the island's rows, K rows and columns, their
  broadcasts inside the island. Nothing of X or W crosses and no page is asked across:
  a row of X is read only where it is held. A broadcast reaches a straight run of
  cores that make the same rows (X) or columns (W), so each group of cores sharing
  broadcasts is a rectangle, but the island need not be one: where the link cores take
  the end of the last row, the plan may cover the ranks with two rectangles sharing one
  block and K block (beside: columns split; below: rows split), each in bands of whole
  blocks, chosen by a cost of blocks a core and the block's shape.
- **The placements give the form.** W split along N with Y split the same: each island
  makes its own columns (output dim sharded). W split along K with Y a copy: each
  island makes its partial of all of Y and the two are added (input dim sharded). X
  and Y split along M with W a copy: each island makes its rows (W gathered). One
  island: every tensor is whole and the kernels run the one-card plan. W split along
  K with Y split along N is refused.

The two standard forms and their numbers are in "A linear layer on one card and two"
below. The first version of the kernels beside the project's own grid kernels (`kernels/grid`), the same plan in
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
kernels, whose form came from the island's own part of W alone; the curve below is of a
later version, whose text on one card is 87,888 B (`--kernel-bytes
90112`).

### One card and two, 128 to 16384 cubed

The same script and kernels on card 0 and on cards 0 and 1, one run a point (three
single calls, then calls back to back for about three seconds), on 2026-11-15 between
05:15 and 05:38, host load 5 to 8; every check passed (relative RMS 3.94e-3 to
3.96e-3). Two cards, rows form: X and Y split along M, W a copy on each card
(`--place rows`): the replicated weight, nothing crosses. Microseconds, the median
single call / the loop's time a call; "x" is one card's single over two cards' (above
1: two cards are faster); the clock during the loop was not read.

| Cube | One card | Two cards, rows form (W copied) | x | Kohaku-TT: one card / two cards (W copied) / two cards (K split) |
|---|---|---|---|---|
| 128 | 7.2 / 16.1 | 7.6 / 41.3 | 0.95 | |
| 256 | 11.6 / 49.3 | 10.0 / 24.3 | 1.16 | |
| 512 | 15.9 / 20.3 | 19.8 / 25.6 | 0.80 | |
| 1024 | 43.8 / 47.2 | 40.6 / 46.6 | 1.08 | |
| 2048 | 167.1 / 195.6 | 125.4 / 134.3 | 1.33 | |
| 4096 | 655.0 (a) | 420.7 / 543.9 | 1.56 | 819.8 / 487.3 / 868.8 |
| 8192 | 4,488.1 (a) | 2,357.6 (b) | 1.90 | 4,549.5 / 2,482.8 / 3,930.1 |
| 16384 | 34,544.8 (a) | 17,731.5 (b) | 1.95 | 32,489.3 / 17,067.8 / 32,970.6 |

(a) single calls of the one-card table above (`results/one-card-*.json`), not run
again; (b) single calls of the rows form below (`results/linear-rows-*.json`, an earlier
head), not run again. Kohaku-TT: the table "Shapes, one card and two" below, its
default two-card form (rows split, W all-gathered once to a kept copy on each chip) and
its K split. The loops from 4096 up lowered the AI clock (one card 1,162 MHz at 2048;
two cards 950 to 1,025 at 4096). Two cards win in the rows form from 2048 up (1.33 to
1.95; Kohaku-TT's default form 1.68 to 1.90 at 4096 to 16384); below 2048 a two-card
call is the command over two islands.

```
python projects/dense/linear.py --cube 16384 --cards 0
python projects/dense/linear.py --cube 4096 --cards 0,1 --place rows
```

### The rows form: W replicated (Kohaku-TT's default form)

`--place rows` places X and Y split along M (each island half the rows) and W as a
copy on every island: the replicated weight, Kohaku-TT's default two-card form. The
same kernels take it from the placements alone: an island makes the rows of X it
holds; each island runs the product of its rows with its own plan. Nothing crosses in a call (the
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

## A linear layer on one card and two

The two standard tensor-parallel forms of Y = X W (X is M x K, W is K x N), each taken
by the linear kernels above from the placements alone (`--place` only says how the
script places the tensors; the kernels read the tensor table):

- **Output dim sharded** (`--place out`): W and Y split along N, X a copy on both
  islands. Each island runs the one-card product on its half of N. Nothing crosses.
- **Input dim sharded** (`--place in`): X and W split along K, Y a copy. Each island
  runs the one-card product on its half of K and writes its partial of the whole M x N
  output to its own DRAM; then one addition across the cards: the collectives'
  all-reduce of Y (`run.all_reduce`), after the product.

Decode is M = 32: one tile row, so one token is one row of it (up to 32 tokens a
call). At M of at most two tile rows the plan picker gives the kernels the K split:
groups of ranks inside an island, a block of Y a group, each rank a share of K read on
both movers in batches of at most 16 pages, the parts summed into the group's first
rank by a tree. Prefill is M = 1024 and 4096 with the grid plan.

### The card's two ceilings

`ceilings.py`, this framework's kernels on card 0: every rank reading its share of a
W-sized tensor with both movers, nothing computed; and every rank's matrix unit on
resident tiles, nothing read.

| Ceiling | Measured |
|---|---|
| DRAM, 128 MiB read by 120 ranks | 495.6 GB/s (batches of 8 pages a mover; 490.2 at 4, 488.2 at 16), 62.0 GB/s a bank of 8 |
| DRAM, 32 MiB (W of 4096 x 4096) | 486.3 GB/s, 69.0 us |
| Matrix unit, 120 ranks, 8 x 8 x 8 tile blocks, HiFi2, fp32 accumulation | 4.70e9 tile products a second (34.44 cycles a tile product at 1,350 MHz) |

A shape's hardware time is the larger of W's bytes at 495.6 GB/s and its tile products
at 4.70e9 a second (on two cards: an island's half of each).

```
python projects/dense/ceilings.py --card 0
```

### The account

With `--account` the kernels stamp the wall clock at their phase edges (compiled out by
default), and a call's time is a sum of named terms on its slowest core, the one whose
command took longest. Terms marked **HW** are the hardware's (W's bytes at the DRAM
ceiling, the matrix unit's cycles, Y's bytes); **OURS** is everything else, named.

- *Way in and out* (OURS): the dispatcher's part less the slowest core's span: the
  command reaching the ranks, the last rank's end reaching the dispatcher.
- *W read*: the rank's slower mover reading its batches of W. HW = the island's W at
  495.6 GB/s; OURS = the rest (a rate under the ceiling).
- *Reduce in the island* (K split, OURS): the group's fp32 parts summed into its first
  rank over the NoC, with the wait for the group's last mover.
- *W free, W cast* (grid, OURS): the column's sender waiting for its receivers to free
  the buffer, and multicasting the block down the column; *Y wait, Y written*: waiting
  for compute's last subblocks, and Y's pages written (HW).
- *Matrix unit* (grid, HW, beside): the slowest core's tile products at 34.44 cycles; it
  overlaps the movers.
- *Unaccounted*: the span less the writer's terms (its clock covers the span).

The host's single synced call adds 250 to 300 us on one card and 300 to 450 us on two
past the dispatcher's part (the host's event path, OURS, outside the call).

### One card

Card 0, the median of three single calls (the dispatcher's part), every check passed
(relative RMS 3.94e-3 to 3.97e-3). Microseconds; GB/s is the island's W over W read.
The terms are of the second call (so a way in and out can come out below 0 by the
calls' spread).

Decode, the K split (groups of S ranks, block width ow tiles):

| M x K x N | Plan | Hardware | Measured | Way in and out | W read: HW + OURS (GB/s) | Reduce in the island | Y written |
|---|---|---|---|---|---|---|---|
| 32 x 4096 x 4096 | S 8, ow 9 | 67.7 | 94.0 | 2.3 | 67.7 + 12.6 (418) | 11.0 | 0.5 |
| 32 x 8192 x 8192 | S 6, ow 13 | 270.8 | 312.3 | 2.2 | 270.8 + 27.6 (450) | 11.0 | 0.6 |
| 32 x 5120 x 17408 | S 5, ow 23 | 359.7 | 409.4 | 2.4 | 358.8 + 0 (497) | 47.2 | 0.9 |
| 32 x 17408 x 5120 | S 6, ow 8 | 359.7 | 422.4 | 2.4 | 359.7 + 42.7 (443) | 17.2 | 0.4 |
| 32 x 3840 x 15360 | S 2, ow 8 | 238.0 | 289.9 | 3.6 | 238.0 + 42.3 (421) | 5.6 | 0.4 |
| 32 x 15360 x 3840 | S 8, ow 8 | 238.0 | 291.4 | -0.6 | 238.0 + 46.0 (415) | 7.5 | 0.4 |
| 32 x 2560 x 9216 | S 5, ow 12 | 95.2 | 121.6 | 2.3 | 95.2 + 13.9 (433) | 9.6 | 0.6 |
| 32 x 9216 x 2560 | S 12, ow 8 | 95.2 | 124.4 | 1.6 | 95.2 + 14.3 (431) | 12.8 | 0.4 |

Decode is W's read, as it should be: 415 to 497 GB/s, the call 1.14 to 1.39 of the
hardware time. What is ours: the read under the ceiling where a block's K rows lie on fewer
banks or the ranks far from the banks finish last (the movers of equal work spread 26
to 82 us at 4096 x 4096), and the group's reduce (5.6 to 47 us; 47 at ow 23, the widest
block). 32 x 4096 x 4096 against the dense project's K split (92.4 us, W at 363 GB/s)
and Kohaku-TT (81.0 us, 414 GB/s), both recorded in "Shapes against Kohaku-TT" above:
our W read is as fast as Kohaku-TT's; the 13 us between are the reduce and the start.

Prefill, the grid plan (rows x columns of cores; one block in flight a column sender):

| M x K x N | Plan | Hardware: DRAM / matrix unit | Measured | Way in and out | W read: HW + OURS (GB/s) | W cast | W free | Y wait + written | Unaccounted | Matrix unit (slowest core) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 x 4096 x 4096 | 8 x 12 | 67.7 / 111.6 | 382.4 | 11.3 | 67.7 + 209.7 (121) | 54.4 | 1.4 | 22.9 + 12.3 | 2.8 | 143.7 |
| 1024 x 8192 x 8192 | 8 x 12 | 270.8 / 446.2 | 1,408.0 | 14.5 | 270.8 + 853.0 (119) | 217.3 | 5.4 | 20.5 + 22.7 | 3.9 | 574.7 |
| 1024 x 5120 x 17408 | 8 x 12 | 359.7 / 592.6 | 1,864.6 | 4.7 | 359.7 + 1,080.4 (124) | 284.4 | 6.8 | 24.0 + 99.6 | 5.0 | 751.0 |
| 1024 x 17408 x 5120 | 8 x 12 | 359.7 / 592.6 | 1,859.3 | 9.6 | 359.7 + 1,148.7 (118) | 293.2 | 6.6 | 19.0 + 19.2 | 3.3 | 777.2 |
| 1024 x 3840 x 15360 | 8 x 12 | 238.0 / 392.2 | 1,289.4 | 17.9 | 238.0 + 689.0 (127) | 184.6 | 4.1 | 28.4 + 123.1 | 4.4 | 489.8 |
| 1024 x 15360 x 3840 | 6 x 12 | 238.0 / 392.2 | 1,300.0 | -1.5 | 238.0 + 820.3 (111) | 174.5 | 20.7 | 38.2 + 4.0 | 5.8 | 734.7 |
| 1024 x 2560 x 9216 | 8 x 12 | 95.2 / 156.9 | 536.1 | 2.4 | 95.2 + 298.1 (120) | 72.9 | 1.7 | 22.2 + 40.4 | 3.2 | 195.9 |
| 1024 x 9216 x 2560 | 8 x 12 | 95.2 / 156.9 | 528.4 | 7.7 | 95.2 + 314.6 (115) | 81.9 | 3.0 | 12.7 + 10.8 | 2.4 | 205.7 |
| 4096 x 4096 x 4096 | 10 x 11 | 67.7 / 446.2 | 656.7 | 2.5 | 67.7 + 242.9 (108) | 92.6 | 108.1 | 25.1 + 70.5 | 47.4 | 509.4 |
| 4096 x 8192 x 8192 | 10 x 11 | 270.8 / 1,784.8 | 2,271.9 | 3.4 | 270.8 + 670.4 (143) | 922.8 | 226.6 | 22.9 + 106.5 | 48.5 | 2,037.6 |
| 4096 x 5120 x 17408 | 10 x 12 | 359.7 / 2,370.5 | 2,960.2 | 11.3 | 359.7 + 516.3 (204) | 1,595.1 | 204.3 | 29.6 + 195.2 | 48.9 | 2,547.0 |
| 4096 x 17408 x 5120 | 10 x 12 | 359.7 / 2,370.5 | 2,965.8 | 7.1 | 359.7 + 1,214.2 (113) | 490.1 | 720.8 | 33.0 + 37.5 | 103.3 | 2,720.1 |

At M = 1024 the call is W's read by the column senders, one row of 12 cores, at 111 to
127 GB/s, a quarter of the ceiling: each sender reads its block of W with one block in
flight and then multicasts it down its column (W cast, not overlapped with the next
read). That term is ours and the largest: at 1024 x 8192 x 8192 853 of 1,408 us. At
M = 4096 the matrix unit is the floor (509 to 2,720 us on the slowest core) and the
call is 1.09 to 1.29 of
it: the multicast and the senders' waits for free buffers fill the rest. The table is
the sender reading alone, as it ran before the spread below.

**W's read spread over the column** (`--spread`, `auto` by default). Every rank of a
column reads its share of the K rows of each W block into its own slice buffer, ahead,
while the block before arrives; once the column's sender has reserved the slot for the
block it marks that in each rank's ready port, the ranks write their rows into the slot
and count themselves at the sender's port, and the sender, having read its own share,
multicasts the whole block down the column as before. `auto` takes it for the grid plan
where a column has more than one rank, the block is at most 7 tile rows high and the K
block at least 2 tiles: the sender alone reads a W tile in 199 ns (10.3 GB/s a core, the
account above: 360 KB in 35 us a block) and a rank's matrix unit makes a tile product in
25.5 ns, so the lone read is the longer of the two below 7.8 rows. `--spread on` and
`off` force it. Card 0, the same columns as the table above (the clock during the loop
was not read):

| M x K x N | Plan: cores, block, K block | Hardware: DRAM / matrix unit | Measured | Loop | Before (above) | Way in and out | Rows into the sender's slot | W cast | W free | Y wait + written | Unaccounted | Matrix unit (slowest core) | Bound by |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024 x 4096 x 4096 | 8 x 12, 4 x 11, 16 | 67.7 / 111.6 | 221.7 | 222.6 | 382.4 | 2.5 | 106.0 | 61.2 | 3.5 | 24.5 + 16.0 | 2.9 | 143.7 | the column's step |
| 1024 x 8192 x 8192 | 8 x 12, 4 x 22, 8 | 270.8 / 446.2 | 693.0 | 698.8 | 1,408.0 | 2.5 | 349.3 | 248.3 | 25.7 | 25.1 + 37.4 | 3.0 | 574.7 | the column's step |
| 1024 x 5120 x 17408 | 8 x 12, 4 x 23, 8 | 359.7 / 592.6 | 956.1 | 961.5 | 1,864.6 | 2.5 | 418.6 | 330.4 | 56.2 | 27.7 + 111.0 | 4.1 | 751.0 | the column's step |
| 1024 x 17408 x 5120 | 8 x 12, 4 x 14, 14 | 359.7 / 592.6 | 918.5 | 924.0 | 1,859.3 | 2.5 | 514.8 | 327.5 | 32.2 | 25.5 + 12.7 | 3.3 | 777.2 | the column's step |
| 1024 x 3840 x 15360 | 7 x 12, 5 x 20, 8 | 238.0 / 392.2 | 766.0 | 772.5 | 1,289.4 | 2.5 | 361.3 | 218.4 | 67.9 | 38.9 + 70.7 | 7.6 | 612.3 | the column's step |
| 1024 x 15360 x 3840 | 8 x 12, 4 x 10, 16 | 238.0 / 392.2 | 592.6 | 602.5 | 1,300.0 | 2.5 | 340.4 | 205.1 | 11.1 | 22.2 + 8.1 | 3.2 | 489.8 | the column's step |
| 1024 x 2560 x 9216 | 8 x 12, 4 x 12, 16 | 95.2 / 156.9 | 352.7 | 356.3 | 536.1 | 2.4 | 190.1 | 80.6 | 9.0 | 27.7 + 36.8 | 3.5 | 195.9 | the column's step |
| 1024 x 9216 x 2560 | 8 x 12, 4 x 7, 16 | 95.2 / 156.9 | 282.1 | 284.1 | 528.4 | 2.4 | 141.2 | 94.5 | 4.5 | 18.8 + 12.6 | 3.5 | 205.7 | the column's step and X's read |

The call is 0.46 to 0.66 of the lone read's. Its bound is now the column's step on the
sender, all OURS: the seven other ranks' rows (315 KB at 1024 x 8192 x 8192) go into
one core at about 28 GB/s, and the whole block goes back down the column at about
46 GB/s, so W crosses the column twice; the ranks' DRAM reads are hidden (read ahead).
A step costs 7.7 us and 70 ns a W tile on every spread row measured (16 sender chains
on one card and two, within 0.8 us). The matrix unit is 0.84 to 0.95 of the step's
time (0.70 at 1024 x 2560 x 9216), the next term. At 1024 x 9216 x 2560 X's read is as long: the row's sender reads
its 4 x 288 tiles of X alone at the same 10.3 GB/s (229 us). 1024 x 3840 x 15360 plans
7 rows of cores where the lone read had 8: the slice buffer took the L1 of the larger
plan.

The pairs at M = 4096 (the lone read: no plan there has blocks under 13 rows; the clock
during the loop was not read):

| M x K x N | Plan: cores, block, K block | Hardware: DRAM / matrix unit | Measured | Loop | Way in and out | W read | W cast | W free | Y wait + written | Unaccounted | Matrix unit (slowest core) | Bound by |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4096 x 3840 x 15360 | 10 x 12, 14 x 10, 6 | 238.0 / 1,568.7 | 2,134.0 | 2,166.3 | 6.5 | 522.8 | 1,203.8 | 180.0 | 51.7 + 106.6 | 67.5 | 1,714.3 | the matrix unit |
| 4096 x 15360 x 3840 | 10 x 12, 14 x 10, 7 | 238.0 / 1,568.7 | 1,875.9 | 1,916.7 | 2.5 | 1,040.3 | 291.3 | 404.8 | 32.4 + 13.6 | 89.7 | 1,714.3 | the matrix unit |
| 4096 x 2560 x 9216 | 10 x 12, 13 x 12, 5 | 95.2 / 627.5 | 882.8 | 893.5 | 2.6 | 258.8 | 335.8 | 76.6 | 33.5 + 125.4 | 50.1 | 636.8 | the matrix unit, then Y |
| 4096 x 9216 x 2560 | 10 x 10, 13 x 8, 10 | 95.2 / 627.5 | 938.9 | 978.3 | 2.5 | 460.8 | 342.3 | 13.6 | 24.2 + 30.0 | 42.3 | 764.1 | the matrix unit and X's read (744) |

At M = 4096 the matrix unit is the floor (1.01 to 1.22 of the hardware time on the
slowest core, which holds more tiles than the average) and the call 1.20 to 1.50 of
the hardware time: the cast waits for the column to free its slots, at the matrix
unit's pace, and Y is written after the last block. Every check passed (relative RMS
3.96e-3 to 3.97e-3). These and the two-card rows below are the first card runs after
the K form left the kernels; all were exact.

### Two cards

Cards 0 and 1, one island each (118 ranks: two cores of each card are link cores), one
session; every check passed (4.12e-3 in the in form: each partial rounded to bf16 once),
Y equal on both islands after the all-reduce, nothing crossed in the products.

Output dim sharded (each island its half of N; "of one card" is the two-card call over
the one-card call; "bound by" is the largest term of the slower island):

| M x K x N | Hardware an island | Measured | Of one card | Way in and out | W read: HW + OURS (GB/s) | Reduce in the island / W cast | Y | Bound by |
|---|---|---|---|---|---|---|---|---|
| 32 x 4096 x 4096 | 33.9 | 53.8 | 0.57 | 2.4 | 33.9 + 7.2 (409) | 9.0 | 1.3 | W's read from DRAM |
| 32 x 8192 x 8192 | 135.4 | 172.8 | 0.55 | 2.3 | 135.4 + 21.9 (427) | 12.0 | 1.3 | W's read from DRAM |
| 32 x 5120 x 17408 | 179.8 | 223.1 | 0.54 | 2.4 | 179.8 + 25.8 (433) | 13.1 | 1.9 | W's read from DRAM |
| 32 x 17408 x 5120 | 179.8 | 227.4 | 0.54 | 1.6 | 179.8 + 26.3 (432) | 18.3 | 1.4 | W's read from DRAM |
| 32 x 3840 x 15360 | 119.0 | 152.3 | 0.53 | 1.1 | 119.0 + 16.7 (434) | 13.7 | 1.7 | W's read from DRAM |
| 32 x 15360 x 3840 | 119.0 | 152.4 | 0.52 | 2.4 | 119.0 + 9.8 (458) | 19.3 | 1.9 | W's read from DRAM |
| 32 x 2560 x 9216 | 47.6 | 68.0 | 0.56 | 2.5 | 47.6 + 7.8 (426) | 9.0 | 1.2 | W's read from DRAM |
| 32 x 9216 x 2560 | 47.6 | 71.0 | 0.57 | 2.4 | 47.6 + 7.4 (429) | 12.3 | 1.3 | W's read from DRAM |
| 1024 x 4096 x 4096 | 33.9 / 55.8 | 216.3 | 0.57 | 2.6 | 33.9 + 119.4 (109) | 31.3 (cast) | 16.2 + 4.4 | the column senders' W read (OURS) |
| 1024 x 8192 x 8192 | 135.4 / 223.1 | 734.3 | 0.52 | 2.6 | 135.4 + 441.3 (116) | 108.8 (cast) | 19.5 + 19.8 | the column senders' W read (OURS) |
| 1024 x 5120 x 17408 | 179.8 / 296.3 | 964.2 | 0.52 | 5.3 | 179.8 + 575.8 (118) | 141.7 (cast) | 21.4 + 32.3 | the column senders' W read (OURS) |
| 1024 x 17408 x 5120 | 179.8 / 296.3 | 1,006.2 | 0.54 | 3.3 | 179.8 + 620.0 (111) | 163.4 (cast) | 12.7 + 16.4 | the column senders' W read (OURS) |
| 1024 x 3840 x 15360 | 119.0 / 196.1 | 668.7 | 0.52 | 2.7 | 119.0 + 391.3 (116) | 90.2 (cast) | 24.2 + 32.2 | the column senders' W read (OURS) |
| 1024 x 15360 x 3840 | 119.0 / 196.1 | 671.3 | 0.52 | 2.7 | 119.0 + 410.8 (111) | 111.2 (cast) | 9.3 + 9.2 | the column senders' W read (OURS) |
| 1024 x 2560 x 9216 | 47.6 / 78.4 | 288.7 | 0.54 | 2.7 | 47.6 + 155.1 (116) | 36.7 (cast) | 21.3 + 20.5 | the column senders' W read (OURS) |
| 1024 x 9216 x 2560 | 47.6 / 78.4 | 286.8 | 0.54 | 4.1 | 47.6 + 165.2 (111) | 52.3 (cast) | 7.6 + 3.1 | the column senders' W read (OURS) |
| 4096 x 5120 x 17408 | 179.8 / 1,185.2 | 1,739.8 | 0.59 | 4.6 | 179.8 + 504.3 (130) | 257.9 (cast), W free 535.0 | 45.1 + 200.0 | the matrix unit: 1,469.4 on the slowest core (HW 1,185.2) |

Each island takes its half of the one-card work in 0.52 to 0.59 of the one-card time:
the terms over half the one-card hardware time are the same terms as on one card, each
about halved (decode: W read under the ceiling 7 to 26 us, the reduce 9 to 19 us;
prefill: the column senders' read at 109 to 118 GB/s). The islands' plans are an
island's (118 ranks, 10 columns at prefill where one card has 12). Nothing is left
unaccounted at decode; at prefill 3.8 to 12.9 us of the writer's span is. The prefill
rows of this table are the column's sender reading alone.

Prefill with W's read spread over the column (the default, `auto`), and M = 4096 (no
spread: blocks of 13 to 16 rows). "Of one card" is over the one-card call of the same
setting (the spread rows above); the terms are the slower island's; the clock during the
loop was not read:

| M x K x N | Plan an island: cores, block, K block | Hardware an island: DRAM / matrix unit | Measured | Loop | Before (above) | Of one card | Way in and out | W read (spread: rows into the sender's slot) | W cast | W free | Y wait + written | Unaccounted | Matrix unit (slowest core) | Bound by |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024 x 4096 x 4096 | 6 x 10 + 6 x 1, 6 x 6, 16 | 33.9 / 55.8 | 158.7 | 162.3 | 216.3 | 0.72 | 2.6 | 47.4 | 52.7 | 21.9 | 16.2 + 3.5 | 13.5 | 117.6 | X's read (6 x 128 tiles a row sender, 153) |
| 1024 x 8192 x 8192 | 8 x 10 + 8 x 2, 4 x 11, 16 | 135.4 / 223.1 | 376.1 | 380.6 | 734.3 | 0.54 | 2.6 | 199.0 | 124.0 | 9.0 | 24.4 + 13.1 | 3.6 | 287.4 | the column's step |
| 1024 x 5120 x 17408 | 8 x 10 + 8 x 2, 4 x 23, 7 | 179.8 / 296.3 | 512.0 | 518.8 | 964.2 | 0.54 | 2.8 | 247.8 | 173.2 | 13.2 | 20.7 + 50.2 | 4.1 | 375.5 | the column's step |
| 1024 x 17408 x 5120 | 8 x 10 + 8 x 2, 4 x 7, 16 | 179.8 / 296.3 | 554.0 | 549.6 | 1,006.2 | 0.60 | 2.6 | 223.3 | 262.4 | 18.6 | 14.2 + 9.7 | 4.7 | 388.6 | the column's step, X's read (433) |
| 1024 x 3840 x 15360 | 8 x 12, 4 x 20, 8 | 119.0 / 196.1 | 350.3 | 355.0 | 668.7 | 0.46 | 2.7 | 171.3 | 106.8 | 4.5 | 22.4 + 39.1 | 3.5 | 244.9 | the column's step |
| 1024 x 15360 x 3840 | 8 x 10 + 8 x 2, 4 x 5, 16 | 119.0 / 196.1 | 437.6 | 442.8 | 671.3 | 0.74 | 2.8 | 130.4 | 306.4 | 7.2 | 9.5 + 3.7 | 10.7 | 244.9 | X's read (4 x 480 tiles, 382) |
| 1024 x 2560 x 9216 | 4 x 10 + 4 x 2, 8 x 12, 10 | 47.6 / 78.4 | 304.7 | 309.1 | 288.7 | 0.86 | 2.5 | 211.3 | 33.9 | 5.6 | 40.4 + 8.4 | 3.7 | 195.9 | the sender's lone read (not spread, below) |
| 1024 x 9216 x 2560 | 8 x 10, 4 x 4, 16 | 47.6 / 78.4 | 241.0 | 238.1 | 286.8 | 0.85 | 2.7 | 77.1 | 122.8 | 3.5 | 7.6 + 1.5 | 11.4 | 117.6 | X's read (4 x 288 tiles, 229) |
| 4096 x 4096 x 4096 | 10 x 8, 13 x 8, 10 | 33.9 / 223.1 | 451.0 | 455.9 | | 0.69 | 2.8 | 164.2 | 203.8 | 11.9 | 24.1 + 19.1 | 39.9 | 339.6 | the matrix unit and X's read (331) |
| 4096 x 8192 x 8192 | 8 x 10 + 8 x 2, 16 x 11, 5 | 135.4 / 892.4 | 1,309.3 | 1,321.9 | | 0.58 | 2.6 | 574.8 | 326.4 | 268.5 | 16.2 + 114.9 | 5.2 | 1,149.4 | the matrix unit |
| 4096 x 17408 x 5120 | 10 x 10, 13 x 8, 10 | 179.8 / 1,185.2 | 1,851.2 | 1,834.9 | | 0.62 | 5.8 | 764.2 | 968.1 | 17.0 | 13.5 + 34.9 | 39.2 | 1,443.3 | the matrix unit and X's read (1,406) |
| 4096 x 3840 x 15360 | 10 x 10, 13 x 12, 6 | 119.0 / 784.3 | 1,163.1 | 1,174.4 | | 0.55 | 2.7 | 368.8 | 492.7 | 123.5 | 38.4 + 88.5 | 48.2 | 955.1 | the matrix unit |
| 4096 x 15360 x 3840 | 10 x 8, 13 x 8, 10 | 119.0 / 784.3 | 1,654.6 | 1,603.5 | | 0.88 | 2.7 | 617.3 | 933.1 | 18.7 | 29.4 + 18.8 | 34.6 | 1,273.5 | X's read (13 x 480 tiles, 1,241) and the matrix unit |
| 4096 x 2560 x 9216 | 9 x 12, 15 x 12, 4 | 47.6 / 313.7 | 522.5 | 528.8 | | 0.59 | 2.6 | 205.9 | 60.6 | 97.1 | 29.8 + 117.2 | 9.2 | 367.4 | the matrix unit, then Y |
| 4096 x 9216 x 2560 | 10 x 10, 13 x 4, 12 | 47.6 / 313.7 | 955.9 | 949.2 | | 1.02 | 2.7 | 205.1 | 671.2 | 14.3 | 17.7 + 6.1 | 21.8 | 382.1 | X's read (13 x 288 tiles, 744) |

The spread halves the M = 1024 rows where W's column is wide (0.51 to 0.55 of the lone
read at 8192 x 8192, 5120 x 17408, 17408 x 5120, 3840 x 15360). Where an island's half
of N is narrow (bn 4 to 7) the next read shows: X's, by the one sender of each row of
cores, alone, at the same 10.3 GB/s a core (bm x K tiles of X a row sender; the figure
in brackets is that read's time). It is the same term as W's was, on the row
broadcast, and it is OURS; it bounds 4096 x 4096 at M = 1024 too, where the island's
plan is 6 rows of 6 x 6 blocks. At M = 4096 the matrix unit is the floor (1.17 to 1.62
of an island's hardware time on the slowest core; 1.52 and 1.62 on the plans of 80
cores) except where X's read is longer
(9216 x 2560: 744 us of X against 382 of products). 1024 x 2560 x 9216 is the one row
the default made slower (304.7 against 288.7): the spread plan the lone read used
(8 x 12 cores, 4 x 12 blocks, K block 16) is 6,416 B over the island's L1 budget with
its slice buffer (1,262,592 + 65,728 B against 1,321,904), and the planner, whose cost
model knows nothing of the spread, then takes 4 x 12 cores of 8 x 12 blocks, which do
not spread (8 rows). Every check passed (relative RMS 3.96e-3 to 3.97e-3), nothing
crossed in the products; the island builds were 96,240 B of 110,592.

**A chain of two layers**, as a model's MLP runs them at decode: layer 1 output dim
sharded (hidden to mlp, W1 split along N), its Y gathered to a copy on both islands,
then layer 2 output dim sharded on that copy (mlp to hidden). The gather is the
collectives' all-gather (`km::all_gather`), called by every rank at the end of layer 1's
kernel, so it is a term of layer 1's call; layer 2 is the down projection above. Loop,
microseconds a call (the clock during the loop was not read); the gather term is the
stamped crossing on each island (from its first rank's start of the gather to its end),
Y's half each way:

| Hidden x mlp | Layer 1 alone | Layer 1 with the gather | Gather term, island 0 / 1 (bytes each way, GB/s) | Layer 2 | Chain | Layers alone | Gather's share |
|---|---|---|---|---|---|---|---|
| 5120 x 17408 | 226.1 | 252.8 | 27.8 / 26.4 (557 KB, 20.0) | 231.4 | 484.3 | 457.3 | 5.9 % |
| 3840 x 15360 | 155.6 | 190.4 | 34.5 / 33.7 (492 KB, 14.2) | 156.0 | 346.4 | 311.7 | 11.1 % |
| 2560 x 9216 | 71.4 | 92.8 | 20.6 / 20.9 (295 KB, 14.3) | 75.6 | 168.3 | 147.7 | 14.0 % |

Every check passed (layer 2 on the gathered Y against the host's product of the two
layers). The gather's term is the collective's own (its meeting of the island's ranks,
its leader's one line, its release), not the linear kernel's: it is the collectives'
to shrink. Before it, the same gather as a separate host call took about 330 us a call
(island parts 76.9 to 89.7 and 49.6 to 62.7 us, the rest one host round trip): in the
kernel it is 20.6 to 34.5 us.

**Use the output dim sharded form.** The input dim sharded form below is recorded and
not taken further: it is exact, but each island makes a partial of the whole M x N
output, so the whole output must cross the link and be added after the product, and
that crossing and addition are its whole difference from the output form (at decode
69 to 149 us on its best path, the addition in the kernels; 165 us and more through the
all-reduce call). The output form crosses nothing in the product and, chained, crosses
half of a layer's Y once (the gather above).

Input dim sharded, the product (each island its partial of the whole Y) and then the
all-reduce of Y (bf16) as a separate term:

| M x K x N | Hardware an island | Product | Its W read: HW + OURS (GB/s) | All-reduce of Y, island 0 / 1 (Y's bytes) | Product + all-reduce | Of one card |
|---|---|---|---|---|---|---|
| 32 x 4096 x 4096 | 33.9 | 52.5 | 33.9 + 6.3 (418) | 207.7 / 169.1 (262 KB) | 260.2 | 2.77 |
| 32 x 8192 x 8192 | 135.4 | 176.2 | 135.4 + 24.9 (419) | 287.6 / 248.6 (524 KB) | 463.8 | 1.49 |
| 32 x 17408 x 5120 | 179.8 | 217.2 | 179.8 + 6.2 (479) | 212.6 / 173.6 (328 KB) | 429.8 | 1.02 |
| 32 x 15360 x 3840 | 119.0 | 159.7 | 119.0 + 21.2 (421) | 211.0 / 172.0 (246 KB) | 370.7 | 1.27 |
| 32 x 9216 x 2560 | 47.6 | 68.5 | 47.6 + 9.1 (416) | 205.5 / 167.0 (164 KB) | 274.0 | 2.20 |
| 1024 x 4096 x 4096 | 33.9 / 55.8 | 216.7 | 33.9 + 108.2 (118) | 1,623.5 / 1,584.3 (8.4 MB) | 1,840.2 | 4.81 |
| 1024 x 8192 x 8192 | 135.4 / 223.1 | 735.6 | 135.4 + 430.4 (119) | 3,047.3 / 3,008.5 (16.8 MB) | 3,782.9 | 2.69 |
| 1024 x 17408 x 5120 | 179.8 / 296.3 | 964.2 | 179.8 + 585.0 (117) | 1,960.8 / 1,920.8 (10.5 MB) | 2,925.0 | 1.57 |

The product takes what the output-dim-sharded form takes (52.5 to 964 us, 0.51 to 0.57
of one card). The all-reduce is the whole difference, and it is OURS: about 165 us
fixed (its decode times barely move with Y's bytes), then 5.2 to 5.5 GB/s at prefill
sizes: the bf16 adds are made by the ranks' movers in software (integer instructions), a
few pages at a time, between the gathers. "Product + all-reduce" takes the slower
island's.

Input dim sharded, the addition in the kernels (the default; `--add reduce` gives the
all-reduce above). Still the standard form: each island makes its partial of the whole
Y, then one addition across. Each rank writes its blocks of the partial (bf16) to its
island's set of a scratch tensor S (a copy on each island), then posts them to the same
pages of the other island's S (`km::gather_pages`: the link cores' page CPUs read them
from this island's DRAM and write them to the other's), naming the rank that makes the
same blocks there as their keeper. Once its keeper count says the other island's
partial of its blocks has landed, its reader hands each subblock of both partials to
compute, which sums them in fp32 (the second packed onto the first in L1) and packs
bf16 for Y; the writer writes Y. Nothing crosses in the product, and the add is the
ranks' own matrix engine's, not a mover's. S has a set an island for each of two
command parities, so that a call's partial never lands on the set the other island is
still reading. Every check passed (relative RMS 3.87e-3: both partials rounded to bf16
once, the sum once; Y equal on both islands).

| M x K x N | Measured | Product, S written, posted | Wait for the other partial (Y's bytes, GB/s each way) | Add pass, Y written | Way in and out | All-reduce form (above) |
|---|---|---|---|---|---|---|
| 32 x 4096 x 4096 | 162.6 | 51.6 | 105.6 (262 KB, 2.5) | 3.0 | 2.4 | 260.2 |
| 32 x 8192 x 8192 | 327.0 | 174.5 | 149.4 (524 KB, 3.5) | 3.3 | -0.3 | 463.8 |
| 32 x 17408 x 5120 | 324.8 | 214.0 | 105.9 (328 KB, 3.1) | 3.6 | 1.3 | 429.8 |
| 32 x 15360 x 3840 | 240.1 | 158.9 | 75.5 (246 KB, 3.3) | 3.4 | 2.4 | 370.7 |
| 32 x 9216 x 2560 | 143.2 | 68.3 | 69.1 (164 KB, 2.4) | 3.3 | 2.5 | 274.0 |
| 1024 x 4096 x 4096 | 1,215.8 | 214.5 | 968.3 (8.4 MB, 8.7) | 22.3 | 10.8 | 1,840.2 |
| 1024 x 8192 x 8192 | 2,048.9 | 756.2 | 1,276.3 (16.8 MB, 13.1) | 44.3 | -27.9 | 3,782.9 |
| 1024 x 17408 x 5120 | 2,150.5 | 960.8 | 1,158.5 (10.5 MB, 9.1) | 28.7 | 2.4 | 2,925.0 |

The slower island's terms (island 0's, at every shape), median single calls. The
product is the product of the table above (51.6 against 52.5 us at 32 x 4096 x 4096).
The add pass is small (3 to 44 us). What is left is the crossing, and it is OURS: Y's
bytes move at 2.4 to 3.5 GB/s at decode, where 16 to 30 ranks each post a line of 8 to
13 pages and the lines' fixed cost is the time (inferred: the link core's lead page CPU
takes the lines one after another); and at 8.7 to 13.1 GB/s at prefill, 80 ranks a line of 44
to 88 pages each, against the 23.4 GB/s each way the same calls reach with lines of 139
pages from 118 ranks (64 MiB) and 72.5 GB/s for ring puts on the same links. Island 1
waits 14 to 95 us at decode where island 0 waits 69 to 149: the direction 1 to 0 is
the slower (measured, not explained). Against the one-card call the two-card call is
0.77 to 1.73 at decode (0.77 at 32 x 17408 x 5120, 1.73 at 32 x 4096 x 4096) and 1.16 to
3.18 at prefill (1,215.8 against 382.4 us at 1024 x 4096 x 4096): that crossing.

### The four cells: plain kernel and aware kernel

The plain kernel knows nothing of islands: `flat.py` runs the dense project's grid
kernels unchanged on one flat group of both cards' ranks (above), the tensors placed
by hint (`--hints`) or by the session's default. The aware kernel is the linear kernel,
which takes the form from the placements.

| | One card | Two cards |
|---|---|---|
| Plain | the dense project's kernels on card 0: 32 x 4096 x 4096 92.4 (its K split); M = 1024: 5120 x 17408 2,091.9, 17408 x 5120 2,054.7, 3840 x 15360 1,399.6, 15360 x 3840 1,302.5, 2560 x 9216 528.5; 4096 cubed 664.9, 4096 x 8192 x 8192 2,959.6, 4096 x 5120 x 17408 8,184.1 | `flat.py`, exact, slow (below) |
| Aware | the linear kernel on one island, the one-card tables above: 94.0; 1,864.6, 1,859.3, 1,289.4, 1,300.0, 536.1; 656.7, 2,271.9, 2,960.2 | the two forms above |

On one card the aware kernel takes the plain kernel's time or less: it is the same
kernel form on one island, and where the times differ the plans differ (the same plan,
beside, gave the same time: "A linear layer written for islands" above). Awareness costs
nothing on one island: the island's shard is the whole tensor and the kernel's island
code is a range that is empty.

The plain kernel on two cards (`flat.py --cards 0,1`, the grid plan over one tall grid of
both islands; a column spanning the islands is a broadcast relayed between them):

| M x K x N | Placement | Measured | Crossed a call | What crossed |
|---|---|---|---|---|
| 32 x 4096 x 4096 | hints | refused: with one row tile no split hint keeps X's and Y's pages home | | |
| 32 x 4096 x 4096 | none | 8,192.3 | 7.7 MB (3,662 pages) | X's pages asked by island 1's rows, W's blocks relayed |
| 32 x 8192 x 8192 | none | 51,010.8 | 66.9 MB | the same |
| 1024 x 4096 x 4096 | hints | 1,899.1 | 34.6 MB | W's blocks up the columns that span the islands, once each |
| 1024 x 4096 x 4096 | none | 16,130.2 | 78.6 MB | W's blocks and X's pages |
| 1024 x 8192 x 8192 | hints | 7,376.1 | 138.4 MB | W's blocks |
| 1024 x 8192 x 8192 | none | 34,140.5 | 235.9 MB | W's blocks and X's pages |

It runs and is exact everywhere it can be placed; it is slow because W (and without
hints X) crosses the link at 1 to 19 GB/s a call, where the aware kernel's forms cross
nothing in the product.

**What is left, with its term** (all OURS):
- *The column's step* (prefill with the spread, M = 1024): the column's rows go into
  the sender's slot at about 28 GB/s and the block back down at about 46 GB/s, so W
  crosses the column twice (7.7 us and 70 ns a W tile a step). A multicast from each
  rank's own rows would cross it once.
- *X's read by the row senders* and *the planner's model knowing no spread* were
  terms here on 2026-11-15; both are done since (`--xspread` and the priced planner,
  below).
- *Decode*: the group's reduce (5.6 to 47 us) and W's read under the ceiling where a
  block's K rows lie on fewer banks.

Measured on 2026-11-15 between 06:20 and 07:15, X280 at 1,350 MHz. JSON:
`results/ceilings-0.json`, `results/linear-shapes-0.json` (one card),
`results/linear-forms-0x1.json` (two cards, with the all-reduce's twenty calls),
`results/linear-in-add-0x1.json` (the addition in the kernels, measured at 08:11 and
08:14), `results/linear-out-rest-0x1.json` (the output form's other shapes) and
`results/linear-chain-0x1.json` (the chains, with the gather by the host and in the
kernel; measured between 08:25 and 08:36),
`results/flat-forms-0x1.json`. The spread and the M = 4096 pairs, at 09:36 to 09:48:
`results/linear-spread-0.json` (one card, 8192 x 8192 and 5120 x 17408),
`results/linear-left-0.json` (one card, the other spread rows and the M = 4096 pairs),
`results/linear-spread-0x1.json` (two cards, M = 1024) and
`results/linear-4096-0x1.json` (two cards, M = 4096).

```
python projects/dense/linear.py --shapes "32,4096,4096;1024,8192,8192" --cards 0 --account
python projects/dense/linear.py --shapes "1024,8192,8192" --cards 0 --account --spread off
python projects/dense/linear.py --shapes "32,4096,4096:out;32,4096,4096:in" --cards 0,1 --account
python projects/dense/flat.py --m 1024 --k 4096 --n 4096 --cards 0,1 --hints
python projects/dense/linear.py --shapes "32,5120,17408:chain" --cards 0,1 --gather kernel
```

### Beside Kohaku-TT, the same shapes and forms

[Kohaku-TT](https://github.com/KohakuBlueleaf/Kohaku-TT) (commit ac81ff2) at the
table's shapes, in the form ours takes: on one card its default `x @ w`; on two cards
X a copy on both and W and Y split along N (its `colwise` placement, the output dim
sharded form). Loops, microseconds a call (calls back to back, the last output checked;
Kohaku-TT's card clock 1,333 to 1,350 MHz on average in every loop). Ours is the
tables above (the spread by default); two marked rows are single calls, where our loop
was throttled (4096 cubed, recorded at 1,018 MHz for Kohaku-TT too, and 4096 x 17408 x
5120, AICLK down to 1,075 MHz in ours). Every point exact (Kohaku-TT relative RMS
3.96e-3 to 3.97e-3).

| M x K x N | One card: ours | Kohaku-TT | Ours / Kohaku-TT | Two cards: ours | Kohaku-TT | Ours / Kohaku-TT | Where ours is slower: the term |
|---|---|---|---|---|---|---|---|
| 32 x 4096 x 4096 | 95.3 | 81.0 (recorded) | 1.18 | 58.0 | 44.7 | 1.30 | the group's reduce (11.0 / 9.0) and W's read under the ceiling (12.6 / 7.2), one card / two |
| 32 x 8192 x 8192 | 314.7 | 302.7 | 1.04 | 176.1 | 155.5 | 1.13 | the same: reduce 11.0 / 12.0, W read OURS 27.6 / 21.9 |
| 32 x 5120 x 17408 | 412.3 | 400.4 | 1.03 | 227.0 | 204.3 | 1.11 | the reduce at ow 23 (47.2) / 13.1 and W read OURS 25.8 |
| 32 x 17408 x 5120 | 425.3 | 414.7 | 1.03 | 231.4 | 224.7 | 1.03 | W read OURS 42.7 / 26.3, the reduce 17.2 / 18.3 |
| 32 x 3840 x 15360 | 293.2 | 273.4 | 1.07 | 156.5 | 134.3 | 1.17 | W read OURS 42.3 / 16.7, the reduce 5.6 / 13.7 |
| 32 x 15360 x 3840 | 294.9 | 262.9 | 1.12 | 156.0 | 145.6 | 1.07 | W read OURS 46.0 / 9.8, the reduce 7.5 / 19.3 |
| 32 x 2560 x 9216 | 124.9 | 108.2 | 1.15 | 71.8 | 54.1 | 1.33 | W read OURS 13.9 / 7.8, the reduce 9.6 / 9.0 |
| 32 x 9216 x 2560 | 127.0 | 128.6 | 0.99 | 75.6 | 64.9 | 1.16 | two cards: the reduce 12.3, W read OURS 7.4 |
| 1024 x 4096 x 4096 | 222.6 | 190.3 | 1.17 | 162.3 | 110.3 | 1.47 | one card: the column's step (106.0 into the slot); two: X's read (153) |
| 1024 x 8192 x 8192 | 698.8 | 711.2 | 0.98 | 380.6 | 382.7 | 0.99 | |
| 1024 x 5120 x 17408 | 961.5 | 1,015.9 | 0.95 | 518.8 | 500.4 | 1.04 | two cards: the column's step (247.8 into the slot, 173.2 cast) |
| 1024 x 17408 x 5120 | 924.0 | 884.3 | 1.04 | 549.6 | 485.5 | 1.13 | the column's step (514.8 / 223.3 into the slot); two: X's read (433) |
| 1024 x 3840 x 15360 | 772.5 | 697.5 | 1.11 | 355.0 | 340.2 | 1.04 | the column's step (361.3 / 171.3 into the slot) |
| 1024 x 15360 x 3840 | 602.5 | 655.4 | 0.92 | 442.8 | 328.9 | 1.35 | two cards: X's read (382) |
| 1024 x 2560 x 9216 | 356.3 | 307.6 | 1.16 | 309.1 | 159.2 | 1.94 | one card: the column's step (190.1); two: the sender's lone read (211.3, the plan does not spread) |
| 1024 x 9216 x 2560 | 284.1 | 271.2 | 1.05 | 238.1 | 153.4 | 1.55 | X's read (229 on both) |
| 4096 x 4096 x 4096 | 656.7 (single) | 819.8 (single) | 0.80 | 455.9 | 412.9 | 1.10 | two cards: X's read (331) beside the matrix unit (339.6) |
| 4096 x 8192 x 8192 | 2,312.7 | 2,371.3 | 0.98 | 1,321.9 | 1,428.0 | 0.93 | |
| 4096 x 5120 x 17408 | 3,011.7 | 6,192.8 | 0.49 | 1,762.9 | 2,665.6 | 0.66 | |
| 4096 x 17408 x 5120 | 2,965.8 (single) | 3,510.0 (single) | 0.84 | 1,834.9 | 1,542.2 | 1.19 | two cards: X's read (1,406) |
| 4096 x 3840 x 15360 | 2,166.3 | 3,613.6 | 0.60 | 1,174.4 | 1,758.5 | 0.67 | |
| 4096 x 15360 x 3840 | 1,916.7 | 1,921.4 | 1.00 | 1,603.5 | 1,009.9 | 1.59 | two cards: X's read (1,241) |
| 4096 x 2560 x 9216 | 893.5 | 1,035.9 | 0.86 | 528.8 | 537.0 | 0.98 | |
| 4096 x 9216 x 2560 | 978.3 | 875.7 | 1.12 | 949.2 | 436.8 | 2.17 | X's read (744 on both) |

Where ours is slower the term is one of the list above, all OURS. At decode the group's
reduce and W's read under the ceiling: 11 to 32 us on one card (7 shapes of 8), 7 to 23
on two.
At M = 1024 on one card the column's step: W crosses the column twice. On two cards,
wherever an island's half of N is narrow, X's read by the row senders; Kohaku-TT's
two-card calls there are about half its one-card calls (0.50 to 0.58 at M = 1024, 0.50
to 0.53 at M = 4096 for 15360 x 3840 and 9216 x 2560), and ours are 0.72 to 1.02 of
our one-card calls ("of one card" above). At M = 4096 with wide N ours is 0.49 to 0.98 of Kohaku-TT's.

Measured on 2026-11-15 between 09:53 and 09:59, Kohaku-TT's side in one process a
card set (its `compare/ktt_shapes.py` point, one session for many shapes), cards 0 and
0, 1; JSON in `results/shape-<M>x<K>x<N>-ktt-<0|0x1>-<default|colwise>.json`. The
4096 cubed and 32 x 4096 x 4096 one-card points are the ones recorded above.

### The terms where ours was slower, one by one

The terms of the list above, taken in the order of what they cost. Loops are in
microseconds a call; every point is exact (relative RMS 3.96e-3 to 3.97e-3).

**X's read by the row senders.** In this form at narrow N, Kohaku-TT calls TT-NN's
`minimal_matmul`. Its grid puts one row of injector cores on X. Each injector reads its
block of X alone, gives it to its own compute first, then unicasts it down its column,
store and forward. So Kohaku-TT also has one reader a line of cores, and the line moves
the same tiles as ours. Ours had one row sender per row and panel, reading its block
alone and then multicasting it to the row. The island's 20 senders read 76.7 MB in
749 us at 4096 x 9216 x 2560. That is 102 GB/s for the island: the senders share NoC
links, and they finish in the order of their row.

X's read now spreads over the row's ranks (`--xspread`, `auto` by default), as W's does
over the column. Each rank reads its share of the block into a slice buffer ahead. When
the sender has reserved the slot, it marks the row's ready port. The ranks write their
tiles into the slot and count themselves at the sender's port, and the sender then
multicasts the block. The slice is sized for the fewest ranks a panel leaves a row
(4 of 10). The row sender's X, slower island, before and after:

| M x K x N | Cards | Loop: before / after | The sender's X read: before / after (its hand-over now) | Its cast: before / after | Kohaku-TT |
|---|---|---|---|---|---|
| 4096 x 9216 x 2560 | 2 | 949.2 / 681.5 | 761 / 426 | 166 / 335 | 436.8 |
| 4096 x 15360 x 3840 | 2 | 1,603.5 / 1,374.7 | 1,137 / 1,019 | 293 / 570 | 1,009.9 |
| 4096 x 17408 x 5120 | 2 | 1,834.9 / 1,576.4 | 1,323 / 1,120 | 343 / 632 | 1,542.2 |
| 4096 x 4096 x 4096 | 2 | 455.9 / 430.8 | 274 / 264 | 81 / 146 | 412.9 |
| 1024 x 9216 x 2560 | 2 | 238.1 / 191.9 | 176 / 116 | 60 / 103 | 153.4 |
| 1024 x 15360 x 3840 | 2 | 442.8 / 399.8 | 318 / 285 | 107 / 229 | 328.9 |
| 1024 x 17408 x 5120 | 2 | 549.6 / 503.6 | 419 / 349 | 126 / 295 | 485.5 |
| 4096 x 9216 x 2560 | 1 | 978.3 / 885.7 | 704 / 592 | 176 / 328 | 875.7 |
| 1024 x 9216 x 2560 | 1 | 284.1 / 273.3 | 175 / 166 | 62 / 115 | 271.2 |

What is left of X's term is the row's step, and it is OURS. The ranks' tiles go into
the sender's slot, and the whole block then goes back out along the row, one after the
other. A step costs about 26 us at 156 tiles, 180 ns a tile, more than W's (W: 3.25 us
and 116 ns a tile). The last paragraph of this section says what would cut it.

**The planner prices the spreads.** Before, the plan's model had no spread. At 1024 x
2560 x 9216 on two cards it took a plan that was 6,416 B over L1 once spread, and that
plan fell back to the lone read. Now the model prices a K block as the largest of
compute, X's feed and W's feed. A feed is lone (4.75 us and 203 ns a tile), or spread
(W: 3.25 us and 116 ns a tile, fitted to 13 W-bound rows; X: 180 ns a tile, fitted to 5
X-bound rows). Each candidate plan is priced with every setting of `--spread` and
`--xspread` that it allows, each with its slices' L1, and the cheapest is kept (no
spread on a tie). Every piece of a plan takes the same setting, because one program has
one set of buffers. Against the 52 measured grid rows the model's ratio is 0.75 to 1.39.
The high end is the short-K shapes (K = 2560: 1.31 to 1.39), because the model has no
fill or tail for a call. The rows the planner now picks differently:

| M x K x N | Cards | Plan before / after (block, K block, spreads) | Loop: before / after | Kohaku-TT |
|---|---|---|---|---|
| 1024 x 2560 x 9216 | 2 | 8 x 12, 10, none / 5 x 12, 14, W | 309.1 / 193.3 | 159.2 |
| 1024 x 4096 x 4096 | 2 | 6 x 6, 16, W / 4 x 6, 16, W and X | 162.3 / 126.3 | 110.3 |
| 4096 x 8192 x 8192 | 2 | 16 x 11, 5, none / 8 x 11, 8, W | 1,321.9 / 1,295.0 | 1,428.0 |
| 4096 x 5120 x 17408 | 2 | 15 x 12, 4, none / 7 x 8, 10, W and X | 1,762.9 / 1,637.3 | 2,665.6 |
| 4096 x 3840 x 15360 | 2 | 13 x 12, 6, none / 13 x 8, 8, X | 1,174.4 / 1,156.9 | 1,758.5 |
| 4096 x 15360 x 3840 | 2 | 13 x 8, 10, none / 14 x 6, 10, X | 1,603.5 / 1,159.3 | 1,009.9 |
| 4096 x 17408 x 5120 | 2 | 13 x 8, 10, none / 13 x 8, 8, X | 1,834.9 / 1,572.6 | 1,542.2 |
| 4096 x 8192 x 8192 | 1 | 13 x 12, 6, none / 13 x 8, 8, X | 2,312.7 / 2,283.4 | 2,371.3 |
| 4096 x 5120 x 17408 | 1 | 13 x 12, 6, none / 13 x 8, 10, X | 3,011.7 / 2,958.6 | 6,192.8 |
| 4096 x 3840 x 15360 | 1 | 14 x 10, 6, none / 13 x 8, 10, X | 2,166.3 / 1,950.1 | 3,613.6 |
| 4096 x 2560 x 9216 | 2 | 15 x 12, 4, none / 8 x 12, 10, W | 528.8 / 526.1 | 537.0 |
| 4096 x 2560 x 9216 | 1 | 13 x 12, 5, none / 13 x 8, 10, X | 893.5 / 870.0 | 1,035.9 |

The plans of the X spread's rows above are the planner's picks too. A first fit, with
one spread constant for both feeds, took 4096 x 2560 x 9216 on two cards to 8 x 6
blocks with both spreads, at 617.2 us against 522.5. After the refit, with X's own
constants, it keeps W's spread there (526.1).

**Decode: the group's reduce.** Kohaku-TT's decode kernel (`ktt_gemv`, tried first for
few rows) has no reduce at all. Its output columns are spread over the grid, one core a
column up to the grid. Each core takes all of K for its columns and sums it in fp32 Dst.
x is multicast once a panel by one sender. Each core reads its W in blocks of at most
64 KB, up to 4 in flight on each mover, in K order. Its NoC 0 and NoC 1 share the blocks
5 : 3, spread evenly. Ours is the K split: groups of S ranks, a block of ow columns a
group, each rank K/S. That form has two terms Kohaku-TT has not: every group reads its
X share again (1/ow of W's bytes, 12.5% at ow 8), and the parts are summed by a tree of
ceil(log2 S) rounds into the group's first rank.

With `--account`, compute and the writer now stamp wall clocks: compute's own part
packed, each round's part taken and added, and the writer holding its part. Every
core's clocks are in the JSON. The tail, from a group's last own part to its Y written,
is 8 to 15 us. It is the rounds on the critical path, each a hop, then the root's own
end. The hop's steps are medians over all hops of a shape:

| Step | ow 8 to 9 | ow 12 to 13 | ow 23 |
|---|---|---|---|
| The part packed, to the writer holding it | 0.52 to 0.63 | 0.58 to 0.59 | 0.54 |
| Held, to landed (written, acknowledged, counted) | 1.10 to 1.21 | 1.37 to 1.45 | 1.95 to 2.08 |
| Landed, to the parent's compute taking it | 0.05 to 0.11 | 0.08 to 0.11 | 0.08 to 0.15 |
| Taken, to added (the combine) | 0.80 to 0.87 | 1.17 to 1.22 | 2.20 |
| The root: its sum in bf16, to the writer | 1.31 to 1.55 | 1.73 to 1.82 | 2.66 to 2.90 |
| The root: Y written | 0.46 to 1.46 | 1.61 to 2.09 | 0.94 to 1.17 |

At two cards 32 x 4096 x 4096 (S 14, ow 8) the tail is 10.4 us of 58.1: three to four
hops of 2.6 us, then 1.3 and 1.2 at the root. The combine and its pack take 1.4 us of a
hop. They are the packer's fp32 L1 accumulation, about 100 ns a tile: four tiles
through Dst at once in place of one changed nothing (58.8, 227.1 and 156.9 against
58.1, 226.8 and 156.2), so that change was taken out again. The send takes 1.2 us: about
0.8 us fixed, because the part is acknowledged before it is counted, plus the bytes at
about 80 GB/s. The root's lap that the earlier tables called the reduce (31 us here,
47.2 at ow 23) is mostly the root waiting for its own group's reads. The own parts end
19 to 44 us apart at 4096 x 4096 and 153 to 399 at ow 23, in the order of the cores'
place.

What is left: the tree, all OURS, at log2 S hops of 2.6 us. Inside the K split it would
go only with another reduce form. One is a reduce-scatter inside the group (each rank
sums and writes its own Y tiles), about 3 us of the 10.4. The other is Kohaku-TT's form,
with no split of K and X multicast, which removes the tree and X's second read alike. It
is now one more plan, below.

**Decode: the read under the ceiling.** The chunks' movers are now a plan word
(`--pattern`, AB by default). AAAB holds three chunks in mover A's half, as the decode
project's. ABABAABA is Kohaku-TT's 5 : 3. Loops:

| M x K x N | Cards | AB | AAAB | ABABAABA |
|---|---|---|---|---|
| 32 x 4096 x 4096 | 2 | 58.1 | 59.9 | 58.1 |
| 32 x 5120 x 17408 | 2 | 226.8 | 228.6 | 222.8 |
| 32 x 4096 x 4096 | 1 | 95.5 | 96.2 | 94.9 |
| 32 x 5120 x 17408 | 1 | 413.6 | 415.7 | 415.7 |
| 32 x 8192 x 8192 | 2 | 176.1 | | 178.9 |
| 32 x 17408 x 5120 | 2 | 231.4 | | 232.3 |
| 32 x 3840 x 15360 | 2 | 156.5 | | 155.8 |
| 32 x 2560 x 9216 | 2 | 71.8 | | 73.6 |
| 32 x 9216 x 2560 | 2 | 75.6 | | 78.2 |

AAAB never wins on these plans (112 to 120 cores, where AB is already near its rate).
ABABAABA wins on one shape of nine (4.0 us) and loses up to 2.8 on others, so the
plans stay on AB. The read itself is near the ceiling on the bytes it moves. At two
cards 32 x 4096 x 4096 the own parts end at 43 to 44 us: W at 495.6 GB/s is 33.9
(HARDWARE), X read again by every group is 4.2 (OURS, the form), and the rest is about
5. The last five rows of the table above ran with the four-tile combine as well; it did
not change the combine.

**Decode: Kohaku-TT's form, as one more plan.** The whole-K plan is the K split with
groups of one core and X cast. Each used core takes all of K for its ow columns, so there
is no reduce and no second read of X. The used cores are the island's first cores. In
each panel, its columns down to the last row it uses form one multicast rectangle (all
its cores ranks, a box on NoC 0). The rectangle's first core is the sender. Its writer,
on NoC 0, reads X in chunks of 32 tiles into its own buffer, once every used core of the
rectangle has said its buffer is free. It casts each chunk to the rectangle and marks the
count in every core's port. A receiver's compute waits for each chunk's mark in its own
port, so both of a receiver's movers are free for W: 5 of each 8 chunks on its writer
(NoC 0), 3 on its reader (NoC 1), Kohaku-TT's 5 : 3. The sender's reader, on NoC 1,
moves all of its own W. W comes in chunks of up to 64 KB, 4 deep a mover. X stays whole
in L1: 256 KB at K = 4096, 1.1 MB at K = 17408. The plan fits at all 16 decode shapes (L1
a core 0.5 to 1.34 MB). It does not fit when X whole leaves too little room for W's
chunks, above about K = 19,000 at M = 32 or K = 9,500 at M = 64; there the K split stays.
Its kernels need 115,248 B on two cards, so a two-card session with a decode shape opens
a 116 K kernel buffer. The cast's addresses lie below the L1's end. The picker prices
both plans and keeps the cheaper, with no flag (`--decode` forces one; `--cast-nocs`
and `--cast-pattern` set the movers' NoCs and the receivers' share). Loops:

| M x K x N | Cards | K split | Whole K: W on one mover | Whole K: W on both | Kohaku-TT | Both: W's read, its last core (GB/s) | X's cast to its last core |
|---|---|---|---|---|---|---|---|
| 32 x 4096 x 4096 | 1 | 95.3 | 86.8 | **86.1** | 81.0 | 77.0 (436) | 76.0 |
| 32 x 8192 x 8192 | 1 | 314.7 | 343.3 | **306.8** | 302.7 | 294.4 (456) | 279.1 |
| 32 x 5120 x 17408 | 1 | 412.3 | 435.8 | **399.6** | 400.4 | 376.8 (473) | 326.0 |
| 32 x 17408 x 5120 | 1 | 425.3 | 434.5 | **390.5** | 414.7 | 382.3 (466) | 339.8 |
| 32 x 3840 x 15360 | 1 | 293.2 | 288.6 | **257.9** | 273.4 | 248.6 (474) | 233.7 |
| 32 x 15360 x 3840 | 1 | **294.9** | 360.8 | 289.2 | 262.9 | 280.7 (420) | 280.7 |
| 32 x 2560 x 9216 | 1 | 124.9 | 124.7 | **113.5** | 108.2 | 104.8 (450) | 103.2 |
| 32 x 9216 x 2560 | 1 | **127.0** | 152.0 | 135.9 | 128.6 | 127.5 (370) | 127.5 |
| 32 x 4096 x 4096 | 2 | 58.0 | 64.7 | **58.4** | 44.7 | 47.8 (351) | 41.0 |
| 32 x 8192 x 8192 | 2 | 176.1 | 182.9 | **176.7** | 155.5 | 166.6 (403) | 147.9 |
| 32 x 5120 x 17408 | 2 | **227.0** | 261.5 | 221.9 | 204.3 | 213.4 (418) | 182.9 |
| 32 x 17408 x 5120 | 2 | **231.4** | 279.9 | 219.5 | 224.7 | 214.4 (416) | 200.5 |
| 32 x 3840 x 15360 | 2 | **156.5** | 169.5 | 152.0 | 134.3 | 142.8 (413) | 116.5 |
| 32 x 15360 x 3840 | 2 | **156.0** | 183.6 | 164.0 | 145.6 | 153.2 (385) | 138.4 |
| 32 x 2560 x 9216 | 2 | 71.8 | 70.8 | **67.9** | 54.1 | 58.2 (405) | 51.6 |
| 32 x 9216 x 2560 | 2 | **75.6** | 87.8 | 90.9 | 64.9 | 80.2 (294) | 58.3 |

W on both movers is faster than W on one at 15 of the 16 shapes, by 0.7 to 71.6 us; at
two cards 9216 x 2560 it is 3.1 us slower, its sender's W on NoC 1 the end. The bold
loop is the picker's choice between the K split and W on both. It is the faster plan at
10 of the 16 shapes. The six misses cost 28.2 us in all: on two cards 0.4 (4096 x 4096),
0.6 (8192 x 8192), 5.1 (5120 x 17408), 11.9 (17408 x 5120, priced 227.5 against 238.7
and measured 231.4 against 219.5) and 4.5 (3840 x 15360), on one card 5.7 (15360 x
3840). Five of them are within 4 % of a tie by measurement. The prices use the rates
fitted to the 16 loops of W on both (root mean square 4.0 %, the worst 8.9 %). The
whole-K plan reads W at 400 / 465 GB/s on one card and 390 / 415 on two, at ow 1 /
wider, and at most 7.2 GB/s on each core. That last term is the few-core shapes: two
cards 9216 x 2560 has 40 tiles of N an island, so 40 cores at ow 1, which read 295 GB/s
together. Its fixed term, 9 us, is the first cast and the tail after the last core's W:
the loop less the last core's own end is 3.7 to 12.9 us. The K split's prices are as
before (W and its second X at 470 GB/s); on two cards they run 1 to 3 % under its loops.

How the three whole-K forms compare, by the account. With W on one mover (the reader on
NoC 0, the writer taking X), the cores end in the order of their row, and W at the last
core ran at 307 to 427 GB/s. Moving a receiver's W onto both movers with X cast on NoC 1
(both movers' share 1 : 1) left the sender as the last core at every point: its X cast
on NoC 1 under half of every receiver's W took 81.6 us at two cards 4096 x 4096 (33.2
before). 5 : 3 toward NoC 0 shortened that cast (67.1 us) but then the sender's own W,
alone on NoC 0 with 5/8 of every receiver's, ended last. Kohaku-TT casts X on NoC 0 and
moves the sender's W on NoC 1. That is the form now: X's cast at two cards 4096 x 4096 is
41.0 us, the sender's W ends at 41.8 to 45.4. The last mover to end is a receiver's NoC 0
share at 10 of the 16 points, its NoC 1 share at one, and the sender at five: its W on NoC
1 at three two-card points, its X at one card's two ow 1 points. The loops of the 1 : 1 and 5 : 3 forms are in
`results/linear-whole-ab-*.json` and `results/linear-whole-5to3-*.json`.

What is left in the whole-K plan, against Kohaku-TT's loops:

- The tail, 3.7 to 12.9 us after the last core's W: the core's sum packed, Y written,
  and on two cards the islands' add. Kohaku-TT's two-card numbers have no such add (each
  card its half of N).
- W's rate at the last core, 351 to 418 GB/s on two cards and 420 to 474 on one, against
  Kohaku-TT's 364 to 449 for the same bytes. At ow 1 a core's chunk is K tiles of one
  column, which sit on one DRAM bank.
- X's cast is now paced with W, not after it: on one card it ends 0 to 14 % before the
  last core's W, and at the two ow 1 shapes it is the end. The sender reads X as one
  requester among the cores reading W.
- The few-core shapes: at two cards 9216 x 2560 there are 40 cores; the K split is faster
  there (75.6 against 90.9).

What is left at decode, and what would take it:
- The form only this framework's ports allow: the K split with a reduce-scatter inside
  the group in place of the tree. Each rank sends each other rank of its group that
  rank's share of its fp32 part, counted in that rank's port, and each rank adds the S
  shares of its own ow / S columns and writes them to Y. That is one exchange in place of
  log2 S hops of the whole block, each rank combining 1 / S of it. At two cards 32 x
  4096 x 4096 the estimate is a tail of about 7.5 us in place of 10.4 (the send's fixed
  0.8 us once, the bytes spread over S - 1 sends), so a loop of about 55 against 58.0.
  Not built.

**The column's step at M = 1024: the direct form.** Each rank of a column multicasting
its own rows does not fit the broadcast's order, for two reasons:

- A slot is free when every receiver has popped the block a window back. Only the
  sender knows that: the receivers mark it in the sender's table.
- VALID is one line a block, written by the sender alone, and it means "the whole block
  has landed". A receiver takes the block when VALID reaches it. With several ranks
  writing parts, the first part's VALID would release a block whose other parts are
  still in flight.

What would allow it is a counted VALID. Each rank would multicast its share into the
slot at its offset and wait for the acknowledgement. It would then add one to a count
on every receiver, with the NoC's multicast atomic increment (which leaves the sender
out of its rectangle, so each rank adds its own count locally). A receiver would take
the block at a count of R. The slot's freedom would reach the ranks as the spread's
ready marker already does. The parts of one column's multicasts go over the same links,
so the step would drop from the hand-over plus the cast to about the cast alone: at
156 tiles about 11 of 26 us for X's row (the hand-over 15) and W's alike. The same holds
for X's row step above. That is a new mode of the broadcast, not built here.

The same-form table again, after these changes (a dash: unchanged):

| M x K x N | One card: ours before / after | Kohaku-TT | Ours / Kohaku-TT: before / after | Two cards: ours before / after | Kohaku-TT | Ours / Kohaku-TT: before / after | What is left where ours is slower |
|---|---|---|---|---|---|---|---|
| 32 x 4096 x 4096 | 95.3 / 86.1 | 81.0 | 1.18 / 1.06 | 58.0 / 58.4 | 44.7 | 1.30 / 1.31 | (whole K) the tail (8.1 / 10.7: the sum, Y, two cards the islands' add) and W at the last core (436 / 351 GB/s) |
| 32 x 8192 x 8192 | 314.7 / 306.8 | 302.7 | 1.04 / 1.01 | 176.1 / 176.7 | 155.5 | 1.13 / 1.14 | (whole K) the tail (10.9 / 10.0) and W at the last core (456 / 403 GB/s) |
| 32 x 5120 x 17408 | 412.3 / 399.6 | 400.4 | 1.03 / 1.00 | 227.0 / - | 204.3 | 1.11 | two cards: the tree (13.3 / 8.8), the read under the ceiling, the picker's miss (whole K 221.9: 1.09) |
| 32 x 17408 x 5120 | 425.3 / 390.5 | 414.7 | 1.03 / 0.94 | 231.4 / - | 224.7 | 1.03 | two cards: the read under the ceiling, the tree (about 10), the picker's miss (whole K 219.5: 0.98) |
| 32 x 3840 x 15360 | 293.2 / 257.9 | 273.4 | 1.07 / 0.94 | 156.5 / - | 134.3 | 1.17 | two cards: the read under the ceiling, the tree (about 11), the picker's miss (whole K 152.0: 1.13) |
| 32 x 15360 x 3840 | 294.9 / - | 262.9 | 1.12 | 156.0 / - | 145.6 | 1.07 | the read under the ceiling, the tree (13.8); one card the picker's miss (whole K 289.2: 1.10) |
| 32 x 2560 x 9216 | 124.9 / 113.5 | 108.2 | 1.15 / 1.05 | 71.8 / 67.9 | 54.1 | 1.33 / 1.26 | (whole K) the tail (7.6 / 9.6) and W at the last core (450 / 405 GB/s) |
| 32 x 9216 x 2560 | 127.0 / - | 128.6 | 0.99 | 75.6 / - | 64.9 | 1.16 | the tree (about 12); two cards: 40 tiles of N an island, so the whole-K plan has 40 cores |
| 1024 x 4096 x 4096 | 222.6 / - | 190.3 | 1.17 | 162.3 / 126.3 | 110.3 | 1.47 / 1.15 | the column's and the row's steps (W 53 + 38, X 55 + 37) |
| 1024 x 8192 x 8192 | 698.8 / - | 711.2 | 0.98 | 380.6 / - | 382.7 | 0.99 | |
| 1024 x 5120 x 17408 | 961.5 / - | 1,015.9 | 0.95 | 518.8 / - | 500.4 | 1.04 | two cards: the column's step |
| 1024 x 17408 x 5120 | 924.0 / - | 884.3 | 1.04 | 549.6 / 503.6 | 485.5 | 1.13 / 1.04 | the column's step; two cards also the row's |
| 1024 x 3840 x 15360 | 772.5 / - | 697.5 | 1.11 | 355.0 / - | 340.2 | 1.04 | the column's step |
| 1024 x 15360 x 3840 | 602.5 / - | 655.4 | 0.92 | 442.8 / 399.8 | 328.9 | 1.35 / 1.22 | two cards: the row's step (285 + 229) |
| 1024 x 2560 x 9216 | 356.3 / - | 307.6 | 1.16 | 309.1 / 193.3 | 159.2 | 1.94 / 1.21 | the column's step; two cards: X's cast (125) |
| 1024 x 9216 x 2560 | 284.1 / 273.3 | 271.2 | 1.05 / 1.01 | 238.1 / 191.9 | 153.4 | 1.55 / 1.25 | two cards: the row's and the column's steps |
| 4096 x 4096 x 4096 | 656.7 (single) / - | 819.8 (single) | 0.80 | 455.9 / 430.8 | 412.9 | 1.10 / 1.04 | the row's step beside the matrix unit |
| 4096 x 8192 x 8192 | 2,312.7 / 2,283.4 | 2,371.3 | 0.98 / 0.96 | 1,321.9 / 1,295.0 | 1,428.0 | 0.93 / 0.91 | |
| 4096 x 5120 x 17408 | 3,011.7 / 2,958.6 | 6,192.8 | 0.49 / 0.48 | 1,762.9 / 1,637.3 | 2,665.6 | 0.66 / 0.61 | |
| 4096 x 17408 x 5120 | 2,965.8 (single) / - | 3,510.0 (single) | 0.84 | 1,834.9 / 1,572.6 | 1,542.2 | 1.19 / 1.02 | the row's step (1,116 + 598) |
| 4096 x 3840 x 15360 | 2,166.3 / 1,950.1 | 3,613.6 | 0.60 / 0.54 | 1,174.4 / 1,156.9 | 1,758.5 | 0.67 / 0.66 | |
| 4096 x 15360 x 3840 | 1,916.7 / - | 1,921.4 | 1.00 | 1,603.5 / 1,159.3 | 1,009.9 | 1.59 / 1.15 | the row's step (718 + 548) |
| 4096 x 2560 x 9216 | 893.5 / 870.0 | 1,035.9 | 0.86 / 0.84 | 528.8 / 526.1 | 537.0 | 0.98 | |
| 4096 x 9216 x 2560 | 978.3 / 885.7 | 875.7 | 1.12 / 1.01 | 949.2 / 681.5 | 436.8 | 2.17 / 1.56 | the row's step (426 + 335: 24 steps of 26 us) |

On two cards the largest ratio went from 2.17 to 1.56. 16 of the 24 two-card rows
changed (two of them slower, by 0.4 and 0.6 us: the picker's misses at 32 x 4096 x 4096
and 32 x 8192 x 8192), and 12 one-card rows. All that is left where ours is slower is
OURS. At decode it is the tree and X's second read, which Kohaku-TT's form has not, and
the read under the ceiling; where the picker takes Kohaku-TT's form, its tail and W's
rate at the last core; and at five two-card shapes and one one-card shape the picker's
miss, 0.4 to 11.9 us. At prefill it is the step of a broadcast: the hand-over into the
sender, then the cast. Casting every rank's share itself, each into every core's slot
(several casters a block), was tried and is slower: 1024 x 4096 x 4096 went from 126.3
to 260.9 us and 4096 x 4096 x 4096 from 430.8 to 565.0 on two cards, because each step
then waits for the slot to be free on every core and for every caster's increment
([experiments/noc/casters](../../experiments/noc/casters/README.md#tried-in-the-linear-prefill),
`results/linear-counted-0x1.json`).

Measured on 2026-11-15 between 10:26 and 11:06, the whole-K plan between 12:05 and
12:16, with W on both movers between 12:47 and 12:59. JSON:
`results/linear-xspread-0x1.json` and `-0.json` (X's spread),
`results/linear-priced-0x1.json` and `-0.json` (the priced plans; at 4096 cubed,
4096 x 8192 x 8192 and the M = 4096 2560 x 9216 / 9216 x 2560 rows on two cards, the
first fit's plans),
`results/linear-tree-0x1.json` (the reduce's clocks, and the refit's 4096 x 8192 x 8192
and 4096 x 2560 x 9216), `results/linear-tree-0.json`, `results/linear-tree-aaab-*.json`
and `results/linear-tree-5to3-0x1.json`,
`results/linear-hops-*.json` (the hops; the four-tile combine),
`results/linear-hops-5to3-*.json`, and `results/linear-whole-0x1.json` and `-0.json`
(the whole-K plan at all 16, as `--decode whole` runs it, W on one mover),
`results/linear-whole-movers-0x1.json` and `-0.json` (W on both movers, as the plan now
runs), `results/linear-whole-ab-*.json` and `results/linear-whole-5to3-*.json` (X cast on
NoC 1, a receiver's W 1 : 1 and 5 : 3), `results/linear-whole-depth8-*.json` (8 deep) and
`results/linear-whole-serial-*.json` (its first run, the cast not overlapped). The unpacker's clocks of the first of these runs
did not land. The firmware sets a buffer's read pointer only on the unpacker and its
write pointer only on the packer. Compute's row of T went through the write pointer, so
on the unpacker it pointed near address 0, and the grid compute's input-wait word had
done the same. The unpacker now uses the read pointer.

```
python projects/dense/linear.py --shapes "1024,9216,2560:out" --cards 0,1 --xspread on --account
python projects/dense/linear.py --shapes "32,4096,4096:out" --cards 0,1 --account --pattern ABABAABA
python projects/dense/linear.py --shapes "32,4096,4096:out" --cards 0 --account --decode whole
python projects/dense/linear.py --shapes "32,4096,4096:out" --cards 0,1 --account --decode whole --cast-nocs 0,1 --cast-pattern AB
```

## Shapes, one card and two, against Kohaku-TT

Y = X x W in bfloat16 over fifteen shapes: ours on one card, and
[Kohaku-TT](https://github.com/KohakuBlueleaf/Kohaku-TT) (commit ac81ff2) on one card and
two, on the same inputs and the same check (relative RMS 3.96e-3; Kohaku-TT 3.96e-3 and,
split along K, 3.86e-3; every point exact to that bound). Our two-card rows of this
sweep were of a form that is neither standard one (X and W split along K, Y along N,
the other island's columns put across from inside the kernel) and left with it; the
standard forms are in "A linear layer on one card and two" above
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
- Its K-split form (X and W split along K, the partial completed to Y split along N;
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

Times in microseconds, "single" the median of synced calls and "loop" calls back to back for
about three seconds (a cool-down of the same length between points), one run each; ratios are
Kohaku-TT's one card over its two cards on the single calls, the loops' in brackets (a one-card loop is
throttled by the card's power limit, so the loop ratio flatters two cards). Ours: three
single calls. Kohaku-TT: one card three calls; on two cards the steady median of nine calls
(the first calls after the first product are slow and the median of three had caught them:
2048 x 8192 x 2048 gave 747 / 1,021 / 315 us), and its loop of the first run. On two cards
its synced call has a floor of about 1.5 to 3 ms in some runs (2048 x 16384 x 2048: 3,000 us in a
run of nine calls, 508 us in a run of three, 851 us a call in the loop): below about 5 ms
the two-card single is the synchronization, not the product, and its loop is the better figure.

| M x K x N | Ours 1 card | Kohaku-TT 1 card | Kohaku-TT 2 cards | ratio | Kohaku-TT 2 cards, K split | ratio |
|---|---|---|---|---|---|---|
| 8192x65536x8192 | 43,698.4 / 107,034.9 | 34,544.0 / 95,826.2 | 20,452.8 / 58,286.5 | 1.69 (1.64) | 18,357.9 / 48,696.6 | 1.88 (1.97) |
| 4096x32768x4096 | 4,421.3 / 12,210.0 | 5,393.5 / 14,906.9 | 2,794.7 / 7,427.0 | 1.93 (2.01) | 3,152.8 / 7,446.3 | 1.71 (2.00) |
| 2048x16384x2048 | 883.9 / 1,097.0 | 792.2 / 1,823.0 | 2,999.7 / 851.3 | 0.26 (2.14) | 2,997.6 / 765.0 | 0.26 (2.38) |
| 8192x32768x8192 | 22,034.0 / 52,647.2 | 17,331.4 / 47,951.6 | 10,324.4 / 29,266.7 | 1.68 (1.64) | 9,889.3 / 24,675.5 | 1.75 (1.94) |
| 4096x16384x4096 | 2,271.0 / 6,232.9 | 2,824.0 / 7,365.9 | 1,503.3 / 1,435.8 | 1.88 (5.13) | 1,842.5 / 3,525.2 | 1.53 (2.09) |
| 2048x8192x2048 | 471.4 / 576.0 | 472.7 / 872.0 | 317.0 / 197.5 | 1.49 (4.42) | 476.3 / 269.2 | 0.99 (3.24) |
| 4096x16384x16384 | 23,388.8 / 27,295.1 | 8,955.3 / 23,754.6 | 4,936.4 / 12,588.7 | 1.81 (1.89) | 6,076.8 / 12,774.3 | 1.47 (1.86) |
| 16384x16384x16384 | 75,169.4 / 90,231.8 | 32,489.3 / 95,676.9 | 17,067.8 / 47,303.6 | 1.90 (2.02) | 32,970.6 / 43,996.1 | 0.99 (2.17) |
| 8192x16384x8192 | 11,161.3 / 25,719.0 | 8,777.9 / 24,630.6 | 5,257.1 / 14,485.0 | 1.67 (1.70) | 5,672.0 / 11,923.8 | 1.55 (2.07) |
| 4096x8192x4096 | 1,199.8 / 2,961.6 | 1,551.6 / 3,551.8 | 865.1 / 1,745.2 | 1.79 (2.04) | 1,113.8 / 1,190.4 | 1.39 (2.98) |
| 2048x4096x2048 | 248.3 / 307.8 | 311.0 / 387.8 | 255.9 / 109.2 | 1.22 (3.55) | 331.5 / 190.5 | 0.94 (2.04) |
| 8192x8192x8192 | 5,716.0 / 12,463.3 | 4,549.5 / 12,094.0 | 2,482.8 / 5,905.9 | 1.83 (2.05) | 3,930.1 / 4,796.2 | 1.16 (2.52) |
| 4096x4096x4096 | 664.6 / 1,539.5 | 819.8 / 1,531.3 | 487.3 / 345.1 | 1.68 (4.44) | 868.8 / 679.7 | 0.94 (2.25) |
| 8192x2048x8192 | 1,713.5 / 2,358.1 | 1,725.8 / 3,010.0 | 879.0 / 778.1 | 1.96 (3.87) | 2,164.0 / 2,103.1 | 0.80 (1.43) |
| 16384x4096x16384 | 19,557.4 / 22,825.5 | 14,782.2 / 20,692.6 | 7,784.0 / 10,721.3 | 1.90 (1.93) | 12,240.6 / 14,096.5 | 1.21 (1.47) |

The card clock of each loop:

| M x K x N | AICLK mean, MHz: ours 1, Kohaku-TT 1, Kohaku-TT 2 |
|---|---|
| 8192x65536x8192 | 1,023 / 1,020 / 1,030 |
| 4096x32768x4096 | 1,051 / 1,010 / 1,010 |
| 2048x16384x2048 | 1,081 / 1,008 / 997 |
| 8192x32768x8192 | 1,030 / 1,021 / 1,026 |
| 4096x16384x4096 | 1,017 / 1,014 / 1,299 |
| 2048x8192x2048 | 1,079 / 1,010 / 1,298 |
| 4096x16384x16384 | 1,150 / 1,037 / 1,016 |
| 16384x16384x16384 | 1,115 / 1,012 / 1,016 |
| 8192x16384x8192 | 1,011 / 1,003 / 1,032 |
| 4096x8192x4096 | 1,048 / 1,025 / 1,015 |
| 2048x4096x2048 | 1,093 / 1,003 / 1,350 |
| 8192x8192x8192 | 1,023 / 1,013 / 1,022 |
| 4096x4096x4096 | 1,037 / 1,018 / 1,344 |
| 8192x2048x8192 | 981 / 999 / 1,314 |
| 16384x4096x16384 | 1,154 / 951 / 958 |

Measured on 2026-11-15 between 02:11 and 03:08, host load average 4 to 11, X280 at 1,350 MHz; JSON in `results/shape-<M>x<K>x<N>-<ours|ktt>-<0|0x1>[-<variant>].json`
(the cubes of ours are the curve's above; the nine-call runs are in `results/singles9/`).
Ours: `python projects/dense/linear.py --m M --k K --n N --cards 0 --seconds 3`. Kohaku-TT:
`python projects/dense/compare/ktt_shapes.py --cards 0|0,1 --shapes MxKxN,... --variants default[,ksplit]`.

Kohaku-TT's best two-card ratio on single calls is 1.96 at 8192 x 2048 x 8192 for its
row split (1.93 at 4096 x 32768 x 4096; its K split 1.88 at 8192 x 65536 x 8192). Our one
card against its one card is the next section.

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
| `linear.py`, `kernels/linear/linear.h`, `reader.cpp`, `compute.cpp`, `writer.cpp` | the linear layer written for islands: the form from the placements (output dim sharded, input dim sharded with the addition, W gathered), the grid plan and the K split, the gather into a copy of Y, the account's stamps |
| `ceilings.py`, `kernels/ceilings/` | the card's two ceilings: DRAM read by every rank on both movers, the matrix unit on resident tiles |
| `flat.py` | the grid form, kernels unchanged, on one flat group over the islands: one plan for the islands' grids one below the other, every tensor held once |
| `results/` | where the runs write their JSON (not kept in git; each table names its command) |
| `compare/shapes.py`, `compare/ktt_shapes.py` | the shapes of the comparison with Kohaku-TT on one card and two, their table, and Kohaku-TT's side of it |
| `reference/ktt_bench.py` | Kohaku-TT's product on the same inputs, timed and checked the same way |
| `bench_ttnn.py` | TT-NN's product and SwiGLU on the same inputs; it imports `ttnn` and `torch` from the tt-metal tree `TT_METAL_HOME` names (Komorebi needs neither) |
| `reference/gpu_bench.py`, `reference/gpu/gemm_gpu.cu` | cuBLAS on one idle GPU (fp32 and fp16) |
