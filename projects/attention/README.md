# Attention

`out = softmax(q k^T scale) v` (forward), q `[B, H, S, D]`, k and v `[B, Hk, S, D]`
with H a multiple of Hk (grouped heads), scale 1 / sqrt(D), optionally causal (query
row i sees keys up to i), bf16 operands, fp32 accumulation, bf16 output, HiFi2, on the
persistent kernels of one Blackhole p150a. Commands reach the grid through the dispatcher on X280 cluster 0 (`komorebi.Session`, `Run.submit`). The compute body is
Kohaku-TT's `ktt_flash` kernel's.

## What each core does

One call is one `ATTEND` command on every core. Each core works through its job table
(one DRAM page per core, named by the command: `job_tables`), a job being a
(batch, head, row range) of query rows:

1. `kernels/reader.cpp` (NoC0): reads the core's job table and hands it to compute
   (the writer reads its own copy). Per job it reads the job's query rows into the Q
   buffer, then, for each block of `BLOCK_KEYS` key tiles of the job's head, takes part
   in the broadcast (`km::Broadcast`): the sender (rank 0) reserves the slot, reads K
   and V of the block from DRAM into it and `cast`s it; every other core of the
   broadcast `arrive`s. After the last block, `finish`.
2. `kernels/compute.cpp` and `kernels/sfpu.hpp`: Kohaku-TT's `ktt_flash` compute body.
   For each block taken from the broadcast (`km::broadcast_receive`), in chunks of 3 key
   tiles and, within a chunk, for the core's query rows two rows at a time: S = Q K^T
   into Dst with the row state, the row's reference and the exponential on the vector
   unit, P V and the row sum accumulated in L1 (fp32) in column groups. The block is
   handed back (`km::broadcast_done`, and `km::broadcast_end` after the last).
3. `kernels/writer.cpp` (NoC1): writes each finished row's `VALUE_TILES` tiles to O.

A broadcast is L cores down one grid column; the core on top (rank == 0) reads each K / V
block from DRAM once and multicasts it to the others. The compute is a running softmax:
the row's reference, P = exp(scale s - reference), O += P V and the row sum, both scaled
when the reference moves. Nothing is merged across cores: each core finishes its own
output rows. Rows are shared contiguously, or for a causal call L rows apart, so that
the triangle's work spreads over the broadcast.

The command goes to the group of every core, with `KM_FLAG_EXCLUSIVE` (a command
with broadcasts is never in flight beside an overlapping group). A K / V block is one broadcast
block: K's tiles head-dim-major `[HEAD_TILES][BLOCK_KEYS]`, then V's key-major `[BLOCK_KEYS][VALUE_TILES]`, two blocks
in flight per broadcast. The broadcasts and their senders come from `komorebi.layout.broadcast_args`
(one band of L rows of the grid after the other).

| Handler | Arguments |
|---|---|
| `ATTEND` | a0 q_address, a1 k_address, a2 v_address, a3 o_address (Q, K, V, O), a4 job_tables, a5 mode (0: the full call, 1: broadcast only, 2: compute only), a6 requested_passes over the jobs (0: one), a7 first_job, a8 job_count (jobs from it, 0: to the end) |
| `PROF` | a0 profile_address (the profile tensor), a1 zero_profile (nonzero: zero the pages after storing them) |

The cut-down modes are for the benchmark: broadcast only reads Q and multicasts every K / V block
but compute only takes and pops the blocks, and no row is written; compute only works on
Q, K and V as they are in L1 (no reads, no broadcast, no writes). Passes repeat the command's
jobs, so that a loop of commands lasts seconds.

`plan.py` is the plan as pure functions, with Kohaku-TT's rules: MAX_ROWS query tile rows per
job and BLOCK_KEYS key tiles per block fitted to the L1 budget (from 13 and 12 down), the broadcast
length L with the fewest score tiles on the busiest core (then the longest broadcast), broadcasts
taking even ranges of the (batch, head, row) order (causal: chunks largest work first
onto the least loaded broadcast), at most 46 jobs per core. `card.py` loads the program and
submits the commands; `run.py` runs one shape and checks it. The references in
`reference/` are the float64 attention (`sdpa.py`) and Kohaku-TT's own kernel on the
same inputs (`ktt_bench.py`, by hand, not a test).

`kernels/sfpu.hpp` is an unchanged copy of `ktt_flash`'s `sfpu.hpp` (Kohaku-TT, Apache
2.0, at ac81ff2); `kernels/compute.cpp` is its `compute.cpp` with K and V taken from the
broadcast buffer, the jobs from the table, its waits on the other kernels timed (see
Limits), and the parts this program does not use left out (masks, packed variable
lengths, statistics for the backward, ragged lengths).

## Run

```
python projects/attention/run.py --shape t1
python projects/attention/run.py --shape long64 --calls 4
python projects/attention/run.py --shape t1 --cards 0,1 [--far] --save o.npy   # two cards
python projects/attention/bench.py --shape long64 --cuts            # time, power, profile
python projects/attention/bench.py --shape wide128 --noc 1 --broadcast 5  # the broadcast's NoC, length
python projects/attention/graph.py --shape short64 --rounds 40       # commands and graphs
python projects/attention/bench.py --shape causal64 --broadcast 5 --chunks 6 --cuts
KMR_CARD=1 python -m pytest tests/hardware/test_attention_host.py
python projects/attention/reference/ktt_bench.py --shape long64   # Kohaku-TT's env
```

`--cards 0,1` runs the same kernels on a session of two cards: the two grids one below
the other as one grid of every column but the last (an island's last core is its link
core), broadcasts inside an island, one ATTEND on every flat rank; the job tables, the
profile pages and Q, K, V, O are read and written by the page calls (`km::read_pages`,
`km::write_pages`), so a page another island holds crosses. `--far` puts O on island 1
alone. `--save` keeps O's pages: t1 on cards 0 and 1, with O split or on island 1, gave
O equal to one card's bit for bit (2026-11-15).

`--shape` names a preset of `plan.SHAPES` (B x H x S x D, K / V heads Hk): `t1` 1 x 1 x
128 x 64, `t2` 1 x 4 x 512 x 64, `t3` 1 x 2 x 256 x 128 causal with Hk 1, `long64` 1 x 32
x 4096 x 64, `wide128` 1 x 24 x 4096 x 128, `short64` 8 x 16 x 256 x 64, `causal64` 1 x 8
x 1024 x 64 causal, `gqa128` 2 x 8 x 1024 x 128 with Hk 2. Setup is that of
[01_mapcollect](../../examples/01_mapcollect/README.md#what-it-needs). Inputs are
standard normal rounded to bf16, from a seed (`reference/sdpa.py`, `inputs`). Lengths
and head dims are whole tiles (multiples of 32).

`run.py` reads back the output tiles of 16 sampled query tile rows (32 query rows each),
not the whole tensor, compares them with the float64 attention of the same bf16
operands and exits 1 when the check fails. `reference/ktt_bench.py` runs Kohaku-TT's
`ktt_flash` (forced with `kohakutt.sdpa_kernel("flash")`) on the same inputs in
Kohaku-TT's own environment and checks the same rows the same way.

**The check's bound.** Relative RMS error at most 1.5e-2 and largest error at most 6e-2
of the largest output, with the accurate exponential. Kohaku-TT documents about 6e-3
RMS for `ktt_flash` at HiFi2 (the matrix unit keeps 7 of Q's 8 significand bits; the
exponential's polynomial is within 1.75e-3); the runs below give 4.2e-3 to 6.7e-3 RMS
and 4.7e-3 to 1.5e-2 largest, and Kohaku-TT's kernel gives the same numbers on the same
rows.

## Results

Correctness, measured on 2026-11-13 on one p150a (card clock 1,350 MHz), ours and
Kohaku-TT's `ktt_flash` at ac81ff2 one after the other in the same session. Rows `t1` to
`t3`: `run.py --shape <preset> --calls 4 --json results/run-<preset>.json` and
`reference/ktt_bench.py --shape <preset> --seconds 1 --json results/ktt-<preset>.json`;
the other rows: the `check` of `results/ours-<preset>.json` and
`results/ktt-<preset>.json` (see "Time and power"). Both are compared with float64 on
the same sampled query tile rows (16, or all 4 of `t1`).

| Preset | Plan: rows, block keys, length | Ours: RMS / largest | Kohaku-TT: RMS / largest |
|---|---|---|---|
| `t1` | 13, 12, 10 | 5.50e-3 / 9.02e-3 | 5.50e-3 / 9.02e-3 |
| `t2` | 13, 12, 10 | 6.19e-3 / 7.88e-3 | 6.19e-3 / 7.88e-3 |
| `t3` (causal, Hk 1, D 128) | 13, 12, 10 | 4.18e-3 / 4.72e-3 | 4.18e-3 / 4.72e-3 |
| `long64` | 13, 12, 5 | 6.65e-3 / 1.47e-2 | 6.65e-3 / 1.47e-2 |
| `wide128` | 13, 12, 10 | 6.32e-3 / 9.46e-3 | 6.32e-3 / 9.46e-3 |
| `short64` | 13, 12, 2 | 5.87e-3 / 7.52e-3 | 5.87e-3 / 7.52e-3 |
| `causal64` | 13, 12, 10 | 5.71e-3 / 7.53e-3 | 5.71e-3 / 7.53e-3 |
| `gqa128` | 13, 12, 5 | 6.28e-3 / 8.91e-3 | 6.28e-3 / 8.91e-3 |

The two kernels' statistics agree to every printed digit: the compute body and its
operand order are the same.

### Time and power

Measured on 2026-11-13 on one p150a with the X280 at 1,000 MHz (the session's default),
each row's output committed in `results/` (`ours-<preset>[-approx].json` from `bench.py
--json`, `ktt-<preset>[-approx].json` from `reference/ktt_bench.py --json`),
ours (`bench.py --shape <preset> [--approx] --cuts`) and Kohaku-TT's `ktt_flash` at
ac81ff2 (`reference/ktt_bench.py`) one after the other in the same session, 15 s apart;
host load (1 min) 3 to 9. Rows 8x16x256x64, 1x8x1024x64 causal and 2x8x1024x128 were
measured again on 2026-11-14 (X280 at 1,000 MHz, host load 6 to 7; `bench.py --shape
<preset> --cuts`, best of three, Kohaku-TT's columns not run again): their per call fell
by 8 to 21 %, the single command did not change (the commands now follow each other
1.0 to 1.2 us apart instead of 15.9 and 16.5). The other two exact rows (4096 x 64
and 4096 x 128) moved by less than 1 % in one run and stay. Of the approximate rows,
8x16x256x64 and 1x8x1024x64 causal were measured again on 2026-11-14 the same way
(`--approx --cuts`, best of three): their per call fell by 19 and 23 %, the single
command did not change; the approximate 4096 x 64 row moved by 1 % and stays. Per
call: ours, single-pass commands back to back (the time of 10 to 60 commands less
that of 5, divided); ours per pass, the power loop of multi-pass
commands (no command boundary); Kohaku-TT, its calls back to back for about 4 s. Clock
(AICLK mean), power (whole card, mean of the loop) and temperature are the power
loops', ours / Kohaku-TT's. RMS: relative RMS error of the sampled rows, the same for
both kernels on every row.

| Shape (B x H x S x D) | Ours per call (us) | Ours per pass (us) | Kohaku-TT per call (us) | MHz | W | C | RMS |
|---|---|---|---|---|---|---|---|
| 1x32x4096x64 | 2675.3 | 2646.0 | 2918.5 | 1347 / 1340 | 138 / 135 | 73 / 76 | 6.65e-3 |
| 1x24x4096x128 | 2328.3 | 2439.1 | 3001.8 | 1267 / 1244 | 144 / 132 | 72 / 70 | 6.32e-3 |
| 8x16x256x64 | 76.7 | 65.1 | 80.7 | 1349 / 1350 | 123 / 110 | 68 / 65 | 5.87e-3 |
| 1x8x1024x64 causal | 58.1 | 48.3 | 53.7 | 1350 / 1350 | 103 / 99 | 66 / 66 | 5.71e-3 |
| 2x8x1024x128, Hk 2 | 149.9 | 133.9 | 261.8 | 1303 / 1350 | 136 / 105 | 69 / 66 | 6.28e-3 |
| 1x32x4096x64, approx | 1932.5 | 1996.2 | 2183.7 | 1295 / 1284 | 143 / 140 | 69 / 71 | 1.94e-2 |
| 8x16x256x64, approx | 66.5 | 55.3 | 67.5 | 1326 / 1350 | 124 / 117 | 69 / 67 | 2.14e-2 |
| 1x8x1024x64 causal, approx | 47.0 | 38.7 | 44.4 | 1350 / 1350 | 108 / 103 | 66 / 66 | 2.21e-2 |

The 4096 x 128 loops, the 2 x 8 x 1024 x 128 one of 2026-11-14 and the approximate 4096 x 64 ones ran below 1,350 MHz (the card's
own clock control under load), both kernels alike. With the approximate exponential the
check's relative RMS bound is 3e-2 (Kohaku-TT documents about 1.9e-2).

**What bounds each row.** Each core keeps a profile page (`PROF`): wall-clock cycles of
its five threads' parts, summed over the commands, read after 15 to 65 single-pass
commands. The shares below are of the slowest core's longest thread; threads run side by
side, so they add up to more than 1. Cut-down runs: broadcast only and compute only, us per
pass.

| Shape | Slowest core's shares | Broadcast only / compute only | Bound |
|---|---|---|---|
| 1x32x4096x64 | score 0.48, P V 0.45 (math); exp 0.55, pack P 0.26 (pack) | 310 / 2639 | compute, both threads |
| 1x24x4096x128 | score 0.45, P V 0.45; exp 0.52, pack P 0.22 | 323 / 2453 | compute |
| 8x16x256x64 | score 0.29, P V 0.28, exp 0.32; Q reads 0.15, compute waiting for Q 0.16 and for K / V 0.16 | 55 / 65 | the command: 10.6 us of each command's 75.6 is outside the passes (table, broadcast start and end, the first Q and K / V reads), and back-to-back commands add 1.0 us each |
| 1x8x1024x64 causal | score 0.35, P V 0.39, exp 0.43; waiting for the broadcast 0.58 | 31 / 59 | the command as above (56.9 us a command, 1.2 us between commands), and balance: the mean core is busy 0.56 of the slowest |
| 2x8x1024x128, Hk 2 | compute waiting for K / V 0.67; score 0.14, P V 0.14 | 83 / 134 | compute (compute only = per pass) |
| 8x16x256x64, approx | score 0.24, P V 0.23, exp 0.24; Q reads 0.19, compute waiting for Q 0.20 and for K / V 0.20 | 55 / 54 | the command: 10.5 us of each command's 65.2 is outside the passes, and back-to-back commands add 1.3 us each |
| 1x8x1024x64 causal, approx | score 0.33, P V 0.33, exp 0.34; waiting for the broadcast 0.60 | 32 / 49 | the command (45.8 us a command, 7.0 outside the passes, 1.2 us between commands), and balance: the mean core is busy 0.57 of the slowest |

Compute only runs on whatever Q, K and V are left in L1, which takes the rescaling
branch more often than real inputs do; it is an upper bound on the compute. The short
rows are ahead of Kohaku-TT per pass. Per call, 8x16x256x64 is ahead (76.7 against 80.7
us) and the causal row is 8 % behind (58.1 against 53.7): what is behind is the cost of a
command, now 1.0 to 1.3 us between commands. The approximate rows, measured again on
2026-11-14, are alike: per call the approximate 8x16x256x64 is ahead of Kohaku-TT (66.5
against 67.5 us) and the causal one 6 % behind (47.0 against 44.4).

### The broadcast's NoC, the broadcast length, commands and graphs

Measured on 2026-11-13 as above (X280 at 1,000 MHz), `wide128` (1 x 24 x 4096 x 128);
the JSON in `results/`: `noc0-wide128.json`, `noc1-wide128.json`, `broadcast-wide128-L<L>.json`
(`bench.py --json`) and `graph-<preset>.json` (`graph.py --json`).
NoC1: `--noc 1` builds the program with `BROADCAST_ON_WRITER`, so that the writer runs the
broadcast (K / V reads and multicasts before each job's rows) and the reader reads only Q. The
broadcast length is `--broadcast L` (L = 1: every core reads its own K / V).

| `wide128` | Single command (us) | Per call (us) | Per pass (us) | Broadcast only / compute only (us) | MHz | W |
|---|---|---|---|---|---|---|
| NoC0, L 10 | 2312.3 | 2330.3 | 2337.3 | 310 / 2393 | 1321 | 152 |
| NoC1, L 10 | 2276.5 | 2358.0 | 2309.6 | 597 / 2447 | 1315 | 153 |
| NoC0, L 5 | 2300.1 | 2315.3 | 2318.5 | 384 / 2352 | 1318 | 154 |
| NoC0, L 2 | 2322.4 | 2338.6 | 2333.5 | 773 / 2504 | 1307 | 149 |
| NoC0, L 1 | 2404.8 | 2418.0 | 2381.8 | 1533 / 2642 | 1289 | 153 |

The broadcast alone takes twice as long from the writer on NoC1 as from the reader on
NoC0, and five times as long with no broadcast (L 1) as with broadcasts of 10; the call, bound by
compute, moves by at most 4 %.

`graph.py` submits the same call, rounds back to back: one command walking the job table,
or one command per job index (a7, a8: each core's job j in command j); each from the host
(commands back to back, the time from the first submit to the last event over the rounds)
and from a kept graph (the call's commands in a chain, then a CPU node that goes round
again; the engine on the X280 submits each next command; the time of a restart over its
rounds, the median of three).

| Shape | Rounds | Commands per call | Host, table | Graph, table | Host, per job | Graph, per job |
|---|---|---|---|---|---|---|
| 8x16x256x64 | 40 | 3 | 97.4 | 85.1 | 210.2 | 134.5 |
| 1x8x1024x64 causal | 40 | 1 | 79.5 | 66.4 | 90.6 | 65.7 |
| 1x32x4096x64 | 8 | 4 | 2701.5 | 2685.4 | 3430.5 | 3331.2 |

Times are us per call. From the kept graph a call comes 12 to 13 us sooner than from
the host (part of that is the host's first submit, spread over the rounds). A command per
job costs 25 us per extra command on the short shape and 215 us on the long one: each
command starts its broadcasts with no block in flight and ends them with every core drained,
which the job table inside one command overlaps.

### The causal row's balance

In the default plan of `causal64` (1 x 8 x 1024 x 64 causal) each of the 8 heads is one
chunk of 32 query tile rows, so 4 of the 12 broadcasts of 10 have nothing to do and a broadcast's
cores take 3 or 4 rows: the mean core is busy 0.56 of the slowest. `--chunks n` cuts
each head into n chunks of about equal area of the triangle (`plan.causal_cuts`) and
places them largest first by the busiest core's work; with `--broadcast` it gives the
candidates below (`bench.py --shape causal64 --broadcast L --chunks n --cuts --seconds 2`,
the JSON in `results/causal64-L<L>-c<n>.json`; 2026-11-13, X280 at 1,000 MHz, AICLK
1,349 to 1,350 MHz).

| Broadcast length, chunks per head | Score tiles, busiest core | Jobs per core | Balance | Single command (us) | Per call (us) | Per pass (us) | Broadcast only / compute only (us) | W |
|---|---|---|---|---|---|---|---|---|
| 10, 1 (the default) | 68 | 1 | 0.56 | 56.9 | 72.3 | 48.4 | 36 / 63 | 104 |
| 10, 3 | 56 | 2 | 0.88 | 59.9 | 76.5 | 51.9 | 45 / 61 | 101 |
| 5, 6 | 50 | 2 | 0.91 | 60.2 | 75.9 | 46.9 | 47 / 57 | 114 |
| 2, 8 | 54 | 2 | 0.70 | 66.9 | 80.9 | 48.9 | 62 / 59 | 122 |
| 1, 31 | 40 | 2 | 0.66 | 107.1 | 127.0 | 97.3 | 101 / 52 | 95 |

The chunks even the cores (0.56 to 0.91) and cut the busiest core's score tiles by up to
41 %, but the call does not get faster: a second job per core adds its own start (Q
rows, the first K / V block, the rows' normalization), more broadcasts read K / V (the cast
only time rises from 36 to 47 us), and without a broadcast (L 1) the reads bound it. The best
candidate gains 3 % per pass and loses 5 % per call, so the default stays; `--chunks`
remains for shapes where a head's work is large against those costs.

## Limits

Timed waits (a give-up is recorded with its code and place, stops the core's three
kernels and is reported by the run):
- the reader: room in the job table, Q and broadcast buffers (`reader.reserve`); its NoC reads
  (`flushed`); every `km::Broadcast` wait (a free slot, the receivers' markers, VALID, the
  end of the command);
- the writer: each finished output row (`writer.wait_front`); its NoC writes (`flushed`);
- the compute kernel's waits on the data-movement kernels: the job table, Q rows and K /
  V blocks (`compute.wait_front`, decided by the unpacker and passed to the other two) and
  room for an output row (`compute.reserve_back`, the packer).

Not timed: the compute kernel's waits between its own three threads, as in
`ktt_flash`: the score and P tiles and their tokens (`cb_wait_front`, `cb_reserve_back`
on `cb_p`, `cb_stok`, `cb_otok`), the destination-register handshake (`tile_regs_*`) and
the semaphore and SFPU mutex waits in the vector sections. They wait only on work the
same kernel has already started.

## Future work

Not in this program: decode against a kept K / V cache (one query row against the
cache: a reduce over a group and routed messages), the backward, packed
variable-length batches, masks, ragged lengths.

## Files

| File | What it is |
|---|---|
| `kernels/attention.h` | handler, command words, buffers, the job table, the timed NoC barrier |
| `kernels/reader.cpp`, `compute.cpp`, `writer.cpp` | the three kernels |
| `kernels/sfpu.hpp` | Kohaku-TT's vector-unit steps, unchanged |
| `plan.py` | the plan as pure functions |
| `card.py` | the program, its broadcast words and its commands on a session |
| `run.py` | one shape run and checked |
| `bench.py` | time, power, clock, the profile and the cut-down runs of one shape |
| `graph.py` | the same call as one command or one command per job, from the host and from a kept graph |
| `results/` | the JSON behind the tables of results |
| `reference/sdpa.py` | the float64 attention, inputs, sampled rows |
| `reference/ktt_bench.py` | Kohaku-TT's kernel on the same inputs, checked the same way |
| `bench_ttnn.py` | TT-NN's attention on the same inputs; it imports `ttnn` and `torch` from the tt-metal tree `TT_METAL_HOME` names (Komorebi needs neither) |
