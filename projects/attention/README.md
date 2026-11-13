# Attention

`out = softmax(q k^T scale) v` (forward), q `[B, H, S, D]`, k and v `[B, Hk, S, D]`
with H a multiple of Hk (grouped heads), scale 1 / sqrt(D), optionally causal (query
row i sees keys up to i), bf16 operands, fp32 accumulation, bf16 output, HiFi2, on the
resident workers of one Blackhole p150a. Records reach the grid through the dispatch
kernel on X280 cluster 0 (`komorebi.Session`, `Run.issue`). The compute body is
Kohaku-TT's `ktt_flash` kernel's.

## The thread program

```
thread attend(lane, i):                        # one thread per core: lane, place in lane
    for job in jobs(lane):                     # time: (batch, head, row range)
        rows = share(job.rows, i)              # space: query rows over the lane's cores
        for kb in key_blocks(job.head, KB):    # time: keys, KB tiles per block
            if i == 0:  kv = read_dram(kb);  emit kv -> threads(lane, *)    # breakpoint
            else:       kv = wait(1)                                        # breakpoint
            for sub in chunks(kv, 3):          # time: 3 key tiles per step
                for r in rows:                 # time: my rows, paired
                    acc[r] = merge(acc[r], score(Q[r], sub))   # D summed inside score
        write out[rows]
```

A lane is L cores down one grid column; the core on top (i == 0) reads each K / V
block from DRAM once and casts it to the others. `merge` is the running softmax: the
row's reference, P = exp(scale s - reference), O += P V and the row sum, both scaled
when the reference moves. Nothing is merged across cores: each core finishes its own
output rows. Rows are shared contiguously, or for a causal call L rows apart, so that
the triangle's work spreads over the lane.

## How it maps onto the runtime

| Thread program | Kernel | What it does |
|---|---|---|
| `jobs(lane)` | all three | the member's job table: one DRAM page per core named by the record (a4), read by the reader (handed to compute) and by the writer |
| `emit kv -> threads(lane, *)`, `wait(1)` | `kernels/reader.cpp` (NoC0) | `km::Lane`: the sender `reserve`, reads K and V of block k into the slot, `cast`; a receiver `arrive`; `finish` |
| `Q[r]` | `kernels/reader.cpp` | the job's query rows into the Q buffer |
| `score`, `merge` | `kernels/compute.cpp`, `kernels/sfpu.hpp` | Kohaku-TT's `ktt_flash` compute body: S = Q K^T into Dst with the row state, the row reference and the exponential on the vector unit, P V and the row sum in column groups accumulated in L1 (fp32), steps in pairs of rows; `km::lane_take` / `lane_done` / `lane_end` around each block |
| `write out[rows]` | `kernels/writer.cpp` (NoC1) | each finished row's DV tiles to O |

One call is one `ATTEND` record on the team of every core, with `KM_F_EXCL` (a record
with lanes is never in flight beside an overlapping team). A K / V block is one lane
block: K's tiles head-dim-major `[DT][KB]`, then V's key-major `[KB][DV]`, two blocks
in flight per lane. The lanes and their senders come from `komorebi.layout.lane_args`
(one band of L rows of the grid after the other).

| Segment | Arguments |
|---|---|
| `ATTEND` | a0 Q, a1 K, a2 V, a3 O, a4 the job tables, a5 mode (0: the full call, 1: cast only, 2: compute only), a6 passes over the jobs (0: one), a7 the first job, a8 jobs from it (0: to the end) |
| `PROF` | a0 the profile tensor, a1 nonzero: zero the pages after storing them |

The cut-down modes are for the benchmark: cast only reads Q and casts every K / V block
but compute only takes and pops the blocks, and no row is written; compute only works on
Q, K and V as they are in L1 (no reads, no lane, no writes). Passes repeat the record's
jobs, so that a loop of records lasts seconds; a record moves fewer than 65,536 pages
through a lane (`Plan.passes_max`, see "Limits").

`plan.py` is the plan as pure functions, with Kohaku-TT's rules: R query tile rows per
job and KB key tiles per block fitted to the L1 budget (from 13 and 12 down), the lane
length L with the fewest score tiles on the busiest core (then the longest lane), lanes
taking even ranges of the (batch, head, row) order (causal: chunks largest work first
onto the least loaded lane), at most 46 jobs per core. `card.py` loads the program and
issues the records; `run.py` runs one shape and checks it. The references in
`reference/` are the float64 attention (`sdpa.py`) and Kohaku-TT's own kernel on the
same inputs (`ktt_bench.py`, by hand, not a test).

`kernels/sfpu.hpp` is an unchanged copy of `ktt_flash`'s `sfpu.hpp` (Kohaku-TT, Apache
2.0, at ac81ff2); `kernels/compute.cpp` is its `compute.cpp` with K and V taken from the
lane buffer, the jobs from the table, its waits on the other kernels bounded (see
Limits), and the parts this program does not use left out (masks, packed variable
lengths, statistics for the backward, ragged lengths).

## Run

```
python projects/attention/run.py --shape t1
python projects/attention/run.py --shape long64 --calls 4
python projects/attention/bench.py --shape long64 --cuts            # time, power, profile
python projects/attention/bench.py --shape wide128 --noc 1 --lane 5  # the lane's NoC, length
python projects/attention/graph.py --shape short64 --rounds 40       # records and graphs
KMR_CARD=1 python -m pytest tests/hardware/test_attention_host.py
python projects/attention/reference/ktt_bench.py --shape long64   # Kohaku-TT's env
```

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

Correctness, measured on 2026-11-13 on one p150a (card clock 1,350 MHz), ours at commit
"Projects: attention on resident workers with lane broadcast", Kohaku-TT at ac81ff2, one
after the other in the same session; host load (1 min) 4 to 13. Ours: `run.py --shape
<preset> --calls 4`; Kohaku-TT: `reference/ktt_bench.py --shape <preset>`. Both are
compared with float64 on the same 16 sampled query tile rows.

| Preset | Plan: R, KB, L | Ours: RMS / largest | Kohaku-TT: RMS / largest |
|---|---|---|---|
| `t1` | 13, 12, 10 | 5.50e-3 / 9.02e-3 | 5.50e-3 / 9.02e-3 |
| `t2` | 13, 12, 10 | 6.19e-3 / 7.88e-3 | 6.19e-3 / 7.88e-3 |
| `t3` (causal, Hk 1, D 128) | 13, 12, 10 | 4.18e-3 / 4.72e-3 | 4.18e-3 / 4.72e-3 |
| `long64` | 13, 12, 5 | 6.65e-3 / 1.47e-2 | |
| `wide128` | 13, 12, 10 | 6.32e-3 / 9.46e-3 | |
| `short64` | 13, 12, 2 | 5.87e-3 / 7.52e-3 | |
| `causal64` | 13, 12, 10 | 5.71e-3 / 7.53e-3 | |
| `gqa128` | 13, 12, 5 | 6.28e-3 / 8.91e-3 | |

Where both were run, the two kernels' statistics agree to every printed digit: the
compute body and its operand order are the same.

### Time and power

Measured on 2026-11-13 on one p150a with the X280 at 1,000 MHz (the session's default),
each row's output committed in `results/` (`ours-<preset>[-approx].json` from `bench.py
--json`, `ktt-<preset>[-approx].json` from `reference/ktt_bench.py --json`),
ours (`bench.py --shape <preset> [--approx] --cuts`) and Kohaku-TT's `ktt_flash` at
ac81ff2 (`reference/ktt_bench.py`) one after the other in the same session, 15 s apart;
host load (1 min) 3 to 9. Per call: ours, single-pass records back to back (the time of
10 to 60 records less that of 5, divided); ours per pass, the power loop of multi-pass
records (no record boundary); Kohaku-TT, its calls back to back for about 4 s. Clock
(AICLK mean), power (whole card, mean of the loop) and temperature are the power
loops', ours / Kohaku-TT's. RMS: relative RMS error of the sampled rows, the same for
both kernels on every row.

| Shape (B x H x S x D) | Ours per call (us) | Ours per pass (us) | Kohaku-TT per call (us) | MHz | W | C | RMS |
|---|---|---|---|---|---|---|---|
| 1x32x4096x64 | 2675.3 | 2646.0 | 2918.5 | 1347 / 1340 | 138 / 135 | 73 / 76 | 6.65e-3 |
| 1x24x4096x128 | 2328.3 | 2439.1 | 3001.8 | 1267 / 1244 | 144 / 132 | 72 / 70 | 6.32e-3 |
| 8x16x256x64 | 91.7 | 65.0 | 80.7 | 1349 / 1350 | 122 / 110 | 67 / 65 | 5.87e-3 |
| 1x8x1024x64 causal | 73.6 | 48.8 | 53.7 | 1350 / 1350 | 102 / 99 | 66 / 66 | 5.71e-3 |
| 2x8x1024x128, Hk 2 | 163.3 | 129.9 | 261.8 | 1345 / 1350 | 143 / 105 | 68 / 66 | 6.28e-3 |
| 1x32x4096x64, approx | 1932.5 | 1996.2 | 2183.7 | 1295 / 1284 | 143 / 140 | 69 / 71 | 1.94e-2 |
| 8x16x256x64, approx | 81.7 | 54.7 | 67.5 | 1336 / 1350 | 128 / 117 | 68 / 67 | 2.14e-2 |
| 1x8x1024x64 causal, approx | 60.7 | 38.8 | 44.4 | 1350 / 1350 | 108 / 103 | 65 / 66 | 2.21e-2 |

The 4096 x 128 loops and the approximate 4096 x 64 ones ran below 1,350 MHz (the card's
own clock control under load), both kernels alike. With the approximate exponential the
check's relative RMS bound is 3e-2 (Kohaku-TT documents about 1.9e-2).

**What bounds each row.** Each core keeps a profile page (`PROF`): wall-clock cycles of
its five threads' parts, summed over the records, read after 15 to 65 single-pass
records. The shares below are of the slowest core's longest thread; threads run side by
side, so they add up to more than 1. Cut-down runs: cast only and compute only, us per
pass.

| Shape | Slowest core's shares | Cast only / compute only | Bound |
|---|---|---|---|
| 1x32x4096x64 | score 0.48, P V 0.45 (math); exp 0.55, pack P 0.26 (pack) | 310 / 2639 | compute, both threads |
| 1x24x4096x128 | score 0.45, P V 0.45; exp 0.52, pack P 0.22 | 323 / 2453 | compute |
| 8x16x256x64 | score 0.29, P V 0.28, exp 0.32; Q reads 0.15, compute waiting for Q 0.16 and for K / V 0.16 | 59 / 67 | the record: 10.8 us of each record's 75.8 is outside the passes (table, lane start and end, the first Q and K / V reads), and back-to-back records add 15.9 us each |
| 1x8x1024x64 causal | score 0.35, P V 0.39, exp 0.43; waiting for the lane 0.58 | 34 / 62 | the record as above (57.1 us a record, 16.5 us between records), and balance: the mean core is busy 0.56 of the slowest |
| 2x8x1024x128, Hk 2 | compute waiting for K / V 0.67; score 0.14, P V 0.14 | 83 / 133 | compute (compute only = per pass) |
| 8x16x256x64, approx | score 0.24, P V 0.23, exp 0.24; Q reads 0.19, compute waiting for Q 0.20 and for K / V 0.20 | 63 / 58 | the record: 10.5 us of each record's 65.2 is outside the passes, and back-to-back records add 16.5 us each |
| 1x8x1024x64 causal, approx | score 0.33, P V 0.33, exp 0.34; waiting for the lane 0.60 | 36 / 49 | the record (45.8 us a record, 7.0 outside the passes, 14.9 us between records), and balance: the mean core is busy 0.57 of the slowest |

Compute only runs on whatever Q, K and V are left in L1, which takes the rescaling
branch more often than real inputs do; it is an upper bound on the compute. The four
short rows (both exponentials) are ahead of Kohaku-TT per pass and behind per call: what
is behind is the cost of a record, not the kernel.

### The broadcast's NoC, the lane length, records and graphs

Measured on 2026-11-13 as above (X280 at 1,000 MHz), `wide128` (1 x 24 x 4096 x 128);
the JSON in `results/`: `noc0-wide128.json`, `noc1-wide128.json`, `lane-wide128-L<L>.json`
(`bench.py --json`) and `graph-<preset>.json` (`graph.py --json`).
NoC1: `--noc 1` builds the program with `LANE_ON_WRITER`, so that the writer runs the
lane (K / V reads and casts before each job's rows) and the reader reads only Q. The
lane length is `--lane L` (L = 1: every core reads its own K / V).

| `wide128` | Single record (us) | Per call (us) | Per pass (us) | Cast only / compute only (us) | MHz | W |
|---|---|---|---|---|---|---|
| NoC0, L 10 | 2312.3 | 2330.3 | 2337.3 | 310 / 2393 | 1321 | 152 |
| NoC1, L 10 | 2276.5 | 2358.0 | 2309.6 | 597 / 2447 | 1315 | 153 |
| NoC0, L 5 | 2300.1 | 2315.3 | 2318.5 | 384 / 2352 | 1318 | 154 |
| NoC0, L 2 | 2322.4 | 2338.6 | 2333.5 | 773 / 2504 | 1307 | 149 |
| NoC0, L 1 | 2404.8 | 2418.0 | 2381.8 | 1533 / 2642 | 1289 | 153 |

The broadcast alone takes twice as long from the writer on NoC1 as from the reader on
NoC0, and five times as long with no lane (L 1) as with lanes of 10; the call, bound by
compute, moves by at most 4 %.

`graph.py` issues the same call, rounds back to back: one record walking the job table,
or one record per job index (a7, a8: each core's job j in record j); each from the host
(records back to back, the time from the first issue to the last event over the rounds)
and from a kept graph (the call's records in a chain, then a CPU node that goes round
again; the engine on the X280 issues each next record; the time of a restart over its
rounds, the median of three).

| Shape | Rounds | Records per call | Host, table | Graph, table | Host, per job | Graph, per job |
|---|---|---|---|---|---|---|
| 8x16x256x64 | 40 | 3 | 97.4 | 85.1 | 210.2 | 134.5 |
| 1x8x1024x64 causal | 40 | 1 | 79.5 | 66.4 | 90.6 | 65.7 |
| 1x32x4096x64 | 8 | 4 | 2701.5 | 2685.4 | 3430.5 | 3331.2 |

Times are us per call. From the kept graph a call comes 12 to 13 us sooner than from
the host (part of that is the host's first issue, spread over the rounds). A record per
job costs 25 us per extra record on the short shape and 215 us on the long one: each
record starts its lanes with no block in flight and ends them with every core drained,
which the job table inside one record overlaps.

## Limits

Bounded waits (a give-up is recorded with its code and place, stops the core's three
kernels and is reported by the run):
- the reader: room in the job table, Q and lane buffers (`rd.reserve`); its NoC reads
  (`flushed`); every `km::Lane` wait (a free slot, the receivers' stamps, VALID, the
  end of the record);
- the writer: each finished output row (`wr.wait_front`); its NoC writes (`flushed`);
- the compute kernel's waits on the data-movement kernels: the job table, Q rows and K /
  V blocks (`cp.wait_front`, decided by the unpacker and passed to the other two) and
  room for an output row (`cp.reserve_back`, the packer).

Not bounded: the compute kernel's waits between its own three threads, as in
`ktt_flash`: the score and P tiles and their tokens (`cb_wait_front`, `cb_reserve_back`
on `cb_p`, `cb_stok`, `cb_otok`), the destination register hand-over (`tile_regs_*`) and
the semaphore and SFPU mutex waits in the vector sections. They wait only on work the
same kernel has already started.

A known runtime defect, being fixed: a lane's sender reads its buffer's 16-bit
popped-page counter only in `km::Lane::finish()`, so a record that moves 65,536 pages
or more through one lane leaves the sender waiting in `finish()` for blocks it counts
as not popped; that record still reports done and the next one times out. Until the fix
lands, `plan.make` refuses such a plan and `Plan.passes_max` keeps the benchmark's
passes below it; both go with the fix.

## Future work

Not in this program: decode against a kept K / V cache (one query row against the
cache: a reduce over a team and the dynamic crossing), the backward, packed
variable-length batches, masks, ragged lengths.

## Files

| File | What it is |
|---|---|
| `kernels/attention.h` | segment, record words, buffers, the job table, the bounded NoC barrier |
| `kernels/reader.cpp`, `compute.cpp`, `writer.cpp` | the three kernels |
| `kernels/sfpu.hpp` | Kohaku-TT's vector-unit steps, unchanged |
| `plan.py` | the plan as pure functions |
| `card.py` | the program, its lane words and its records on a session |
| `run.py` | one shape run and checked |
| `bench.py` | time, power, clock, the profile and the cut-down runs of one shape |
| `graph.py` | the same call as one record or one record per job, from the host and from a kept graph |
| `results/` | the JSON behind the tables of results |
| `reference/sdpa.py` | the float64 attention, inputs, sampled rows |
| `reference/ktt_bench.py` | Kohaku-TT's kernel on the same inputs, checked the same way |
