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
| `ATTEND` | a0 Q, a1 K, a2 V, a3 O, a4 the job tables, a5 mode (0: the full call) |

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
| `reference/sdpa.py` | the float64 attention, inputs, sampled rows |
| `reference/ktt_bench.py` | Kohaku-TT's kernel on the same inputs, checked the same way |
