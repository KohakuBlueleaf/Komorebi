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

## Run

```
python projects/dense/run.py --shape lin384 --calls 100
python projects/dense/run.py --m 256 --k 64 --n 128 --inputs int --form stream
python projects/dense/bench.py --shape lin384 --json lin384.json
KTT_GEMM=1 python projects/dense/reference/ktt_bench.py --shape lin384
python -m projects.dense.reference.gpu_bench --shape lin384
```

`--shape` names a preset of `plan.SHAPES` (M x K x N): `kvproj` 128 x 896 x 128,
`lin384` 16,384 x 384 x 384, `lin1024` 16,384 x 1024 x 1024, `sq4096` 4096 cubed,
`sq8192` and `sq16384` cubed, `dec4096` 32 x 4096 x 4096. Setup is that of
[01_mapcollect](../../examples/01_mapcollect/README.md#what-it-needs). Inputs are
uniform in [-1, 1) rounded to bf16, from a seed (`ref/dense.py`, `inputs`).

`run.py` and `bench.py` compare Y with the float64 product of the same bf16 operands
on 256 sampled rows (all rows when there are fewer) and exit 1 when the check fails.
`reference/ktt_bench.py` runs Kohaku-TT's product on the same inputs in Kohaku-TT's
own environment (fast dispatch), with the route its caller selects (`KTT_GEMM=1`: its
gemm; unset: its automatic choice), and checks the same rows the same way.

**The check's bound.** Relative RMS error at most 1e-2 and largest error at most 2e-2 of
the largest output. The output's rounding to bf16 alone is up to 2^-9 relative per
element; the runs in the table below give 3.96e-3 to 3.98e-3 RMS and 5.0e-3 to 5.1e-3
largest, the same numbers as Kohaku-TT's routes on the same rows (each run's JSON line
carries them). With `--inputs int` (integers in [-2, 2], K at most 64) every product
and sum is exact, and the check is bit for bit.

## Results

Measured on 2026-11-13 on one p150a, ours at commit "Projects: the matrix product's
resident and streamed forms", Kohaku-TT at ac81ff2, the two sides one after the other
in the same session; host load average (1 min) 3.6 to 12.1 on a shared 96-thread host.
HiFi2 on bf16 with fp32 accumulation. Ours: `bench.py --shape <preset>`, time per call
with records issued back to back for about four seconds (host clock, first issue to
last event). Kohaku-TT: `KTT_GEMM=1` or unset `reference/ktt_bench.py --shape
<preset>`, calls back to back for about four seconds and one synchronize; the faster
route is shown. Power: mean whole-card power over the loop; AICLK sampled every 20 ms
(min / mean). Only the product is timed on both sides: uploads and read-back are not
part of a call.

| Preset | M x K x N | Ours: form, us per call | W, MHz | Kohaku-TT best: route, us | W, MHz | Bound of ours |
|---|---|---|---|---|---|---|
| `kvproj` | 128 x 896 x 128 | resident, 6.34 | 65 W, 1350 / 1350 | auto, 28.2 (gemm 29.3) | 63 W, 1343 / 1350 | dispatch: a call issues every 6.3 us; the product itself is 4.8 us on the slowest core |
| `lin384` | 16,384 x 384 x 384 | resident, 60.9 | 130 W, 1237 / 1301 | gemm, 94.1 (auto 146) | 119 W, 1343 / 1349 | DRAM: A in and Y out, 25 MB per call; reads only 32.8 us, compute only 20.8 us |

The clock of our 16,384-row loop left its nominal 1,350 MHz (mean 1,301): ours ran at
the lower clock and was 1.55 times as fast.

## Files

| File | What it is |
|---|---|
| `kernels/dense.h` | segments, record words, buffers, profile words, the bounded NoC barrier |
| `kernels/reader.cpp`, `compute.cpp`, `writer.cpp` | the three kernels |
| `plan.py` | the plan as pure functions |
| `card.py` | the program and its records on a session |
| `run.py`, `bench.py` | one shape run and checked; one shape measured |
| `reference/ktt_bench.py` | Kohaku-TT's product on the same inputs, timed and checked the same way |
| `reference/gpu_bench.py`, `reference/gpu/gemm_gpu.cu` | cuBLAS on one idle GPU (fp32 and fp16) |
