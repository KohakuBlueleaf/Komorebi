# Dense: SwiGLU

`y = (silu(x W1) * (x W3)) W2` with x `[M, D]`, W1 and W3 `[D, H]`, W2 `[H, D]`, bf16
operands, fp32 accumulation, HiFi2 (or HiFi4), on the persistent kernels of one Blackhole p150a,
with the matrix product's persistent kernels and one more handler, `LAYER`. A call is a
chain of dependent products with no DRAM round trip between them: a core's rows stay
on its core through the call, so the intermediates never leave L1.

## What each core does

W1, W3 and W2 are resident: three `LOAD` commands, once per session (`kernels/reader.cpp`
reads each into `kCbW`, `kCbW3` or `kCbW2`). A call is then three `LAYER` commands
(`swiglu_plan.commands`), each over the row tiles the core owns (`km::partition` of the
row tiles over the group):

1. `x W1` with `silu` on the product. The reader (`kernels/reader.cpp`, NoC0) reads
   x's row from DRAM into `kCbA`; compute (`kernels/compute.cpp`, `layer`) runs
   `matmul_block` over the row's K tiles into DST (fp32), `silu_tile` on DST, and packs
   the block into a held buffer (`kCbHold0`).
2. `x W3`, multiplied by the held `silu(x W1)`. The reader reads x's row again; compute
   runs `matmul_block`, waits for the held tile, multiplies it into DST (DST to source
   A, times the held tile) at HiFi4 whatever the program's fidelity, and packs into
   the next held buffer (`kCbHold1`).
3. `h W2`, with h the held result of command 2 as A: the reader reads nothing, compute
   runs `matmul_block` and packs into `kCbOut`, and the writer (`kernels/writer.cpp`,
   NoC1) writes the column blocks to Y's pages in DRAM.

The held buffers are circular buffers that only compute uses: one command packs into
them and the next command on the same core takes from them.

There are no breakpoints: no thread waits for another, and every value a command takes
from an earlier one was made on the same core. The rows of one core are the "row
block" of the call: all of them are held between commands, which bounds the shapes
(below). With `--gate-pass` the gate is a command of its own (four commands: x W1 and
x W3 held, then silu(a) * b, then the output product).

| Handler | Arguments |
|---|---|
| `LAYER` | a0 a_address (A, x in DRAM), a1 y_address (Y), a2 m_tiles, a3 k_tiles (at most 32), a4 n_tiles, a5 block_width (column block), a6 w_buffer (0: the gate pass), a7 epilogue (none, silu, times B, the gate), a8 held_buffers: where A, B and Y live (`A \| B << 8 \| Y << 16`, a held buffer or `kDram`) |

`LOAD` and `PROF` are the matrix product's. Profile words are summed over all commands;
`LAYER` adds the epilogue's time (`kPrMathEp`).

Cores take the commands of a group in sequence order, and the held buffers make compute
wait for what the previous command packed, so the commands of a call need no completion
boundary between them and are submitted without `exclusive`.

`swiglu_plan.py` holds the plan (`Swiglu`: cores, row tiles held per core, buffers, the
L1 fit, the commands of a call); `swiglu.py` loads the program, submits the commands,
reads the profile and checks the result; `reference/ktt_swiglu.py` runs the same layer
with Kohaku-TT's device operators.

## Run

```
python projects/dense/swiglu.py --shape sw256
python projects/dense/swiglu.py --shape sw16k --gate-pass --seconds 3 --json out.json
python projects/dense/swiglu.py --shape sw16k --fidelity HiFi4
python projects/dense/reference/ktt_swiglu.py --shape sw16k        # Kohaku-TT, its env
KTT_GEMM=1 python projects/dense/reference/ktt_swiglu.py --shape sw16k
```

`--shape` names a preset of `swiglu_plan.SHAPES` (M x D x H): `sw256` 256 x 256 x 512,
`sw4k` 4096 x 256 x 512, `sw16k` 16,384 x 256 x 512, `sw16k384` 16,384 x 384 x 384.
Setup is that of [01_mapcollect](../../examples/01_mapcollect/README.md#what-it-needs).
Inputs come from a seed (`reference/dense.py`, `swiglu_inputs`): x uniform in [-1, 1), the W
uniform in [-1, 1) times 1 / sqrt(fan in), all rounded to bf16. With `--seconds`, the
driver times 20 single calls with the profile around them, then calls back to back for
about that long under the power sampler, then a second with every command `exclusive`.

**The check's bound.** Y on 256 sampled rows (8 row tiles) against the float64 layer
of the same bf16 inputs: relative RMS error at most 2e-2, largest error at most 3e-2 of
the largest output. The runs below give 1.21e-2 to 1.23e-2 RMS and 1.19e-2 to 1.61e-2
largest at HiFi2 (`--fidelity HiFi4`: 0.43e-2 to 0.47e-2 and 0.49e-2 to 0.66e-2);
Kohaku-TT gives 1.00e-2 to 1.01e-2 and 0.98e-2 to 1.32e-2 on the same rows. The bf16
roundings of the intermediates and of Y alone, with exact products and an exact silu,
would give 0.29e-2 RMS.

Where the error comes from, measured step by step on `sw4k` (each step's output
written to DRAM instead of a held buffer, RMS against float64): x W1 at HiFi2 0.40e-2
(at HiFi4 0.17e-2, the bf16 rounding of the output); silu on DST adds 0.25e-2 against
silu of the card's own x W1; the multiply by g, an element-wise product on the matrix
unit, took the program's fidelity and gave 0.92e-2 at HiFi2 against 0.37e-2 at HiFi4,
since no K sum averages the low bits HiFi2 leaves out. The multiply therefore always
runs at HiFi4 (a copy of `eltwise_binary.h`'s dest-reuse multiply with the fidelity
fixed), which took Y from 1.48e-2 to 1.22e-2 for 0.5 to 3 % more time per call (the
same rows measured before and after the change). The rest of the gap
to Kohaku-TT is the three products at HiFi2 in a chain; with every product at HiFi4 Y
is at 0.43e-2 to 0.47e-2, under Kohaku-TT's, for 20 to 30 % more time per call.

## Results

Measured on 2026-11-13 on one p150a (card 0), ours at commit "Projects: SwiGLU as
LAYER commands on the resident matrix-product kernels", Kohaku-TT at ac81ff2, the two
sides one after the other in one session, 15 to 20 s apart; X280 at 1000 MHz; host
load average (1 min) 4.3 to 13.4 on a shared 96-thread host. Ours: `swiglu.py --shape
<preset> [--gate-pass] --fidelity <f> --seconds 3`, host clock from the first submit to
the last event over the loop (`results/swiglu-<preset>-<fused|gate>-<f>.json`).
Kohaku-TT: `reference/ktt_swiglu.py --shape <preset>`, unset and `KTT_GEMM=1`, calls
back to back for about four seconds and one synchronize; the faster route is shown
(`results/swiglu-<preset>-ktt-auto.json`, `-ktt-gemm.json`). Power: mean whole-card
power over the loop (67 to 72 W open and idle); AICLK sampled every 20 ms (min /
mean). Only the call is timed on both sides; the three W are loaded once (0.25 ms on
120 cores), and uploads and the read-back are not part of a call.

| Preset | M x D x H | Ours HiFi2: form, us per call | W, MHz | Ours HiFi4: form, us | W, MHz | Kohaku-TT best: route, us | W, MHz |
|---|---|---|---|---|---|---|---|
| `sw256` | 256 x 256 x 512 | fused, 42.6 (gate pass 46.3) | 76, 1350 / 1350 | fused, 51.7 (55.1) | 74, 1350 / 1350 | gemm, 181.6 (auto 186.2) | 73, 1350 / 1350 |
| `sw4k` | 4096 x 256 x 512 | fused, 73.7 (76.3) | 114, 1343 / 1349 | fused, 91.8 (94.1) | 119, 1350 / 1350 | auto, 183.5 (gemm 192.9) | 95, 1343 / 1350 |
| `sw16k` | 16,384 x 256 x 512 | gate pass, 180.2 (fused 186.6) | 137, 1256 / 1310 | gate pass, 231.9 (232.5) | 141, 1212 / 1290 | auto, 542.1 (gemm 604.5) | 109, 1343 / 1349 |
| `sw16k384` | 16,384 x 384 x 384 | gate pass, 161.7 (fused 186.8) | 142, 1175 / 1278 | fused, 222.5 (225.9) | 144, 1125 / 1246 | gemm, 458.1 (auto 477.0) | 108, 1343 / 1349 |

Ours was measured again on 2026-11-14 with the same commands (HiFi2, both forms of every
preset): every cell is within 3 % of the table, except the gate pass of `sw4k`, which
is replaced (76.3 us, in three runs of 76.3, 77.5 and 77.5, against 79.5). The HiFi4
columns and Kohaku-TT's were not run again.

Ours is 2.5 to 4.3 times as fast at HiFi2 and 2.0 to 3.5 times at HiFi4, and the
16,384-row loops ran at the lower clock (mean 1,246 to 1,310 MHz against Kohaku-TT's
1,349).

| Preset | Bound of ours (HiFi2, best form) |
|---|---|
| `sw256` | the gate on DST, 24.3 us of the slowest core's 33.7; the call is 42.6, the rest is the three commands' submit |
| `sw4k` | the gate, 48.5 of 66.3 us |
| `sw16k` | the gate, 110.9 of 164.1 us; the products 48.2 |
| `sw16k384` | the gate, 83.4 of 141.9 us; the products 53.8 |

The bound is the profile of the slowest core over the loop, cycles at 1,350 MHz per
call: pack's body (the longest of the five kernels' bodies) and within it the epilogue
(silu and the multiply on DST, or the gate pass). Reads of x take 1.4 to 9.4 us per
call and the writes of Y 0.4 to 4.0 (29 and 34 in the fused `sw16k384`): the best form
of every row does not wait on DRAM.

Fused or gate pass. From 16,384 rows the gate pass is faster at HiFi2 (3.5 to 16 %), on
short calls the fused form is (2.6 to 3.7 us: a command less); at HiFi4 the two are
within 3 % from 4096 rows. Our reading of the profile, not a measurement of the waits: fused, pack
waits for the vector unit inside each column block of the second product; as a pass,
the products pack at the matrix unit's pace and the gate runs from L1.

Exclusive commands. A call's commands overlap the same group, and none needs
`exclusive`: every value a command takes was made on its core, and the held buffers make
compute wait for it. Submitted with `exclusive` (each command held until the previous one
completed), a call costs 43 to 124 us more (`sw256` fused 86.0 against 42.6 us,
`sw16k384` gate pass 286.0 against 161.7).

## sw256 as a class chain

`swiglu_split.py` is the `sw256` layer written as a class whose call is ONE chain
command: `up` on 64 cores (8 row tiles of x, 2 hidden columns a core, its slices of W1
and W3 held in fields filled once) writes `gate(h, g)` of its two DST sums to the hidden;
`down` on 8 cores (a row tile each, W2 held in two fields) reads its row's 16 hidden
tiles once up's 8 cores of that row have marked them, and writes y. The LAYER program
above runs the gate on 8 cores (24.3 us of a call); here each of 64 cores gates 2 tiles.

Card 0, 2026-11-18, AICLK 1,350 MHz at every read, host load 62 to 74, HiFi2; y against
float64 on all 256 rows: relative RMS 9.35e-3, largest 9.50e-3 (bound 2e-2 / 3e-2).

| sw256, us a call | single call: on the card | 32 calls, the loop fed on the card (a kept graph) | 32 calls queued from the host |
|---|---:|---:|---:|
| LAYER commands, fused gate (3 commands) | 95.2 | 58.5 | 59.6 |
| the class chain (1 command) | 19.8 | 29.4 | 34.4 |

```
python projects/dense/swiglu_split.py --seconds 1
```

## Limits

- K is one chunk in `LAYER`: D and H at most 1,024 (32 tiles).
- The three W and all of a core's held rows share one core's L1 (1,380,000 bytes):
  a layer of model width 1,024 and hidden width 4,096 needs 8 MB per W and does not
  fit. `sw16k384` is the largest D x H with H at least D for which both forms fit with
  16,384 rows on 120 cores (5 row tiles held each); the fused form alone would fit
  320 x 512.
- The gate, silu and the multiply on DST, is the bound at every shape above.
- The multiply's fidelity is fixed in the kernel through the low-level calls that
  `eltwise_binary.h` wraps; the compute API takes one fidelity per program.

## Files

| File | What it is |
|---|---|
| `kernels/dense.h` | `LAYER`'s words, epilogues, held buffers and profile word, beside the matrix product's |
| `kernels/reader.cpp`, `compute.cpp`, `writer.cpp` | `LAYER`'s bodies beside `LOAD`, `PRODUCT` and `PROF` |
| `swiglu_plan.py` | the plan as pure functions: shapes, buffers and the L1 fit, the commands of a call |
| `swiglu.py` | one shape run, checked and optionally measured |
| `swiglu_split.py` | `sw256` as a class chain: held weights, the gate on 64 cores |
| `results/swiglu-*.json` | where the runs write their JSON (not kept in git) |
| `reference/ktt_swiglu.py` | Kohaku-TT's layer on the same inputs, timed and checked the same way |
