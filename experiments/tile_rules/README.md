# Tile rules: emitted bodies against hand bodies, and the cost model against the card

Question: do the compute bodies that `komorebi.dsl.tile` emits for SiLU, a residual add,
a scale by a constant, the SwiGLU gate `silu(a) * b` and the row norm
`x * rsqrt(mean(x^2) + eps) * w` compute the same expression as the hand bodies they
stand in for, what does a unit cost on one core with each, and how close is the printed
cost to the card? Then the same for attention's exponent step and its online softmax
step at the error the program states, and the whole per-core attention body assembled
from the library's matrix calls and the emitted softmax step. The page for the tile layer
is [docs/reference/tile.md](../../docs/reference/tile.md).

## Bodies

| BODY | case | form | what |
|---|---|---|---|
| 0 | silu | hand | tt-metal `silu_tile` (as the dense project's SwiGLU calls it) |
| 1 | silu | emitted | `trace.silu(x)` |
| 2 | add | hand | tt-metal `add_binary_tile(0, 1, 0)` |
| 3 | add | emitted | `a + b` |
| 4 | add | macro | `a + b` with the load-macro rule |
| 5 | scale | hand | tt-metal `mul_unary_tile(0, bits)`, the constant 1 / sqrt(128) |
| 6 | scale | emitted | `x * 0.0883883...` |
| 7 | scale | macro | the same with the load-macro rule |
| 8 | gate | hand | the decode project's gate (`projects/decode/kernels/gate`): `silu_tile`, then its HiFi4 product with b on the matrix unit; less the norm's scale |
| 9 | gate | emitted | `trace.silu(a) * b` |
| 10 | norm | hand | the decode project's norm: the squares as a product `y y^T` over the row's tiles times the identity, `rms_scale` (a product with the mean tile, + eps, `rsqrt_tile`), the row times the scale tile, then times the weight tile (decode folds the weight into the next product's matrix) |
| 11 | norm | emitted | the row form, a row of 2 tiles |
| 12 | exp | hand | the attention project's `exp_step<1>`: p = exp(s scale - m), degree-2 polynomial, its state tile in `Dst` 3 |
| 13 | exp | emitted | `trace.exp(s * scale - m.col())`, accurate |
| 16 | exp | stated | the same at `error=2e-3` (the flash template) |
| 14 | softmax | hand | the attention project's `state_max<3>`, `state_update`, `exp_step<3>` on 3 score tiles and the state tile |
| 15 | softmax | emitted | the step form at `error=2e-3` (the tile function on the reference page) |
| 17 | softmax | accurate | the same with the accurate exp2 |
| 18 | attn | hand | the per-core body (`kernels/attention.cpp`): per step of 3 key tiles the score products, BODY 14's step, P packed bf16, P V and l = P 1 added into O |
| 19 | attn | emitted | the same with BODY 15's step (l from the step's state) |

The hand norm keeps decode's sequence with fp32 buffers where decode has bf16, and
reads the squares from a second copy of the row as decode reads its `DC_CB_Y`.

## How it runs

| part | file |
|---|---|
| reader (RISCV_1): the P pages once; per unit its tiles of A (and of Y for the hand norm), one tile of B | `kernels/reader.cpp` |
| compute: per unit the copies into `Dst`, the body (define `BODY`), the pack | `kernels/compute.cpp` |
| writer (RISCV_0): the output tiles, then the info page | `kernels/writer.cpp` |
| shared layout of buffers and the info page | `kernels/rules.hpp` |
| the per-core attention body (BODY 18, 19) and its reader | `kernels/attention.cpp`, `kernels/attn_reader.cpp` |
| the attention body in stages (score product, + the step, the whole body, body 19), one session, each stage's name printed before its launch | `attn_stages.py` |
| driver: emits `kernels/generated/*.hpp`, builds, runs, checks | `rules.py` |

```
python rules.py build                                   # mock device: every body built, host checks
python rules.py card --cases silu add scale             # one core: values and cycles
python rules.py card --cases norm --forms hand --save out.npz
```

- **Timing.** The math CPU reads the wall clock (AICLK cycles) around the body between
  two `tensix_sync()` calls; the tables give the median of 16 units less the empty span
  (10 cycles, measured 16 times). The pack CPU times the whole unit (copies, body, pack).
- **Values.** 16 units of fp32 (`--seed 1`): normal values scaled by 8 and one tile over
  magnitudes 2^-40 .. 2^40 with random signs; for the norm 16 rows of 2 tiles of
  normal values, the first row times 100, a weight `1 + 0.1 N(0, 1)`. Each output
  against float64 of the same fp32 inputs, as the largest relative error.
- **The gate.** The tolerance is fixed before the run: 2^-23 for one fp32 rounding (add,
  scale), 1e-6 for SiLU and the gate, 1e-5 for the norm. It applies to the emitted
  bodies; the hand bodies are reported against the same reference.

## Results

Card 0, one core, AICLK 1350 MHz. Cycles a unit (a tile; a row of 2 tiles for the norm).

| case | body: hand | emitted | load macro | estimate (emitted / macro) | whole unit: hand / emitted / macro | max rel. error vs float64: hand / emitted / macro |
|---|---|---|---|---|---|---|
| silu | (1720) | 1592 | refused | 1593.6 | - / 1744 / - | - / 1.8e-7 / - |
| add | 184 | 183.5 | 93 | 182.6 / 91.6 | 1260 / 1255 / 1248 | 6.0e-8, all three bits equal |
| scale | 163 | 155 | 62 | 150.4 / 59.5 | 617 / 606 / 612 | 5.9e-8, all three bits equal |
| gate | 1901.5 | 1656.5 | refused | 1657.6 | 2143 / 1921 / - | 2.7e-3 / 2.1e-7 / - |
| norm | 3201.5 | 771.5 | - | 753.0 | 2640 / 1277 / - | 4.1e-3 / 1.8e-7 / - |

- Every emitted body is inside its tolerance. The emitted norm and the hand norm, on
  the same inputs, differ by at most 4.1e-3 relative: the hand norm's error. The hand
  gate and norm multiply on the matrix unit, whose operands are tf32 (about 1e-3).
- The hand SiLU was not run again: the tile-front experiment measured it at 1730 cycles
  a tile with a 10-cycle empty span (1720 net) on other inputs; its cycles do not depend
  on the values.
- The load macro halves the vector unit's time for add (183.5 to 93) and scale (155 to
  62) with the same bits, but a whole unit barely changes: an add or a scale is bound by
  the copies into `Dst` and the pack.
- The load-macro rule refuses SiLU ("the row computes with 45 instructions (16
  MAD-class); a macro arms at most one a sub-unit") and the gate (46, 17).
- The hand norm's math span (3201.5) is longer than its whole unit on the pack CPU
  (2640): it waits for its own two intermediate packs, and the pack CPU's unit span
  starts when it is done with the previous unit. The whole-unit column is the
  comparable one: 2640 against 1277.

Attention's steps, the same core, cycles a unit (a tile for exp, 3 score tiles and the
state for softmax):

| case | hand | emitted, first rules | emitted, now | estimate now | max rel. error: hand / emitted |
|---|---|---|---|---|---|
| exp, accurate | 351.5 | 1210 | 1115 | 1113.5 | 1.77e-3 / 3.0e-7 |
| exp at 2e-3 | 351.5 | 542.5 | 444 | 443.2 | 1.77e-3 / 1.77e-3 |
| softmax at 2e-3 | 1590.5 | 2818.5 | 2369 | 2350.0 | p, alpha, l within 2e-3 both |
| softmax, accurate | - | - | 5630 | 4950.0 | p 8.2e-7 |
| norm (for the rules' effect) | - | 771.5 | 689.5 | 677.0 | - / 1.8e-7 |

What the hand exp step has that the emitted one at 2e-3 does not (19 against 24
instructions a 4-row block, 351.5 against 443.2 cycles a tile):

| hand | emitted | cycles a tile |
|---|---|---|
| the state tile holds `b = 639 - A m`, computed once a step | a multiply-add for `639 - m log2 e` each block | 16 |
| `log2 e` and 639 folded into that state | `SFPLOADI` of both each block (3 words) | 48 |
| the address step in the store's address mode | an `SFPINCRWC` each block | 16 |
| the same clamp (`SFPSWAP`) and its stall | the same | 0 |

Not built: the rule that closes the first two keeps a per-row value that depends only
on the state in the state tile, computed once a step; the third needs the emitter to
own the address modes (the follow-up form that lowers the matrix words too).

What the hand softmax step has that the emitted one does not (1590.5 against 2369 on
the card; the emitted side's cycles are its estimate's terms, 2350 in all):

| hand | emitted | emitted cycles a step |
|---|---|---|
| l = P 1 as one more product on the matrix unit, after the step | the row sum on the vector unit: 112 `SFPADD`, 119 `SFPSHFT2` lane rotations and their 119 stalls | 350 |
| the row max by `SFPLOADMACRO`, load and swap in one word a vector | 112 `SFPSWAP` and their stall slots | 224 |
| the polynomial's constants in programmable registers (LReg 11 used too, restored to -1.0 at the end) | `SFPLOADI` of the two constants each block, with 639 and the lag | 88 |
| other work in the multiply-add stall slots | 128 `SFPNOP` the mad-reader check inserts | 128 |
| the state tile already in the exp's form (`b`, a flag) | the selects of alpha's `exact_at_zero` and of the lagging max (56 words) and the word's running max (21) | 77 |

The interleave rule finds independent work for few of the stall slots: the step holds 9
to 11 live values where the 8 work registers allow 3 more. The emitted vector part
stays 1.48 times the hand's.

**The per-core body did not run.** BODY 18 stopped the core both times it was
launched ("waiting for physical cores to finish", the host's 20 s timeout), first with
fp32 operands and `tensix_sync` around each product, then with bf16 operands and no
sync where the unpacker can be ahead. BODY 19 was never launched. The cause is not
found; its cycles and values on the card are not measured. Off the card (the host's
model of the products with the step's instructions emulated, bf16 inputs and P): O / l
within 3.1e-3 of float64 (tolerance 4e-3); estimate a step 2350 for the vector part and
199 (6 or more `Dst` blocks rotating) to 1154 (none) for each product set.

## The cost model against the card

Each price is a one-core measurement (`../tile_probes/README.md`, section 8). The
first run showed every plain add and scale estimate 32 cycles over: exactly the 32
stalls priced where an `SFPSTORE` reads the multiply-add result just before it. Without
a stall before a store, every body is within 4% (table on the reference page); the old
SiLU code shape estimates 1721.6 against 1720. Attention's steps are within 1%, except
the accurate softmax step (-12%: 26 KB of straight-line code, unexplained).

## What the card showed that the off-card checks did not

- **`SFPSHFT2` lane rotations are not held back behind a multiply-add.** The first
  emitted norm was wrong (mean relative error 0.67). An emulator of the ISA's functional
  models gives the right answer for that program, and the same emulator with "a lane
  rotation right after a multiply-add reads the register as it was before" gives a mean
  error of 0.667. The ISA's `SFPMAD` page lists the lane modes of `SFPSHFT2` among the
  readers the stall logic does not see; the `mad-reader` check now follows that list
  (32 `SFPNOP` in the norm), and the norm is right.
- **The hand bodies must set their source formats.** tt-metal's `state_configure` does
  nothing without its sentinel, so after `compute_kernel_hw_startup` on a buffer
  unpacked to `Dst` in fp32, the matrix unit's sources keep that buffer's format
  (Float32 for the unpacker, wrong for products). The first hand norm gave even
  columns 0 and odd columns scaled by about 10^3 or 10^-19; the first hand gate gave 58
  non-finite values. With `reconfig_data_format` before each product both are right.

## The row state is not a queue

The attention body keeps its row state in one page of `kCbSt`, written in place by the
pack CPU and never pushed or popped. A buffer with two producers (the reader's first
page, then the pack CPU's) hangs: the pack CPU's push writes its own count of pushes
into the buffer's received counter, not an increment, so after its first push the
counter reads 1 although two pages were pushed, equals the pops, and the unpacker's wait
for the next page never returns. The first version of the body did that and left the
core stuck twice.
