# Tile rules: emitted bodies against hand bodies, and the cost model against the card

Question: do the compute bodies that `komorebi.dsl.tile` emits for SiLU, a residual add,
a scale by a constant, the SwiGLU gate `silu(a) * b` and the row norm
`x * rsqrt(mean(x^2) + eps) * w` compute the same expression as the hand bodies they
stand in for, what does a unit cost on one core with each, and how close is the printed
cost to the card? Then the same for attention's exponent step and its online softmax
step at the error the program states, and the whole per-core attention body assembled
from the library's matrix calls and the emitted softmax step. Then the five rules that
close the softmax step's gap to the hand step, one at a time on the card, and the
matrix unit's words of a tile product emitted against the library call. The page for
the tile layer is [docs/reference/tile.md](../../../docs/reference/tile.md).

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
| 20 | softmax | ruled | the step form with every rule: the row sums as products with a tile of ones on the matrix unit (HiFi4), in a second part after them |
| 21 | exp | ruled | BODY 16 with the store's address mode in place of the `INCRWC` |
| 22 | product | hand | the library's `matmul_block_no_mop` of one bf16 tile by another at HiFi2, as the dense project's compute order calls it (`kernels/product.cpp`) |
| 23 | product | emitted | the library's unpack, and the matrix words `komorebi.dsl.tile.matrix` emits for its math |
| 24 | add | band | BODY 4 emitted with `rows=True`: the live rows a runtime argument (`--rows`) |
| 25 | gate | band | `trace.silu(a) * b` with the load macro and `rows=True` |
| 26 | norm | band | BODY 11's norm with `rows=True` (the row form's guarded groups) |
| 27 | product | trace | `a @ w` through the trace at HiFi2: BODY 23's words (`build` refuses it otherwise); not run again |
| 28 | mm_silu | emitted | `trace.silu(a @ w1)`: the product's words and the vector code in one body, HiFi4 (`kernels/fused.cpp`) |
| 29 | mm_gate | emitted | `trace.silu(a @ w1) * (a @ w3)`: two products and the gate in one body, HiFi4 |
| 30 | mm_gate | band | BODY 29 with `rows=True`: the vector code on the live rows, the products whole |
| 31 | clamp | hand | BODY 12's exponent step (512 in LReg 11) on arguments far below 0 |
| 32 | clamp | emitted | `trace.exp2(a, error=2e-3, over=(-inf, 0))` on the same arguments |
| 33 | rsqrt_col | emitted | `a * trace.rsqrt(s.col())`: a per-row operand into rsqrt's constants |
| 34 | sigmoid_col | emitted | `trace.sigmoid(a + s.col())` |

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
| the product (BODY 22, 23): unpack, math, pack in the dense project's order | `kernels/product.cpp` |
| the attention body in stages (score product, + the step, the whole body, body 19), one session, each stage's name printed before its launch | `attn_stages.py` |
| driver: emits `kernels/generated/*.hpp`, builds, runs, checks | `rules.py` |
| the rules in stages (the hand step, the emitted step with none, one rule added a stage, BODY 20, 21, then 22 and 23), one session, each stage's name printed before its launch; a rule is switched off by replacing its entry point while its header is built | `rule_stages.py` |

```
python rules.py build                                   # mock device: every body built, host checks
python rules.py card --cases silu add scale             # one core: values and cycles
python rules.py card --cases norm --forms hand --save out.npz
python rule_stages.py build                             # mock device: every stage built
python rule_stages.py card --stages 4 5 6 8 9 --save out.npz
```

A card job compiles nothing: `build` fills the runtime's kernel cache
(`.logs/jit-rules/mock`) and `card` reads it (the device's build key is the same).

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

The third is closed now (the store's address mode, BODY 21: 443.2 to 401.0 priced, 402
on the card). The rule for the first two (the per-row term computed once a block in a
pass first) is built and refused by the cost model: its pass costs 68 cycles a tile
against the 64 words it saves.

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

### The rules, one at a time

`rule_stages.py`, the same core, cycles a step (3 score tiles and the state), the median
of 16 less the empty span; values against float64 (tolerance 2e-3 each). The smallest
saving first; the row sums on the matrix unit last: the cost model takes them only with
the interleave on, and they alone add waits and buffers.

| stage | estimate | card | error | p / l / alpha / m |
|---|---|---|---|---|
| hand step (BODY 14) | - | 1591.5 | - | p 1.76e-3, m 2.9e-5 abs. |
| emitted step, no rule | 2350.0 | 2369 | -0.8% | 1.76e-3 / 1.12e-3 / 1.73e-3 / 5e-8 |
| + constants held for the step | 2291.0 | 2311 | -0.9% | the same |
| + LReg 11 a fourth constant register | 2289.0 | 2308 | -0.8% | the same |
| + interleave (the live-value count as the allocator's) | 2015.0 | 2034 | -0.9% | the same |
| + max macro | 1939.0 | 2045 | -5.2% | the same |
| + row sums on the matrix unit (BODY 20): first part | 1659.0 | 1761 | -5.8% | 1.76e-3 / 1.03e-3 / 1.73e-3 / 5e-8 |
| the same, second part from the products' issue | 246.0 | 366 | -33% | |
| exp step at 2e-3, + the store's address mode (BODY 21) | 401.0 | 402 | -0.2% | 1.77e-3 |

- The interleave is most of it (274 cycles): before, the pressure count it moves under
  counted one value more than the allocator does, so no pair ever moved.
- The max macro saved nothing on the card: 88 load-and-max pairs cost about 3 cycles
  each in both forms, where the price list gives the macro 2.05. A candidate, not
  measured: the template's swap issues in the `SFPNOP` after the macro, and the next
  macro is then a word right after an `SFPSWAP`, which stalls one; the price list has
  that stall for a plain swap only. With it the cost model would refuse the rule here.
- The row sums on the matrix unit make the step's first part 284 cycles shorter than
  the vector sums (2045 to 1761), but its second part costs 366 where 246 was priced:
  P goes out through a buffer and back through the unpacker before its products. In
  all 2127 against 2045. The hand attention body adds l = P 1 as one more product
  beside P V, with no second copy of P.
- Left against the hand step (1591.5), from the best measured step (2034): 442.5 cycles.
  The terms, from its estimate: the row sum on the vector unit (lane rotations, their
  stalls and the adds), about 350, which the hand body does as a product outside the
  step; the selects of `exact_at_zero`, the lagging max and the word, 77; the model's
  1%, about 19.

### The matrix unit's words of a tile product

BODY 22 and 23 over 16 units of bf16 tiles with values in [0.5, 1.5) at HiFi2: BODY
23's output is bit-equal to BODY 22's (both 4.83e-3 of float64, tolerance 2^-4), and a
unit measured 1147.0 cycles against 1145.5 (math CPU, from the product's issue to a
`tensix_sync` after it, less the empty span). Both are bound by the reader's tile reads:
the 34 words are priced 41.0 (`matrix.block_cost`) and are not seen under that wait.

### A product inside the body

BODY 28 to 30 over 16 units, A in [0.5, 1.5) and W1, W3 in [0.5, 1.5) / 16, all bf16,
at HiFi4; every unit's tiles are in L1 before the first unit (`kernels/fused.cpp`), so
the span is the body's (math CPU, from the products' issue to a `tensix_sync` after
the body, less the empty span). Tolerances, stated before the run: 2^-8 for
silu(a @ w1), 2^-7 for the gate (each term exact at HiFi4; the matrix unit's sum is not
one rounding; two products' errors add).

| Body | Cycles a unit | Estimate | Whole unit | Max rel. error vs float64 |
|---|---:|---:|---:|---:|
| silu(a @ w1) | 1579 | 1548.6 | 1598 | 3.0e-4 |
| silu(a @ w1) (a @ w3) | 1707 | 1678.6 | 1726 | 4.6e-4 |
| the same, band, 32 rows | 1707 | 1678.6 | 1726 | 4.6e-4 |
| the same, band, 1 row | 356 | 349.3 | 454 | 3.8e-4 (past the band: a @ w1, 2.4e-4) |

Beside the same work as separate bodies (rows already measured): the gate's vector code
alone ran 1568 cycles a tile (BODY 25 at 32 rows) and 220 at one row; BODY 29 less it is
139.0, the two products' estimate exactly (66 + 73 words' cycles), and the band at one row
less 139 is 217. The 28.4 over the estimate is the gate code's own (1568 against
1539.6). Separate bodies would also pay the products' packs and the gate's two copies
into `Dst`: the emitted gate's whole unit was 1921 against its 1656.5 body (two copies and
a pack, 264.5), and each product alone was a unit of its own (1147, bound by the reader).

### The exponent's clamp and the per-row path's constants

BODY 31 and 32: unit 0 cycles through the arguments t below, the other units take t in
[-30, 0]; the hand body gets x = t / A with its bias 639, the emitted one t. Held to the
stated 2e-3 where 2^t >= 2^-126, below that to within 2^-126 and finite. The first run
had the clamp's `SFPSWAP` write `VD` 12 (512 in constant register 12):

| t | hand | emitted, `VD` 12 | emitted, `VD` a work register |
|---:|---:|---:|---:|
| -1 | 0.50086 | 0.50086 | 0.50086 |
| -64 | 5.430e-20 | 5.430e-20 | 5.430e-20 |
| -126 | 1.1775e-38 | 1.1775e-38 | 1.1775e-38 |
| -127.5 | 0 | NaN | 0 |
| -128 | 0 | 1.761e38 | 0 |
| -150 | 0 | 1.001e25 | 0 |
| -200 | 0 | 7.898e-6 | 0 |
| -300 | 0 | 5.507e11 | 0 |
| -1e4 | 0 | 5.796e-17 | 0 |
| cycles a tile (estimate) | 351.5 | 478.0 (478.1) | 511.0 (510.2) |

On the ordinary arguments all three are within 1.75e-3 of float64. The copy of 512 is
one word a vector (32 a tile); in the softmax step it is 118 words, and the step's first
part ran 1878.5 cycles (estimate 1777.0) against 1761 before, its second part 368, its
values as before (p 1.76e-3, l 1.03e-3, alpha 1.73e-3). The step's earlier runs and BODY
16 and 21 were right with the clamp never running because no score was more than 127
below its row's reference.

The per-row path (a `.col()` operand) dropped the load of each constant register's
value (the `SFPCONFIG` that copies LReg 0 was not counted as its reader), so the
register took whatever LReg 0 held. With the loads kept, over 16 units, a in 3 N(0, 1),
s one value a row:

| Body | Cycles a tile | Estimate | Whole unit | Max rel. error vs float64 |
|---|---:|---:|---:|---:|
| `a * rsqrt(s.col())`, s in [0.5, 4) | 490.0 | 487.4 | 1254 | 1.2e-7 |
| `sigmoid(a + s.col())`, s in [-3, 3) | 1454.5 | 1453.5 | 1712 | 5.8e-7 |

## The cost model against the card

Each price is a one-core measurement (`../../core/tile_probes/README.md`, section 8). The
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
- **A word whose `VD` is 12 + k stores load-macro template k.** The step's clamps
  against constant register 12 (`SFPSWAP` with `VD` 12) overwrote the max macro's
  template 0: from the second unit on every macro swapped the wrong register (rows 0 to
  3 kept their old reference max; P off by 10^5). The emulator ran the init before each
  unit, the card runs it once a stream. The rule now takes a template slot no word of
  the unit writes, and `emit` refuses a macro whose template its init does not store or
  its tile's code writes.
- **Such a word runs nothing.** The same clamp never clamped: an argument more than 127
  below the reference gave NaN or values up to 1.8e38 (the table above). The clamp's
  `VD` is now a work register, and `emit` refuses any word with a `VD` of 12 or more that
  is not one of the init's template stores.
- **A reader told the wrong number of inputs leaves the core stuck.** The first launch
  of BODY 20 passed the reader the case's input count (1), not the body's (2): the tile
  of ones was never read and the unpacker's wait for it never returned. The mock build
  runs no kernel and could not see it; the reader now takes the body's define.
- **A wall-clock stamp with no sync is the RISC-V's issue time.** The first product
  timing stamped after `tile_regs_acquire` and after `tile_regs_wait` with no sync:
  both are waits of the coprocessor, the RISC-V runs ahead, and the spans came out
  negative. The product is now timed as the attention body times its products.
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
core stuck twice. The tile emitter now refuses such a
buffer in a unit's table (`emit`'s `one-producer` check).
