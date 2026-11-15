# Tile rules: emitted bodies against hand bodies, and the cost model against the card

Question: do the compute bodies that `komorebi.dsl.tile` emits for SiLU, a residual add,
a scale by a constant, the SwiGLU gate `silu(a) * b` and the row norm
`x * rsqrt(mean(x^2) + eps) * w` compute the same expression as the hand bodies they
stand in for, what does a unit cost on one core with each, and how close is the printed
cost to the card? Two rules are new here: the load-macro chain (schedule) and "a
per-row value is lanes of one tile" (layout, the row form). The page for the tile layer
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

The hand norm keeps decode's sequence with fp32 buffers where decode has bf16, and
reads the squares from a second copy of the row as decode reads its `DC_CB_Y`.

## How it runs

| part | file |
|---|---|
| reader (RISCV_1): the P pages once; per unit its tiles of A (and of Y for the hand norm), one tile of B | `kernels/reader.cpp` |
| compute: per unit the copies into `Dst`, the body (define `BODY`), the pack | `kernels/compute.cpp` |
| writer (RISCV_0): the output tiles, then the info page | `kernels/writer.cpp` |
| shared layout of buffers and the info page | `kernels/rules.hpp` |
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

## The cost model against the card

Each price is a one-core measurement (`../tile_probes/README.md`, section 8). The
first run showed every plain add and scale estimate 32 cycles over: exactly the 32
stalls priced where an `SFPSTORE` reads the multiply-add result just before it. Without
a stall before a store, every body is within 4% (table on the reference page); the old
SiLU code shape estimates 1721.6 against 1720.

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
