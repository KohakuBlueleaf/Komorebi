# Tile functions on Wormhole

Question: does a tile function that a Wormhole chip loads (its words pass the chips
table's `tile_refusal`, in the Wormhole forms the tile compiler writes: the stall by
its `p_stall` names, the vector words' address modes with `ADDR_MOD_SET_Base` set, an
`SFPNOP` where Wormhole does not hold a read) compute on a Wormhole chip what it
computes on a Blackhole card, and what numpy computes?

`wormhole_check.py` makes one class a function, each a handler `run` over four float32
tiles of a and b on one core, run as a plain command, the result compared with numpy
(float64) on the values the tensors hold:

| Function | Body | What it exercises |
|---|---|---|
| `add` | `a + b` | a load macro, the base bracket |
| `thrice` | `a + a + a` | two MAD results read the next cycle, inside replayed words: Wormhole's `SFPNOP`s and replay count |
| `affine` | `a * 0.5 + 1.25` | a MAD with constants |
| `multiply` | `a * b` | a load macro's MAD |
| `absolute` | `abs(a)` | `SFPABS` |
| `relu` | `km.relu(a)` | a comparison and a conditional write |
| `larger` | `km.maximum(a, b)` | `SFPSWAP` |
| `chain` | 20 lines of `t = t * 0.5 + a` | a row of more than 32 replayed entries: recorded in pieces |
| `difference` | `a - b` | `SFPADD`'s `NEGATE_VC` in a load macro's template: on Wormhole the template stores `a + b`, each load of b is followed by `-b` |
| `exp`, `exp2` | `km.exp(a)`, `km.exp2(a)` | a shift of another register (Wormhole's form) |
| `sigmoid`, `silu`, `reciprocal` | `km.sigmoid(a)`, `km.silu(a)`, `km.reciprocal(a)` | `SFPARECIP` (Wormhole's estimate), `SFPGT` against 0, a negated operand |
| `swiglu` | `km.silu(a) * b` | the same, and a product after it |
| `tanh` | `km.tanh(a)` | `SFPGT` of two registers at a loop body's end: a register free because the next pass writes it first |
| `log` | `km.log(abs(a) + 0.25)` | `SFPGT` of two registers (one free), Blackhole's integer cast |
| `sqrt`, `rsqrt` | `km.sqrt(abs(a))`, `km.rsqrt(abs(a) + 0.25)` | a negated operand |
| `where` | `km.where(a > b, a, b * 0.5)` | `SFPGT` of two registers (two free) |

A function the chip refuses is a JSON line with its refusal; the others a line with
the largest error over the largest value, the lanes beyond 1e-6 of it, the lanes
exactly equal, the relative RMS error and a hash of the output's words (equal hashes:
the chips wrote the same words).

`--products LoFi HiFi2 HiFi3 HiFi4` adds a product (`a @ w`, one tile by one tile,
four tiles) at each fidelity for each operand format a class product takes
(`--sources`: bfloat16, bfp8_b, bfp4_b, bfp2_b, float16, bfp8, bfp4, bfp2), against
numpy on the values the tensors hold; `--dump` keeps every output in an `.npz`. On
Wormhole a product runs as tt-llk's own Wormhole image, in the address modes its init
sets (docs/reference/tile.md, On Wormhole).

    python wormhole_check.py              # the session's chip (Wormhole: the Tensix dispatcher)
    python wormhole_check.py --products LoFi HiFi2 HiFi3 HiFi4 --dump out.npz
    kmr prebuild --cards 0 wormhole_check.py
    kmr prebuild --arch wormhole_b0 --cards 0 wormhole_check.py

## Measured

2026-11-19: Wormhole chip 0 of the Galaxy box (Tensix dispatcher, the kernels compiled
at the loads) and Blackhole card 0 (prebuilt from the cache). The largest error over
the largest value, the lanes beyond 1e-6 of it, and the largest relative error of a
lane, of 4,096, against numpy in float64; "the same": Blackhole's words, every one:

| Function | Wormhole | Blackhole |
|---|---|---|
| `add` | 3.5e-8, 0, 5.9e-8 | the same |
| `thrice` | 4.3e-8, 0, 6.0e-8 | the same |
| `affine` | 3.1e-8, 0, 6.0e-8 | the same |
| `multiply` | 2.6e-8, 0, 5.8e-8 | the same |
| `absolute`, `relu`, `larger`, `where` | 0, 0, 0 (exact) | the same |
| `chain` | 6.4e-8, 0, 1.2e-7 | the same |
| `exp` | 3.1e-8, 0, 6.6e-8 | the same |
| `exp2` | 3.5e-7, 0, 4.7e-7 | the same |
| `log` | 6.3e-8, 0, 8.3e-6 (at log near 0) | the same |
| `sqrt` | 7.4e-8, 0, 1.1e-7 | the same |
| `rsqrt` | 6.5e-8, 0, 8.9e-8 | the same |
| `reciprocal` | 4.7e-8, 0, 1.12e-7 | 3.0e-8, 0, 1.09e-7 |
| `sigmoid` | 1.3e-7, 0, 1.48e-7 | 1.4e-7, 0, 1.50e-7 |
| `silu` | 1.1e-7, 0, 1.82e-7 | 1.1e-7, 0, 1.69e-7 |
| `swiglu` | 7.7e-8, 0, 2.03e-7 | 7.7e-8, 0, 2.01e-7 |
| `difference` | 4.9e-8, 0, 6.0e-8 | the same |
| `tanh` | 1.4e-7, 0, 2.3e-7 (its stated error: 3e-7) | 1.5e-7, 0, 1.8e-7 |

Fifteen functions write Blackhole's words on Wormhole, every one (0 of 4,096 words
differ, the dumps compared): the comparisons, the casts, the negations (a template's
too) and the shifts in Wormhole's forms are exact. The five with a reciprocal differ
in the last bits (1,509 to 1,978 words): Wormhole's estimate is a cubic, not
Blackhole's table, and the two Newton steps after it leave errors of the same size on
both chips.

`experiments/language/step_lines/row_softmax.py card` (`softmax_step`: a lagging max,
two `exp2`, the sums; 12 cores) on chip 0: P 1.79e-3 relative RMS, 5.8e-3 largest; m
4.8e-3; l 6.7e-3; the softmax 5.7e-3; a row's lanes 1.4e-6 apart: card 0's figures
(its README). A command takes 33.7 to 34.8 us (card 0: 20.5 to 21.4 us).

### Products

2026-11-19, the same chip and card, `--products LoFi HiFi2 HiFi3 HiFi4`: all 32
products (8 formats, 4 fidelities) ran on both, and Wormhole's output is Blackhole's
word for word in every one: 0 of 131,072 words differ (the dumps compared), as are the
seven functions above. The figures, the same on both chips: the largest error over the
largest value and the relative RMS error.

| Operands | LoFi | HiFi2 | HiFi3 | HiFi4 |
|---|---|---|---|---|
| bfloat16 | 3.2e-2, 2.7e-2 | 5.0e-3, 4.0e-3 | 4.6e-4, 3.3e-4 | 4.4e-4, 3.2e-4 |
| bfp8_b | 2.2e-2, 1.9e-2 | 4.3e-4, 2.6e-4 | the same | the same |
| bfp4_b, bfp2_b, bfp4, bfp2 | 0 (exact) | 0 | 0 | 0 |
| float16 | 3.4e-2, 3.1e-2 | 7.4e-3, 6.6e-3 | 1.8e-3, 8.6e-4 | 1.8e-3, 8.6e-4 |
| bfp8 | 2.1e-2, 1.9e-2 | 9.5e-4, 6.7e-4 | the same | the same |

The differences an audit of the two chips' documents leads to expect (Wormhole
truncating a source datum where Blackhole rounds it; an fp32 accumulator of exactly
-1 ulp normalized otherwise) did not show here: these operands fit the source
registers as they are (a float32 operand, read as tf32, is refused in this form), and
no sum met the -1 ulp case. With bfp4 or narrower operands every product is exact on
both chips.

One real line, `projects/dense/run.py --m 160 --k 64 --n 96 --split 2,2,2,1 --reduce
tree --inputs int --calls 4` (the card list's smallest shape: hand kernels, the
library's product, a plain command through the Tensix dispatcher, 12 cores), on
Wormhole chip 0: its check exact (160 rows, bit for bit), 10.0 us a command (median).
