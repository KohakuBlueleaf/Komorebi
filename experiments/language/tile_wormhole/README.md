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
| `difference` | `a - b` | `SFPADD`'s `NEGATE_VC`: refused on Wormhole by name |
| `silu` | `km.silu(a)` | `SFPGT`, `SFPARECIP`: refused on Wormhole by name |

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
at the loads) and Blackhole card 0 (prebuilt, 56 of 56 kernels from the cache). The
largest error over the largest value, the lanes beyond 1e-6 of it and the lanes exactly
equal, of 4,096, against numpy in float64:

| Function | Wormhole | Blackhole |
|---|---|---|
| `add` | 3.5e-8, 0, 2,426 | the same |
| `thrice` | 4.3e-8, 0, 1,431 | the same |
| `affine` | 3.1e-8, 0, 2,854 | the same |
| `multiply` | 2.6e-8, 0, 0 | the same |
| `absolute` | 0, 0, 4,096 | the same |
| `relu` | 0, 0, 4,096 | the same |
| `larger` | 0, 0, 4,096 | the same |
| `difference` | refused: `SFPADD instr_mod1 2: NEGATE_VA, NEGATE_VC are Blackhole's` | 4.9e-8, 0, 2,457 |
| `silu` | refused: `REPLAY of 36 words` (its 32 recorded words and Wormhole's 4 `SFPNOP`s) | 1.1e-7, 0, 0 |

Every figure of the seven functions Wormhole runs is Blackhole's, digit for digit, the
relative RMS errors too: the errors are float32 rounding against float64, the same on
both chips.

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
