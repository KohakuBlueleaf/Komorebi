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
exactly equal and the relative RMS error.

    python wormhole_check.py              # the session's chip (Wormhole: the Tensix dispatcher)
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
