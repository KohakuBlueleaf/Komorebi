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

Not yet measured on a chip.
