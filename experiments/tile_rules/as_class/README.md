# Tile rules, as tile functions of a class

Question: do the load-macro bodies of [../README.md](../README.md), the residual add
`a + b` and the scale `x * 1/sqrt(128)`, and its row norm in the row form, run the same
when they are a class's tile functions ([Programs written as classes](../../../docs/reference/classes.md#tile-functions)),
called on the tiles of a stream on many cores, as they ran in the one-core harness?

`rules_class.py` makes the class `Rules`: two tile functions, `residual` and `scaled`,
and two handlers, `add` (two streams read in step) and `scale` (one stream), each
`@km.handler(stamps="stamps")`: the math CPU reads the wall clock around each call
between two `tensix_sync()`, as the harness does. Its files are under `build/rules/`
beside the script (`tile_residual.hpp`, `tile_scaled.hpp` and their unit programs as
text).

```bash
python rules_class.py --cores 12 --tiles 64
```

The values are the harness's: float32 normal numbers times 8, and one page over
magnitudes 2^-40 .. 2^40 with random signs; each output against float64 of the same
inputs, the stated tolerance one fp32 rounding (2^-23).

## Results

Card 1, a Blackhole p150a, 12 cores, 64 tiles a core (768 calls each), seed 1, host
load 2.75, one run. Cycles a call on the math CPU: the median span less the median
empty span (10 cycles).

| Body | Printed at the build | On the card, class | Harness (../README.md) | Largest relative error | Command |
|---|---|---|---|---|---|
| add, load macro | 91.6 | 91.0 (raw 101 to 105) | 93 | 5.96e-8 | 36.6 µs |
| scale, load macro | 59.5 | 59.0 (raw 69 to 74) | 62 | 5.95e-8 | 21.8 µs |

The build printed, for each, that the load-macro rule fired ("1 operation(s) and the
store in one SFPLOADMACRO a row") and the replay recorded 4 of 4 instructions of a
row. The class's calls are 2 and 3 cycles shorter than the harness's units of the same
bodies; the harness timed 16 units on one core, the class 768 calls on 12 cores.
Kernel text 47,520 B for the two stamped handlers.

## The row norm

`norm_class.py` makes the class `RowNorm`: the tile function `norm(row, weight)`,
`y = x rsqrt(mean(x^2) + 1e-6) g` over a row of 2 tiles, and the stamped handler
`norm_rows`, which fills the field `weight` with the row form's per-column tile of g
(`p_tile` of ../rules.py) and reads x with `row=2`. Its files are under
`build/rownorm/` (`tile_norm.hpp`, `tile_norm.txt`).

```bash
python norm_class.py --cores 12 --rows 32
```

The values are ../rules.py's for the norm: normal numbers, the first row of tiles
times 100, a weight `1 + 0.1 N(0, 1)`; the output against float64, the stated
tolerance 1e-5.

Card 1, 12 cores, 32 rows of 2 tiles a core (384 calls), seed 1, host load 1.55, one
run.

| Body | Printed at the build | On the card, class | Harness (../README.md) | Largest relative error | Command |
|---|---|---|---|---|---|
| norm, row form | 753.0 a row | 771.0 (raw 780 to 804) | 771.5 (hand norm 3,201.5) | 2.33e-7 | 38.92 µs |

The build printed 649 instructions and that the product-into-add rule fired 8 times.
Kernel text 41,376 B.
