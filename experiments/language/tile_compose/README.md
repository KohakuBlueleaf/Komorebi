# Tile functions composed

Question: does a tile function's result taken as another's operand (nested, through a
name bound earlier in the loop, or inside a tile expression) give on the card what
numpy gives, as one unit program a tile?

`compose.py` makes the class `Composed`: three handlers, each a stream loop over four
tiles of x, y and z on one core, pushing one composed call a tile into a float32
tensor:

```python
@km.tile_function
def larger(x, y):
    return km.where(x > y, x, y * 0.5)


@km.tile_function
def scaled(t, z):
    return t * z + 1.0
```

`run_nested` pushes `scaled(larger(a, b), c)`; `run_named` binds `t = larger(a, b)`
and pushes `scaled(t, c)`; `run_expression` pushes `km.silu(larger(a, b)) * c`;
`run_field_row` pushes `biased(larger(a, b), self.bias)` (`biased(t, b)`: `t +
b.row()`) and `run_field_col` binds `t = larger(a, b)` and pushes `by_column(t,
self.bias)` (`t * b.col()`), the field filled from a float32 tile w. Each is one unit
program (`tile_line_N`). x, y and z are normals times 4, y equal to x in a
quarter of the lanes; the class is made for float32 and bfloat16 inputs, each result
compared with numpy (float64 for the silu) on the stored values.

    python compose.py                # one card (--cards N: that card)
    kmr prebuild --cards 0,1 compose.py --cards 0,1   # the same class for two cards, on the mock

## Measured (one card, one core)

2026-11-18, card 1, AICLK 800 MHz before the job, 0 compiled (prebuilt), 13 s with the
open. The largest error over the largest value, and lanes beyond 1e-6 of it, against
numpy on the stored values (float32 arithmetic; float64 for the silu):

| Handler | float32 | bf16 |
|---|---|---|
| `run_nested` | 4.2e-8, 0 lanes | 0 (exact), 0 lanes |
| `run_named` | 4.2e-8, 0 lanes (the same figures as nested) | 0, 0 lanes |
| `run_expression` | 7.3e-8, 0 lanes | 1.2e-7, 0 lanes |

The float32 differences are the fused `t * z + 1` (one rounding where numpy rounds
twice). Each handler's call is one unit program; in the build's list nested and named
are the same words (334 instructions a tile, 3 of 8 work registers), the expression
1,560 (silu's exp and reciprocal; 8 of 8 work registers).
