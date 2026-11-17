# Tile where and floor

Question: does `km.where(cond, a, b)` under each of the six comparisons, and
`km.floor(x)`, give on the card what numpy gives on the values the tensors hold, and
where they differ (signed zeros, NaN, infinities, subnormals, the ties of floor)?

`where_floor.py` makes the class `Where`: eight handlers, each a stream loop over two
tiles of x and y on one core, pushing one tile function a tile into a float32 tensor:

```python
@km.tile_function
def greater_equal(x, y):
    return km.where(x >= y, 1.0, 0.0)
```

`greater`, `less`, `greater_equal`, `less_equal`, `equal` and `not_equal` give 1 or 0
a lane, `smaller` gives `km.where(x < y, x, y)`, and `floor` pushes `km.floor(x)`.
Tile 0 holds every pair of 32 edge values (x the row's, y the column's): +0, -0, NaN,
+-inf, +-1, +-0.5, +-1.5, +-2.5, +-3, -0.75, +-1e-40 (subnormal), the smallest normal,
+-(2^23 - 0.5), +-2^23, +-(2^23 + 1), +-(2^22 + 0.5), 2^24 + 2, +-1e30, 100.25 and
-100.75. Tile 1 holds random values of either sign, y equal to x in half the lanes.
The class is made for float32 and for bfloat16 inputs; each lane is compared with
numpy's result on the stored values (a NaN equal to a NaN), and a lane equal in value
but not in sign is counted apart.

    python where_floor.py                # one card (--cards N: that card)
    kmr prebuild --cards 0,1 where_floor.py --cards 0,1   # the same class for two cards, on the mock

## Measured (one card, one core)

2026-11-18, card 1, AICLK 800 MHz before the job; 0 compiled (prebuilt), 14 s with the
open. Lanes of 2,048 a handler that differ from numpy (a NaN equal to a NaN), and those
equal in value but not in sign:

| Handler | float32 | bf16 |
|---|---|---|
| greater, less | 32: NaN above every value (31 lanes), +0 > -0 | 35: NaN, and subnormals read as 0 |
| greater_equal, less_equal | 33: NaN (32), -0 >= +0 false | 38: NaN, subnormals as 0, NaN read as +inf |
| equal, not_equal | 3: +0 == -0 false (2), NaN == NaN true | 13: subnormals equal to 0, NaN equal to +inf |
| smaller | 97: a subnormal picked is stored as a zero of its sign; NaN above every value | 126: also -0 read as +0, 2^-126 stored as 1.5 x 2^-126 |
| floor | 32 (-1e-40 gives 0, not -1); 32 lanes -0 gives +0 | 32 (NaN read as +inf gives +inf); 32 lanes -0 gives +0 |

Every other lane is numpy's: the comparisons of the random pairs and of every normal
and infinite edge value, and floor's ties (2.5 -> 2, -2.5 -> -3, -0.5 -> -1),
+-(2^23 - 0.5), +-2^23, +-(2^23 + 1), +-(2^22 + 0.5), 2^24 + 2, +-1e30, +-inf and NaN
(float32). The comparisons are SFPGT's order of fp32 as sign-magnitude integers (-0
below +0, numpy's NaN above +inf); the rest is how the unpack reads bf16 and how the
pack stores a subnormal. The same class builds for two cards on the mock (both islands'
kernels, all 16 handlers).
