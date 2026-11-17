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
