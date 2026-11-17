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
and pushes `scaled(t, c)`; `run_expression` pushes `km.silu(larger(a, b)) * c`. Each
is one unit program (`tile_line_N`). x, y and z are normals times 4, y equal to x in a
quarter of the lanes; the class is made for float32 and bfloat16 inputs, each result
compared with numpy (float64 for the silu) on the stored values.

    python compose.py                # one card (--cards N: that card)
    kmr prebuild --cards 0,1 compose.py --cards 0,1   # the same class for two cards, on the mock
