# Tile columns: a value a column in a call on a stream's tiles

Question: does a field of one plain row, read by `.row()` in a tile function called on a
stream's tiles (a bias, a scale a column), give every row of the tile that row's values,
in float32 and in bfloat16?

`tile_columns.py` makes the class `Columns` with two handlers, on 12 cores of 16 tiles:

- `affine_line`: float32 tiles of a, `affine(x, self.scale, self.shift)`, two float32
  fields;
- `bias_line`: bfloat16 tiles of c, `biased(x, self.bias)`, a bfloat16 field, the result
  in bfloat16.

```python
@km.tile_function
def affine(x, scale, shift):
    return x * scale.row() + shift.row()


@km.tile_function
def biased(x, bias):
    return km.silu(x + bias.row())
```

Each field is filled from a tile whose row 0 holds the 32 values (the other rows are
not read). a is normal; c is 2 times a normal and the bias normal, both rounded to
bfloat16, so float64 computes from the same values. `build` builds the class on a mock
device into the jobs' cache, `card` runs each handler three times and compares against
float64, `made_up` runs the host side on a made-up run.

## Measured (one card, 12 cores, 192 tiles a command)

2026-11-16, card 0, AICLK 800 MHz read before the job.

| Handler | rms rel. | largest error / rms | largest rel. | us a command |
|---|---:|---:|---:|---:|
| affine_line (float32) | 2.6e-8 | 3.0e-7 | 5.6e-2 | 14.7 to 15.7 |
| bias_line (bfloat16 out) | 1.7e-3 | 1.8e-2 | 3.9e-3 | 24.9 to 25.1 |

- affine_line's largest relative error is where `x s + b` cancels to near 0; against
  the largest values it is 3.0e-7.
- bias_line's largest relative error, 3.9e-3, is half a bfloat16 step (2^-8 of the value
  at its least): the rounding of the bfloat16 result.
- A first run had bias_line's values wrong (about 1e35): a bfloat16 tile's row broadcast
  goes through SrcB, whose format was still the stream's float32. The call now sets
  both source formats to the field's before the broadcast.
- The build's estimate (vector unit only): affine 123.7 cycles a tile, biased 1,571.6.
