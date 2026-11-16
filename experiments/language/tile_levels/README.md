# Tile levels: two results, and the layer norm

Question: do a tile function of two results (a state and an output, pushed into two
write streams) and a layer norm (two sums in turn over the row, a weight and a bias row)
compute what float64 computes?

`tile_levels.py` makes the class `Levels` with two handlers, on 12 cores:

- `decay_line`: each core's 16 float32 tiles of a and h; `s2, y = decay(x, s)` with
  `decay` returning `0.9 h + x` and its SiLU, `s2` pushed into state, `y` into out.
- `layer_rows`: each core's 16 rows of 2 float32 tiles (64 values a row);
  `layer_norm(row, self.weight, self.bias)`, the weight and bias fields filled from
  rows laid out as the row form's per-column operand (`row_operand`).

```python
@km.tile_function
def layer_norm(row, weight, bias):
    mean = km.reduce_sum(row, exact=True) * (1.0 / (32 * WIDTH))
    d = row - mean.col()
    var = km.reduce_sum(d * d, exact=True) * (1.0 / (32 * WIDTH))
    return d * km.rsqrt(var + EPS).col() * weight.row() + bias.row()
```

a and h are normals; x is 2 times a normal plus 0.7 (a mean away from 0: the variance
is taken of x - mean); weight and bias are normals. `build` builds the class on a mock
device into the jobs' cache, `card` runs each handler three times and compares against
float64, `made_up` runs the host side on a made-up run.

## Measured (one card, 12 cores)

2026-11-16, card 0, AICLK 800 MHz read before the job, host load 6.3.

| Result | rms rel. | largest error / rms | largest rel. | us a command |
|---|---:|---:|---:|---:|
| decay state `0.9 h + x` | 3.1e-8 | 2.3e-7 | 1.7e-2 | 30.5 to 31.5 |
| decay output `silu(...)` | 6.7e-8 | 1.0e-6 | 1.7e-2 | (same command) |
| layer norm, 192 rows of 64 a command | 4.7e-8 | 9.2e-7 | 4.5e-2 | 27.2 to 27.4 |

- The largest relative errors are at values near 0: `0.9 h + x` where the two cancel
  (the constant 0.9 is rounded to fp32, 1.5e-8 of 0.9 h), and the layer norm's
  `d w + b` where the two cancel. Against the largest values the errors are 2.3e-7 to
  1.0e-6.
- The build's estimate (vector unit only): decay 1,635.6 cycles a tile (both results),
  layer norm 1,225.0 cycles a row of 2 tiles (the row norm of one sum: 673).
