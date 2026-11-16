# Tile wide rows: norms over 768 values

Question: do the RMS norm and the layer norm, written as tile functions on rows of 24
tiles (768 values, a hidden size of real models) and run in chunks of 2 tiles, compute
what float64 computes?

`tile_wide.py` makes the class `Wide` with two handlers, on 12 cores of 4 rows each (48
rows of 768 values a command):

```python
@km.tile_function
def rms_norm(row, weight):
    ms = km.reduce_sum(row * row, exact=True) * (1.0 / (32 * WIDE))
    return row * km.rsqrt(ms + EPS_RMS).col() * weight.row()


@km.tile_function
def layer_norm(row, weight, bias):
    mean = km.reduce_sum(row, exact=True) * (1.0 / (32 * WIDE))
    d = row - mean.col()
    var = km.reduce_sum(d * d, exact=True) * (1.0 / (32 * WIDE))
    return d * km.rsqrt(var + EPS).col() * weight.row() + bias.row()
```

A row is one stream block. Per row: one acquire runs the sums' passes over the 12
chunks (the layer norm: two passes, the mean's then the variance's) and makes the
per-row values, packed to a page of their own; then each chunk is a call of its own,
the values copied back beside the chunk and the weight's (and bias') chunk. The weight
and bias fields hold a tile a chunk, chunk c's 64 values laid out as the row form's
per-column operand (`row_operand`). x is 2 times a normal plus 0.7 (a mean away from 0);
weight and bias are normals. `build` builds the class on a mock device into the jobs'
cache, `card` runs each handler three times and compares against float64, `made_up` runs
the host side on a made-up run.

## Measured (one card, 12 cores)

2026-11-16, card 0, AICLK 800 MHz read before the job.

| Handler | rms rel. | largest error / rms | largest rel. | us a command |
|---|---:|---:|---:|---:|
| rms_rows | 5.5e-8 | 1.6e-6 | 2.6e-7 | 64.9 to 65.8 |
| layer_rows | 5.1e-8 | 1.1e-6 | 1.3e-2 | 99.2 to 99.6 |

- The layer norm's largest relative error is where `d w + b` cancels to near 0; against
  the largest values it is 1.1e-6.
- The build's estimate (vector unit only, the chunk parts once a chunk): RMS norm 7,723
  cycles a row of 24 tiles, layer norm 13,736. A row of 2 tiles in the row form: 673 and
  1,225.
