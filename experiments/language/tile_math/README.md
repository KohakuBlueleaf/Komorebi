# Tile math: tanh, erf and log

Question: do `km.tanh`, `km.erf` and `km.log` in a handler's tile line compute what
float64 computes, within the bounds the tile compiler prints for them? And GELU and
softplus written with them?

`tile_math.py` makes the class `Math` with five handlers. Each core takes 16 float32
tiles from rank x 16, on 12 cores (192 tiles, 196,608 values a command), and pushes
one line a tile into a float32 tensor:

```python
for x in km.Stream(a.pages(first, TILES), block=8):
    results.push(km.tanh(x))
```

`gelu_line` pushes `gelu(x)`, `softplus_line` `softplus(x)`, two tile functions:

```python
@km.tile_function
def gelu(x):
    return 0.5 * x * (1.0 + km.erf(x * SQRT_HALF))


@km.tile_function
def softplus(x):
    return km.maximum(x, 0.0) + km.log(1.0 + km.exp(-abs(x)))
```

a is 3 times a normal (both sides of tanh's 0.625 and erf's tails); b is e^u, u uniform
on [-20, 20] (log's input). `build` builds the class on a mock device into the jobs'
cache, `card` runs each handler three times and compares against float64, `made_up`
runs the host side on a made-up result.

## Measured (one card, 12 cores, 192 tiles a command)

2026-11-16, card 0, AICLK 800 MHz read before the job, host load 4.5.

| Handler | Line | rms rel. | largest rel. | largest error / rms | us a command |
|---|---|---:|---:|---:|---:|
| tanh_line | `km.tanh(x)` | 5.2e-8 | 2.3e-7 | 2.0e-7 | 35.9 to 36.9 |
| erf_line | `km.erf(x)` | 7.8e-8 | 3.6e-7 | 4.0e-7 | 35.9 to 36.0 |
| log_line | `km.log(x)` | 2.7e-8 | 8.1e-8 | 8.9e-8 | 19.4 to 19.6 |
| gelu_line | `gelu(x)` | 6.4e-8 | (tail) | 9.1e-7 | 39.2 to 39.3 |
| softplus_line | `softplus(x)` | 2.9e-8 | 3.7e-2 | 2.4e-7 | 36.5 to 36.7 |

- tanh, erf and log are within the bounds the build prints (3e-7, 4e-7, 1e-7).
- GELU's relative error is large only where GELU is under about 1e-6 (x below -5):
  1 + erf(x / sqrt 2) there is the difference of two numbers within an fp32 rounding
  of 1. Its error against the largest values is 9.1e-7.
- softplus's 3.7e-2 is at its most negative inputs, where 1 + e^-|x| rounds away most
  of e^-|x| before the log; against the largest values it is 2.4e-7.
- The build's estimate (vector unit only): tanh 2,463.6 cycles a tile, erf 2,469.6,
  log 1,058.0, GELU 2,723.6, softplus 2,501.6.
