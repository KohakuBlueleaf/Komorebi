# Tile lines of a handler

Question: does a tile expression written in a handler's line (`results.push(x / y)`)
give the same words as a `@km.tile_function` of the same expression? And do the vector
functions x / y, `km.sqrt`, `abs`, `km.minimum` and `km.reciprocal`, and a function of
four tiles, compute what float64 computes? How fast are x / y and `km.sqrt` next to the
inline body of tt-metal's own library call?

`tile_lines.py` makes the class `Lines` with nine handlers. Each core takes 64 tiles
from rank x 64, on 12 cores (768 bf16 tiles of each input a command), and pushes one
line a tile into a float32 tensor:

```python
for x, y in zip(
    km.Stream(a.pages(first, TILES), block=8),
    km.Stream(b.pages(first, TILES), block=8),
):
    results.push(x / y)
```

The inline-body forms are the library's calls on the tiles copied into `Dst`
(`div_binary_tile(0, 1, 0)` after `div_binary_tile_init()`, and `sqrt_tile(0)` after
`sqrt_tile_init()`), as a `km.cxx(..., gives=km.Tile)` body. a and d are normal, b is
normal with its magnitude moved to 0.5 and above (the divisor), and c is the absolute
value of a normal (the root's input). `build` builds the class on a mock device into
the jobs' cache. `card` runs each handler three times and compares against float64.
`--only NAME ...` runs only the named handlers.

## Measured (one card, 12 cores, 768 tiles a command)

| Handler | Line | rms rel. | largest rel. | us a command |
|---|---|---:|---:|---:|
| divide | `results.push(x / y)` | 5.0e-8 | 1.6e-7 | 30.3 |
| divide_body | `div_binary_tile` (inline body) | 2.6e-8 | 7.0e-8 | 75.5 |
| root | `results.push(km.sqrt(x))` | 3.9e-8 | 9.5e-8 | 40.2 |
| root_body | `sqrt_tile` (inline body) | 3.0e-8 | 8.9e-8 | 53.9 |
| smaller | `results.push(km.minimum(abs(x), y))` | 0 | 0 | 23.2 |
| inverse | `results.push(km.reciprocal(y))` | 4.3e-8 | 1.1e-7 | 23.5 |
| four | `results.push(x * y + z * w)` | 3.2e-9 | 6.0e-8 | 34.9 |
| gated | `results.push(km.silu(x) * y)` | 6.1e-8 | 2.5e-7 | 82.1 |
| gated_function | `results.push(self.gate(x, y))` | 6.1e-8 | 2.5e-7 | 82.1 |

- `gated` and `gated_function` are equal datum for datum: the pushed expression is the
  tile function's unit program, and both take the same time.
- The line's x / y takes 0.40 of the time of the inline body's `div_binary_tile`. The
  line's sqrt takes 0.75 of `sqrt_tile`'s time. The line's results differ from the
  bodies' in 42 % and 36 % of datums, all within the errors above.
- `four` is from its own run (`tile_lines_four.json`, `--only four`). In the first run
  the script's reference took c where the tensor holds |c|, so that run's `four` row is
  marked unchecked in `tile_lines.json`.
