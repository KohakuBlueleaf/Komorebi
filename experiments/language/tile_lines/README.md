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

## A product and a tile function in one handler

Question: when a handler writes a tile function of a product's sums
(`y.pages(first, n).write(act(acc))` of DST sums, `y.tiles[...].write(act(self.part))`
of a field of sums), is the result silu(x W), and what does the function add to the
product alone?

`linear_act.py` makes the class `LinearAct` with three handlers on 12 cores. Each
core computes 4 columns of y = x W from rank x 4 (x 32 x 1024 bf16, W 1024 x 1536 bf16,
two K tiles a block, HiFi4) and writes them into a float32 tensor:

```python
acc = km.tile.zeros(km.fp32, WIDTH)
for xb, wb in zip(
    km.Stream(x.tiles[0:1, 0:K], block=2),
    km.Stream(w.tiles[0:K, first : first + WIDTH], block=W_BLOCK),
):
    acc += self.project(xb, wb)
y.pages(first, WIDTH).write(act(acc))
```

`plain` writes acc, `in_dst` writes act(acc), and `in_field` adds the product into an
fp32 field (`self.part += ...`) and writes act(self.part). `card` runs each handler
five times and compares against float64.

### Measured (one card, 12 cores)

| Handler | Line | rms rel. | largest rel. | us a command |
|---|---|---:|---:|---:|
| plain | `y.pages(first, 4).write(acc)` | 1.7e-4 | 1.8e-4 | 18.1 to 18.8 |
| in_dst | `y.pages(first, 4).write(act(acc))` | 1.7e-4 | 1.7e-4 | 22.6 to 22.9 |
| in_field | `y.tiles[0:1, ...].write(act(self.part))` | 1.7e-4 | 1.7e-4 | 23.1 to 23.6 |

- The errors are the product's own (bf16 operands, fp32 sums). silu adds nothing that
  shows at this size.
- act over the 4 DST tiles adds 4.4 us a command, about 1.1 us a tile. The field form
  costs 0.6 us more than the DST form: the copy of each field tile into DST.

## Run-time scalars in a tile function

Question: does a float word of the command, used in a tile expression
(`results.push(x * scale)`) or passed to a tile function (`affine(x, scale, shift)`,
`x * scale + shift`), give what the inline body of tt-metal's `mul_unary_tile` gives,
with each command its own value?

`scalars.py` makes the class `Scalars` with three handlers on 12 cores, 64 bf16 tiles a
core, each into a float32 tensor: `line` (`results.push(x * scale)`), `called`
(`results.push(affine(x, scale, shift))`) and `body` (`mul_unary_tile(0,
__builtin_bit_cast(uint32_t, scale))` after `binop_with_scalar_tile_init()`, an inline
body that takes `scale` as a C++ float). The class is loaded once, and each handler runs
one command for each (scale, shift) of (2.0, 0.5), (-0.75, 3.0), (0.001, -1.0) and
(15000.0, 0.0). Each result is compared against float64.

### Measured (one card, 12 cores, 768 tiles a command)

| scale | line: rms rel. | line == body | called (shift): rms rel. | us a command: line / called / body |
|---:|---:|:---:|---:|---|
| 2.0 | 0 | yes | 1.7e-11 (0.5) | 18.6 / 16.1 / 34.0 |
| -0.75 | 0 | yes | 2.4e-10 (3.0) | 15.8 / 16.9 / 33.9 |
| 0.001 | 2.5e-8 | yes | 2.7e-8 (-1.0) | 15.5 / 16.3 / 33.8 |
| 15000.0 | 0 | yes | 0 (0.0) | 16.3 / 16.3 / 33.6 |

- `line` equals the body datum for datum at every scale: one fp32 product a datum,
  rounded once. A product that float32 holds exactly is exact.
- Each command took its own word with no build between them: the loads before the rows
  are two `TT_SFPLOADI` a word, and the rows replay as for a number.
- The line takes 0.47 of the body's time. The body copies each tile into Dst and runs
  the library's loop; the line's eight tiles a call share one setup.
- `called` adds the shift in the same `SFPMAD` (one rounding), at the line's time.
- The first command (18.6 us) includes the first use of the class's code.
