# Neural network layers as classes

Question: how does a project write the common layers of a network with the language as
it is today, and are the values right? Each script here is one family of layers. It
makes random bfloat16 inputs, runs the layers on one card, reads the results back and
compares them with float64 numpy. It prints one JSON line: for each layer the rms and
the largest error, each divided by the reference's rms, and the microseconds of each
command (its event). The exit code is 0 when every check is under its bound.

`forms.py` holds what the scripts share: the log, tensors made from numpy arrays and
read back, the error and the JSON line.

```bash
python elementwise.py --cores 8 --tiles 8
python conv_shift.py
python conv_more.py
python norms.py
python resample.py
python head.py --cores 8
python embedding.py --cores 8
python state.py --cores 8 --tiles 4 --tokens 6
python chain.py --layers 4
python conv_shift.py --cards 0,1  # also conv_more.py, norms.py
```

All the numbers below are from one Blackhole p150a (card 1), slow dispatch, with the
kernels prebuilt (0 compiled in the run). Every script ran once as it is now. Where a
table gives HiFi2 beside HiFi4, the HiFi2 column is the run before the products by a
0/1 matrix were moved to HiFi4: the same script with only that fidelity changed.

## What bounds the errors

Two formats explain every error measured here, and each is checked against a model:

- A bfloat16 result keeps 8 significant bits: about 2^-9 of a value's size, so 1e-3 to
  2e-3 of the rms. Elementwise layers come out at that.
- A product at `fidelity="HiFi2"` keeps 7 of the left operand's 8 bits in the matrix
  unit (the build's placement line: `plain (left in SrcB): left keeps 7, right 8`). A
  product by a 0/1 matrix (a shift, a nearest resample, a selection) is then the
  left operand truncated to 7 bits. The measured errors at HiFi2 equal that truncation,
  done in numpy on the same inputs, to every printed digit: rms 4.09e-3 for the shift,
  4.08e-3 for up nearest, 4.12e-3 for down nearest.

The rule: a product by a 0/1 matrix is an exact copy only at HiFi4. At HiFi4 the
shifts, the stride-2 selection and both nearest resamples measure error 0, and none
was slower than at HiFi2 here (the shift 11.6 us against 12.3, the selection 10.7
against 11.1). A class's products share one fidelity: it is the compute kernel's, and
two in one class are refused. So the shifts and the selection are classes of their own
(`Shift`, `Select`), and the convs over them stay at HiFi2 (`Conv`, `Stride2`).
A subclass cannot change it: a tile function redefined in a subclass is not taken (the
parent's is built), and `compute={"math_fidelity": ...}` on the subclass is refused
against the products' fidelity.

## Elementwise (`elementwise.py`)

A tile function a layer over DRAM streams, 8 cores, 8 tiles a core:

| layer | rms | max | us |
|---|---|---|---|
| GELU, erf form, over 2 chunks of the cores' pages | 1.27e-3 | 5.6e-3 | 38.0 |
| GELU, tanh form written as `x * km.sigmoid(1.5957691216 * (x + 0.044715 x^3))` | 1.26e-3 | 5.6e-3 | 15.2 |
| SiLU | 1.65e-3 | 1.19e-2 | 13.2 |
| residual add, then SiLU | 1.66e-3 | 1.63e-2 | 14.2 |
| per-channel scale and shift, a value a row (`g.col()`) | 1.67e-3 | 1.48e-2 | 10.2 |
| the same, a value a column (`g.row()`) | 1.67e-3 | 2.71e-2 | 9.4 |

The tanh form written with `km.tanh` is refused (more live values than the 8 work
registers). The sigmoid identity, 0.5 (1 + tanh z) = sigmoid(2 z), builds.

## Convolutions (`conv_shift.py`, `conv_more.py`)

The feature map is a matrix with rows (y, channel) and columns x, with a border tile
row and a border tile column on each side. A shift along x is a product by a 0/1 matrix
on the right. A shift along y is a choice of tile rows. The channels mix by the weights
on the left. 64 channels in and out, 16 x 64 pixels:

| layer | rms (HiFi4 copies) | max | us | rms (HiFi2 copies) | us |
|---|---|---|---|---|---|
| 3x3, zero border: the shifted copies | 0 | 0 | 11.6 | 4.09e-3 | 12.3 |
| 3x3, zero border: the conv (9 taps, K = 2 tiles each, HiFi2) | 4.40e-3 | 2.32e-2 | 32.1 | 6.40e-3 | 31.0 |
| 3x3, border holding the edge's copy: shifts | 0 | 0 | 11.1 | 4.08e-3 | 11.1 |
| 3x3, border holding the edge's copy: the conv | 4.39e-3 | 2.26e-2 | 32.1 | 6.41e-3 | 32.1 |
| 3x3 stride 2: the selection (its taps checked directly) | 0 | 0 | 10.7 | not checked | 11.1 |
| 3x3 stride 2: the conv on rows 2y + dy | 4.34e-3 | 2.29e-2 | 15.9 | 7.16e-3 | 17.0 |
| depthwise 3x3: a diagonal weight tile a tap and channel tile | 4.52e-3 | 3.79e-2 | 28.4 | 6.56e-3 | 28.8 |
| 1x1 + SiLU in the product's write (no copy) | 4.75e-3 | 4.40e-2 | 7.7 | | |
| bias as its own pass, `x + bias.col()` (no product) | 1.77e-3 | 1.13e-2 | 7.2 | | |

The border is whatever the input tensor's border tiles hold. Zeros give zero padding
and the edge's copy gives replicate padding, with the same kernels. The convs' 4.4e-3
is the HiFi2 weights' products and the bf16 output.

A bias cannot be a second operand of a product's write (`act(acc, self.bias)` is
refused), so it is its own pass over the output.

## Norms (`norms.py`)

Group norm, 2 groups of 32 channels, over 16 x 64 pixels, in four steps:

1. `squares`: x * x of each tile.
2. `column_sums`: a product of a tile of ones on the left by each of the group's tiles,
   summed in DST over y at HiFi4, so each row of the result is a column sum. Then
   `lanes`: the row form's `row * 0.0 + km.reduce_sum(row, exact=True).col()` sums
   along the row. It runs once on x and once on the squares.
3. The host derive. Mean = s1 / n and var = s2 / n - mean^2, from the two sums read
   back. Then scale = gamma rstd and shift = beta - mean scale, written as column 0 of
   two small float32 tensors.
4. `apply`: x * scale.col() + shift.col() over the group's tiles.

| step | where it ran | rms | max | us |
|---|---|---|---|---|
| squares | card | | | 6.6 |
| group sum s1: ones product, then lane sum | card | 2.0e-5 | 2.7e-5 | 14.2 + 4.3 |
| group sum of squares s2 (squares written as bf16) | card | 3.6e-4 | 3.7e-4 | 14.2 + 4.3 |
| mean, rstd, scale and shift | host, from s1 and s2 read back | | | |
| group norm, after the apply | card | 1.82e-3 | 2.00e-2 | 24.6 |

Layer norm over rows of 64 tiles (2,048 values), 4 cores, a tile row (32 rows) each,
read in blocks of 8, float32 in and out, weight and bias by `.row()`: rms 5.4e-8, max 9.9e-7, 67.8 us. In this long-row
form (the chunk form) a `.row()` field holds one tile for each chunk of 2 tiles, made
by `komorebi.dsl.tile.row_operand` from the chunk's 64 values: 32 tiles for a row of
64. The first run wrote row 0 of 64 tiles, as a call on a stream's tiles reads a
field, and was wrong (rms 1.06).

## Resampling (`resample.py`)

Up and down by 2, in the convolutions' layout, 64 channels, 8 x 64 to 16 x 128 and
back. Along x a product by a constant matrix on the right (duplication, the 0.75 / 0.25
interpolation, selection, the pair's mean). Along y a choice of tile rows or a weighted
sum of two. The border holds the edge's copy, which gives bilinear's clamped edges.
The two directions are two classes, because a class's rectangles are built for its
example tensors' widths.

| layer | rms (HiFi4) | max | us (x, then y) | rms (HiFi2) | us |
|---|---|---|---|---|---|
| up, nearest | 0 | 0 | 9.7 + 5.8 | 4.08e-3 | 9.5 + 5.8 |
| up, bilinear (half-pixel centres) | 2.32e-3 | 2.16e-2 | 9.2 + 8.3 | 4.04e-3 | 9.4 + 8.3 |
| down, nearest | 0 | 0 | 8.2 + 4.2 | 4.12e-3 | 8.4 + 4.3 |
| down, mean of 2 x 2 | 2.81e-3 | 1.64e-2 | 7.1 + 4.8 | 3.87e-3 | 7.1 + 4.8 |

The weights of these products (1, 0.75, 0.25, 0.5) are exact in bfloat16. At HiFi4
the only error left is the bf16 rounding of a weighted sum.

## The head (`head.py`)

Logits = x W for one token row, 8 cores, 384 logits a core (3,072 in all), in 4 chunks
of 3 tiles of DST sums. A running per-row maximum is kept in a state field by the step
form with a lag of 0: `km.lagging_max(m_old, km.reduce_max(row, exact=True), 0.0)`.
The language has no comparison or select in a tile function and no `reduce_max` in
the row form, so the argmax is C++ of the script's own. On the pack CPU, `km.cxx`
bodies scan row 0 of each float32 logit tile (face 0 words 0 to 15, face 1 words 256
to 271) and keep the largest value and its index in one page a core. A second handler
has the leader read every core's page in order and keep the largest.

| what | result | us |
|---|---|---|
| logits | rms 4.03e-3, max 2.08e-2 | 15.5 |
| running max (slot 0 of the state page) | equal to the largest written logit of each row: error 0 | |
| argmax of row 0 | index 2872, numpy's, value equal | 19.2 + 4.6 |
| argmax with a tie: 4.0 at 808, 1068 (same core) and 1927 | 808, the first, as `np.argmax` | 18.4 + 4.5 |

A strict `>` in the scan and in the combine, in index and core order, keeps the first
of equal values. The column numbers come from the tile's layout, not from a table:
row 0's columns 0 to 15 are words 0 to 15 of face 0 and columns 16 to 31 are words
256 to 271 of face 1. So the index is (tile) x 32 + i or + 16 + i, the tile counted
from the core's first.

`km.where` and the six comparisons are tile functions now, but the row form has no
maximum of a row. So the index of a row's largest value is still the body above.

## Embedding (`embedding.py`)

A token word of a page (`ids.pages(r, 1).entries(Entry, size=1, fixed=1)`) picks what
each core reads. 8 cores, a 256-entry table of 128 values, two cores on one entry and
one on the last.

- E1: the table in tile layout, one entry a tile row (32 times the memory of the table).
  The token's tiles are one rectangle, `table.tiles[e.token : e.token + 1, 0:E]`, copied
  to row r of the output. Exact (error 0), 5.4 us.
- E2 (`--e2`): the table in row layout, the token's row as 8 items of 32 B, each pushed
  to an item write stream of the output. WRONG: rms 1.06 of the reference. The build
  lowers `results.push(item)` to compute's `copy_tile` and `pack_tile` of a whole tile,
  while its own table gives both buffers pages of 32 B (`kRead0 ... 4 + 4 pages of 32
  B`, `kWrite0 ... 2 + 2 pages of 32 B`). So each push packs a 2 KiB tile into a 32 B
  slot. The run completed. Until this is a refusal or a mover's copy, a row-layout
  lookup is a body on a mover, not this line.

## A recurrent state (`state.py`)

h = a h + u, one token a command, the state a float32 tensor in DRAM read and written
in place, page by page (`state.pages(first, tiles)` as both a read and a write stream
of one command). 8 cores, 4 tiles a core, 6 tokens: rms 3.7e-8, max 3.2e-7 of the
float64 recurrence, which is float32's rounding. 7.1 us a token after the first (7.9).

## One chain of three classes (`chain.py`)

A small model as one chain: `Model(Norm, Proj, Act)` merges three classes into one
program, and 4 layers of layer norm, a product by W1, SiLU, a product by W2 and a
residual add are 20 calls in one `run.chain`. The row is 32 x 2,048. The norm runs on
one core (its call's `cores=(0, 1)`), the rest on 16. The products take W's columns in
blocks of the class constants `WIDTH` and `W_BLOCK`. The residual reads x, written 5
calls back, as `km.past(x)`.

```python
chain = run.chain(
    [handler(*arguments, cores=group) for handler, group, arguments in calls],
    cores=(0, 16),
)
event = run.wait(run.submit(chain))
```

On card 1:

- The chain's x equals, bit for bit, x from the same 20 calls submitted one command at
  a time.
- One chain takes 514.1 us; the 20 commands take 547.0 us summed.
- Against float64 numpy, rms 7.4e-3 and max 3.5e-2 after 4 layers. A model that
  rounds every stored tensor to bf16 and truncates each product's left operand as HiFi2
  does is no closer (rms 7.1e-3), so it does not account for the rest of the error.

A core's buffers take 822,016 B of the 1,352,576 B left for them: the norm's weight and
bias fields take 131,072 B each, in `row_operand`'s layout. On two cards, 1,018,160 B
is left for buffers. There the merged kernels take 164,880 B of the 163,840 B kernel
buffer, so the load places `norm` as an overlay and builds the program again. Cards 0
and 1, 2026-11-18, `--cards 0,1 --layers 1`, after `kmr prebuild`:

- The chain's x equals x from the 5 commands bit for bit.
- One chain takes 162.0 us; the 5 commands take 177.6 us summed.
- Against float64 numpy, rms 4.3e-3 and max 2.5e-2.

Until that date the rebuild lost the defines the two-card load had added, and the load
was refused ("the link plan keeps the link area on the link cores alone").

## Two cards (`--cards 0,1`)

`conv_shift.py`, `conv_more.py` and `norms.py` take `--cards 0,1`. The session is then
the two cards as islands, and each command's group starts at
`session.graph.first(1) - n / 2`, so half its ranks are on each card. The scripts do not
print where the groups sat. Cards 0 and 1, the same inputs and seeds:

- Every error is equal to one card's, to every printed digit. The shifts and the
  selection are exact, and layer norm is 5.4e-8.
- Every command takes 4 to 92 times its time on one card: shift 569.3 us against
  11.6, conv 2,944.8 against 32.1, depthwise 1,979.1 against 28.4, layer norm 284.6
  against 67.8, group norm's apply 150.3 against 24.6. Where the time goes (which pages
  cross the link) is not measured here.

## Not proven

- E2 above. It is not run unless `--e2` is given, and then its check fails.
