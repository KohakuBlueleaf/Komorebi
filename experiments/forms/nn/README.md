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
python sum_operand.py
python halo_exchange.py --cards 0,1  # --chain: also as one chain, slow (a covering range)
python group_norm.py --chain  # also --cards 0,1 --chain
python decode_parts.py --chain  # also --cards 0,1 --chain
python res_block.py --chain  # also --cards 0,1 --chain
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
| 1x1 + bias + SiLU, two passes: the 1x1, then `silu(x + bias.col())` | 3.92e-3 | 4.23e-2 | 6.9 + 9.2 | | |
| 1x1 + bias + SiLU in the product's write, float32 bias field | 3.67e-3 | 3.34e-2 | 9.4 | | |
| 1x1 + bias + SiLU in the product's write, bfloat16 bias field | 3.67e-3 | 3.34e-2 | 8.6 | | |

The border is whatever the input tensor's border tiles hold. Zeros give zero padding
and the edge's copy gives replicate padding, with the same kernels. The convs' 4.4e-3
is the HiFi2 weights' products and the bf16 output.

A bias is a second operand of a product's write: `bias_silu(acc, self.bias)`, with
`silu(a + b.col())`, broadcasts the field's column 0 (each output channel's bias) into
the DST tile after the sums and adds it there. In the write it takes 9.4 us against
16.1 us for the two passes. The second pass reads and writes the 256 KiB output once
more, at 28.5 GB/s. The bias values are bfloat16 values, so the float32 and bfloat16
fields give the same words.

Card 1, 1350 MHz, 2026-11-18, one command a layer.

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

## Group norm on the card, and as one chain (`group_norm.py`)

The same group norm as `norms.py` (2 groups of 32 channels, 16 x 64 pixels), with the
derive on the card as well: one class of six handlers, run as six commands and, with
`--chain`, as one chain of the six calls.

1. `squares`: x * x of each tile row (32 ranks).
2. `sums`: two product loops in one handler, the ones tile by x's tiles and by the
   squares' tiles, each summed in DST over y (one rank a group).
3. `lanes`: the row form's lane sum of both, so every lane holds the group's sum and
   sum of squares.
4. `derive_scale`: `gamma * km.rsqrt(s2 / n - (s1 / n)^2 + eps)`, three streams of one
   tile; `derive_shift`: `beta - s1 / n * scale`. gamma and beta sit in column 0 of
   float32 tiles, so scale and shift come out in column 0.
5. `apply`: x * scale.col() + shift.col(), scale and shift filled into fields.

On card 0 (2026-11-18, at 1a175215): rms 1.82e-3, max 2.46e-2 against numpy, the same
error as `norms.py`'s host derive. As commands: 5.3 + 25.2 + 5.2 + 6.0 + 5.1 + 25.4 =
72.2 us; as one chain 67.9 us, the same values. On cards 0 and 1 (groups across the
edge, tensors where the session puts them): the same values; commands 81.8 + 365.9 +
25.8 + 32.6 + 32.6 + 150.9 = 689.6 us, one chain 662.9 us. The time there is reads
across the link (see "Where a two-card command's time goes").

Three things the existing forms do not take, and what this script writes instead:

- `x.tiles[first : first + H * CT : CT, ...]`, `norms.py`'s rows (conv_shift's layout,
  a group's tile rows CT apart), in a chain: "a rectangle with a step is not
  described; it still runs as a command (run.submit)". Here the image is group-major
  (tile row g H + y), so every rectangle is a plain one. After a conv in the
  conv_shift layout the stats steps run as commands.
- A step that reads the outputs of two earlier calls neither of which waits on the
  other, even marked `km.past`: "core 0 reads out ... tile 0, written by call 2
  (GroupNorm.lanes) (block 0), and nothing it waits for tells it that block is
  complete". This was the derive reading the sum (from the lane sum of x) and the sum of
  squares (from the lane sum of the squares, two calls before). Here both sums are one
  handler's two loops and both lane sums one handler's, so the derive reads one call.
- `t.tiles[g : g + 1, J : 2 * J].write(acc2)`, a rectangle not from column 0, was
  refused ("acc2 is 2 tile(s), 2 page(s); write pages(first, 2)"): the length 2 * J -
  J was not worked out. Slice lengths now fold, and the script writes it so.

## An attention step and an RMS norm (`decode_parts.py`)

An attention step and an RMS norm at small sizes, written from the existing forms.
Attention: 8 heads of 64 values (2 tiles) a core each, 32 queries (one tile row,
positions 96 to 127) over 128 keys (4 tiles), causal. Q, K^T and V are
bfloat16; K is kept as K^T (D x keys), since `.T` is a cast's and a stream does not
take it. One class of four handlers, run as four commands and, with `--chain`, as one
chain of the four calls:

1. `scores`: a product loop of Q's tile row by K^T's tile rows (a block of the 4 key
   tiles), summed in DST over D, written float32.
2. `masked`: `s * SCALE + mask` of two zipped streams, the mask a float32 tile row of 0
   and -inf every head reads.
3. `softmax`: the row form over the 4 tiles, `m = km.reduce_max(row)`, `e =
   km.exp(row - m.col())`, `e * km.reciprocal(km.reduce_sum(e)).col()`, written as
   bfloat16 P.
4. `weigh`: a product loop of P's tiles by V's tile rows, an acc of 2 tiles.

RMS norm: 128 tokens of 2,048 values (64 tiles), bfloat16, 4 cores a tile row each,
`row * km.rsqrt(mean(row^2) + eps).col() * weight.row()`, the weight a `.row()` field of
`norms.py`'s chunk form.

On card 0 (2026-11-18): attention rms 1.70e-3, max 1.05e-2 against numpy; the chain
the same values. Each handler is one command over 8 cores (a head each): scores 6.5,
masked 6.0, softmax 16.6, weigh 6.3, so 35.4 us as commands and 35.2 us as one chain.
RMS norm: rms 6.5e-8, max 1.7e-6, 34.4 us. On cards 0 and 1 (heads across the edge,
tensors where the session puts them): the same values; commands 71.7 + 45.8 + 41.5 +
120.3 = 279.3 us, one chain 240.4 us, RMS norm 193.3 us.

To size an attention block: a head is one core, 32 query rows over 128 keys, so a
command's time is a head's. In query rows a second, a head:

| handler | card 0, us | rows a second | cards 0 and 1, us | rows a second |
|---|---|---|---|---|
| scores | 6.5 | 4.9 M | 71.7 | 0.45 M |
| masked | 6.0 | 5.3 M | 45.8 | 0.70 M |
| softmax | 16.6 | 1.9 M | 41.5 | 0.77 M |
| weigh | 6.3 | 5.1 M | 120.3 | 0.27 M |
| the step, as commands | 35.4 | 0.90 M | 279.3 | 0.11 M |
| the step, one chain | 35.2 | 0.91 M | 240.4 | 0.13 M |

The 8 heads run side by side, so on card 0 the step makes 7.2 M query rows a second
over all of them.

The error is bfloat16 P's. In numpy, with only P rounded to bfloat16 and the rest in
float64, a row of P sums to 1 within 9.4e-4 (rms 2.5e-4). O then has rms 1.63e-3 and
max 1.07e-2, nearly all of the card's error. The row sums are this host model's, not
read from the card: the card's O error matches the model's, so the card's P is the
model's P to its rounding. The script now reports P's row sums (`p_row_sum`) on its
next run.

Nothing here was refused by the forms. One count is spelled as a constant:
`self.weight.fill(km.Stream(w.pages(0, WIDE // 2)))` is refused ("(64 / 2) is not known
to be 32"), so the script fills `CHUNKS`.

## A residual block and the upsample (`res_block.py`)

One block of a UNet-style model, written from the existing forms: conv 3x3, group
norm and SiLU, conv 3x3, the skip added, then the nearest 2x upsample. 64 channels (2
groups of 32), 16 x 64 pixels up to 32 x 128, bfloat16 activations, no bias. Two
classes, run in three runs:

- `Convs`: `copies` (the left and right neighbours by 0/1 products, as
  `conv_shift.py`), `conv`, `skip`, `up_x` and `up_y` (as `resample.py`).
- `Norm`: `group_norm.py`'s steps, with `km.silu` in the apply.
- The runs: the first conv; the norm; then the second conv, the skip and the upsample.

On card 0 (2026-11-18, aiclk 1,350 MHz while the session was open): rms 2.51e-3, max
1.72e-2 against float64 numpy, for the block and after the upsample. The three chains
give the same values. As commands:

| step | us | step | us |
|---|---|---|---|
| copies 1 | 13.3 | shift | 5.3 |
| conv 1 | 20.9 | norm and SiLU | 10.5 |
| squares | 5.2 | copies 2 | 13.3 |
| sums | 25.0 | conv 2 | 21.6 |
| lanes | 5.1 | skip | 5.3 |
| scale | 6.1 | up x, up y | 9.8 + 5.9 |

That is 131.6 us for the block and 15.7 us for the upsample. As a chain each, the
three runs take 37.3, 58.3 and 76.4 us, against 34.2, 57.2 and 55.9 us as commands.

On cards 0 and 1 (half of each group on each card, tensors where the session puts
them; aiclk 1,350 MHz): the same values. The block takes 7,900.8 us and the upsample
725.3 us as commands, with each conv at 2,881 us. The chains take 4,356.8, 744.1 and
8,086.3 us, against 3,520.3, 728.0 and 4,377.8 us as commands.

What the forms did not take, and what the script writes instead:

- One class of all 13 handlers is refused with "ResBlock: more pairs than buffers": its
  streams need more circular buffers than a program has. So there are two classes, and
  the block is three runs, each a chain, not one chain.
- A class's products share one fidelity, so the convs run at HiFi4 beside the copies'
  0/1 products. `conv_shift.py` runs them at HiFi2 in a class of their own.
- `conv_shift.py`'s conv reads 6 streams (three copies, by dy). Here the copies are one
  tensor, z, with rows 3 r + dx, and the weights' tile columns are in (dy, ci, dx)
  order. So a conv is one product loop over 9 x CT tiles.
- These are refused in a chain, though they run as commands:
  - a call that writes three rows of z a rank, with z bordered: "group 0 writes z
    pages 1 to 18 with a gap or an overlap". Here z has no border columns, and a rank's
    three rows are adjacent.
  - the norm's per-row loop over the conv layout's rows, CT apart: "y's rectangles over
    the loop's row make no rectangle together". Here a conv writes its output
    group-major (tile row g H + y) and the norm reads it as `group_norm.py` does.
  - `resample.py`'s up_y, two output rows a rank: "u's rectangles over the loop's k
    make no rectangle together". Here a rank writes one output row.
- `9 * CT` as a slice's length and `block=2 * J` are refused or not known as a count.
  They are module constants (`TAPS`, `UP_J`) instead.
- On two cards, 32 tensors at once fill the tensor table ("the tensor table is full").
  The commands' tensors are freed before the chains' are made.

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
- E2 (`--e2`): the table in row layout, the token's row as 4 items of 64 B, each pushed
  to an item write stream of the output. It builds as an item copy on the movers: mover
  A reads the items into the loop's buffer, mover B writes them from it, and compute
  takes none. It has not been run on the card in this form. Before, as 8 items of 32 B,
  it built as compute's `copy_tile` and `pack_tile` of a whole tile through buffers of
  32 B pages, packed 2 KiB into each 32 B slot and ran WRONG (rms 1.06 of the
  reference); then it was refused at the build. Items of 32 B of a DRAM tensor are now
  refused (Blackhole reads DRAM at 64-B alignment). The same copy is proven on the card
  by [item_gather](../../language/item_gather/README.md).

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
`session.graph.first(1) - n / 2`, so half its ranks are on each card. The tensors have
no placement hint, so each lies wholly on card 0, and the session says so at the
placement ("not in proportion to their ranks"). Card 1's ranks read every page across
the link: "shift on ranks 100..135 moved 360 pages across the link (asks 144, puts
144)". Cards 0 and 1, the same inputs and seeds:

- Every error is equal to one card's, to every printed digit. The shifts and the
  selection are exact, and layer norm is 5.4e-8.
- Every command takes 4 to 92 times its time on one card: shift 569.3 us against
  11.6, conv 2,944.8 against 32.1, depthwise 1,979.1 against 28.4, layer norm 284.6
  against 67.8, group norm's apply 150.3 against 24.6. How much of that the reads
  across the link take is not split out here.

## Where a two-card command's time goes

The same commands on cards 0 and 1 with `--place`: `inside` puts the group and every
tensor on island 0; `edge` is the run above; `own` straddles the edge and places each
tensor on its command's group (`split=0, group=<the group>`), with the small weights
and shift matrices `copy=True`. The pages across are what `run.crossings()` counted
around the command. Values are equal to one card's in every placement.

| Command | One card | inside | edge (pages across) | own (pages across) |
|---|---|---|---|---|
| shift (36 ranks) | 11.6 | 12.3 | 569.6 (360) | 10.7 (0) |
| conv (32 ranks) | 32.1 | 38.7 | 2,938.3 (1,184) | 137.2 (48) |
| layer norm (4 ranks) | 67.8 | 67.6 | 284.9 (576) | 67.1 (0) |
| squares (36 ranks) | 6.6 | 6.4 | 170.9 (108) | 6.1 (0) |
| column sums (2 ranks) | 14.2 | 20.1 | 184.6 (49) | 64.9 (32) |
| apply (2 ranks) | 24.6 | 24.6 | 151.8 (50) | 89.9 (48) |

A group that crosses the edge costs nothing measurable by itself (`own` against
`inside` where nothing crosses). The pages read across the link are the cost. The conv,
the column sums and the apply still cross in `own`: each reads a tensor placed on
another command's group (the conv on 32 ranks reads the rows split over the shift's 36;
the column sums and the apply on 2 ranks read the image split over the squares' 36).

## A halo brought over first (`halo_exchange.py`)

The conv of `conv_shift.py` on cards 0 and 1 with its halo rows brought over in one
step. Its tensors hold 4 more tile rows: island 0 holds rows 0 to 19 and island 1
rows 16 to 35. One command on a rank of each island reads the other island's two halo
rows in one `km::read_pages` and writes them into its own copy. Then the shifts (40
ranks) and the conv (32 ranks, island 1's rows 4 further on) read only their own
island. Cards 0 and 1, 2026-11-18, us:

| Step | us | Pages across |
|---|---|---|
| the halo step (2 ranks) | 11.6 | 16 (2 asks) |
| the shifts | 10.9 | 0 |
| the conv | 42.6 | 0 |

The conv's values equal `conv_shift.py`'s (rms 4.40e-3, the shift exact). The halo
and the conv take 54.2 us, against 137.2 us for the conv placed on its group with
the halo read a page an ask, and 38.7 us with everything on island 0.

This is the way to handle a halo across the cards' edge: bring it over in one step,
then read locally. As one chain of the three calls (`--chain`) the result is right
(rms 1.7e-3, the conv at HiFi4) but the chain took 837.2 us, against 65.1 us for the
three commands. The cause is the conv's covering range: its reads of `zl` and `zr` are
not `base + step x g`, so each of its 32 ranks waits for every block of the shift, and
each of those marks from the other card crosses the link one after another, about 1.2 us
each ("A covering range on islands" in
[classes](../../../docs/reference/classes.md)). To be one chain: one fidelity for the
class's products (the conv at HiFi4), y without halo rows (its rows written at a pitch),
W's columns in (dx, dy, ci) order so that each stream's taps form one rectangle, the
halo body's write stated as `writes=[(x, target, pages)]`, and `km.past(x)` where the
conv reads x.

## A sum as a product's operand (`sum_operand.py`)

(x + r) @ W, W a field of 4 x 2 tiles, 120 cores, 2 tile rows each, card 0,
2026-11-18. The direct form `acc += self.project(xb + rb, self.w)` is refused:
"`xb + rb` is not the loop's tile or `self.w` is not a field; write acc +=
self.project(block, self.w)".

| Way round | us | rms | max |
|---|---|---|---|
| two products into one sum, x @ W + r @ W | 21.8 | 3.1e-4 | 1.7e-3 |
| x + r written as bf16 by a pass, then (x + r) @ W | 20.1 + 15.1 | 1.8e-3 | 1.0e-2 |

Errors are against float64 (the second run: two products 20.9 us, the rest equal).
The second form's error is the bf16 store of x + r, which the script reads back: of
983,040 datums, the 264,144 whose nearest and truncated bf16 differ all read the
nearest, and 104,722 read neither, which is the count of ties that ties to even
rounds down: the card's pack rounds to nearest, ties away from zero
([formats](../../../docs/reference/formats.md#unpack-and-pack)). Against
numpy's ties to even the product is 2.2e-3 from the card's.

## Not proven

- E2 above. It is not run unless `--e2` is given, and has not been run on the card as
  the movers' copy.
