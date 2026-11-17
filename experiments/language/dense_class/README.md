# The linear layer, as a class

Question: can the dense project's linear layer, Y = X W, be written as a class in the
dense project's own two plans
([Programs written as classes](../../../docs/reference/classes.md#products)), its product
a tile function `return x @ w`, its SwiGLU gate a second tile function after it; and
what does it cost beside the project's hand kernels, term by term?

`linear_class.py` makes the class `Linear`: two fields and three handlers. The plan is
picked by the shape, as the dense project's plan picker does:

- `split(x, w, y)`, for one row tile of X, is the K split in the dense project's plan
  for that shape (14,1,16): 112 cores in groups of 14 consecutive ranks, each group 16
  columns of Y, each core a fourteenth of K (9 or 10 tiles). X's tiles and W's rows of
  the group's columns stream in step (`zip` of two streams, 2 K tiles a block), and
  `self.sums += self.project(xb, wb)` adds each pair's product into the float32 field
  `sums` of 16 tiles; `km.reduce(self.sums, form="tree", group=14)` adds the group's
  fields, and the group's first rank writes Y's 16 tiles.
- `grid(x, w, y)`, for more row tiles, is the grid form on the hand's 12 x 8 cores:
  core (x, y) makes Y's block of M / 256 x 11 tiles in the float32 field `part`, the
  last column of cores 7 (`left:min(left + OW, N_TILES)`, a rectangle whose bounds are
  words); for each chunk of 2 K tiles, X's block is cast along the row and W's down the
  column, and `self.part += self.project(xs, ws)` adds their product into the field.
- `gate(a, b, out, tiles)` is `silu(a) * b` over two streams, as the
  [22 gate](../../../examples/22_gate/as_class/README.md).

The product is at HiFi2, as the dense project's rows (`@km.tile_function(fidelity=
"HiFi2")`). The host lays out nothing: X, W and Y are the plain tensors, read as
rectangles of tiles. One program serves both shapes (the grid's block of 1024 rows), so
one load runs them; its files are under `build/linear-<hash>/` beside the script.

```bash
python linear_class.py --rows 1 32 --check exact swiglu  # the K split, then the grid
python linear_class.py --rows 1 32 --check exact --k-tiles 128 64  # two programs
python linear_class.py --cards 0,1 --rows 1 --check exact random --block 8  # two cards
python linear_parts.py --repeat 3          # the first version's plans' parts, alone
```

The sizes are constants of the class (`K_TILES = km.constant(128)`, `W_BLOCK = CHUNK *
WIDTH` derived; [Constants](../../../docs/reference/classes.md#constants)), so
`--k-tiles 128 64` makes two programs of `Linear`, `Linear(..., K_TILES=64)` a build of
its own, loaded one after the other in one session, each command named through its
program (`program.split`). On card 0, `exact` at both: K_TILES 128, split 0 wrong
(94.96 µs), grid 0 wrong (360.52 µs); K_TILES 64 (K = 2048), split 0 wrong (58.68 µs),
grid 0 wrong (197.82 µs); kernel text 90,624 B and 90,608 B.

The checks: `exact`, x in {-1, 0, 1} and w in {-1, 0, 0, 0, 1}, every sum an integer of
at most 256, exact in bfloat16, so Y must equal the host's product; `swiglu`, normal
bfloat16 values (W / 64), Y against float64 of the same inputs by relative RMS, as the
dense project judges, then a second product and the gate, against the gate of the
card's two products and against float64's whole SwiGLU.

## Results

Card 1, a Blackhole p150a, seed 1, one load for both shapes (a shape: one exact
command, three timed commands of the random values, the median). Kernel text is the
packed text of the five CPUs, in a kernel buffer of 89 KiB (the dense project's linear
kernels take 88); the hand kernels' by a build on a mock device, at the same shapes.

| M x K x N | Plan | Exact | Relative RMS: product / gate / SwiGLU | Class, us a command | gate, us | Kernel text | The dense project | The first version |
|---|---|---|---|---|---|---|---|---|
| 32 x 4096 x 4096 | K split, 112 cores | 0 wrong | 4.38e-3 / 1.64e-3 / 8.03e-3 | 94.2 | 7.3 | 90,192 B | 92.4 us (112 cores), 61,584 B | 111.9 us (96 cores) |
| 1024 x 4096 x 4096 | grid form, 12 x 8 | 0 wrong | 4.38e-3 / 1.66e-3 / 8.03e-3 | 358.6 | 92.7 | 90,192 B | 382.4 us (12 x 8), 69,296 B | 434.3 us (8 x 8) |

The error is the dense project's at HiFi2 (its 3.94e-3). The K split alone, with the
grid's chunks of 4 and a kernel buffer of 88 KiB (an earlier build of the same split):
93.6 us.

On two cards (cards 0 and 1, four links; `--cards 0,1 --block 8`), the output dimension
sharded and the activation a copy: the same K split over 112 flat ranks centred on
island 1's first rank, four groups of 14 on each card, W and Y split by tile columns so
each island holds its groups' 64 columns (16 MiB of W an island, 320 KiB read a core,
as on one card), all at 32 x 4096 x 4096. On two cards the script builds the class
with CHUNK 1 (`--chunk`; one card keeps 2): an island's L1 left for buffers no longer
holds CHUNK 2's read pair (1,165,312 B against 1,126,064 B), CHUNK 1's takes 903,168 B.
At 58.6 us, 32 MiB of W is read at 573 GB/s over both cards (18.3 TFLOP/s):

| | Exact | Relative RMS | us a command (median of 3) | Each island's part, us |
|---|---|---|---|---|
| The class, one card | 0 wrong | 4.38e-3 | 94.2 | |
| The class, two cards (CHUNK 1, 2026-11-17) | 0 wrong | 4.38e-3 | 58.6 | 56.7 to 57.5 and 58.5 to 59.2 |
| The class, two cards (CHUNK 2, 2026-11-15) | 0 wrong | 4.38e-3 | 65.3 (65.1) | 61.9 to 63.5 and 64.3 to 65.4 |
| The dense project's hand program, two cards, output form | | | 53.7 (one call) | |

The command's time is its slower part's. Island 1's part ends 3.2 to 3.4 us after
island 0's every time: the host stores island 0's part first, so this is the skew
between the two parts' starts, not the link: by placement no page of W, X or Y crosses
(the links' counts were not read). The
two cards' times are not split into terms here (no part run). Taken at face value,
from 94.2 us for 32 MiB of W on one card and 61.6 us for 16 MiB on one island, about
29 us does not scale with W's bytes (the tree's reduce, Y's write, the command's
start) and the rest is W's read at about 2 us a MiB. The hand program's 53.7 us is
11.4 us under the class's.

## Where the time goes

The first version of the class lost to the hand program by three terms, each a line
the class could not write; each is a line now
([Products](../../../docs/reference/classes.md#products)).
`linear_parts.py` runs the first version's plans' parts as commands of their own, at
the same shapes: `fill`, its K split's W read alone into a field; `products`, its X
stream and products on that field, each core's sums written to pages of its own (no
reduce); `casts`, its grid form's two casts on 8 x 8 cores, the blocks received and
freed (no product). The ceilings are the card's DRAM, 495.6 GB/s, and the matrix
unit's word a cycle at 1,350 MHz.

| K split, 32 x 4096 x 4096 | First version, us | Now | Whose |
|---|---|---|---|
| W, 33.55 MB | 87.5 (383 GB/s) into a field before the first product, then X's stream and the products: 24.4 | read under the products, a block used as it arrives: 94.2 for the whole command, W at 356 GB/s over it | the lowering's: gone (a product over two streams) |
| X, 8.45 MB read by groups of 4 columns (6.2 MB more than the hand's, 12.5 us at the ceiling) | in the 24.4 | 2.1 MB, read once a group of 16 columns, as the hand's | the form's: gone (a reduce of a field of sums; 4 columns were the most a sum in DST takes with the reduce's tile) |
| left over the hand's | | 1.8 us (2 %) | not split by term (no part run): W in blocks of 2 K rows (64 KiB), three of four on mover A (the reads' pattern), where the hand reads chunks of 5 (160 KiB) on both movers in turn; after the last block its products, the tree's 4 rounds of 16 float32 tiles, Y's write, the hand's in form |

| Grid form, 1024 x 4096 x 4096 | First version, us | Now | Whose |
|---|---|---|---|
| W's casts, 33.55 MB | 402.6 by 8 column senders (10.4 GB/s a sender) | 12 senders: at that rate a sender, 269 | the form's: gone (12 columns of 11, the last 7: a rectangle whose bounds are words) |
| the rest: X's casts, the products, the last pair, Y's write (8.4 MB) | 31.7 after the casts | 358.6 in all, 23.8 under the hand's | not split by term (no part run): K chunks of 2, 64 casts a column, where the hand casts 8 chunks of 16 |

The grid's chunks are 2 K tiles, not the first version's 4: at 4, the casts' slots,
the K split's buffers (W's stream pair 512 KiB, the tree's slots 256 KiB) and the
runtime's own (its control block, slots and transfer buffer, 58.5 KB) are over a
core's L1, one program holding both plans.
