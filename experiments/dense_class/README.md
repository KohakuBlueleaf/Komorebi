# The linear layer, as a class

Question: can the dense project's linear layer, Y = X W, be written as a class in the
dense project's own two plans
([Programs written as classes](../../docs/reference/classes.md#products)), its product
a tile function `return x @ w`, its SwiGLU gate a second tile function after it; and
what does it cost beside the project's hand kernels, term by term?

`linear_class.py` makes the class `Linear`: two fields and three handlers. The plan is
picked by the shape, as the dense project's plan picker does:

- `split(x, w, y)`, for one row tile of X, is the K split: 96 cores in groups of 3
  consecutive ranks, each group 4 columns of Y, each core a third of K. The field
  `weight` is the core's K rows of the group's columns (`w.tiles[k0:k1, c0:c1]`); X's
  row of tiles streams past it, the 4 sums held in DST; `km.reduce(acc, form="tree",
  group=3)` adds the group's three, and the group's first rank writes Y's 4 tiles.
- `grid(x, w, y)`, for more row tiles, is the grid form on 8 x 8 cores: core (x, y)
  makes Y's block of M / 256 x 16 tiles in the float32 field `part`; for each chunk of
  4 K tiles, X's block is cast along the row and W's down the column, and
  `self.part += self.project(xs, ws)` adds their product into the field in L1.
- `gate(a, b, out, tiles)` is `silu(a) * b` over two streams, as the
  [22 gate](../../examples/22_gate/as_class/README.md).

The product is at HiFi2, as the dense project's rows (`@km.tile_function(fidelity=
"HiFi2")`). The host lays out nothing: X, W and Y are the plain tensors, read as
rectangles of tiles. Its files are under `build/linear/` beside the script.

```bash
python linear_class.py --rows 1 --check exact swiglu    # 32 x 4096 x 4096, the K split
python linear_class.py --rows 32 --check exact swiglu   # 1024 x 4096 x 4096, the grid form
python linear_parts.py --repeat 3                       # the plans' parts, alone
```

The checks: `exact`, x in {-1, 0, 1} and w in {-1, 0, 0, 0, 1}, every sum an integer of
at most 256, exact in bfloat16, so Y must equal the host's product; `swiglu`, normal
bfloat16 values (W / 64), Y against float64 of the same inputs by relative RMS, as the
dense project judges, then a second product and the gate, against the gate of the
card's two products and against float64's whole SwiGLU.

## Results

Card 1, a Blackhole p150a, seed 1, one load a shape (one exact command, three timed
commands of the random values, the median). Kernel text is the packed text of the five
CPUs, in a kernel buffer of 88 KiB, as the dense project's linear kernels; the hand
kernels' by a build on a mock device, at the same shapes.

| M x K x N | Plan | Exact | Relative RMS: product / gate / SwiGLU | Class, us a command | gate, us | Kernel text | The dense project |
|---|---|---|---|---|---|---|---|
| 32 x 4096 x 4096 | K split, 96 cores | 0 wrong | 4.38e-3 / 1.64e-3 / 8.03e-3 | 111.9 | 7.3 | 88,032 B | 92.4 us (112 cores), 61,584 B |
| 1024 x 4096 x 4096 | grid form, 8 x 8 | 0 wrong | 4.38e-3 / 1.66e-3 / 8.03e-3 | 434.3 | 95.9 | 88,096 B | 382.4 us (8 x 12), 69,296 B |

The error is the dense project's at HiFi2 (its 3.94e-3).

## Where the time goes

`linear_parts.py` runs the plans' parts as commands of their own, at the same shapes:
`fill`, the K split's W read alone into the field; `products`, its X stream and
products on that field, each core's sums written to pages of its own (no reduce);
`casts`, the grid form's two casts, the blocks received and freed (no product). The
ceilings are the card's DRAM, 495.6 GB/s, and the matrix unit's word a cycle at
1,350 MHz.

| K split, 32 x 4096 x 4096 | us | At the ceilings | Whose |
|---|---|---|---|
| W into the fields, 33.55 MB | 87.5 (383 GB/s) | 67.7 | the read's rate; the hand reads W at 363 GB/s |
| X's stream and the products, after the fill: X 8.45 MB, 172 tile products a core | 24.4 | X 17.0, products 4.1 | the lowering's: the field is filled whole before the first product, where the hand reads X and multiplies under W's read; the form's: 6.2 MB of X more than the hand's (12.5 us at the ceiling), read by each group of 4 columns, the most a sum in DST takes with the reduce's tile |
| the tree in groups of 3, the repack to bfloat16, Y's write | 0.1 (111.9 less the two) | | |

The K split's 19.5 us over the hand's is the second row: the fill alone is faster than
the hand's whole command.

| Grid form, 1024 x 4096 x 4096 | us | At the ceilings | Whose |
|---|---|---|---|
| the two casts: W 33.55 MB by 8 column senders, X 8.39 MB by 8 row senders | 402.6 (W 83 GB/s, 10.4 GB/s a sender) | 84.5 for both from DRAM | the form's: 8 senders of W at the hand's rate a sender (its 12 at 10.1, 121 GB/s, would cast W in 277 us) |
| the products not under the casts, Y's write (8.4 MB) | 31.7 (434.3 less the casts) | products 194 (8,192 tile products a core, under the casts); Y 17.0 | the last chunk's products and the write after them |

The grid form's casts alone take longer than the hand's whole command: the class has 8
columns of cores to the hand's 12. The hand's 12 columns of 11 tiles leave its last
column 7 wide, and a rectangle's columns are a constant, so the class takes 8 columns
of 16.
