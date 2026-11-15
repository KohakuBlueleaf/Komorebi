# The linear layer, as a class

Question: can the dense project's linear layer, Y = X W, be written as a class
([Programs written as classes](../../docs/reference/classes.md#products)), its product
a tile function `return x @ w` summed over K, and its SwiGLU gate a second tile
function after it; and what does it cost beside the project's hand kernels?

`linear_class.py` makes the class `Linear`: a field `weight`, W's columns a core holds
(K rows of 2 tiles, 512 KiB), and two handlers. `apply(x, w, y, rows)` fills the field
from W, then for each of X's `rows` row tiles sums `self.project(block, self.weight)`
over the blocks of the row (the library's `matmul_block`, fp32 in DST) and writes the 2
tiles of Y. `gate(a, b, out, tiles)` is `silu(a) * b` over two streams, as the
[22 gate](../../examples/22_gate/as_class/README.md). Its files are under
`build/linear/` beside the script.

```bash
python linear_class.py --rows 1 --check exact swiglu    # 32 x 4096 x 4096
python linear_class.py --rows 32 --check exact swiglu   # 1024 x 4096 x 4096
```

The form is the page's: each of 64 cores holds its 2 columns of W, and every row of X
streams past them (N = 64 x 64 = 4096). The host lays W out by core, each core's K x 2
tiles one after another, and reads Y back by core: a class reads a page range of a
tensor, and a slice of its columns is not built. Neither the dense project's K split
(its reduce of partial sums is not a line of a class) nor its broadcast of W down
columns of cores is written.

The checks: `exact`, x in {-1, 0, 1} and w in {-1, 0, 0, 0, 1}, every sum an integer of
at most 168, exact in bfloat16, so Y must equal the host's product; `swiglu`, normal
bfloat16 values (W / 64), Y against float64 of the same inputs by relative RMS, as the
dense project judges, then a second product and the gate, against the gate of the
card's two products and against float64's whole SwiGLU.

## Results

Card 1, a Blackhole p150a, 64 cores, seed 1, one run a shape (one exact command, four
timed commands of the random values). Kernel text is the packed text of the five CPUs;
the hand kernels' by a build on a mock device, at the same shapes.

| M x K x N | Exact | Relative RMS: product / gate / SwiGLU | apply, us a command (median) | gate, us | Kernel text | The dense project |
|---|---|---|---|---|---|---|
| 32 x 4096 x 4096 | 0 wrong | 1.69e-3 / 1.66e-3 / 3.12e-3 | 124.8 | 7.2 | 47,664 B | K split, 92.4 us, 61,584 B |
| 1024 x 4096 x 4096 | 0 wrong | 1.69e-3 / 1.66e-3 / 3.12e-3 | 1,212.7 | 92.6 | 47,664 B | grid form, 382.4 us, 69,296 B |

The build prints the product's cost: 2 tile products a stream tile, 128 MVMUL words at
HiFi4, 135.0 to 770.0 cycles a stream tile (the tile compiler's `matrix_cost`, from
blocks rotating to a block alone; the library's order is not known to it; the unpack
and the pack not priced).

Where the time goes is the form's reads. Each core reads all of X. At M = 32 the
command reads W once (33.6 MB) and X 64 times (16.8 MB): 50.3 MB, 103 us at the card's
490 GB/s (the dense project's ceilings), measured 403 GB/s; the K split reads W alone,
33.6 MB in 92.4 us, and the 32.4 us between are X's 63 extra reads (34 us at the
ceiling). At M = 1024 it reads 537 MB in 1,212.7 us, 443 GB/s; the grid form reads X
once and casts W down columns of cores, 382.4 us. The matrix unit is not what bounds
it: 400 cycles a stream tile at 1,350 MHz, inside the printed range. The class's error
is smaller than the dense project's (3.94e-3): HiFi4 against its HiFi2.

What the class lacks for the dense project's forms: a slice of a tensor's columns
(W and Y laid out by core here); the sum of partial products across cores (the K
split); a field cast from one core to a column of cores (the grid form); a field held
from one command to the next (W is read again each command); the fidelity as a line
(HiFi4 always); a product and a gate in one loop (the gate is a second command, its
operands through DRAM).
