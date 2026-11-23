# Transport: a hierarchical light-transport matrix applied on one card

One apply `y = M x` of a light-transport matrix stored as a hierarchical (block
low-rank) matrix, on all 120 cores of one Blackhole p150a, as ONE command of one class.
`x` is every element's outgoing light (three colours) and `y` the light each element
receives. Each block of `M` is either dense or a low-rank pair. The result is checked
against the host's emulation of the same numbers and against the set's own float32
apply.

## The files it reads (the host step's layout)

A set is a folder written by the host step that builds the matrix. Its `README.md`
gives the exact layout:

- `blocks.npz`: each block's receiver rows `row0:row1` and source columns `col0:col1`
  (elements, multiples of 32), `kind` (0 dense, 1 low rank), `rank_padded`, and
  `offset_a` / `offset_b` into `weights.f32`.
- `weights.f32`: every block stored TRANSPOSED for `y_row = x_row @ W`. A dense block
  is `n x m` (sources by receivers). A low-rank block is `V` (`n x kp`), then `U^T`
  (`kp x m`).
- `x.npy`, `y_exact.npy`, `y_float.npy`, `y_q8.npy` (`[N, 3]` float32), `perm.npy`.

## What the host does (`plan.py`)

- **Dealing.** The receiver tiles go to the cores in contiguous ranges, with the
  largest cost as small as possible (a binary search on the cap). A range's cost is the
  weight tiles its slices read: a dense slice reads `K n`, a low-rank one its whole `V`
  (`K R`) and its `U^T` columns (`R n`). For the 64-leaf set this is exactly the host
  step's `cores.npz`.
- **w.** Every block's tiles, one block after another, in bfp8_b (1,088 B a tile),
  each tile stored transposed (`transposed_tiles`, the weight of the turned product).
  Within a block the tiles go column by column, so any core's slice is one run of
  pages.
- **Tables.** Each core gets its entries, one per block slice, in the order of the
  block's first source tile. An entry is 8 words: `src, k` (x's tiles), `w, a` (the
  first stream: a dense slice's columns, or `V`), `u, b` (the slice's `U^T` columns),
  `r` (the first own receiver) and `rank` (R tiles; 0 for dense). They are written to
  4 KiB pages of 127 entries, `pages` pages a core. Words 0 to 2 of a page are its
  count and the core's range.
- **Emulation.** `emulate` computes `y` from the decoded bfp8_b tiles, with x and t
  rounded to bfloat16, every product exact and float64 sums.

## The handler (`transport.py`, `transport.hpp`)

`Transport.apply(x, w, y, tables, profile, pages)` is one handler, submitted once to
cores 0 to 119. Each core:

1. reads its table pages (`entries`, a class line). For each entry:
   - x's tiles `src : src + k` arrive as a held stream, one block;
   - the first stream of weight pages is read, a block of 8 tiles at a time. A dense
     slice's columns are summed into DST, K tiles each, and each column is packed onto
     the own sums tile `r + j` with L1 accumulation. For a low-rank slice, V's columns
     give `t = x V`, packed in bfloat16 into an own field;
   - the `U^T` stream: R tiles a column, `t U^T` packed onto the sums the same way;
2. writes its own sums to y's pages `first .. first + own`, then its stamps.

Lines that are `km.cxx` (`transport.hpp`), and why:

- **The products and the sums.** Adding a product's tile into a field of sums at a
  tile offset that an entry chooses is no class line, so the body calls the library's
  `matmul_block` and `pack_tile<true>` with L1 accumulation, as the tile compiler's
  own output for a field of sums does.
- **t and its token.** t's pack into an own bf16 field, and the token from pack to
  unpack before the `U^T` products read it.
- **The write.** Copy, transpose back, pack.
- **The stamps.** Unpack's words go to pack by mailbox.

**Operand placement.** Every product is the turned one: the weight is in SrcB, so it
keeps its 7 bits at HiFi2, and the vector (x or t) is in SrcA. x's tile is read
transposed, so DST holds `(x W)^T`. The sums stay transposed and are turned back once,
at the write. t is packed as `t^T` and read untransposed. Products run at HiFi2 with
float32 sums.

## Running

```
python projects/transport/run.py <set folder> [--cache <folder>] [--calls 32] [--again 2]
```

`--cache` keeps w's encoded bytes and `--reverse` deals the ranges to the ranks in
reverse (a diagnostic). The script prints one JSON line with:

- the sizes;
- the check against the emulation, `y_q8` and `y_float`: relative error of the
  luminance over the elements whose exact luminance is above 1e-4 (median, 99th
  percentile, most), and the ratio of sums;
- the time over `--calls` calls;
- every core's stamps, with fits of a core's span;
- `--again` applies with y fed back as x (through the host), each checked against the
  emulation and the float32 apply of the same input.

## Measured

Card 1, 2026-11-23, 20:56 to 20:57, 1,350 MHz. Both sets have 120 cores and 1,225
receiver tiles.

| | leaves of 64 | leaves of 256 |
|---|---|---|
| blocks (low rank / dense) | 38,374 / 37,926 | 8,564 / 13,072 |
| weight tiles (bfp8_b) | 404,852 (440.5 MB) | 463,565 (504.4 MB) |
| entries a core, most / mean | 1,048 / 708 | 381 / 257 |
| weight tiles a core, most / mean | 4,610 / 4,340 | 5,136 / 4,878 |
| x tiles read a core, most / mean (bf16, 2,048 B) | 2,642 / 2,149 | 2,319 / 1,758 |
| check against the emulation: median / p99 / most | 3.5e-5 / 6.9e-4 / 1.3e-2 | 1.3e-5 / 3.9e-4 / 5.5e-3 |
| check against y_float: median / p99 / sum ratio | 1.451e-3 / 4.05e-2 / 0.99966 | 1.208e-3 / 3.44e-2 / 0.99969 |
| (the emulation against y_float) | 1.454e-3 / 4.05e-2 / 0.99965 | 1.213e-3 / 3.44e-2 / 0.99968 |
| a command on the card, median / least (32 calls) | 3,099.3 / 3,083.2 us | 2,754.1 / 2,748.3 us |
| host wall a call, median / least | 3,310.4 / 3,285.0 us | 2,960.2 / 2,950.7 us |
| weight bytes over the command | 142 GB/s (29 % of 495.6) | 183 GB/s (37 %) |
| weight and x bytes over the command | 313 GB/s | 340 GB/s |
| slowest core: span, unpack busy, waits | 3,101 / 360 / 2,741 us | 2,749 / 277 / 2,471 us |
| fit of a core's span, a term per grid row (20:49 to 20:52, two runs each): a a tile, b an entry | 0.145 us, 1.90 us (13 tiles) | 0.112 us, 2.23 us (20 tiles) |

The emulation reproduces `y_float.npy` exactly from the float numbers (max 1.8e-7)
before quantisation. `y_q8.npy` is about 1.1e-3 to 1.5e-3 (median) away from both the
card and this emulation. That gap comes from the stored layout, not from lost bits:
with the tiles transposed, a bfp8_b exponent is shared by 16 sources of one receiver,
where `y_q8` shares it among 16 receivers of one source. Three applies in a row, with
y fed back as x, agree with the emulation of the same input to a median of 2.1e-5 to
3.4e-5 on both sets.

What limits the time:

- **Compute is not the limit.** It is busy 12 to 13 % of a core's span.
- **Entries.** Each entry costs about 2 us, about two DRAM round trips in series: the
  held x block is read only after the last entry's block is freed, then the first
  weight block.
- **x.** x is read once per entry, so it adds 0.9 to 1.2 times the weight bytes.
- **The core's place in the grid.** The same work runs 2 to 2.5 times slower on the
  first grid rows than on the last. The run with the ranges reversed keeps this
  gradient with the position, not with the work.

On the 64-leaf set the slowest core is the one with the most entries. On the 256-leaf
set it is a core on a slow row.

## Limits

- A core owns at most `OWN` receiver tiles (16 here, at most 32: pack keeps one bit
  for each).
- An entry's x range is at most `KMAX` tiles, one held block of a pair (77 tiles,
  315 KB of L1).
- R is at most `RANK` tiles.
- Every scale must be 1 (refused otherwise).
- The table pages a core are a command word.
