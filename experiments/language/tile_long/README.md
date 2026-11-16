# Tile long rows: norms over 1,024 to 4,096 values

Question: do the RMS norm and the layer norm, written as tile functions on rows longer
than their stream's block (rows of 32, 64 and 128 tiles in blocks of 8), compute what
float64 computes, and what does a row cost?

`tile_long.py` makes three classes, one a width: `Long1024` (RMS norm and layer norm),
`Long2048` (RMS norm) and `Long4096` (both). Each core normalizes one row (32 rows of
values) with the weight (and bias) fields, a float32 tile a chunk of 2 tiles. The loop
is the wide row's, with a block shorter than the row:

```python
for row in km.Stream(x.pages(first, 128), block=self.BLOCK, row=128):
    results.push(rms_4096(row, self.weight))
```

The range is read once a level of sums (RMS norm 1, layer norm 2), then once for the
writes. A sum pass holds DST for the row, adds each block's chunks into the sums' slots
and packs the per-row values' and the sums' tiles to a page pair of the row's; the next
pass unpacks them again. The last pass writes each chunk with the row's values and the
fields' chunk.

A fourth class, `Split2048`, runs the last step of the hand decode norm (a row of 2,048
split over 16 cores, 4 tiles each): `x * rsqrt(s.col() / 2048 + eps) * w.row()`, `s` a
field holding each row's sum of squares in column 0 (the host's, where a reduce over the
cores would leave it), read by `.col()`; `w` a stream of the weight row's tiles, read by
`.row()`, in blocks of 2 so both halves of each buffer pair are read.

Run: `kmr prebuild tile_long.py card`, then `python tile_long.py card` on a card
(`made_up` checks the host side without a card).

## Results

2026-11-16, card 0. Errors against float64, microseconds a command (three commands):
on 12 cores (a row each), and on one core (one row).

| Values a row | Function | rms rel. | largest error / rms | largest rel. | us, 12 cores | us, one row |
|---:|---|---:|---:|---:|---:|---:|
| 1,024 | RMS norm | 5.3e-8 | 1.3e-6 | 2.2e-7 | 25.6 to 26.6 | 23.1 |
| 1,024 | layer norm | 5.0e-8 | 1.3e-6 | 5.3e-3 | 38.9 to 39.3 | 35.6 to 35.8 |
| 2,048 | RMS norm | 6.3e-8 | 1.7e-6 | 3.1e-7 | 45.1 to 46.5 | 41.6 |
| 4,096 | RMS norm | 7.4e-8 | 2.4e-6 | 3.7e-7 | 84.9 to 85.9 | 78.9 to 79.1 |
| 4,096 | layer norm | 6.4e-8 | 1.3e-6 | 2.7e-2 | 137.6 | 126.6 to 126.7 |
| 2,048 | split apply, 16 cores | 4.5e-8 | 7.1e-7 | 1.5e-7 | 8.8 to 9.9 | |

The layer norm's largest relative error is at values near zero after the mean is taken
away (cancellation); its error over the rms stays at 1.3e-6.

- A row costs about 20 us a 1,024 values for the RMS norm on one core (two reads of the
  row, one write), and about 31 us for the layer norm (three reads).
- The hand decode norm took 12.6 us at 2,048 in its kernels (projects/decode README,
  "The norms inside the products"): one tile row on 16 cores, each reading the whole row
  for its sums of squares, then writing its 4 tiles. One core here takes 41.6 us for the
  row. The split apply alone takes 8.8 to 9.9 us on 16 cores; the hand's other steps
  (each core's partial sums, the sums meeting at every core) are not lines a class
  writes today: a reduce leaves the group's sum at its root only.
- The build's L1 for a core: 540,672 B at 1,024 values (layer norm), 933,888 B at 4,096
  (two float32 fields 524,288 B, the read pair 262,144 B, the write pair 131,072 B).
- The hand decode norm is no longer a command in projects/decode (folded into the
  products' roots), so it did not run in this job.
