# Tile split norm: the decode's norm of one row over 16 cores

Question: can a class write the decode's RMS norm of one row of 2,048 values split
over 16 cores, at the hand's speed, and do bf16 weight and bias fields work in a wide
row?

`tile_split.py` makes two classes. `SplitNorm.norm`, on 16 cores, is the hand's form:
each core reads the whole row for its sums of squares, then writes its own 4 tiles.

```python
@km.tile_function
def square_sums(row):
    return km.reduce_sum(row * row, exact=True).col()


for row in km.Stream(x.pages(0, 64), block=8, row=64):
    self.part.write(square_sums(row))
first = self.rank * 4
results = km.Stream(out.pages(first, 4), write=True, block=2)
for xt, wt in zip(
    km.Stream(x.pages(first, 4), block=2), km.Stream(w.pages(first, 4), block=2)
):
    results.push(rms_apply(xt, wt, self.part))
```

`square_sums` returns a per-row value alone: its sum parts run over the row's blocks,
then its part `value` puts each row's sum in every column of one tile, packed into the
1-tile field `self.part`. The next loop reads it by `s.col()` (`rms_apply`). x, w and
out are bf16, as the decode's are; w's row 0 holds the weight (uniform in [0.5, 1.5)).
No reduce: every core sums the whole row itself, as the hand does.

`SplitNormAll.norm` is the same norm with the sums all-reduced: each core sums the
squares of its own 4 tiles, then one line leaves the row's sum at every core of the
group, in the same command:

```python
for row in km.Stream(x.pages(first, 4), block=4, row=4):
    self.part.write(square_sums(row))
km.all_reduce(self.part)
```

`HalfFields.layer`, on 12 cores (a row each), is the layer norm over 4,096 values with
weight and bias fields of bf16 tiles: each field tile is unpacked in its own format
setup, then the per-row values' page's again.

Run: `kmr prebuild tile_split.py card`, then `python tile_split.py card` on a card. `card_all`
runs the two split norms side by side (`results/tile_split_all.json`).

## Results

2026-11-16, card 0, AICLK 800 MHz read idle (the busy clock was not read). Against float64 of the same bf16 inputs.

| Case | rms rel. | largest error / rms | largest rel. | us a command |
|---|---:|---:|---:|---:|
| split RMS norm, 2,048, bf16 out, 16 cores | 1.7e-3 | 1.5e-2 | 3.9e-3 | 15.7 to 16.6 |
| layer norm, 4,096, bf16 fields, 12 cores | 6.6e-8 | 2.6e-6 | 7.0e-2 | 125.6 to 126.5 |

`card_all`, 2026-11-16, card 0, AICLK 800 MHz read idle (the busy clock was not read):

| Case | rms rel. | largest error / rms | largest rel. | us a command |
|---|---:|---:|---:|---:|
| split RMS norm, every core reads the row | 1.7e-3 | 1.5e-2 | 3.9e-3 | 15.9 to 17.0 |
| split RMS norm, sums all-reduced | 1.7e-3 | 1.5e-2 | 3.9e-3 | 16.5 to 17.7 |

The two outputs are the same: a core's sums of its own tiles, added over the group in
float32, give the row's sum as the whole row's read does. Reading 4 tiles instead of
64 does not make the command shorter: the all-reduce (a reduce by tree over 16 cores,
then the cast back) costs what the 60 more tiles' read and squares cost.

`card_islands`: the same class on card 0 split in 2, its 16 cores 8 on each island
(cores 51 to 66), x and w a copy on each island: the sums' all-reduce crosses the link
(rank 0 writes the sum to island 1's first core of the group, which casts it on). The
output is the one island's: rms rel. 1.7e-3, largest error / rms 1.5e-2, largest rel.
3.9e-3; 78.9 to 80.5 us a command (its kernels need a 192 KiB kernel buffer there,
`results/tile_split_islands.json`).

- The split norm's error is its output's bf16: the largest relative error, 3.87e-3, is
  under one bf16 step (2^-8 = 3.91e-3).
- Its 15.7 to 16.6 us a command stand beside the hand's norm command, which completed at
  13.1 us (12.6 in its kernels; projects/decode README, "The norms inside the
  products"). The build's estimate of the sums is 13,468 cycles a row, 10.0 us at
  1,350 MHz; the command's fixed term (3.9 us on 8 cores) and the passes' fp32 copies of
  the row into Dst (126 cycles a tile a pass) are the rest.
- On main the decode folds both norms into its products: 3.4 and 4.5 us at two roots,
  plus 0.7 us of squares (same README). A norm as its own command does not reach that.
- The bf16 fields take 663,552 B of L1 at 4,096 against 933,888 B with float32 fields.
  The layer norm's largest relative error is at values near zero after the mean is taken
  away; its error over the rms is 2.6e-6.
