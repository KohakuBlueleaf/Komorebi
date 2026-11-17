# A 27B-class layer's products against the card's bounds

Question: written as Python classes, how near do a 27B-class layer's products come to
the memory bound in decode and to the compute bound in prefill, with the weights in
bfp8_b or bfp4_b, on one card and on two?

**The shapes are assumed** until a model is named: hidden 5,376, MLP 21,504, 32 heads,
16 KV heads, head dimension 128, 62 layers (constants at the top of `bounds.py`). A
layer's five products, K x N: qkv 5,376 x 8,192, o 4,096 x 5,376, gate and up 5,376 x
21,504, down 21,504 x 5,376; 403,200 weight tiles a layer. The attention step is not
here (its prefill form is not written), nor the norms, the cache, the embedding and the
head.

`bounds.py` has two classes, each a product `return x @ w` at HiFi2 into a float32
field:

- `Split` (decode, one row tile of X): the dense project's K split. Groups of SPLIT
  cores, a group WIDTH columns of Y, a core 1/SPLIT of K; X's and W's blocks stream in
  step, and the group's sums meet by `km.reduce(..., form="tree")`. One card: qkv in
  16 groups of 7 (112 cores), o and down 12 of 10 (120), gate and up 24 of 5 (120).
- `Grid` (prefill, many row tiles): the dense project's grid plan on 12 x 8 cores, a
  core OH row tiles and OW columns; X's chunks cast along each row of cores, W's down
  each column. A command makes BLOCK columns of Y (qkv 128, o and down 84, gate and up
  112), so a product is 2 or 6 commands.

On two cards the output dimension is sharded as `dense_class/linear_class.py` does: X a
copy on each card, W and Y split by tile columns so that each card holds its half, and
each card computes its half (the K split as one command over both cards' groups
centred on the second card's first rank, groups of 14 for qkv, 19 for o and down, 8
for gate and up; the grid a command on each card's own rectangle).

W is normal / sqrt(K) in float32, encoded once into its format's pages (`results/`,
made before the first run, not committed); X is normal in bfloat16. Y's first and last
row tile, on 256 columns, are checked against float64 of the same X and the float32 W.

```
python experiments/language/bounds/bounds.py --cards 0 --format bfp8_b --rows 1 8 32 128 256
python experiments/language/bounds/bounds.py --cards 0,1 --format bfp4_b --rows 1 8 32 128
```

## The bounds

From the dense project's measured ceilings
([the card's two ceilings](../../../projects/dense/README.md#the-cards-two-ceilings)):
DRAM 495.6 GB/s with 120 ranks reading on both movers, and the matrix unit at 4.70e9
tile products a second on 120 ranks at HiFi2 with float32 sums. No LoFi or block-float
peak is recorded, so the compute bound is HiFi2's, and the products run at HiFi2. A
shape's bound is the larger of its W bytes at 495.6 GB/s (bfp8_b 1,088 B a tile, bfp4_b
576) and its tile products at 4.70e9 a second; on two cards each card's half of both.

| | bfp8_b | bfp4_b |
|---|---:|---:|
| W a layer | 438.68 MB | 232.24 MB |
| W, 62 layers | 27.20 GB | 14.40 GB |
| decode bound a layer, one card / two | 885.2 / 442.6 us | 468.6 / 234.3 us |
| tokens a second at that bound, 62 layers, one card / two (no head, embedding, attention) | 18.2 / 36.4 | 34.4 / 68.8 |
| rows from which prefill is compute-bound | 330 | 175 |

The prefill compute bound is 85.79 us a layer for each 32 rows, on one card.

## Results

Main at 222abf11 with this script, 2026-11-17, Blackhole p150a cards at AICLK 1,350
MHz, every load built before the run. A shape's time is the sum of its commands'
microseconds, the slower card's on two, the median of 3; in brackets the time over the
shape's bound. M is the rows of X (32 a row tile); M 32 is the decode shape.

One card, bfp8_b:

| Product | M 32 | M 256 | M 1,024 | M 4,096 | M 8,192 |
|---|---:|---:|---:|---:|---:|
| qkv | 136.3 (1.44x) | 581.4 (6.16x) | 589.5 (2.01x) | 1,847.9 (1.58x) | refused (L1) |
| o | 78.3 (1.66x) | 298.2 (6.32x) | 347.5 (2.37x) | 1,048.5 (1.79x) | 2,073.0 (1.77x) |
| gate | 320.1 (1.29x) | 1,463.2 (5.90x) | 1,623.7 (2.11x) | 5,001.2 (1.63x) | refused (L1) |
| up | 320.9 (1.29x) | 1,477.5 (5.96x) | 1,621.6 (2.11x) | 5,003.5 (1.63x) | refused (L1) |
| down | 318.1 (1.28x) | 1,428.0 (5.76x) | 1,619.6 (2.11x) | 4,847.6 (1.58x) | 9,821.5 (1.60x) |
| layer (5 products) | 1,173.6 (1.33x) | 5,248.3 (5.93x) | 5,801.8 (2.11x) | 17,748.7 (1.62x) | |

One card, bfp4_b:

| Product | M 32 | M 256 | M 1,024 | M 4,096 | M 8,192 |
|---|---:|---:|---:|---:|---:|
| qkv | 100.8 (2.02x) | 415.1 (5.67x) | 498.3 (1.70x) | 1,846.1 (1.58x) | refused (L1) |
| o | 61.4 (2.46x) | 267.1 (7.30x) | 338.6 (2.31x) | 1,037.7 (1.77x) | 2,119.5 (1.81x) |
| gate | 228.0 (1.74x) | 1,171.5 (6.10x) | 1,371.3 (1.78x) | 4,995.4 (1.62x) | refused (L1) |
| up | 224.4 (1.71x) | 1,171.3 (6.10x) | 1,399.7 (1.82x) | 4,997.6 (1.63x) | refused (L1) |
| down | 234.5 (1.79x) | 1,303.8 (6.78x) | 1,548.3 (2.01x) | 4,730.2 (1.54x) | 9,531.9 (1.55x) |
| layer (5 products) | 849.1 (1.81x) | 4,328.8 (6.31x) | 5,156.1 (1.88x) | 17,606.9 (1.60x) | |

Two cards, bfp8_b:

| Product | M 32 | M 256 | M 1,024 | M 4,096 |
|---|---:|---:|---:|---:|
| qkv | 85.6 (1.81x) | 437.4 (9.27x) | 468.9 (3.20x) | 926.2 (1.58x) |
| o | 54.6 (2.31x) | 244.4 (10.35x) | 263.5 (3.60x) | 570.0 (1.95x) |
| gate | 201.4 (1.63x) | 1,235.9 (9.97x) | 1,323.1 (3.44x) | 2,538.4 (1.65x) |
| up | 201.9 (1.63x) | 1,236.5 (9.98x) | 1,321.8 (3.44x) | 2,536.9 (1.65x) |
| down | 195.6 (1.58x) | 1,258.6 (10.16x) | 1,301.5 (3.39x) | 2,582.3 (1.68x) |
| layer (5 products) | 739.1 (1.67x) | 4,412.8 (9.97x) | 4,678.8 (3.41x) | 9,153.8 (1.67x) |

Two cards, bfp4_b:

| Product | M 32 | M 256 | M 1,024 | M 4,096 |
|---|---:|---:|---:|---:|
| qkv | 75.6 (3.03x) | 418.4 (11.43x) | 449.2 (3.07x) | 927.5 (1.58x) |
| o | 48.9 (3.91x) | 237.6 (12.98x) | 258.7 (3.53x) | 561.8 (1.92x) |
| gate | 159.3 (2.43x) | 1,183.8 (12.32x) | 1,265.9 (3.29x) | 2,534.1 (1.65x) |
| up | 161.4 (2.46x) | 1,182.8 (12.31x) | 1,266.1 (3.29x) | 2,534.1 (1.65x) |
| down | 163.1 (2.49x) | 1,221.0 (12.71x) | 1,258.9 (3.28x) | 2,534.8 (1.65x) |
| layer (5 products) | 608.3 (2.60x) | 4,243.5 (12.37x) | 4,498.7 (3.28x) | 9,092.3 (1.66x) |

Y against float64 (relative RMS): 8.2e-3 to 8.6e-3 with bfp8_b and 0.113 to 0.120 with
bfp4_b, every shape, as the formats' stated errors at HiFi2 (8.33e-3 and 0.118:
[placement](../../../docs/reference/formats.md#placement)). No page crossed a link on
two cards (the run's crossing lines). The largest M is set by L1: the grid's float32
field and its casts' slots; M 8,192 of qkv, gate and up is refused at the load (gate
bfp8_b: its buffers take 1,657,344 B, 1,367,792 B left), and so is M 8,192 of every
product on two cards, where 162 KB less L1 is left for buffers.

**The placement on two cards.** W and Y split by tile columns over all 236 ranks
(`split=1, group=(0, 236)`) do not give each card its half: the partition gives its
remainder to the first ranks, so the first card holds more than half of the columns,
and its share of the commands' pages is read across the link. In that form o's decode
took 1,796 us (76 times its bound, 1,683 pages across the link) and down's 9,325 us.
The form above (a group centred on the second card's first rank, the same count of
columns a rank) puts exactly half on each card; the session's crossing line names the
first form ("moved 1,683 pages across the link").

## Where the time goes (by reading; no part run)

- **Decode** reads W at 384 GB/s for gate on one card in bfp8_b (122.8 MB in 320 us)
  and carries about 30 us a product that does not scale with W (o: 78.3 us against
  47.2). With bfp4_b the fixed part stays and W's bound halves, so the ratios grow (1.71
  to 2.46); a block of the same tile count holds half the bytes, so fewer bytes are in
  flight a mover. On two cards the bound halves again and the command spans both cards,
  its time the slower card's part (1.58 to 3.91).
- **Prefill at M 256 and 1,024** is the grid's W read: W comes only through the 12
  column senders, about 84 GB/s for gate in bfp8_b at M 256, so the time hardly moves
  between M 256 and 1,024. On two cards a grid command at M 256 takes about twice one
  card's at the same bytes (gate bfp8_b: 412 against 244 us a command); not explained
  here.
- **Prefill at M 4,096 and 8,192** is 1.54 to 1.95 times the HiFi2 bound: the grid takes
  96 of the 120 cores (1.25 at best), the rest is the casts and the waits, not split by
  term here.

## Every core reads its own W

`owned.py` runs the prefill shapes with no core sending W to another. The rectangle is
the whole card, 12 x 10 cores, in H groups of rows of cores, each group its block of
X's rows; N is cut in bands across a group, and S consecutive cores of a row split a
band's K, their float32 sums met by `km.reduce(..., form="tree", group=S)` at the
first. X comes down each column of a group from one sender (`km.Broadcast` along the
column, length the group's rows); W by a row broadcast of length 1, so every core is
its own W sender (`Owned` for S = 1, `OwnedSplit` for S > 1). A command makes B row
tiles of Y; the plan (H, S, KC, B) a product and format is `PLANS` at the top of the
file. W's pages are `bounds.py`'s (`results/`), read; Y is checked as there.

```
python experiments/language/bounds/owned.py --format bfp8_b bfp4_b --rows 8 32 128
python experiments/language/bounds/owned.py --probe
```

Main at 8ccdf6d9 with this script, 2026-11-17, one Blackhole p150a card at AICLK 1,350
MHz, every load built before the run. A shape's time is the sum of its commands'
microseconds, the median of 3, in brackets the time over the bound; beside it the
`Grid` time of the Results above.

One card, bfp8_b (owned / grid):

| Product | M 256 | M 1,024 | M 4,096 |
|---|---:|---:|---:|
| qkv | 206.8 (2.19x) / 581.4 | 707.9 (2.42x) / 589.5 | 2,831.4 (2.42x) / 1,847.9 |
| o | 109.8 (2.33x) / 298.2 | 391.0 (2.67x) / 347.5 | 1,565.2 (2.67x) / 1,048.5 |
| gate | 424.3 (1.71x) / 1,463.2 | 1,067.5 (1.39x) / 1,623.7 | 4,256.3 (1.38x) / 5,001.2 |
| up | 431.1 (1.74x) / 1,477.5 | 1,063.5 (1.38x) / 1,621.6 | 4,233.9 (1.38x) / 5,003.5 |
| down | 411.7 (1.66x) / 1,428.0 | 1,816.6 (2.36x) / 1,619.6 | 7,246.1 (2.36x) / 4,847.6 |
| layer (5 products) | 1,583.6 (1.79x) / 5,248.3 | 5,046.5 (1.84x) / 5,801.8 | 20,132.8 (1.83x) / 17,748.7 |

One card, bfp4_b (owned / grid):

| Product | M 256 | M 1,024 | M 4,096 |
|---|---:|---:|---:|
| qkv | 175.0 (2.39x) / 415.1 | 551.4 (1.88x) / 498.3 | 2,180.6 (1.86x) / 1,846.1 |
| o | 108.5 (2.96x) / 267.1 | 385.5 (2.63x) / 338.6 | 1,548.6 (2.64x) / 1,037.7 |
| gate | 383.5 (2.00x) / 1,171.5 | 1,464.7 (1.91x) / 1,371.3 | 5,890.9 (1.92x) / 4,995.4 |
| up | 386.7 (2.01x) / 1,171.3 | 1,458.0 (1.90x) / 1,399.7 | 5,879.1 (1.91x) / 4,997.6 |
| down | 408.3 (2.12x) / 1,303.8 | 1,458.6 (1.90x) / 1,548.3 | 5,885.5 (1.91x) / 4,730.2 |
| layer (5 products) | 1,462.0 (2.13x) / 4,328.8 | 5,318.3 (1.94x) / 5,156.1 | 21,384.6 (1.95x) / 17,606.9 |

Y against float64 (relative RMS): 8.16e-3 to 8.55e-3 with bfp8_b, 0.113 to 0.119 with
bfp4_b. The layer takes 0.30 (bfp8_b) and 0.34 (bfp4_b) of the grid's time at M 256,
0.87 and 1.03 at M 1,024, 1.13 and 1.21 at M 4,096.

**Where the time goes.** A command's time follows X's column sender, not W: a sender
reads its RB x K/S tiles of X (bfloat16) at 3.6 to 6.7 GB/s (gate bfp8_b at M 1,024:
5.51 MB in 1,064 us, 5.2 GB/s), against the 10.3 GB/s of a lone mover reading alone.
The probes (`--probe`, gate's tensors, the median of 10): a command whose cores each
make one tile of Y from one K block takes 6.3 us (7.6 us with S = 2 and its tree);
M 1,024's gate plan takes 1,058.2 us at KC 4 and 1,086.1 us at KC 2, so twice the
field's packs add 27.9 us, 3.5 ns a tile a K block, in a command not bound by the
matrix unit (the pack's own cost is not isolated by this).
