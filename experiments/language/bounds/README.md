# A large layer's weight products against the card's bounds

Question: written as Python classes, how near do a large layer's products come to
the memory bound in decode and to the compute bound in prefill, with the weights in
bfp8_b or bfp4_b, on one card and on two?

**No language model runs here.** The weights are synthetic, and the attention step, the
norms, the cache, the embedding and the head are absent; no token is produced. A step
is 62 layers of these products at the chosen shapes; where a table below says steps/s
it means 1 / (62 x the time of these products), a rate of this benchmark.

**The shapes are chosen**, not a model's: hidden 5,376, MLP 21,504, 32 heads,
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
| steps a second at that bound, 62 layers, one card / two (no head, embedding, attention) | 18.2 / 36.4 | 34.4 / 68.8 |
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

- **Decode**: measured by term below ([Where a decode layer's time
  goes](#where-a-decode-layers-time-goes)). The fixed part is 13 to 20 us a product,
  and most of the gap is the rate at which the cores take W.
- **Prefill at M 256 and 1,024** is the grid's W read: W comes only through the 12
  column senders, about 84 GB/s for gate in bfp8_b at M 256, so the time hardly moves
  between M 256 and 1,024. On two cards a grid command at M 256 takes about twice one
  card's at the same bytes (gate bfp8_b: 412 against 244 us a command); not explained
  here.
- **Prefill at M 4,096 and 8,192** is 1.54 to 1.95 times the HiFi2 bound: the grid takes
  96 of the 120 cores (1.25 at best), the rest is the casts and the waits, not split by
  term here.

## Where a decode layer's time goes

Question: is a decode product's time above the memory bound a fixed cost, which a layer
as one chain would hide, or something that scales with W? And what keeps two cards
further from the bound than one?

`decode_cost.py` has the five decode products in one class, `Layer`, with the chain's
walk:
- qkv: 16 groups of 7 at width 16.
- o and down: 12 groups of 10 at width 14.
- gate and up: 48 groups of 2 at width 14.
- On two cards the groups are 14, 19 and 4, centred on the edge.

Gate and up take width 14 because the class at the widths of `Split` above (16, 14 and
28) needs 1,860,352 B of L1 against 1,352,656 B. The class makes a read pair for each W
block size (278,528 + 243,712 + 487,424 B) and sizes the reduce's arrival as the most
rounds times the widest field (458,752 B).

The chain is qkv(x), o(q's first 128 tiles), gate(h), up(`km.past(h)`), down(u). Each
product's words are checked against float64 of its input as the card left it: bfp8_b
8.2e-3 to 8.5e-3, bfp4_b 0.113 to 0.116, the same as a command a step.

```
python experiments/language/bounds/decode_cost.py --cards 0 --part chain --format bfp8_b bfp4_b --repeat 10 --kernel-bytes 101120
python experiments/language/bounds/decode_cost.py --cards 0 --part terms --format bfp8_b bfp4_b --repeat 10 --kernel-bytes 98304
python experiments/language/bounds/decode_cost.py --cards 0,1 --part chain --format bfp8_b bfp4_b --repeat 10 --kernel-bytes 204800
python experiments/language/bounds/bounds.py --cards 0 --format bfp4_b --rows 1 --repeat 5 --kernel-bytes 57344 --chunk 4
```

The two-card row below ran at a kernel buffer of 180,224 B. Since then the two-card
kernels take 184,128 B, so the command gives 204,800.

Main at 8ccdf6d9 with these scripts, 2026-11-17, p150a cards at AICLK 1,350 MHz,
dispatcher microseconds, median of 10:

| A layer (5 products) | As one chain | As 5 commands | Bound |
|---|---:|---:|---:|
| one card bfp8_b | 1,117.5 us (1.26x) | 1,167.3 us (1.32x) | 885.2 us |
| one card bfp4_b | 814.3 us (1.74x) | 875.0 us (1.87x) | 468.6 us |
| two cards bfp8_b | 14,845.4 us (33.5x) | 13,871.6 us (31.3x) | 442.6 us |
| two cards bfp4_b | 14,910.6 us (63.6x) | 13,835.9 us (59.1x) | 234.3 us |

As commands at width 14, gate and up take 316.2 and 317.5 us in bfp8_b (320.1 and 320.9
at width 28, above) and 236.9 and 239.3 in bfp4_b (228.0 and 224.4).

**On two cards** each product's input is the step before's output, and each card holds
half of it. Every product after qkv reads half its input across the link, at about 2
GB/s: 12,864 pages and 26.8 MB a layer. A layer split by columns over two cards needs a
gather of each output onto both cards (`km.all_gather`, not used here). Over both
cards, a command with its input local takes as long as its slower card's part: qkv
87.1 us over both, 84.4 and 87.1 on each card alone.

**A product by term** (`--part terms`, one card): commands at full K, K/2, K/4 and 2 K
tiles a core, with the run's trace (each core's take, outputs written and completion
sent).

| | o bfp8_b | o bfp4_b | gate bfp8_b | gate bfp4_b |
|---|---:|---:|---:|---:|
| full K, us | 78.2 | 62.2 | 316.8 | 238.0 |
| a straight line over K: fixed, us | 19.9 | 20.3 | 15.1 | 12.5 |
| a straight line over K: W's rate | 399 GB/s | 295 GB/s | 405 GB/s | 288 GB/s |
| at 2 K tiles a core: before the first take and after the completion | 0.98 | 0.95 | 0.96 | 0.97 |
| the take's spread over the cores | 0.18 | 0.18 | 0.19 | 0.19 |
| a leaf's take to its outputs written (X's chunk, W's first block, one product, the send) | 11.0 | 9.3 | 10.6 | 8.9 |
| the root after the last leaf (tree rounds: o 4, gate 1; then Y's write) | 13.2 | 13.4 | 4.9 | 5.1 |
| at full K: the last leaf after the median leaf | 18.5 | 13.6 | 84.3 | 63.9 |

So the fixed part of a product is 13 to 20 us. Over the five it is 0.08 of the bound in
bfp8_b and 0.18 in bfp4_b. The rest is the rate at which W is taken: 1.23 times the
bound's time in bfp8_b and 1.70 in bfp4_b. In bfp8_b the command waits for the slowest
cores: the last leaf ends 1.4 times after the median one. In bfp4_b a core takes a K
tile of gate (14 W tiles) in 2.66 us against bfp8_b's 3.55, for half the bytes: about
160 cycles a tile do not follow the bytes.

**bfp4_b with a larger chunk** (`bounds.py --chunk`, `Split`, median of 5): the layer
takes 850.8 us at chunk 2, 867.7 at 4, and 914.6 at 8 (gate and up at 5; refused by L1
at 6 and 8). More tiles in flight a mover is slower, so it is not the bytes in flight
that bfp4_b lacks. Which CPU spends the cycles a tile is not measured here.

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

## The two-card layer with its hand-over

Question: as one chain over two cards, each card on its own plan, with each product's
output gathered onto both cards, how near does a decode layer come to the two-card
memory bound?

`two_card.py` has the five products in one class, `TwoCard`, each card computing its
half of N in groups as large as one card's: qkv 16 groups of 7 at width 8, o and down
12 of 9 at width 7, and gate and up 48 of 2 at width 7 (a card has 118 ranks, so o
cannot take groups of 10). Its `km.chain` method `layer` is qkv(x), o(a), then
`km.all_gather(h, 0, 168)`, gate(h), up(`km.past(h)`), down(m), then
`km.all_gather(y, 0, 168)`. X, a and m are copies of host values on both cards.

Two hand-overs are not in it. A gather's record holds `first + pages` in 8 bits (at
most 255), and a decode row tile's output is a page a tile, so q (256 pages, of which
o reads 128) and gate's and up's outputs (672) cannot be gathered. o reads a and down
reads m instead.

```
python experiments/language/bounds/two_card.py --cards 0,1 --format bfp8_b bfp4_b --repeat 10 --kernel-bytes 213504
```

The kernel buffer is the one the session names on main bdd6b963 (2026-11-18). The
class's kernels took 187,344 B in the run below and take 203,904 B there (mock builds
of the class, empty caches): +2,352 B by 4014bcac (pages of k tiles, the island page
runs), +6,688 B in the gated read (4e438a49 and 1316d3a3: BRISC +3,068, NCRISC
+3,608), +5,424 B by cc9552c8 (the shared arrival and the shared read pair among
those merges, not split further), +2,096 B by bdd6b963 (the share of K and the write
by island among them).

Main at 3ab4c15a with this script, 2026-11-17, two p150a cards at AICLK 1,350 MHz,
the slower card's microseconds, median of 10:

| | The chain with its two gathers | Bound | b1's 5 commands | The chain reading the outputs across |
|---|---:|---:|---:|---:|
| bfp8_b | 770.4 us (1.74x) | 442.6 us | 739.1 us (1.67x) | 14,845.4 us (33.5x) |
| bfp4_b | 580.6 us (2.48x) | 234.3 us | 608.3 us (2.60x) | 14,910.6 us (63.6x) |

The two gathers moved 336 pages across the link. Words against float64 of each
product's input as the card held it: bfp8_b 8.2e-3 to 8.5e-3, bfp4_b 0.113 to 0.119.
The same chain without its gathers takes 2,862.7 and 2,707.7 us: each record then
marks the next record's cores on the other card too (2,304 puts across), which the
gather's leaders do only on their own card.

## A W tile's cost by CPU

`tile_cost.py` gives each of the 120 cores a 42 x 8 tile share of gate's W (336
tiles) and runs four handlers on it:
- `streamed`: X and W streamed and multiplied into 8 DST tiles;
- `drained`: W's stream read and dropped, the movers alone;
- `load`: W read once into a kept L1 field;
- `held`: X streamed against that field, compute alone.

```
python experiments/language/bounds/tile_cost.py --card 0 --format bfp8_b bfp4_b --repeat 20
```

Card 0, same day and clock, median of 20:

| | streamed | drained | held (W in L1) | drained, cycles a W tile |
|---|---:|---:|---:|---:|
| bfp8_b | 126.6 us | 99.5 us | 31.3 us | 400 |
| bfp4_b | 94.6 us | 71.0 us | 31.3 us | 285 |

The compute takes W from L1 at 126 cycles a tile in either format. The movers alone
take about 156 cycles a page plus 0.22 cycles a byte, so a bfp4_b tile costs 71% of a
bfp8_b tile for 53% of the bytes. The cost a tile that does not follow the bytes is the
movers' read of a page: one `noc_async_read_page` a tile, with its bank and address
found a page at a time.

## W in DRAM pages of several tiles

A class cannot ask for it today: a tile tensor's page is one tile. `page_tiles/probe.py`
is a hand program that asks what a page of k tiles would give. Each of 120 cores reads
gate's share (336 tiles) from a `uint8` row tensor whose page is k tiles' bytes. A
block of 16 tiles is 16 / k NoC reads and one barrier, as the movement engine does a
pass. The tiles are pushed to compute, which frees them.

```
python experiments/language/bounds/page_tiles/probe.py --pages 1 2 4 8
```

Card 0, AICLK 1,350 MHz, 8 repetitions, the best span of 3. The line is 156 cycles a
page plus 0.22 cycles a byte; DRAM's bound is 495.6 GB/s.

| k | bfp8_b GB/s | the line | bfp4_b GB/s | the line |
|---:|---:|---:|---:|---:|
| 1 | 406.7 | 445.8 | 290.5 | 330.1 |
| 2 | 430.3 | 555.4 | 411.0 | 455.8 |
| 4 | 431.2 | 633.2 | 430.3 | 563.1 |
| 8 | 417.7 | 680.9 | 431.6 | 638.2 |

- **k = 1 is the classes' rate** (b2's 400 and 290 GB/s).
- **Pages of 2 tiles (bfp8_b) or 4 (bfp4_b)** take both formats to about 431 GB/s,
  87% of the bound.
- **The rate then stays flat.** The slowest and median cores meet: the card's bytes
  set the rate, not a core's pages.
- **That 431 GB/s is mover A's limit, not the card's.** Mover A reads on NoC 0 alone,
  and 431 is NoC 0's limit over the whole grid (the routes probe measured 427.9).

With both movers, blocks of 28 tiles, `--patterns A AAAB`, the probe splits the work
by the engine's pattern: mover B takes every fourth block, on NoC 1. Pages of 7 tiles
fit the decode plans' widths of 14 and 7.

```
python experiments/language/bounds/page_tiles/probe.py --pages 1 4 7 --block 28 --patterns A AAAB
```

GB/s over 120 cores, card 0, same clock, the best of 3. The bound is 495.6 GB/s.

| k | bfp8_b, mover A | bfp8_b, AAAB | bfp4_b, mover A | bfp4_b, AAAB |
|---:|---:|---:|---:|---:|
| 1 | 410.7 | 472.3 | 290.8 | 348.5 |
| 4 | 434.4 | 474.1 | 431.5 | 475.0 |
| 7 | 425.9 | 477.4 | 434.4 | 488.8 |

With both NoCs, pages of 7 tiles read W at:
- bfp4_b: 488.8 GB/s, against 348.5 at one tile a page;
- bfp8_b: 477.4 GB/s, against 472.3.

### The decode layer with W in pages of k tiles

A tensor takes `page_tiles=k` and a class's read stream of its tiles reads a page of k
tiles a NoC read. `page_tiles/decode.py` runs decode_cost.py's chain of the five
products unchanged, then the same class with W in pages of k tiles: `2` is every W,
`4/7` is qkv's 4 and the others' 7. W's stream is on the default pattern (AAAB) or AB.

```
python experiments/language/bounds/page_tiles/decode.py --cards 0 --format bfp4_b --unchanged --pages 1 2 4/2 4/7
```

One card, AICLK 1,350 MHz, the median of 10 chains. steps/s counts 62 layers of the five
products alone; GB/s is a layer's W bytes over the chain's time, beside 495.6.

| W's pages | pattern | bfp8_b us | steps/s | GB/s | bfp4_b us | steps/s | GB/s |
|---|---|---:|---:|---:|---:|---:|---:|
| decode_cost, a tile | AAAB | 1,116.1 | 14.5 | 393.0 | 814.5 | 19.8 | 285.1 |
| 1 | AAAB | 1,114.9 | 14.5 | 393.5 | 814.2 | 19.8 | 285.2 |
| 1 | AB | 1,096.4 | 14.7 | 400.1 | 692.9 | 23.3 | 335.2 |
| 2 | AAAB | 1,077.5 | 15.0 | 407.1 | 673.5 | 23.9 | 344.9 |
| 2 | AB | 1,088.1 | 14.8 | 403.2 | 658.0 | 24.5 | 352.9 |
| 4/2 | AAAB | 1,076.5 | 15.0 | 407.5 | 671.1 | 24.0 | 346.1 |
| 4/2 | AB | 1,085.2 | 14.9 | 404.2 | 657.0 | 24.5 | 353.5 |
| 4/7 | AAAB | 1,124.2 | 14.3 | 390.2 | 670.2 | 24.1 | 346.6 |
| 4/7 | AB | 1,103.2 | 14.6 | 397.7 | 657.1 | 24.5 | 353.4 |

- **The words are the same in every row**, against float64: rms 8.2e-3 to 8.5e-3 in
  bfp8_b and 0.113 to 0.116 in bfp4_b.
- **bfp4_b**: 814.5 to 657.0 us, 19.8 to 24.5 steps/s. The probe's 488.8 GB/s for W would
  give about 552 us: the rest of the layer is the difference.
- **bfp8_b**: 1,116.1 to 1,076.5 us at pages of 2. Pages of 7 are slower than 2.
- **The best setting a format**: bfp8_b pages of 2 on AAAB (15.0 steps/s, 407.1 GB/s),
  bfp4_b pages of 2 on AB (24.5 steps/s, 352.9 GB/s). W in pages of 7 loses in bfp8_b (390 to
  398 GB/s), so k is chosen by format.

### The two-card layer with W in pages of k tiles

`page_tiles/two_card.py` runs two_card.py's chain (each card on its own groups, h and
y gathered) with W in pages of k tiles and W's stream on AAAB or AB. On two cards a W
group is 7 columns wide (8 in qkv), and a page is not split: `2/7` is qkv's 2 and the
others' 7. W's split hint is counted in pages (`split_of`): bounds.py's group, counted
in tiles, leaves every paged W on island 0 (the session's uneven-split warning names
it), and card 1 then reads its W over the link (85 to 97 ms a layer).

```
python experiments/language/bounds/page_tiles/two_card.py --cards 0,1 --kernel-bytes 221184 --format bfp8_b --pages 1
python experiments/language/bounds/page_tiles/two_card.py --cards 0,1 --kernel-bytes 221440 --format bfp4_b --pages 1 2/7 4/7
```

The kernel buffers are the ones the session names on main bdd6b963 (2026-11-18): there
the class's kernels take 205,216 B (bfp8_b) and 205,248 B (bfp4_b, a tile a page; 211,408
B at 4/7), past the 204,800 B the runs below had (not traced here).

Main at 4014bcac with this script, 2026-11-17, cards 0,1 at AICLK 1,350 MHz, a kernel
buffer of 204,800 B, a process a bfp8_b load; the slower card's microseconds, median of
10. steps/s counts 62 layers of the five products and the two gathers; GB/s is a layer's
W bytes over the chain's time, beside 991.2 for two cards.

| W's pages | pattern | bfp8_b us | steps/s | GB/s | bfp4_b us | steps/s | GB/s |
|---|---|---:|---:|---:|---:|---:|---:|
| two_card.py, a tile | AAAB | 709.2 | 22.74 | 618.6 | 581.3 | 27.75 | 399.5 |
| 1 | AAAB | 707.5 | 22.80 | 620.0 | 582.3 | 27.70 | 398.9 |
| 1 | AB | 662.4 | 24.35 | 662.3 | 480.3 | 33.58 | 483.6 |
| 2/7 | AAAB | 722.2 | 22.33 | 607.4 | 482.0 | 33.46 | 481.8 |
| 2/7 | AB | 678.4 | 23.78 | 646.7 | 465.9 | 34.62 | 498.4 |
| 4/7 | AAAB | 701.3 | 23.00 | 625.5 | 478.1 | 33.73 | 485.7 |
| 4/7 | AB | 673.7 | 23.94 | 651.2 | 460.0 | 35.06 | 504.8 |

- **The words are the same in every row**: rms 8.2e-3 to 8.5e-3 in bfp8_b, 0.113 to
  0.119 in bfp4_b.
- **On two cards W wants AB**: 1.07 times in bfp8_b, 1.21 in bfp4_b at a tile a page.
- **The best setting a format**: bfp8_b a tile a page on AB (24.35 steps/s, 662.3 GB/s),
  bfp4_b 4/7 on AB (35.06 steps/s, 504.8 GB/s). Pages of 7 lose in bfp8_b here too.

**All four hand-overs gathered.** With gathers of up to 2,047 pages the chain `whole`
gathers q (256 pages) and g (672) as well: o reads q's first 128 columns in attention's
place, down reads g in the MLP product's, and up runs before gate so g's gather follows
its writer. No input is a host copy but X. `--whole --patterns AB`, main at 6d0ecd50,
the same cards and buffer, in one load beside the chain of two gathers:

| format, W's pages | h and y gathered | steps/s | GB/s | all four | steps/s | GB/s |
|---|---:|---:|---:|---:|---:|---:|
| bfp8_b, 1 | 651.6 | 24.75 | 673.2 | 696.1 | 23.17 | 630.2 |
| bfp4_b, 4/7 | 446.1 | 36.15 | 520.6 | 494.3 | 32.63 | 469.9 |

- The gathers of q and g cost 44.5 us in bfp8_b and 48.2 in bfp4_b.
- Words of the four-gather chain against float64: qkv, o, gate and down 8.3e-3 to
  8.5e-3 (bfp8_b) and 0.116 to 0.117 (bfp4_b). up's output u is a copy no call gathers,
  so the host reads only island 0's half of it: its word is not a check.

### The two-card layer as one chain

`page_tiles/one_chain.py` runs a layer whose o and down are split by their input: each
card computes a partial of all of N over its half of K, the partials cross by
`km.all_gather`, and an add sums them. Each card's ranks write that card's own copy of
the sum, a write by island (classes.md, "A write by island"), so the whole layer is one
chain of 8 records: qkv, o, h's gather, h's add, gate_up, down, y's gather, y's add. It
is timed beside the same layer as four commands in order (a chain, h's add, a chain,
y's add), and beside a chain with the two adds as commands after it. In that last form
gate_up reads the hs of the turn before, so it is not a legal layer and is shown only
for its time.

```
python experiments/language/bounds/page_tiles/one_chain.py --cards 0,1 --format bfp8_b bfp4_b
```

At 8803e454 (the write by island), 2026-11-17, cards 0,1 at AICLK 1,350 MHz, W a
tile a page; the slower card's microseconds, the median of 10. steps/s counts 62
layers; GB/s is a layer's W bytes over its time, beside 991.2 for two cards. The four
commands' time is the cards' (two chains and two adds); their wall time with each
waited is 2.2 to 2.5 ms.

| format | pattern | one chain us | steps/s | GB/s | four commands us | chain + adds us |
|---|---|---:|---:|---:|---:|---:|
| bfp8_b | AAAB | 690.3 | 23.37 | 635.5 | 681.6 | 637.6 |
| bfp8_b | AB | 670.5 | 24.06 | 654.3 | 659.1 | 628.0 |
| bfp4_b | AAAB | 555.8 | 29.02 | 417.8 | 544.0 | 495.9 |
| bfp4_b | AB | 468.2 | 34.45 | 496.0 | 464.4 | 420.0 |

- **The words are those of the commands**: each add's sum 1.7e-3 against float64 on
  both copies; the products 8.0e-3 to 8.5e-3 (bfp8_b) and 0.11 to 0.12 (bfp4_b).
- **One command a layer** costs 4 to 12 us of the cards' time beside four, and saves
  three host round trips.

### The five products in one job

The three layers above in one job, with no setting tuned for it: one card, the
PagedLayer at pages 4/2 on AB; two cards, two_card's chain with all four hand-overs
gathered (`--whole`) on AB at the best pages of each format above; two cards, the layer
as one chain on AB.

```
python experiments/language/bounds/page_tiles/decode.py --cards 0 --format bfp8_b bfp4_b --pages 4/2 --patterns AB --repeat 10
python experiments/language/bounds/page_tiles/two_card.py --cards 0,1 --kernel-bytes 221184 --format bfp8_b --pages 1 --patterns AB --whole
python experiments/language/bounds/page_tiles/two_card.py --cards 0,1 --kernel-bytes 221440 --format bfp4_b --pages 4/7 --patterns AB --whole
python experiments/language/bounds/page_tiles/one_chain.py --cards 0,1 --format bfp8_b bfp4_b --kernel-bytes 249920
```

Main at d93d323d, 2026-11-18, card 0 and cards 0,1 at AICLK 1,350 MHz, every load built
before the run; the slower card's microseconds, the median of 10 chains. steps/s counts
62 layers; GB/s is a layer's W bytes over its time, beside the ceiling of 495.6 (one
card) or 991.2 (two cards). The kernel buffer is the session's default on one card and
the one given on two. The kernels have grown since: the commands above give the size
that holds them at 2026-11-18's build (the size a too small buffer's error names).

| layer | format | W's pages | kernel buffer B | us | steps/s | W GB/s | ceiling |
|---|---|---|---:|---:|---:|---:|---:|
| one card, PagedLayer | bfp8_b | 4/2 | default | 1,054.3 | 15.30 | 416.1 | 495.6 |
| one card, PagedLayer | bfp4_b | 4/2 | default | 626.5 | 25.74 | 370.7 | 495.6 |
| two cards, all four gathered | bfp8_b | 1 | 206,272 | 701.0 | 23.01 | 625.8 | 991.2 |
| two cards, all four gathered | bfp4_b | 4/7 | 212,480 | 496.9 | 32.46 | 467.4 | 991.2 |
| two cards, one chain | bfp8_b | 1 | 249,920 | 670.9 | 24.04 | 653.8 | 991.2 |
| two cards, one chain | bfp4_b | 1 | 249,920 | 471.1 | 34.24 | 493.0 | 991.2 |

- **The words are those of the rows above**: rms 8.2e-3 to 8.5e-3 (bfp8_b) and 0.113
  to 0.118 (bfp4_b) against float64; the one chain's adds 1.7e-3; in the four-gather
  chain up's u is not a check (as above).
- **Against the rows above**: the PagedLayer is 31 us faster in each format (1,085.2
  and 657.0 us at 4014bcac); the two-card rows are within 5 us of theirs.

### X read once

Each group of a decode product reads all of its X share from DRAM. In gate each X page is
read 48 times, so X is 58.2 MB a layer beside W's 438.7 MB (bfp8_b) or 232.2 MB
(bfp4_b). `page_tiles/x_forms.py` builds, on the host, the forms that would read X once
and cast it:
- X's cast block as the product's left side against W's stream, in DST or a field of
  sums: refused ("`xs` is not the loop's tile or `wb` is not a field").
- The cast block written into a field compute writes: refused (a field takes DST sums).
- A filled field as the left side: refused (a filled field is a right side only).
- Turned, Y^T = W^T X^T, a row of W^T streamed against X^T cast once: builds. Its cast
  must be on mover A, so W^T streams on mover B alone.

`page_tiles/direct.py` sends o's result to gate's cores (`km.direct`). It is refused at
`run.chain`, because up reads the same h and a direct input has no DRAM copy.
`page_tiles/wide.py` halves the groups instead (qkv 32 tiles wide in groups of 14, the
others 28 in groups of 20 or 4). It fits in bfp4_b with blocks of one X tile only.

Card 0, AICLK 1,350 MHz, bfp4_b, W in pages of 2, the median of 10 chains. The products
as commands are timed after the chain. The words against float64 are the same in every
row.

| chain | patterns | us | steps/s | W GB/s | as commands us |
|---|---|---:|---:|---:|---:|
| decode.py | W AB | 658.6 | 24.49 | 352.6 | 700.1 |
| decode.py | W and X AB | 667.6 | 24.16 | 347.9 | 690.8 |
| wide.py, `--chunk 1` | W AB | 666.5 | 24.2 | 348.4 | 703.0 |
| wide.py, `--chunk 1` | W and X AB | 675.7 | 23.87 | 343.7 | 707.6 |

The probe on mover B alone, `--pages 1 2 7 --block 28 --patterns A B`, in GB/s:

| mover | format | k 1 | k 2 | k 7 |
|---|---|---:|---:|---:|
| A alone | bfp8_b | 409.9 | 430.7 | 423.7 |
| A alone | bfp4_b | 294.7 | 411.2 | 432.5 |
| B alone | bfp8_b | 245.4 | 247.7 | 245.3 |
| B alone | bfp4_b | 232.6 | 246.2 | 247.2 |

- **Wider groups lose 8 us.** Blocks of one X tile and the deeper trees cost more than
  half the X reads save.
- **X on AB as well** costs the chain 9 us and saves the commands 9 us.
- **Mover B alone reads at about 247 GB/s**, NoC 1's limit over the grid. So the turned
  form, with W^T on B alone, would read W slower than today.
- **No form a class says today beats the chain.** The one that would read X once with W
  on both movers needs a held block as a stream product's left operand.

## X's sender

Why does X's column sender read at 3.6 to 6.7 GB/s? The probes on M 1,024's gate plan
(24 senders, 5.51 MB of X each; `--probe x_...`, the median of 10, one card):

| Probe | us | a sender |
|---|---:|---:|
| the casts alone (no product), mover B, KC 4, window 2 | 896.5 | 6.2 GB/s |
| the same, KC 2, window 2 | 878.0 | 6.3 GB/s |
| the same, KC 2, window 8 | 872.4 | 6.3 GB/s |
| the casts alone, mover A | 349.3 | 15.8 GB/s |
| the reads alone, mover B (bands of one core, no receivers) | 1,030.7 | 5.3 GB/s |
| Owned, KC 2, window 4 (window 2: 1,086.1) | 1,051.2 | |
| Owned, KC 4, X on mover A and W on mover B (as given: 1,058.2) | 1,637.9 | |

The cause is the mover: X's reads on mover B (NoC 1) run at about 6 GB/s a sender,
with or without the product, the window or the multicast; the same casts on mover A
take 0.39 of the time. Swapping the movers puts W's reads on mover B and is slower.
A reader alone is not slow on mover B (below, "A lone reader"): the rate is shared, about
78 GB/s for 12 senders in one row of cores and 125 to 150 GB/s for 24 in two rows. In
these plans the time a command follows X's bytes a sender,
(B/H)(K/S) tiles, against W's pages a core, (K/S) x OW, which mover A reads at about
0.36 us a page (fitted on the bfp4_b shapes, where W's pages and not its bytes set the
time). Shorter bands, or each core reading its own X, add readers of the same block
without cutting a sender's bytes; a handler has one broadcast a mover, so X cannot go
on both while W holds mover A.

With these rates (X 6.2 GB/s a sender, W 0.36 us a page a core, DRAM 495.6 GB/s, a
tile product 25.5 ns, a pack 3.5 ns, the tree 28 GB/s a level, 6.3 us a command) the
arithmetic gives the shapes above within 0.88 to 1.22 of their times. It picks larger
S or B where X's bytes set the time (`PLANS_X`, `--plans x`; `PLANS` stays as given):

```
python experiments/language/bounds/owned.py --plans x --format bfp8_b bfp4_b
```

One card, the best of the three plans a shape (us, in brackets the time over the
bound; Owned as given, Owned with `PLANS_X`, the grid of the Results):

| bfp8_b | M 256 | M 1,024 | M 4,096 |
|---|---:|---:|---:|
| qkv | 194.3 (2.06x) X (1,4,2,8) | 589.5 (2.01x) grid | 1,847.9 (1.58x) grid |
| o | 109.8 (2.33x) owned | 347.5 (2.37x) grid | 1,048.5 (1.79x) grid |
| gate | 424.3 (1.71x) owned | 1,067.5 (1.39x) owned | 4,256.3 (1.38x) owned |
| up | 431.1 (1.74x) owned | 1,063.5 (1.38x) owned | 4,233.9 (1.38x) owned |
| down | 411.7 (1.66x) owned | 1,619.6 (2.11x) grid | 4,847.6 (1.58x) grid |
| layer | 1,571.2 (1.78x) | 4,687.6 (1.71x) | 16,234.1 (1.48x) |

| bfp4_b | M 256 | M 1,024 | M 4,096 |
|---|---:|---:|---:|
| qkv | 156.0 (2.13x) X (1,4,2,8) | 498.3 (1.70x) grid | 1,846.1 (1.58x) grid |
| o | 108.5 (2.96x) owned | 338.6 (2.31x) grid | 1,037.7 (1.77x) grid |
| gate | 383.5 (2.00x) owned | 1,033.7 (1.34x) X (2,1,2,32) | 4,132.2 (1.34x) X (2,1,2,32) |
| up | 386.7 (2.01x) owned | 1,035.0 (1.35x) X (2,1,2,32) | 4,136.5 (1.35x) X (2,1,2,32) |
| down | 339.8 (1.77x) X (1,6,2,8) | 1,458.6 (1.90x) owned | 4,730.2 (1.54x) grid |
| layer | 1,374.5 (2.00x) | 4,364.2 (1.59x) | 15,882.6 (1.45x) |

`PLANS_X` against `PLANS` on the 20 shapes it changes: qkv faster at every M in bfp8_b
(M 1,024: 610.0 against 707.9 us), gate and up in bfp4_b at M 1,024 and 4,096 (1,033.7
against 1,464.7), down in bfp4_b at M 256 (339.8 against 408.3) and in bfp8_b at M
1,024 and 4,096 by 3 %; o at S = 4 slower
(466.5 against 391.0: the tree of two levels costs more than its arithmetic), and so
are qkv in bfp4_b at M 1,024 (627.0 against 551.4) and gate, up, down in bfp8_b at
M 256 by 1 to 8 %. Y against float64 as before on every shape.

### A lone reader

One core reads 1 MiB from one DRAM bank, alone on the card (`--lone`: the four corner
cores of the rectangle and a middle one, each of the 8 banks, blocks of 16 and 64 pages,
the median of 5; one card):

| Block | mover A | mover B |
|---|---:|---:|
| 16 pages | 12.7 to 14.3 GB/s | 12.5 to 13.6 GB/s |
| 64 pages | 16.2 to 16.4 GB/s | 15.9 to 16.0 GB/s |

The same on both movers, from every bank and every place: neither the bank's endpoint
on NoC 1 (the SoC descriptor's subchannel 1 of each bank, at other rows than NoC 0's),
nor the hops, nor the pages in flight make mover B slow. It is slow only with many
readers at once: 24 readers in two rows of cores get 15.8 GB/s each on mover A and 5.3
to 6.2 on mover B. Which links of NoC 1 they share is not measured here.

## Two cards

`--cards 0,1 --plans two`: N split by card (W and Y placed as `bounds.py` places them,
each card exactly its half; X a copy on each card), each card a command on its own
12 x 8 cores, `PLANS_2` from the arithmetic above with the tree's extra slot on islands.
A round's time is the larger card's sum. The two-card grid of the Results beside it.

| bfp8_b (owned / grid) | M 256 | M 1,024 | M 4,096 |
|---|---:|---:|---:|
| qkv | 167.1 (3.54x) / 437.4 | 478.6 (3.27x) / 468.9 | 2,638.8 (4.51x) / 926.2 |
| o | 104.8 (4.44x) / 244.4 | 303.7 (4.15x) / 263.5 | 1,206.8 (4.12x) / 570.0 |
| gate | 325.3 (2.62x) / 1,235.9 | 962.6 (2.50x) / 1,323.1 | 2,931.0 (1.91x) / 2,538.4 |
| up | 324.6 (2.62x) / 1,236.5 | 972.5 (2.53x) / 1,321.8 | 2,943.2 (1.91x) / 2,536.9 |
| down | 403.2 (3.25x) / 1,258.6 | 1,318.8 (3.43x) / 1,301.5 | 5,284.5 (3.44x) / 2,582.3 |
| layer | 1,325.1 (2.99x) / 4,412.8 | 4,036.3 (2.94x) / 4,678.8 | 15,004.3 (2.73x) / 9,153.8 |

| bfp4_b (owned / grid) | M 256 | M 1,024 | M 4,096 |
|---|---:|---:|---:|
| qkv | 164.8 (4.50x) / 418.4 | 376.6 (2.57x) / 449.2 | 2,612.4 (4.46x) / 927.5 |
| o | 95.2 (5.20x) / 237.6 | 296.8 (4.05x) / 258.7 | 1,180.2 (4.03x) / 561.8 |
| gate | 312.3 (3.25x) / 1,183.8 | 891.1 (2.32x) / 1,265.9 | 2,894.8 (1.88x) / 2,534.1 |
| up | 311.9 (3.25x) / 1,182.8 | 891.7 (2.32x) / 1,266.1 | 2,890.4 (1.88x) / 2,534.1 |
| down | 392.6 (4.09x) / 1,221.0 | 1,313.7 (3.42x) / 1,258.9 | 5,242.4 (3.41x) / 2,534.8 |
| layer | 1,276.8 (3.72x) / 4,243.6 | 3,769.9 (2.75x) / 4,498.8 | 14,820.3 (2.70x) / 9,092.3 |

The best of the two a product, the layer: bfp8_b 1,325.1 (2.99x), 3,969.0 (2.89x),
9,153.8 (1.67x); bfp4_b 1,276.8 (3.72x), 3,677.0 (2.68x), 9,092.3 (1.66x) at M 256,
1,024, 4,096. Y against float64 as on one card. A card's commands take 1.3 to 1.8 times
their arithmetic (not explained here). Split by N, each card still reads all of X while
W and the products halve, so X's senders set the time: at M 4,096 a card's command of
64 rows of qkv takes about 1,300 us against the arithmetic's 728.

## Owned with W in DRAM pages of k tiles

A cast reads a tensor in pages of k tiles a page at a time (`km::PagesOf<k, T>`), so
Owned's W, cast by each core to itself, can be stored in pages of 2 or 3 tiles.
`--page-tiles k` writes W so for each M whose plan's band is whole pages (OW and N - OW
multiples of k), and at one tile otherwise; on two cards W's split is counted in pages.

```
python experiments/language/bounds/owned.py --plans x --format bfp8_b bfp4_b --rows 8 32 128 --page-tiles 3
python experiments/language/bounds/owned.py --cards 0,1 --plans two --format bfp8_b bfp4_b --rows 8 32 128 --page-tiles 2
```

Main at 65855041 (two cards at k 2 and 3: ff0ddeae) with this script, 2026-11-18,
AICLK 1,350 MHz, every load built before the run. Each product at its best of `PLANS`
and `PLANS_X` (X: the latter) and of k = 1, 2 and 3; k = 1 is the best of every run
that read it at one tile. On one card qkv stays at one tile (OW 5, 7 or 9). Y against
float64 as above.

One card (us):

| bfp8_b: k = 1 / best k | M 256 | M 1,024 | M 4,096 |
|---|---:|---:|---:|
| qkv | 190.0 / same | 608.2 / same | 2,439.0 / same |
| o | 112.2 / 106.8 (k 2) | 390.3 / 385.6 (k 3) | 1,553.4 / same |
| gate | 427.9 / 394.2 (k 3 X) | 1,053.5 / 1,016.9 (k 3) | 4,245.8 / 4,075.4 (k 3) |
| up | 416.0 / 399.2 (k 2) | 1,068.0 / 1,017.0 (k 3) | 4,259.5 / 4,072.0 (k 3) |
| down | 421.7 / 379.9 (k 2) | 1,763.1 / same | 7,062.7 / same |
| layer | 1,567.7 / 1,470.1 | 4,883.1 / 4,790.8 | 19,560.4 / 19,202.5 |

| bfp4_b: k = 1 / best k | M 256 | M 1,024 | M 4,096 |
|---|---:|---:|---:|
| qkv | 150.6 / same | 544.2 / same | 2,180.8 / same |
| o | 110.3 / 107.2 (k 2) | 384.1 / same | 1,553.5 / 1,549.8 (k 2) |
| gate | 387.2 / 276.9 (k 2) | 1,035.0 / 1,031.3 (k 2 X) | 4,139.6 / 4,128.7 (k 3 X) |
| up | 385.0 / 276.8 (k 3) | 1,033.7 / 1,032.2 (k 2 X) | 4,129.1 / 4,125.7 (k 2 X) |
| down | 339.5 / 304.8 (k 3 X) | 1,464.7 / 1,307.7 (k 3) | 5,865.6 / 5,247.4 (k 3) |
| layer | 1,372.6 / 1,116.4 | 4,461.6 / 4,299.5 | 17,868.6 / 17,232.3 |

Two cards (`PLANS_2`, us):

| bfp8_b: k = 1 / best k | M 256 | M 1,024 | M 4,096 |
|---|---:|---:|---:|
| qkv | 167.4 / same | 490.1 / same | 2,613.3 / same |
| o | 100.1 / same | 303.9 / same | 1,213.7 / same |
| gate | 326.9 / same | 982.4 / 881.7 (k 2) | 2,926.4 / same |
| up | 327.2 / same | 991.6 / 882.7 (k 2) | 2,929.2 / same |
| down | 405.8 / same | 1,327.1 / same | 5,308.7 / same |
| layer | 1,327.4 / 1,327.4 | 4,095.2 / 3,885.6 | 14,991.3 / 14,991.3 |

| bfp4_b: k = 1 / best k | M 256 | M 1,024 | M 4,096 |
|---|---:|---:|---:|
| qkv | 166.6 / 166.0 (k 2) | 377.7 / same | 2,583.5 / same |
| o | 96.7 / same | 296.7 / same | 1,190.9 / same |
| gate | 313.2 / same | 888.0 / same | 2,880.5 / 2,869.8 (k 2) |
| up | 312.7 / same | 887.8 / same | 2,879.5 / same |
| down | 403.2 / 398.5 (k 2) | 1,316.2 / same | 5,271.0 / same |
| layer | 1,292.4 / 1,287.1 | 3,766.5 / 3,766.5 | 14,805.4 / 14,794.7 |

Rows a second through a prefill step (62 layers; 2 x M x 412.9 M FLOPs a layer), Owned at its best
k, and with each product at the faster of it and the grid of the Results (the grid's
times as measured there, not again):

| | M 256 | M 1,024 | M 4,096 |
|---|---:|---:|---:|
| one card bfp8_b, Owned | 2,809 rows/s, 144 TFLOPS | 3,447, 176 | 3,440, 176 |
| one card bfp8_b, with the grid | 2,809, 144 | 3,598, 184 | 4,157, 213 |
| one card bfp4_b, Owned | 3,699, 189 | 3,841, 197 | 3,834, 196 |
| one card bfp4_b, with the grid | 3,699, 189 | 3,925, 201 | 4,163, 213 |
| two cards bfp8_b, Owned | 3,111, 159 | 4,251, 218 | 4,407, 226 |
| two cards bfp8_b, with the grid | 3,111, 159 | 4,348, 223 | 7,217, 369 |
| two cards bfp4_b, Owned | 3,208, 164 | 4,385, 224 | 4,465, 229 |
| two cards bfp4_b, with the grid | 3,208, 164 | 4,499, 230 | 7,266, 372 |

Pages help where W's reads set a command's time: bfp4_b gate and up at M 256 (387 to
277 us), down at every M in bfp4_b (M 4,096: 5,866 to 5,247 us), bfp8_b down at M
256. On two cards X's senders set the time (above), so only bfp8_b gate and up at M
1,024 gain (982 to 882 us).

W stays on mover A. Two forms that would put it on both movers do not build: an own
broadcast with `pattern="AB"` ("a broadcast has no pattern="), and W as a
`km.Stream(..., pattern="AB")` against X's cast block into the field ("`wb` is not a
block a broadcast brought: a sum in a field adds products of two broadcasts' blocks").
Their bound, W alone at the two movers' 475 GB/s against mover A's 431: 924 against
1,018 us a layer in bfp8_b, 489 against 539 in bfp4_b, one card.
