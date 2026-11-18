# Step lines of a handler

## A product of a stream's tile with a broadcast's block

Question: does `acc += project(xb, ws)`, `ws` a block one core of a column reads and
casts down the column, give the same sums as the two streams' form, where every core
reads W itself? And do a window of a wider block (`ws[:, N : N + N]`) and a part cast
transposed (`wt.tiles[...].T`) give them too?

`broadcast_product.py` computes Y = X W on a rectangle of 12 x 4 cores. Core (x, y)
writes Y's tiles (y, x N .. x N + N - 1), X's row y of K = 8 tiles by N = 4 of W's
columns, from x 2 N + N (W is 12 x 2 N tiles wide). X is normal, W normal / sqrt(K x
32), both bfloat16; the sums are float32 in DST.

```python
column = km.Broadcast(self, along="column", window=1)
if column.sender():
    ws = column.cast(w.tiles[0:K, left : left + N].read())
else:
    ws = column.arrive()
acc = km.tile.zeros(km.fp32, N)
for xb in km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2, pattern="A"):
    acc += project(xb, ws)
```

Four handlers: `streams` (X's and W's streams in a zip), `down` (above), `window` (W's
whole column block of 2 N columns cast, its second half multiplied) and `turned` (W
transposed in DRAM, read with `.T`, the part multiplied with the transpose flag). X's
stream is on mover A: a column's broadcast is mover B's, which waits at a cast outside
a loop for compute to free the block, after the stream's last block. A class then ran
one broadcast a mover (the core's arguments held one), so each broadcast handler is
its own class: `Down` (`streams`, `down`), `Window`, `Turned`, loaded in turn. `build`
builds the three on a mock device; `card` runs each handler 5 times into its own Y and
writes `results/broadcast_product.json`.

| Handler | Y against float64 (relative RMS / max) | Equal to `streams` | Time a command (5) |
|---|---|---|---|
| `streams` | 3.11e-4 / 3.38e-4 | | 20.4 to 21.6 us |
| `down` | 3.11e-4 / 3.38e-4 | datum for datum | 18.0 to 18.4 us |
| `window` | 3.11e-4 / 3.38e-4 | datum for datum | 24.9 to 25.9 us |
| `turned` | 3.11e-4 / 3.38e-4 | datum for datum | 17.0 to 18.0 us |

Kernel text (mock): `Down` 56,608 B, `Window` 43,264 B, `Turned` 43,584 B.

The same products in the same order give the same words: the block's form changes
where W comes from, not the sums. `down` reads W once a column (8 x 4 tiles) where
`streams` reads it on every core. `window` casts twice the tiles and multiplies half
of them. Its time is not split by term: no part was run.

## A step function on DST sums: a row softmax

Question: does `pages.write(f(acc, self.state[r]))`, f a tile function in the step form,
carry a row's reference and sum through a state tile in L1 from one chunk to the
next, and give the softmax float64 gives?

`row_softmax.py` computes Z = X W on 12 cores, each core's 32 rows in C = 4 chunks of
N = 2 tiles (K = 8 tiles, X normal and W normal x 3 / sqrt(K x 32), both bfloat16). Each
chunk's step is `softmax_step` (classes.md): x = Z log2 e, m = max(m, the row's max),
P = 2^(x - m) written, l = 2^(m_old - m) l + the row's sum of P. The state starts as a
tile of zeros (m = 0, l = 0: the reference only has to bound each chunk), and is
written after the last chunk. The host takes m and l from the state tile's slots
(`state_slot` of tile_rules) and checks P against 2^(x - m_c), m and l against float64,
and P 2^(m_c - m) / l against the softmax.

| | Card (5 commands) |
|---|---|
| P, relative RMS / largest error | 1.79e-3 / 5.8e-3 |
| m, largest error (log2 units) | 4.8e-3 |
| l, largest relative error | 6.7e-3 |
| softmax, largest error | 5.7e-3 |
| a state's 8 lanes of a row, largest spread | 1.4e-6 |
| time a command | 20.5 to 21.4 us |

Kernel text (mock): 52,352 B.

The products carry 3e-4 of relative error (the broadcast product's proof above
measured the same), so m is off by up to 4.8e-3 in log2 units. That moves each
probability by up to 2^0.0048 - 1 = 3.3e-3, and the exp template adds up to 1.75e-3:
together the l and softmax errors. The first card run gave l up to 11 times too large:
the step form had inlined alpha = exp2(m_old - m_new) into l's level, which reloaded
slot 0 after m's store, so alpha was 1. A model with alpha = 1 gives that run's l
(10.07 against the card's 10.09) and softmax (0.7641 against 0.7643). The step form now
keeps such a value in a free slot (tile.md); tile_rules' step units are emitted
unchanged.

## Two handlers' broadcasts on one mover

Question: do two handlers of one class, each with a column's broadcast on mover B,
each read their own arguments (the second's after the first's), and give the sums
each gives in a class of its own?

`two_broadcasts.py` makes class `Two` of broadcast_product's `down` and `window` on the
same 12 x 4 rectangle and runs them in turn, 5 commands each, each into its own Y.

| Handler | Y against float64 (relative RMS / max) | Equal to `down` | Time a command (5) |
|---|---|---|---|
| `down` | 3.11e-4 / 3.38e-4 | | 18.3 to 20.0 us |
| `window` | 3.11e-4 / 3.38e-4 | datum for datum | 25.2 to 26.0 us |

Kernel text (mock): `Two` 52,944 B.

Both give the words each gave in a class of its own (above): `window` reads its own
buffer (64 pages, number 1's area) and not `down`'s, which ran before it on the same
core. With one set of arguments a mover, as before, both would have run with the
last one's. The times are those of each handler alone, within 0.5 us.

## A product whose left side is a field: a hidden layer kept in L1

Question: does `self.h.write(act(acc))` put a row of DST sums into a field in bfloat16
that `out += project(self.h, wb)` then takes as the product's left side, a turn of a
loop at a time?

`hidden_layer.py` computes Y = relu(X W1) W2 on 12 cores, 2 rows of tiles a core (X 8
tiles wide, H 4, Y 4; X normal, W1 and W2 normal / sqrt of their rows, bfloat16). Each
turn writes H's row into the field `h` (4 bfloat16 tiles) and multiplies it by W2's
4 rows; the second turn's write frees the first's pages. Against Y with H rounded to
bfloat16 (as the field holds it) and against float64:

| | Card (5 commands) |
|---|---|
| Y against H in bfloat16, relative RMS / max | 1.06e-3 / 1.47e-3 |
| Y against float64, relative RMS / max | 1.69e-3 / 1.65e-3 |
| time a command | 15.1 to 16.0 us |

Kernel text (mock): 46,768 B.

The first card run gave Y near 1e36: the field's pack format was never set (a kernel
whose only other pack is float32 packed H as float32 tiles into bfloat16 pages), so
unpack read float32 low halves as bfloat16. The written field is now one of the
formats pack writes.

## Sums started from a field: a decayed running sum

Question: does `acc = decay(self.o, self.state[0])` start a chunk's sums from the
field's tiles scaled by a per-row value of a state slot, the products then add into
them, and `self.o.write(acc)` carry them to the next chunk?

`decayed_sum.py` computes o = o d + X_c W_c over C = 4 chunks of K = 4 tiles on 12
cores (o 2 fp32 tiles, X normal and W normal / sqrt(K x 32), bfloat16), d each row's
decay in [0.5, 1), written by the host into slot 0 of each core's state tile; Y = o d
after the last chunk. Against float64:

| | Card (5 commands) |
|---|---|
| Y, relative RMS / max | 3.09e-4 / 3.07e-4 |
| Y against the sums without the last d (a decay not applied) | 0.249 / 0.315 |
| time a command | 12.6 to 13.5 us |

The error is the products' 3e-4: o and d stay in float32 through DST and the field.
