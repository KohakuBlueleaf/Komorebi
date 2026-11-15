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
a loop for compute to free the block, after the stream's last block. A class runs one
broadcast a mover (the core's arguments hold one), so each broadcast handler is its
own class: `Down` (`streams`, `down`), `Window`, `Turned`, loaded in turn. `build`
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
