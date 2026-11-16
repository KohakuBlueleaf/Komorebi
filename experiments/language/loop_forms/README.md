# Loop forms: an object made at one loop depth and used at another

Question: do the class lines whose object is made outside a loop and used in it (a
running total, a block cast once and read each turn, a weight filled each turn, a gate
of two DST sums) give, on the card, every word numpy gives? Each line is a line of the
class compiler ([classes](../../../docs/reference/classes.md)).

`loop_forms.py --form total` makes `RunningTotal`. `prefix` adds two tiles a turn into
a tile made before the loop and writes it each turn, so page i is the core's sum of
turns 0 to i. (A reduce of the total each turn is built but refused for now: a reduce
repeated in a loop of one command has no flow control between turns, which the card
showed; [hazards](../../../docs/reference/hazards.md#a-reduce-repeated-in-a-loop-of-one-command).) The inputs are k / 16 in [-8, 8), exact in bf16 and summed
exactly in fp32 in any order, with sums that need more bits than tf32 keeps. The
outputs are fp32, so every word compares.

```python
total = km.tile.zeros(km.fp32)
for i in range(turns):
    for tile in km.Stream(x.pages((self.rank * turns + i) * K, K)):
        total += tile
    out.pages(self.rank * turns + i, 1).write(total)
```

On the card (2026-11-16, card 0, 8 cores, 4 turns of 2 tiles): all 32,768 words of the
32 prefix pages equal numpy's, and the command took 8.2 us.

`loop_forms.py --form held` makes `HeldWeight`. `two` casts W's 8 x 4 tiles down each
column of a 2 x 4 rectangle once, before the loop. The block is held, and each turn two
products read it, rows 2i and 2i + 1 of the core's x. Two references run on the same
inputs, each casting the block anew every turn with one product a turn: `CastEachTurn`
(the cast on mover A, x's stream on B, as in `two`) and `CastOnB` (broadcast_product's
arrangement, the cast on B and the stream on A). Each core picks its rows by its place
in the rectangle, `self.y * 2 + self.x`. `self.rank` is its place in the load's group,
which runs along the whole grid row. The inputs are k / 8, and the outputs are fp32.

```python
if column.sender():
    ws = column.cast(w.tiles[0:KW, 0:NW].read())
else:
    ws = column.arrive()
for i in range(turns):
    r = (self.y * 2 + self.x) * 2 * turns + 2 * i
    acc = km.tile.zeros(km.fp32, NW)
    for xb in km.Stream(x.tiles[r : r + 1, 0:KW], block=2, pattern="B"):
        acc += project(xb, ws)
    y.pages(r * NW, NW).write(acc)
    ...  # the same of row r + 1 into a second sum
```

On the card (2026-11-16, card 0, 8 cores, 4 turns): all three gave the same 262,144 words,
equal to float64's (no word wrong on any core, page or face). The times were 35.98 us
for `two`, 40.42 us for `CastEachTurn` and 52.98 us for `CastOnB`.

