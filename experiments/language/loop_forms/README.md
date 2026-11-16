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
