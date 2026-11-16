# Tile results: a row function of two results

Question: does a row function on rows of 2,048 values (64 tiles, read in blocks of 8)
return two rows, each pushed into its own output, and does each match float64?

`tile_results.py` makes one class, `Results`. Each of 4 cores normalizes `rows` rows (a
command word, at most 16) and pushes two results: the RMS norm with the weight field,
and the normalized row alone:

```python
@km.tile_function
def normed_2048(row, weight):
    ms = km.reduce_sum(row * row, exact=True) * (1.0 / 2048)
    n = row * km.rsqrt(ms + EPS_RMS).col()
    return n * weight.row(), n


for row in km.Stream(x.tiles[r0 : min(r0 + MOST, r0 + rows), 0:64], block=8, row=64):
    y, n = normed_2048(row, self.weight)
    ys.push(y)
    ns.push(n)
```

Result 0 is written over the chunk (DST 0, 1) and result 1 into DST 5 and 6, past the
weight's chunk in DST 4. Each chunk's two results are packed after its one call.

The counts 5 and 16 run three commands each, each into its own two outputs that start
at zeros, all on one load; each result is checked against float64 on its rows and for
zeros past them.

Run: `kmr prebuild tile_results.py card`, then `python tile_results.py card` on a card
(`made_up` checks the host side without a card).

## Results

Not yet run.
