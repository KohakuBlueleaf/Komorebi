# Tile row count: a norm over a run-time count of long rows

Question: does the RMS norm, written as a tile function on rows of 2,048 values longer
than their stream's block, take its count of rows from the command (a prefill's norm
has as many rows as tokens) and compute what float64 computes for each count, from one
build and one load?

`tile_count.py` makes one class, `Counted`. Each of 4 cores normalizes `rows` rows (a
command word, at most 64) of 64 tiles from row `rank x 64`, in blocks of 8 tiles. The
rows side of the rectangle states the most:

```python
for row in km.Stream(x.tiles[r0 : min(r0 + MOST, r0 + rows), 0:64], block=8, row=64):
    results.push(rms_2048(row, self.weight))
```

The per-row values' pages (`kChunkRowsRms2048`) are sized at the most: one page a row
(the RMS norm has one level of sums), 64 pages, 256 KiB. A command reads and writes its
`rows` rows alone; the rows past them are left as they were.

The counts 1, 7 and 64 run three commands each, each count into its own output that
starts at zeros, all on one load; each is checked against float64 on its rows and for
zeros past them.

Run: `kmr prebuild tile_count.py card`, then `python tile_count.py card` on a card
(`made_up` checks the host side without a card).

## Results

Not yet run.
