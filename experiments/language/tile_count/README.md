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

2026-11-16, card 0, one build (0 kernels compiled on the card) and one load. Errors
against float64 on the rows each count writes; microseconds a command (three commands a
count, 4 cores).

| Rows a core | rms rel. | largest error / rms | largest rel. | rows past untouched | us a command |
|---:|---:|---:|---:|---|---:|
| 1 | 6.0e-8 | 1.3e-6 | 2.3e-7 | yes | 42.5 to 43.6 |
| 7 | 6.1e-8 | 1.4e-6 | 2.8e-7 | yes | 259.4 to 259.7 |
| 64 | 6.2e-8 | 1.7e-6 | 2.9e-7 | yes | 2,319.3 to 2,321.5 |

About 36.3 us a row of 2,048 values on a core once the command is long; one row a core
is near a constant row on one core in [tile long](../tile_long/README.md) (41.6 us).
L1 of a core: 894,848 B of 1,486,592 B, the per-row pages 262,144 B of it.
