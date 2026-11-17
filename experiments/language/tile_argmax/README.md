# Tile row max and argmax

Question: do `km.reduce_max`, `km.reduce_min` and an argmax written with `km.where`
give, a row of a tile function, what numpy gives: on rows of 2 tiles and of 64, with
the largest value in several columns (which index wins), a NaN in a row, a row of one
value?

`row_max.py` makes a class a row width: three handlers, each a stream loop over one
row of tiles on one core pushing one tile function a row into a float32 tensor. The
argmax picks the column numbers (a field: one per-column tile a chunk of 2 tiles,
`row_operand`) where the row equals its largest value and takes their smallest:

```python
@km.tile_function
def argmax(row, cols):
    m = km.reduce_max(row, exact=True)
    return km.reduce_min(km.where(row == m.col(), cols.row(), NOWHERE)).col()
```

A row of 2 tiles is read in one block (the row form); a row of 64 in blocks of 16 (the
row read once a pass, two passes for the argmax). Rows 0 to 7 of each tile row are edge
rows: the largest value in 2 columns, in 3, in every column, in the first, in the last;
a NaN; a row of -inf; -0 and +0 as the largest. The rest hold integers from -50 to 49,
with ties in about half. Each row's value (column 0 of the broadcast result) is
compared with numpy's (`max`, `min`, `argmax`: the first of the largest).

    python row_max.py                # one card (--cards N: that card)
    kmr prebuild --cards 0,1 row_max.py --cards 0,1   # the same classes for two cards, on the mock

## Cost (the build's model, `UnitProgram.cost("math")`)

A row of 2 tiles: `row_max` 441 instructions, 577 cycles; the argmax 1,513 and 1,785
(two levels: the max, then the picked columns' min). A row of 64 read in parts: 402
instructions a chunk of 2 for the max's pass, 1,050 a chunk for the argmax's second.
