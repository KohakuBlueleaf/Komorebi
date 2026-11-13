---
title: "Tiles and tensors"
part: 3
summary: "the 32 x 32 tile, and a tensor: an array of numbers kept in the card's DRAM"
---

# Tiles and tensors

A **tile** is a block of 32 x 32 numbers, and a **tensor** is an array of numbers that
Python puts into the card's DRAM, cut into tiles. This lesson relies on [[the-card]] and
needs no kernel: it makes a tensor from the host and reads it back.

## Why tiles

A core does not take data one number at a time. Each trip to DRAM has a cost, so a core
takes a whole block per trip, and the arithmetic units of the core work on whole blocks
too. The block has a fixed size, 32 rows by 32 columns, 1024 numbers: a tile. Everything
that moves between DRAM and the cores in this tutorial moves as tiles, and every later
lesson of this part counts in tiles.

The numbers of a tile live in one piece of DRAM, so that a core can read it in one trip.
That piece is a **page**: with the tile layout, a page is exactly one tile.

## The general form

```python
tensor = session.tensor(ARRAY, DTYPE, "tile")
```

- `session` is the open card ([[hello-core]]).
- `ARRAY` is a numpy array of two dimensions: rows by columns. Its numbers are copied to
  the card.
- `DTYPE` is how each number is stored on the card, a name: `"uint32"` is a whole number
  from 0 to 4,294,967,295 in 4 bytes. This part uses it for its first lessons, and
  `"bfloat16"`, a 2-byte number with a fraction, from the lesson on arithmetic.
- `"tile"` is the layout: the array is cut into 32 x 32 tiles, numbered row by row, and
  each tile is one page.
- `tensor` is the handle on the data in DRAM. It has `tensor.shape`, `tensor.pages` (the
  number of tiles), `tensor.page_bytes` (the bytes of one tile) and `tensor.read()`, which
  copies the whole tensor back to a numpy array.

An array whose sides are not multiples of 32 is padded with zeros up to the next multiple.

<figure class="km-diagram">
<script type="application/json">
{"kind": "boxes", "boxes": [{"id": "t0", "x": 0, "y": 0, "w": 142, "h": 64, "t": "tile 0\nrows 0 to 31\ncolumns 0 to 31", "small": true}, {"id": "t1", "x": 150, "y": 0, "w": 142, "h": 64, "t": "tile 1\nrows 0 to 31\ncolumns 32 to 63", "small": true}, {"id": "t2", "x": 300, "y": 0, "w": 142, "h": 64, "t": "tile 2\nrows 0 to 31\ncolumns 64 to 95", "small": true}, {"id": "t3", "x": 0, "y": 72, "w": 142, "h": 64, "t": "tile 3\nrows 32 to 63\ncolumns 0 to 31", "small": true}, {"id": "t4", "x": 150, "y": 72, "w": 142, "h": 64, "t": "tile 4\nrows 32 to 63\ncolumns 32 to 63", "small": true}, {"id": "t5", "x": 300, "y": 72, "w": 142, "h": 64, "t": "tile 5\nrows 32 to 63\ncolumns 64 to 95", "small": true}], "arrows": [], "narrow": {"boxes": [{"id": "t0", "x": 0, "y": 0, "w": 150, "h": 64, "t": "tile 0\nrows 0 to 31\ncolumns 0 to 31", "small": true}, {"id": "t1", "x": 160, "y": 0, "w": 150, "h": 64, "t": "tile 1\nrows 0 to 31\ncolumns 32 to 63", "small": true}, {"id": "t2", "x": 0, "y": 72, "w": 150, "h": 64, "t": "tile 2\nrows 0 to 31\ncolumns 64 to 95", "small": true}, {"id": "t3", "x": 160, "y": 72, "w": 150, "h": 64, "t": "tile 3\nrows 32 to 63\ncolumns 0 to 31", "small": true}, {"id": "t4", "x": 0, "y": 144, "w": 150, "h": 64, "t": "tile 4\nrows 32 to 63\ncolumns 32 to 63", "small": true}, {"id": "t5", "x": 160, "y": 144, "w": 150, "h": 64, "t": "tile 5\nrows 32 to 63\ncolumns 64 to 95", "small": true}], "arrows": []}}
</script>
<figcaption>A 64 x 96 array as six tiles. They are numbered along the rows first, so tile 3 is the first tile of the second row of tiles.</figcaption>
</figure>

## One line at a time

```{ .python }
--8<-- "tutorial/15_tiles_and_tensors/tiles_and_tensors.py:make"
```

`numbers` is an array of 64 rows and 96 columns: 0, 1, 2 ... The next line copies it to
the card. The print shows what the card has: the same shape, 6 tiles (2 rows of 3 tiles),
and 4096 bytes in a tile: 1024 numbers of 4 bytes.

```{ .python }
--8<-- "tutorial/15_tiles_and_tensors/tiles_and_tensors.py:read"
```

`tensor.read()` copies the tensor back, and the `assert` checks that what came back is what
went in. The second line asks the session for the raw pages, the bytes of each tile as
they lie in DRAM, and prints the first number of each. They are 0, 32 and 64 for the
first row of tiles: tile 1 starts 32 columns to the right of tile 0. The second row of tiles
starts at 3072, which is row 32 times 96 columns.

```{ .python }
--8<-- "tutorial/15_tiles_and_tensors/tiles_and_tensors.py:padding"
```

A 40 x 40 array is not a multiple of 32 on either side. The card keeps it as 64 x 64,
four tiles, with zeros around it.

## The whole program

``` { .python }
--8<-- "tutorial/15_tiles_and_tensors/tiles_and_tensors.py"
```

```{ .text .km-output }
--8<-- "tutorial/15_tiles_and_tensors/output.txt"
```

This program loads no kernels. It runs on the card, because DRAM is on the card, and it
is the first program that works without `km.Program`.

## Common mistakes

- Expecting the tiles in the order of the rows of the array. Tile 1 is the next block to
  the right, not the next 1024 numbers of the first row.
- Forgetting the padding. A tensor made from 40 x 40 numbers reads back as 64 x 64.
- A number that does not fit the type. The array is converted to the dtype: `-1` in a
  `"uint32"` tensor is 4294967295.

Next: [[reading-from-dram]] sends a core to fetch one tile.
