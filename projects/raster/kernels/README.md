# The passes' records

The record words of the per-triangle passes (`setup.h`, segments from 11; the fill's
segments are in `raster.h`). Every pass runs on the `a5` cores, each over its share of
the mesh blocks.

| segment | words |
|---|---|
| LANES (11) | `a0` mesh, `a1` output (a page of 32 records per block), `a2` blocks, `a3` camera, `a4` and `a6` the window of blocks written (first, count: block `b` goes to page `b - a4`). The records stay as the vector unit leaves them (words 24 to 31 raw, see `kRawX0`). |
| COUNT (12) | `a0` mesh, `a1` counters (a page of `kCountBytes` per core), `a2` blocks, `a3` camera. |
| SETUP (13) | `a0` mesh, `a1` lists (2 KB pages of 16 slots), `a2` blocks, `a3` camera, `a4` slots (a page of `kSlotsBytes` per core: the list slot of each item's next record), `a6` the lists' slots (a slot past them is not written but gives up), `a7` the items (one page of `kItemsBytes`: per tile its first item and their number, `first \| n << 16`; from byte `kItemRows`, each item's rows, `row0 \| row1 << 8`). |
| HIST (14, writer only) | `a0` the heavy tiles (one page: their number, then the tiles), `a1` out (a page of `a4` bytes per core), `a4` at least 64 bytes per heavy tile. |
| PLACE (15, writer only) | `a0` the directory pages (4 KB per fill core), `a1` the directory tensor, `a2` the marks (`a4` of them, `kMarkBytes` each), `a6` fill cores, `a7` the lists. Needs `a8`. |

`a8`, in every segment: 0, or 1 + the DRAM bank of the X280 thread's regions. When it
is not 0, the camera, counters, slots, items, heavy tiles, histograms, directory pages
and marks are addresses in that bank (pages contiguous); the mesh and the lists stay
tensors.
