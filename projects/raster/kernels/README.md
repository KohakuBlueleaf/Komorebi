# The passes' commands

The command words of the per-triangle passes (`setup.h`, handlers from 11; the fill's
handlers are in `raster.h`). Every pass runs on the `a5` cores, each over its partition of
the mesh blocks.

| handler | words |
|---|---|
| LANES (11) | `a0` mesh, `a1` output (a page of 32 records per block), `a2` blocks, `a3` camera, `a4` and `a6` the window of blocks written (first, count: block `b` goes to page `b - a4`). The records stay as the vector unit leaves them (words 24 to 31 raw, see `kRawX0`). |
| COUNT (12) | `a0` mesh, `a1` counters (a tensor, a page of `kCountBytes` per core), `a2` blocks, `a3` camera. With `kRanges` in `a2` (and `a8`), COUNT and SETUP: each core's blocks are `[b0, b1)` at `kRangesAt + 64 x core` of the camera's region, and its writer leaves the pass's ticks at `kCostsAt + 128 x core` (COUNT), `+ 64` (SETUP). |
| SETUP (13) | `a0` mesh, `a1` lists (2 KB pages of 16 slots), `a2` blocks, `a3` camera, `a4` slots (a tensor, a page of `kSlotsBytes` per core: the list slot of each item's next record), `a6` the lists' slots (a slot past them is not written but gives up), `a7` the items (one page of `kItemsBytes`: per tile its first item and their number, `first \| n << 16`; from byte `kItemRows`, each item's rows, `row0 \| row1 << 8`). With `kSetupPrefix` in `a6` (and `a8`), each core's reader makes its slots page itself: `a4` holds SUM's entries before the core per tile, the items region after its page the items' base words and the address of HSUM's pages (`kBaseAt`, `kHprefixAddrAt`). |
| HIST (14, writer only) | `a0` the heavy tiles (one page: their number, then the tiles), `a1` out (a tensor, a page of `a4` bytes per core), `a4` at least 64 bytes per heavy tile. |
| PLACE (15, writer only) | `a0` the items region (the middle pass's words from `kFrameWordsAt`), `a1` the directory tensor, `a6` fill cores, `a7` the lists: each fill core writes its directory page and its list's end slot, every core the headers of every `a5`-th item from its own. Needs `a8`. |
| SUM (16, writer only) | `a1` counters, `a2` out (a tensor, a page of `kSlotsBytes` per core: the entries of the cores before it per tile), `a3` totals (entries, then rows per tile), `a6` counting cores; 16 tiles a slice, slices across the `a5` cores. Needs `a8`. |
| HSUM (17, writer only) | `a0` heavy tiles, `a1` HIST's histograms (pages of `a4` bytes), `a2` out (a tensor, a page of `kHprefixPageBytes` per core: per heavy tile the cores' before it, 16 starts then 16 ends, each summed over the groups up to its own), `a3` totals (from `kSumsHeavyAt`, the 32 counts per heavy tile over the cores), `a6` counting cores; heavy tiles across the `a5` cores. Needs `a8`. |

`a8`, in every handler: 0, or 1 + the DRAM bank of the X280 thread's regions. When it
is not 0, the camera, items, heavy tiles and totals are addresses in that bank (pages
contiguous). The pages only the cores read and write (counters, slots, histograms,
HSUM's pages), the mesh, the lists and the directory are always interleaved tensors:
every core reading one bank at once is what that bank gives, 3 to 4 times slower than
the pages spread over all of them (SUM: 247 against 59 us on 120 cores).

SETUP with `kSetupPrefix`: the items region holds after its page (`kBaseAt`) each item's
base word, its first record's slot (0 when it has none) and, from bit `kBaseSlotBits`,
1 + its tile's index among the heavy tiles (0 for a tile not cut); then (`kHprefixAddrAt`)
the address of HSUM's pages. A core's next slot of item `i` is the base, plus for a whole
tile SUM's entries of the cores before it, plus for a heavy tile's rows `[row0, row1)`
HSUM's starts before `row1` less its ends before `row0` (both of the cores before it).

PLACE's words in the items region, each part on 64 bytes: from `kFrameWordsAt` the
directory's frame words (16: pages per frame tensor, then their addresses); from
`kItemWordsAt` per item its fill core `| entries << 8` and its header's slot (0 without
entries); from `kCoreWordsAt` per fill core its first list page, pages, items and end
slot. A core's directory lists its items in item (tile) order, an item without entries
marked `kItemEmpty`.
