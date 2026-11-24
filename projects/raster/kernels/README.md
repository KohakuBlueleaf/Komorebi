# The passes' commands

The command words of the per-triangle passes (`setup.h`, handlers from 11; the fill's
handlers are in `raster.h`). Every pass runs on the `cores` (`a5`), each over its partition
of the mesh blocks.

| handler | words |
|---|---|
| LANES (11) | `a0` `mesh_address`, `a1` `output_address` (a page of 32 records per block), `a2` `blocks_word`, `a3` `camera_address`, `a4` `first_written` and `a6` `written_blocks`: the window of blocks written (block `i` goes to page `i - first_written`). The records stay as the vector unit leaves them (words 24 to 31 raw, see `kRawX0`). |
| COUNT (12) | `a0` `mesh_address`, `a1` `counters_address` (a tensor, a page of `kCountBytes` per core), `a2` `blocks_word`, `a3` `camera_address`. With `kRanges` in `blocks_word` (and a `bank`), COUNT and SETUP: each core's blocks are `[first_block, end_block)` at `kRangesAt + 64 x core` of the camera's region, and its writer leaves the pass's ticks at `kCostsAt + 128 x core` (COUNT), `+ 64` (SETUP). |
| SETUP (13) | `a0` `mesh_address`, `a1` `lists_address` (2 KB pages of 16 slots), `a2` `blocks_word`, `a3` `camera_address`, `a4` `slots_address` (a tensor, a page of `kSlotsBytes` per core: the list slot of each item's next record), `a6` `list_slots` (a slot past them is not written but gives up), `a7` `items_address` (one page of `kItemsBytes`: per tile its first item and their number, `first \| n << 16`; from byte `kItemRows`, each item's rows, `first_row \| end_row << 8`). With `kSetupPrefix` in `list_slots` (and a `bank`), each core's reader makes its slots page itself: `slots_address` holds SUM's entries before the core per tile, the items region after its page the items' base words and the address of HSUM's pages (`kBaseAt`, `kHprefixAddrAt`). |
| HIST (14, writer only) | `a0` `heavy_address` (the heavy tiles: one page, their number, then the tiles), `a1` `histograms_address` (a tensor, a page of `histogram_page_bytes` per core), `a4` `histogram_page_bytes`, at least 64 bytes per heavy tile. |
| PLACE (15, writer only) | `a0` `items_region_address` (the middle pass's words from `kFrameWordsAt`), `a1` `directory_address` (the directory tensor), `a6` `fill_cores`, `a7` `lists_address`: each fill core writes its directory page and its list's end slot, every core the headers of every `cores`-th item from its own. Needs a `bank`. |
| SUM (16, writer only) | `a1` `counters_address`, `a2` `prefixes_address` (a tensor, a page of `kSlotsBytes` per core: the entries of the cores before it per tile), `a3` `totals_address` (laid for the frame's `T` tiles in whole slices: entries per tile, then rows from byte `4 T`), `a6` `counting_cores`; 16 tiles a slice, slices across the `cores`. Needs a `bank`. |
| HSUM (17, writer only) | `a0` `heavy_tiles`, `a1` `histograms_address` (HIST's histograms, pages of `histogram_page_bytes`), `a4` `histogram_page_bytes`, `a2` `prefixes_address` (a tensor, a page of `kHprefixPageBytes` per core: per heavy tile the cores' before it, 16 starts then 16 ends, each summed over the groups up to its own), `a3` `totals_address` (from byte `8 T`, after SUM's rows; at most `kSumsHeavyAt`: the 32 counts per heavy tile over the cores), `a6` `counting_cores`; heavy tiles across the `cores`. Needs a `bank`. |

`a8` in every handler (`grid.h`), but SHOW's (whose `a8` is its block's word, so SHOW
takes no grid: a frame's tile is its page): bits 0 to 3 the `bank`; bits 4 to 17 the
frame's tiles a row `TX`, bits 18 to 31 its rows of tiles `TY`, 0 for 32 (1,024 x 1,024).
A tile is `ty TX + tx`; COUNT, SETUP and LANES clip the box to the padded frame
`32 TX x 32 TY` and number tiles so.

No number of tiles is written in the kernels. Every per-tile table is sized from the
capacity `RASTER_TILES` (`kTileCapacity`, `kTiles` in `setup.h`), a multiple of 32 the
load chooses from the sizes it renders (`card_thread.capacity`); a grid of at most that
many tiles renders, tiles past the grid count 0, and a larger grid is loaded again with a
larger capacity. Items are `kTiles + kBandItems` (the bands besides the whole tiles);
the heavy tiles' tables (`kHeavyMax`, HSUM's 128-byte rows, `kHprefixPageBytes`) and
`kBandItems` are `RASTER_HEAVY` and `RASTER_BAND_ITEMS`, from the load's fill cores
(`card.heavy_capacity`: a frame has fewer than cores / 0.33 of either).
What a tile costs a core's L1 (`card_thread.l1_tile_bytes`): COUNT's counters 8 B, the
2-row histograms 64 B, the slots 4 B an item, the items page 4 B and 2 B an item; the
histogram buffer is also PLACE's copy of the items region (4 B a tile, 14 B an item),
whichever is larger. The 16-bit fields that number tiles (an item id's tile, PLACE's
`tile_of`, the items page's `first | n << 16`) hold far more tiles than a core's L1
does (`raster.h` asserts the capacity fits them).

Two forms of the box words. Without `RASTER_WIDE` (a narrow build, today's words): the
box word is tile columns in 6 bits each and pixel rows in 10, and a record's box columns
(words 24, 25) are float16, exact to 2,048, so `TX` is at most 64 and `TY` at most 32
(`kNarrowAcross`, `kNarrowDown`). With `RASTER_WIDE` (the load's choice for a wider
grid): COUNT's box is three words (tile columns 16 bits each, the first row, the last
row; 341 boxes a page), SETUP's writer takes the box from the record (words 24, 25 the
columns as float32, 26, 27 the rows), and FILL loads the float32 columns with two
immediates each.

`RASTER_RECOUNT` (the load's choice when the full histograms do not fit L1): COUNT with
`a4` 0 counts and keeps no histograms; COUNT with `a4` the heavy tiles' page (the
thread's, after the middle pass picks them) walks the same blocks again and counts the
2-row histograms of those tiles alone, through a map of 1 + heavy index a tile in
`kCbCount`'s space, into `kHeavyMax` slots of `kCbHist`; it sends no counters or costs.
HIST then sends heavy tile `i` from slot `i`.

`bank` (`a8` bits 0 to 3), in every handler: 0, or 1 + the DRAM bank of the X280 thread's regions. When it
is not 0, the camera, items, heavy tiles and totals are addresses in that bank (pages
contiguous). The pages only the cores read and write (counters, slots, histograms,
HSUM's pages), the mesh, the lists and the directory are always interleaved tensors:
every core reading one bank at once is what that bank gives, 3 to 4 times slower than
the pages spread over all of them (SUM: 247 against 59 us on 120 cores).

SETUP with `kSetupPrefix`: the items region holds after its page (`kBaseAt`) each item's
base word, its first record's slot (0 when it has none) and, from bit `kBaseSlotBits`,
1 + its tile's index among the heavy tiles (0 for a tile not cut); then (`kHprefixAddrAt`)
the address of HSUM's pages. A core's next slot of an item is the base, plus for a whole
tile SUM's entries of the cores before it, plus for a heavy tile's rows `[first_row, end_row)`
HSUM's starts before `end_row` less its ends before `first_row` (both of the cores before it).

PLACE's words in the items region, each part on 64 bytes: from `kFrameWordsAt` the
directory's frame words (16: pages per frame tensor, then their addresses); from
`kItemWordsAt` per item its fill core `| entries << 8` and its header's slot (0 without
entries); from `kCoreWordsAt` per fill core its first list page, pages, items and end
slot. A core's directory lists its items in item (tile) order, an item without entries
marked `kItemEmpty`.
