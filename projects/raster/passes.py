"""The two passes over the triangles on the card, modelled in numpy: the mesh as
de-indexed blocks of 32 triangles, each core's contiguous partition of blocks, the
counters COUNT writes, and the middle pass between COUNT and SETUP (cost, items per
core, the list positions every core writes its records to)."""

import numpy as np

from projects.raster import lists
from reference import raster

K = lists.K

LANES = 32
# A block's vectors: x, y, z of corners 0, 1, 2 (slots 0 to 8), then red, green,
# blue of corners 0, 1, 2 (slots 9 to 17).
BLOCK_SLOTS = 18
BLOCK_WORDS = BLOCK_SLOTS * LANES
GROUPS = 16  # 2-row groups of a tile: the grain of a band
SPLIT = 0.33  # a tile above this fraction of a core's mean load is cut into bands
MAX_BANDS = 16


def window_words(slots):
    """Window word of (slot, lane) for `slots` slots of 32 lanes: (slots, 32)."""
    slot = np.arange(slots)[:, None]
    lane = np.arange(LANES)[None, :]
    return 64 * (slot // 2) + slot % 2 + 2 * lane


def mesh_blocks(scene):
    """(blocks, 576) uint32: every triangle's corners, 32 triangles per block, in the
    order of the destination register's window (copied in as they are). The last
    block is padded with triangles whose corners are all 0 (zero area: culled)."""
    indices = scene["indices"]
    triangles = len(indices)
    blocks = -(-triangles // LANES)
    positions = np.zeros((blocks * LANES, 3, 3), dtype=np.float32)
    colours = np.zeros((blocks * LANES, 3, 3), dtype=np.float32)
    positions[:triangles] = scene["positions"][indices]
    colours[:triangles] = scene["colors"][indices]
    # vectors (block, slot, lane): slot 3 k + i is coordinate i of corner k
    vectors = np.concatenate(
        [positions.reshape(blocks, LANES, 9), colours.reshape(blocks, LANES, 9)], axis=2
    ).transpose(0, 2, 1)
    block_words = np.zeros((blocks, BLOCK_WORDS), dtype=np.uint32)
    block_words[:, window_words(BLOCK_SLOTS)] = vectors.view(np.uint32)
    return block_words


def partitions(blocks, cores):
    """First block of every core and the end: (cores + 1,), contiguous, ascending."""
    return (np.arange(cores + 1) * blocks) // cores


def entries(records, triangles, cores):
    """Every (record, tile) entry in record order: its tile, the core whose partition
    holds the record, and its first and last row inside the tile."""
    edge = partitions(-(-triangles // LANES), cores) * LANES
    box = records["box"].astype(np.int64)
    first_tile_x, last_tile_x = box[:, 0] // raster.TILE, box[:, 1] // raster.TILE
    first_tile_y, last_tile_y = box[:, 2] // raster.TILE, box[:, 3] // raster.TILE
    tiles_across, tiles_down = (
        last_tile_x - first_tile_x + 1,
        last_tile_y - first_tile_y + 1,
    )
    per_record = tiles_across * tiles_down
    record = np.repeat(np.arange(len(box)), per_record)
    local = np.arange(per_record.sum()) - np.repeat(
        np.cumsum(per_record) - per_record, per_record
    )
    tile_y = first_tile_y[record] + local // tiles_across[record]
    tile = tile_y * raster.TILES_X + first_tile_x[record] + local % tiles_across[record]
    top_row = raster.TILE * tile_y
    first_row = np.maximum(box[record, 2], top_row) - top_row
    last_row = np.minimum(box[record, 3], top_row + 31) - top_row
    core = np.searchsorted(edge, records["tri"][record], side="right") - 1
    return {
        "record": record,
        "tile": tile,
        "core": core,
        "first_row": first_row,
        "last_row": last_row,
    }


def counters(records, triangles, cores):
    """What COUNT writes per core: entries and rows of each tile (cores, tiles) from
    the records of `raster.setup`, each core counting its partition of the triangles."""
    table = entries(records, triangles, cores)
    cell = table["core"] * lists.TILES + table["tile"]
    size = cores * lists.TILES
    entry_counts = np.bincount(cell, minlength=size).reshape(cores, lists.TILES)
    rows = np.bincount(cell, table["last_row"] - table["first_row"] + 1, minlength=size)
    return entry_counts.astype(np.int64), rows.astype(np.int64).reshape(
        cores, lists.TILES
    )


def histograms(records, triangles, cores):
    """What COUNT keeps per core: for every tile, how many of its entries start, and
    how many end, in each 2-row group (cores, tiles, GROUPS) twice."""
    table = entries(records, triangles, cores)
    base = (table["core"] * lists.TILES + table["tile"]) * GROUPS
    size = cores * lists.TILES * GROUPS
    shape = (cores, lists.TILES, GROUPS)
    starts = np.bincount(base + table["first_row"] // 2, minlength=size).reshape(shape)
    ends = np.bincount(base + table["last_row"] // 2, minlength=size).reshape(shape)
    return starts.astype(np.int64), ends.astype(np.int64)


def cost_of(counts, rows, cost=lists.COST):
    """Estimated cycles of each tile from the counters, as `lists.tile_cost`."""
    tile_entries, tile_rows = counts.sum(axis=0), rows.sum(axis=0)
    busy = cost["tri"] * tile_entries + cost["row"] * tile_rows + cost["tile"]
    return np.where(tile_entries > 0, busy, cost["empty"])


def middle(counts, rows, cores, mode="balanced", bands=None):
    """The middle pass between COUNT and SETUP, from the counters and, for bands,
    `items_of`'s result: items to cores by cost, the list pages with headers and end
    slots, the directory of `lists.build_lists`, each setup core's next slot per item
    (setup cores, items), the items in tile order and the lists' slots."""
    if bands is None:
        items = [(tile, 0, 32) for tile in range(lists.TILES)]
        per_core, cost = counts, cost_of(counts, rows)
    else:
        items, per_core, cost = bands
    total = per_core.sum(axis=0)
    partition = lists.assign(cost, cores, mode)
    header = np.full(len(items), -1, dtype=np.int64)
    directory = np.zeros((cores, K["kDirBytes"] // 4), dtype=np.uint32)
    marks, first_page = [], 0
    for core, own in enumerate(partition):
        slot = 16 * first_page
        item_ids = []
        for item in own:
            tile, first_row, end_row = items[item]
            band = (first_row, end_row) != (0, 32)
            rows_bits = (
                first_row << K["kItemRow0Shift"] | end_row << K["kItemRow1Shift"]
            )
            item_ids.append(tile | (rows_bits if band else 0))
            if total[item] == 0:
                item_ids[-1] |= K["kItemEmpty"]
                continue
            header[item] = slot
            marks.append((slot, K["kMarkTile"], tile, total[item], first_row, end_row))
            slot += 1 + total[item]
        marks.append((slot, K["kMarkEnd"], 0, 0, 0, 0))
        pages = -(-(slot + 1 - 16 * first_page) // lists.PAGE_SLOTS)
        directory[core, K["kDirFirst"]] = first_page
        directory[core, K["kDirPages"]] = pages
        directory[core, K["kDirItems"]] = len(own)
        directory[core, K["kDirIds"] : K["kDirIds"] + len(own)] = item_ids
        first_page += pages
    slots = np.zeros((first_page * lists.PAGE_SLOTS, lists.SLOT), dtype=np.uint32)
    for slot, mark, tile, count, first_row, end_row in marks:
        slots[slot, 0] = mark
        if mark == K["kMarkTile"]:
            slots[slot, K["kHdrTile"]] = tile
            slots[slot, K["kHdrCount"]] = count
            slots[slot, K["kHdrRow0"]] = first_row
            slots[slot, K["kHdrRow1"]] = end_row
    before = np.cumsum(per_core, axis=0) - per_core
    next_slots = np.where(header >= 0, header + 1, 0)[None, :] + before
    return {
        "tiles_of": [[items[item] for item in own] for own in partition],
        "items": items,
        "pages": slots.reshape(-1, lists.PAGE_WORDS),
        "directory": directory,
        "next": next_slots.astype(np.uint32),
        "slots": len(slots),
    }


def item_table(items):
    """SETUP's items page (setup.h): per tile its first item and their number, then
    per item its rows (2,048 uint32)."""
    table = np.zeros(2048, dtype=np.uint32)
    tiles = np.array([tile for tile, _, _ in items])
    first_item = np.searchsorted(tiles, np.arange(lists.TILES))
    counts = np.bincount(tiles, minlength=lists.TILES)
    table[: lists.TILES] = first_item | counts << 16
    rows = table[1024:].view(np.uint16)
    rows[: len(items)] = [first_row | end_row << 8 for _, first_row, end_row in items]
    return table


def heavy_tiles(counts, rows, cores, split=SPLIT):
    """(tile, bands) of every tile to cut: above `split` of a core's mean load, in
    up to MAX_BANDS bands; none when no tile is above the mean."""
    cost = cost_of(counts, rows)
    mean = cost.sum() / cores
    if cost.max() <= mean:
        return []
    bands = np.minimum(MAX_BANDS, np.ceil(cost / (split * mean))).astype(int)
    return [(int(tile), int(bands[tile])) for tile in np.flatnonzero(bands >= 2)]


def cover_of(starts, ends):
    """Entries reaching each 2-row group, from the start and end counts (GROUPS,)."""
    return np.cumsum(starts) - np.concatenate([[0], np.cumsum(ends)[:-1]])


def cuts(starts, ends, bands):
    """Band bounds in rows (even, 0 to 32) for `bands` bands of about equal estimated
    cost, from the tile's start and end counts summed over the cores."""
    weights = (
        2 * lists.COST["row"] * cover_of(starts, ends) + lists.COST["tri"] * starts
    )
    cumulative = np.cumsum(weights).astype(np.int64)
    # the first group whose sum reaches j / bands of the whole, in integers
    groups = (
        np.searchsorted(bands * cumulative, cumulative[-1] * np.arange(1, bands)) + 1
    )
    return np.unique(np.concatenate([[0], 2 * groups, [GROUPS * 2]])).astype(int)


def band_count(starts, ends, first_row, end_row):
    """Entries reaching rows [first_row, end_row): starting before end_row less ending
    before first_row (starts, ends: (..., GROUPS))."""
    return starts[..., : end_row // 2].sum(-1) - ends[..., : first_row // 2].sum(-1)


def items_of(counts, rows, heavy, starts, ends):
    """Items in tile order (tile, first_row, end_row), with per item the entries of
    every setup core (cores, items) and the estimated cost. starts, ends: (cores,
    len(heavy), GROUPS), the heavy tiles' counts."""
    per_core, cost, items = [], [], []
    tile_cost = cost_of(counts, rows)
    heavy_index = {tile: position for position, (tile, _) in enumerate(heavy)}
    for tile in range(lists.TILES):
        if tile not in heavy_index:
            items.append((tile, 0, 32))
            per_core.append(counts[:, tile])
            cost.append(tile_cost[tile])
            continue
        position = heavy_index[tile]
        tile_starts, tile_ends = starts[:, position], ends[:, position]
        bounds = cuts(tile_starts.sum(0), tile_ends.sum(0), heavy[position][1])
        cover = cover_of(tile_starts.sum(0), tile_ends.sum(0))
        for first_row, end_row in zip(bounds[:-1], bounds[1:]):
            band_entries = band_count(tile_starts, tile_ends, first_row, end_row)
            items.append((tile, int(first_row), int(end_row)))
            per_core.append(band_entries)
            estimated_rows = int(2 * cover[first_row // 2 : end_row // 2].sum())
            busy = (
                lists.COST["tri"] * band_entries.sum()
                + lists.COST["row"] * estimated_rows
            )
            busy += lists.COST["tile"]
            cost.append(busy if band_entries.sum() > 0 else lists.COST["empty"])
    return items, np.stack(per_core, axis=1), np.asarray(cost, dtype=np.int64)
