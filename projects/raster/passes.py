"""The two passes over the triangles on the card, modelled in numpy: the mesh as
de-indexed blocks of 32 triangles, each core's contiguous share of blocks, the
counters COUNT writes, and the step between COUNT and SETUP (cost, items per core,
the list positions every core writes its records to)."""

import numpy as np

from projects.raster import lists
from ref import raster

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
    s = np.arange(slots)[:, None]
    lane = np.arange(LANES)[None, :]
    return 64 * (s // 2) + s % 2 + 2 * lane


def mesh_blocks(scene):
    """(blocks, 576) uint32: every triangle's corners, 32 triangles per block, in the
    order of the destination register's window (copied in as they are). The last
    block is padded with triangles whose corners are all 0 (zero area: culled)."""
    idx = scene["indices"]
    n = len(idx)
    blocks = -(-n // LANES)
    pos = np.zeros((blocks * LANES, 3, 3), dtype=np.float32)
    col = np.zeros((blocks * LANES, 3, 3), dtype=np.float32)
    pos[:n] = scene["positions"][idx]
    col[:n] = scene["colors"][idx]
    # vectors (block, slot, lane): slot 3 k + i is coordinate i of corner k
    vec = np.concatenate(
        [pos.reshape(blocks, LANES, 9), col.reshape(blocks, LANES, 9)], axis=2
    ).transpose(0, 2, 1)
    out = np.zeros((blocks, BLOCK_WORDS), dtype=np.uint32)
    out[:, window_words(BLOCK_SLOTS)] = vec.view(np.uint32)
    return out


def shares(blocks, cores):
    """First block of every core and the end: (cores + 1,), contiguous, ascending."""
    return (np.arange(cores + 1) * blocks) // cores


def entries(recs, triangles, cores):
    """Every (record, tile) entry in record order: its tile, the core whose share
    holds the record, and its first and last row inside the tile."""
    edge = shares(-(-triangles // LANES), cores) * LANES
    box = recs["box"].astype(np.int64)
    tx0, tx1 = box[:, 0] // raster.TILE, box[:, 1] // raster.TILE
    ty0, ty1 = box[:, 2] // raster.TILE, box[:, 3] // raster.TILE
    nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
    per = nx * ny
    rec = np.repeat(np.arange(len(box)), per)
    local = np.arange(per.sum()) - np.repeat(np.cumsum(per) - per, per)
    ty = ty0[rec] + local // nx[rec]
    tile = ty * raster.TILES_X + tx0[rec] + local % nx[rec]
    top = raster.TILE * ty
    r0 = np.maximum(box[rec, 2], top) - top
    r1 = np.minimum(box[rec, 3], top + 31) - top
    core = np.searchsorted(edge, recs["tri"][rec], side="right") - 1
    return {"rec": rec, "tile": tile, "core": core, "r0": r0, "r1": r1}


def counters(recs, triangles, cores):
    """What COUNT writes per core: entries and rows of each tile (cores, tiles) from
    the records of `raster.setup`, each core counting its share of the triangles."""
    e = entries(recs, triangles, cores)
    at = e["core"] * lists.TILES + e["tile"]
    size = cores * lists.TILES
    cnt = np.bincount(at, minlength=size).reshape(cores, lists.TILES)
    rows = np.bincount(at, e["r1"] - e["r0"] + 1, minlength=size)
    return cnt.astype(np.int64), rows.astype(np.int64).reshape(cores, lists.TILES)


def histograms(recs, triangles, cores):
    """What COUNT keeps per core: for every tile, how many of its entries start, and
    how many end, in each 2-row group (cores, tiles, GROUPS) twice."""
    e = entries(recs, triangles, cores)
    base = (e["core"] * lists.TILES + e["tile"]) * GROUPS
    size = cores * lists.TILES * GROUPS
    shape = (cores, lists.TILES, GROUPS)
    first = np.bincount(base + e["r0"] // 2, minlength=size).reshape(shape)
    last = np.bincount(base + e["r1"] // 2, minlength=size).reshape(shape)
    return first.astype(np.int64), last.astype(np.int64)


def cost_of(cnt, rows, cost=lists.COST):
    """Estimated cycles of each tile from the counters, as `lists.tile_cost`."""
    n, r = cnt.sum(axis=0), rows.sum(axis=0)
    busy = cost["tri"] * n + cost["row"] * r + cost["tile"]
    return np.where(n > 0, busy, cost["empty"])


def middle(cnt, rows, cores, mode="balanced", bands=None):
    """The step between COUNT and SETUP, from the counters and, for bands, `items_of`'s
    result: items to cores by cost, the list pages with headers and end slots, the
    directory of `lists.build_lists`, each setup core's next slot per item (setup
    cores, items), the items in tile order and the lists' slots."""
    if bands is None:
        items = [(t, 0, 32) for t in range(lists.TILES)]
        per_core, cost = cnt, cost_of(cnt, rows)
    else:
        items, per_core, cost = bands
    total = per_core.sum(axis=0)
    part = lists.assign(cost, cores, mode)
    header = np.full(len(items), -1, dtype=np.int64)
    directory = np.zeros((cores, K["kDirBytes"] // 4), dtype=np.uint32)
    marks, first = [], 0
    for core, its in enumerate(part):
        at = 16 * first
        ids = []
        for i in its:
            t, r0, r1 = items[i]
            band = (r0, r1) != (0, 32)
            rws = r0 << K["kItemRow0Shift"] | r1 << K["kItemRow1Shift"]
            ids.append(t | (rws if band else 0))
            if total[i] == 0:
                ids[-1] |= K["kItemEmpty"]
                continue
            header[i] = at
            marks.append((at, K["kMarkTile"], t, total[i], r0, r1))
            at += 1 + total[i]
        marks.append((at, K["kMarkEnd"], 0, 0, 0, 0))
        pages = -(-(at + 1 - 16 * first) // lists.PAGE_SLOTS)
        directory[core, K["kDirFirst"]] = first
        directory[core, K["kDirPages"]] = pages
        directory[core, K["kDirItems"]] = len(its)
        directory[core, K["kDirIds"] : K["kDirIds"] + len(its)] = ids
        first += pages
    slots = np.zeros((first * lists.PAGE_SLOTS, lists.SLOT), dtype=np.uint32)
    for at, mark, t, n, r0, r1 in marks:
        slots[at, 0] = mark
        if mark == K["kMarkTile"]:
            slots[at, K["kHdrTile"]] = t
            slots[at, K["kHdrCount"]] = n
            slots[at, K["kHdrRow0"]] = r0
            slots[at, K["kHdrRow1"]] = r1
    before = np.cumsum(per_core, axis=0) - per_core
    nxt = np.where(header >= 0, header + 1, 0)[None, :] + before
    return {
        "tiles_of": [[items[i] for i in its] for its in part],
        "items": items,
        "pages": slots.reshape(-1, lists.PAGE_WORDS),
        "directory": directory,
        "next": nxt.astype(np.uint32),
        "slots": len(slots),
    }


def item_table(items):
    """SETUP's items page (setup.h): per tile its first item and their number, then
    per item its rows (2,048 uint32)."""
    table = np.zeros(2048, dtype=np.uint32)
    tiles = np.array([t for t, _, _ in items])
    first = np.searchsorted(tiles, np.arange(lists.TILES))
    n = np.bincount(tiles, minlength=lists.TILES)
    table[: lists.TILES] = first | n << 16
    rows = table[1024:].view(np.uint16)
    rows[: len(items)] = [r0 | r1 << 8 for _, r0, r1 in items]
    return table


def heavy_tiles(cnt, rows, cores, split=SPLIT):
    """(tile, bands) of every tile to cut: above `split` of a core's mean load, in
    up to MAX_BANDS bands; none when no tile is above the mean."""
    cost = cost_of(cnt, rows)
    mean = cost.sum() / cores
    if cost.max() <= mean:
        return []
    k = np.minimum(MAX_BANDS, np.ceil(cost / (split * mean))).astype(int)
    return [(int(t), int(k[t])) for t in np.flatnonzero(k >= 2)]


def cover_of(first, last):
    """Entries reaching each 2-row group, from the start and end counts (GROUPS,)."""
    return np.cumsum(first) - np.concatenate([[0], np.cumsum(last)[:-1]])


def cuts(first, last, k):
    """Band bounds in rows (even, 0 to 32) for k bands of about equal estimated cost,
    from the tile's start and end counts summed over the cores."""
    w = 2 * lists.COST["row"] * cover_of(first, last) + lists.COST["tri"] * first
    cum = np.cumsum(w).astype(np.int64)
    # the first group whose sum reaches j / k of the whole, in integers
    at = np.searchsorted(k * cum, cum[-1] * np.arange(1, k)) + 1
    return np.unique(np.concatenate([[0], 2 * at, [GROUPS * 2]])).astype(int)


def band_count(first, last, row0, row1):
    """Entries reaching rows [row0, row1): starting before row1 less ending before
    row0 (first, last: (..., GROUPS))."""
    return first[..., : row1 // 2].sum(-1) - last[..., : row0 // 2].sum(-1)


def items_of(cnt, rows, heavy, first, last):
    """Items in tile order (tile, row0, row1), with per item the entries of every
    setup core (cores, items) and the estimated cost. first, last: (cores, len(heavy),
    GROUPS), the heavy tiles' counts."""
    per_core, cost, items = [], [], []
    tile_cost = cost_of(cnt, rows)
    cut = {t: i for i, (t, _) in enumerate(heavy)}
    for t in range(lists.TILES):
        if t not in cut:
            items.append((t, 0, 32))
            per_core.append(cnt[:, t])
            cost.append(tile_cost[t])
            continue
        i = cut[t]
        f, la = first[:, i], last[:, i]
        bounds = cuts(f.sum(0), la.sum(0), heavy[i][1])
        cover = cover_of(f.sum(0), la.sum(0))
        for r0, r1 in zip(bounds[:-1], bounds[1:]):
            n = band_count(f, la, r0, r1)
            items.append((t, int(r0), int(r1)))
            per_core.append(n)
            est = int(2 * cover[r0 // 2 : r1 // 2].sum())
            busy = lists.COST["tri"] * n.sum() + lists.COST["row"] * est
            busy += lists.COST["tile"]
            cost.append(busy if n.sum() > 0 else lists.COST["empty"])
    return items, np.stack(per_core, axis=1), np.asarray(cost, dtype=np.int64)
