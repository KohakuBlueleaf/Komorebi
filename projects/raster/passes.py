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


def counters(recs, triangles, cores):
    """What COUNT writes per core: entries and rows of each tile (cores, tiles) from
    the records of `raster.setup`, each core counting its share of the triangles."""
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
    rows = np.minimum(box[rec, 3], top + 31) - np.maximum(box[rec, 2], top) + 1
    core = np.searchsorted(edge, recs["tri"][rec], side="right") - 1
    at = core * lists.TILES + tile
    size = cores * lists.TILES
    cnt = np.bincount(at, minlength=size).reshape(cores, lists.TILES)
    rws = np.bincount(at, rows, minlength=size).astype(np.int64)
    return cnt.astype(np.int64), rws.reshape(cores, lists.TILES)


def cost_of(cnt, rows, cost=lists.COST):
    """Estimated cycles of each tile from the counters, as `lists.tile_cost`."""
    n, r = cnt.sum(axis=0), rows.sum(axis=0)
    busy = cost["tri"] * n + cost["row"] * r + cost["tile"]
    return np.where(n > 0, busy, cost["empty"])


def middle(cnt, rows, cores, mode="balanced"):
    """The step between COUNT and SETUP, from the counters (setup cores, tiles):
    tiles per core by cost (no bands); the list pages with every header and end slot
    in place (records still 0), the directory of `lists.build_lists`, each setup
    core's next slot per tile (setup cores, tiles) and the list slots in all."""
    total = cnt.sum(axis=0)
    tiles_of = lists.assign(cost_of(cnt, rows), cores, mode)
    header = np.full(lists.TILES, -1, dtype=np.int64)
    directory = np.zeros((cores, K["kDirBytes"] // 4), dtype=np.uint32)
    marks, first = [], 0
    for core, tiles in enumerate(tiles_of):
        at = 16 * first
        ids = []
        for t in tiles:
            if total[t] == 0:
                ids.append(t | K["kItemEmpty"])
                continue
            ids.append(t)
            header[t] = at
            marks.append((at, K["kMarkTile"], t, total[t]))
            at += 1 + total[t]
        marks.append((at, K["kMarkEnd"], 0, 0))
        pages = -(-(at + 1 - 16 * first) // lists.PAGE_SLOTS)
        directory[core, K["kDirFirst"]] = first
        directory[core, K["kDirPages"]] = pages
        directory[core, K["kDirItems"]] = len(tiles)
        directory[core, K["kDirIds"] : K["kDirIds"] + len(tiles)] = ids
        first += pages
    slots = np.zeros((first * lists.PAGE_SLOTS, lists.SLOT), dtype=np.uint32)
    for at, mark, t, n in marks:
        slots[at, 0] = mark
        if mark == K["kMarkTile"]:
            slots[at, K["kHdrTile"]] = t
            slots[at, K["kHdrCount"]] = n
            slots[at, K["kHdrRow1"]] = 32
    before = np.cumsum(cnt, axis=0) - cnt
    nxt = np.where(header >= 0, header + 1, 0)[None, :] + before
    return {
        "tiles_of": tiles_of,
        "pages": slots.reshape(-1, lists.PAGE_WORDS),
        "directory": directory,
        "next": nxt.astype(np.uint32),
        "slots": len(slots),
    }
