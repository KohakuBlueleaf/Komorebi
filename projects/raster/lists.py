"""Host side of the host-binned rasterizer: records, tile lists per core, decoding.

Pure numpy, no card: the layouts are those of `kernels/raster.h`, whose constants are
read from the header so the two cannot disagree.
"""

import re
from pathlib import Path

import numpy as np

from ref import raster

HERE = Path(__file__).resolve().parent
RASTER_H = HERE / "kernels" / "raster.h"
CONSTEXPR = re.compile(r"constexpr\s+uint32_t\s+(k\w+)\s*=\s*(0x[0-9A-Fa-f]+|\d+)\s*;")
K = {name: int(value, 0) for name, value in CONSTEXPR.findall(RASTER_H.read_text())}

TILES = raster.TILES_X * raster.TILES_Y
SLOT = K["kSlotWords"]
PAGE_SLOTS = K["kSlotsPerPage"]
PAGE_WORDS = SLOT * PAGE_SLOTS
NEG_ZERO_NEIGHBOUR = 0x80000001


def bits(x):
    """float32 values as uint32 bit patterns."""
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def half_bits(x):
    """Integers below 2048 as float16 bit patterns (exact), in uint32."""
    x = np.asarray(x)
    if x.min(initial=0) < 0 or x.max(initial=0) >= 2048:
        raise ValueError("a box column is not exact in float16")
    return x.astype(np.float16).view(np.uint16).astype(np.uint32)


def pack_records(rec):
    """The 32-word card record of every triangle record of `raster.setup`."""
    n = len(rec["tri"])
    out = np.zeros((n, SLOT), dtype=np.uint32)
    out[:, K["kRecA"] : K["kRecA"] + 3] = bits(rec["a"])
    out[:, K["kRecB"] : K["kRecB"] + 3] = bits(rec["b"])
    out[:, K["kRecC"] : K["kRecC"] + 3] = bits(rec["c"])
    out[:, K["kRecT"] : K["kRecT"] + 3] = np.where(
        rec["top_left"], NEG_ZERO_NEIGHBOUR, 0
    )
    out[:, K["kRecW"] : K["kRecW"] + 3] = bits(rec["w_planes"])
    cp = rec["c_planes"]  # (n, coefficient, channel)
    for ch in range(3):
        col = K["kRecCol"] + 3 * ch
        out[:, col : col + 3] = bits(cp[:, :, ch])
    box = rec["box"].astype(np.int64)
    out[:, K["kRecX0"]] = half_bits(box[:, 0])
    out[:, K["kRecX1"]] = half_bits(box[:, 1] + 1)
    out[:, K["kRecY0"]] = box[:, 2].astype(np.uint32)
    out[:, K["kRecY1"]] = box[:, 3].astype(np.uint32)
    out[:, K["kRecTri"]] = bits(rec["tri"].astype(np.float32))
    out[:, K["kRecIndex"]] = np.arange(n, dtype=np.uint32)
    return out


# Measured pack-thread cycles: per (triangle, tile), per row, per tile with triangles;
# an empty tile costs only its writer.
COST = {"tri": 260, "row": 60, "tile": 6800, "empty": 2000}


def tile_cost(cards, offsets, members, cost=COST):
    """Estimated cycles of each tile: its triangles, their rows in it and the tile."""
    n = np.diff(offsets)
    tile = np.repeat(np.arange(len(n)), n)
    top = 32 * (tile // raster.TILES_X)
    y0 = cards[members, K["kRecY0"]].astype(np.int64)
    y1 = cards[members, K["kRecY1"]].astype(np.int64)
    rows = np.minimum(y1, top + 31) - np.maximum(y0, top) + 1
    per_tile = np.bincount(tile, rows, minlength=len(n))
    busy = cost["tri"] * n + cost["row"] * per_tile + cost["tile"]
    return np.where(n > 0, busy, cost["empty"])


def as_item(x):
    """(tile, row0, row1): a tile number alone is the whole tile."""
    if isinstance(x, tuple):
        return x
    return (int(x), 0, 32)


def band_ranks(cards, ranks_t, tile, row0, row1):
    """The members of a tile whose rows reach into its rows [row0, row1)."""
    top = 32 * (tile // raster.TILES_X)
    y0 = cards[ranks_t, K["kRecY0"]].astype(np.int64)
    y1 = cards[ranks_t, K["kRecY1"]].astype(np.int64)
    return ranks_t[(y0 <= top + row1 - 1) & (y1 >= top + row0)]


def band_cost(cards, ranks_t, tile, row0, row1, cost=COST):
    """Estimated cycles of a band of a tile, as `tile_cost` counts a tile."""
    top = 32 * (tile // raster.TILES_X)
    m = band_ranks(cards, ranks_t, tile, row0, row1)
    if len(m) == 0:
        return cost["empty"]
    y0 = cards[m, K["kRecY0"]].astype(np.int64)
    y1 = cards[m, K["kRecY1"]].astype(np.int64)
    rows = np.minimum(y1, top + row1 - 1) - np.maximum(y0, top + row0) + 1
    return cost["tri"] * len(m) + cost["row"] * int(rows.sum()) + cost["tile"]


def split_tiles(cards, offsets, members, tiles, cost, target, max_bands=16):
    """(items, their costs): each tile of `tiles` whole, or, when its cost is above
    `target`, as bands of even rows of about equal estimated cost."""
    items, costs = [], []
    for t in tiles:
        k = min(max_bands, int(np.ceil(cost[t] / target)))
        if k < 2:
            items.append((t, 0, 32))
            costs.append(cost[t])
            continue
        m = members[offsets[t] : offsets[t + 1]]
        top = 32 * (t // raster.TILES_X)
        y0 = np.clip(cards[m, K["kRecY0"]].astype(np.int64) - top, 0, 31)
        y1 = np.clip(cards[m, K["kRecY1"]].astype(np.int64) - top, 0, 31)
        # Row weights: each row's coverage, a triangle's setup on its first row.
        starts = np.bincount(y0, minlength=33)
        cover = np.cumsum(starts - np.bincount(y1 + 1, minlength=33))[:32]
        w = COST["row"] * cover + COST["tri"] * starts[:32]
        cum = np.cumsum(w.reshape(16, 2).sum(axis=1))
        cuts = np.searchsorted(cum, cum[-1] * np.arange(1, k) / k) + 1
        bounds = np.unique(np.concatenate([[0], 2 * cuts, [32]]))
        for row0, row1 in zip(bounds[:-1], bounds[1:]):
            items.append((t, int(row0), int(row1)))
            costs.append(band_cost(cards, m, t, int(row0), int(row1)))
    return items, np.asarray(costs, dtype=np.float64)


def assign(cost, cores, mode="balanced"):
    """Tiles per core: round-robin over the tile numbers, or balanced by the tiles'
    costs (largest first to the least loaded core)."""
    cost = np.asarray(cost, dtype=np.float64).reshape(-1)
    if mode == "round-robin":
        return [list(range(c, len(cost), cores)) for c in range(cores)]
    if mode != "balanced":
        raise ValueError(f"unknown assignment {mode}")
    load = np.zeros(cores)
    out = [[] for _ in range(cores)]
    for t in np.argsort(-cost, kind="stable"):
        c = int(np.argmin(load))
        out[c].append(int(t))
        load[c] += cost[t]
    return [sorted(tiles) for tiles in out]


def build_lists(cards, offsets, members, tiles_of):
    """List pages of all cores and one directory page per core, from packed records,
    `raster.bin_triangles`' offsets and members, and the items per core (tiles or
    bands, `as_item`): (pages (P, 512) uint32, directory (cores, 1024) uint32)."""
    cores = len(tiles_of)
    if max(len(t) for t in tiles_of) > K["kDirMaxItems"]:
        raise ValueError("a core has more tiles than its directory page holds")
    streams, first = [], 0
    directory = np.zeros((cores, K["kDirBytes"] // 4), dtype=np.uint32)
    for core, tiles in enumerate(tiles_of):
        parts, ids = [], []
        for item in tiles:
            t, row0, row1 = as_item(item)
            ranks_t = members[offsets[t] : offsets[t + 1]]
            band = (row0, row1) != (0, 32)
            if band:
                ranks_t = band_ranks(cards, ranks_t, t, row0, row1)
            rows = row0 << K["kItemRow0Shift"] | row1 << K["kItemRow1Shift"]
            ids.append(t | (rows if band else 0))
            if len(ranks_t) == 0:
                ids[-1] |= K["kItemEmpty"]
                continue  # the writer stores empty tiles by itself
            header = np.zeros((1, SLOT), dtype=np.uint32)
            header[0, 0] = K["kMarkTile"]
            header[0, K["kHdrTile"]] = t
            header[0, K["kHdrCount"]] = len(ranks_t)
            header[0, K["kHdrRow0"]] = row0
            header[0, K["kHdrRow1"]] = row1
            parts += [header, cards[ranks_t]]
        end = np.zeros((1, SLOT), dtype=np.uint32)
        end[0, 0] = K["kMarkEnd"]
        parts.append(end)
        slots = np.concatenate(parts)
        pad = -len(slots) % PAGE_SLOTS
        slots = np.concatenate([slots, np.zeros((pad, SLOT), dtype=np.uint32)])
        pages = slots.reshape(-1, PAGE_WORDS)
        directory[core, K["kDirFirst"]] = first
        directory[core, K["kDirPages"]] = len(pages)
        directory[core, K["kDirItems"]] = len(tiles)
        directory[core, K["kDirIds"] : K["kDirIds"] + len(tiles)] = ids
        streams.append(pages)
        first += len(pages)
    return np.concatenate(streams), directory


def window_index(rows, lanes):
    """Word of (vector `rows`, lane `lanes`) in a plane read out of the Dst window."""
    rows, lanes = np.asarray(rows), np.asarray(lanes)
    return 64 * (rows // 2) + 16 * (lanes >> 3) + 2 * (lanes & 7) + rows % 2


ROW, COL = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
TILE_ORDER = window_index(ROW, COL)  # (32, 32): word of pixel (row, col)


def decode_frame(frame):
    """Colour, depth and triangle images from the frame pages (tiles x 5, 1024)."""
    planes = np.asarray(frame, dtype=np.uint32).reshape(TILES, K["kPlanes"], 1024)
    px = planes[:, :, TILE_ORDER].view(np.float32)  # (tiles, plane, 32, 32)
    px = px.reshape(raster.TILES_Y, raster.TILES_X, K["kPlanes"], 32, 32)
    img = px.transpose(2, 0, 3, 1, 4).reshape(K["kPlanes"], raster.HEIGHT, raster.WIDTH)
    tri = img[4].astype(np.int64)
    if not np.array_equal(tri.astype(np.float32), img[4]):
        raise ValueError("a triangle number in the frame is not an integer")
    return {
        "colour": np.ascontiguousarray(np.stack(img[1:4], axis=2)),
        "depth": np.ascontiguousarray(img[0]),
        "tri": tri.astype(np.int32),
    }


def frame_of(images):
    """The frame pages the card would write for these images (inverse of decode)."""
    img = np.stack(
        [images["depth"], *np.moveaxis(images["colour"], 2, 0), images["tri"]]
    ).astype(np.float32)
    px = img.reshape(K["kPlanes"], raster.TILES_Y, 32, raster.TILES_X, 32)
    px = px.transpose(1, 3, 0, 2, 4).reshape(TILES, K["kPlanes"], 32, 32)
    out = np.zeros((TILES, K["kPlanes"], 1024), dtype=np.uint32)
    out[:, :, TILE_ORDER] = px.view(np.uint32)
    return out.reshape(TILES * K["kPlanes"], 1024)


def busy_map(tiles_of, busy):
    """Per-pixel image of each core's busy cycles over the tiles it owns."""
    per_tile = np.zeros(TILES)
    for core, tiles in enumerate(tiles_of):
        for item in tiles:
            t = as_item(item)[0]
            per_tile[t] = max(per_tile[t], busy[core])
    grid = per_tile.reshape(raster.TILES_Y, raster.TILES_X)
    return np.repeat(np.repeat(grid, 32, axis=0), 32, axis=1)
