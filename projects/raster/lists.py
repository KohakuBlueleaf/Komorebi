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


def pack_records(records):
    """The 32-word card record of every triangle record of `raster.setup`."""
    count = len(records["tri"])
    packed = np.zeros((count, SLOT), dtype=np.uint32)
    packed[:, K["kRecA"] : K["kRecA"] + 3] = bits(records["a"])
    packed[:, K["kRecB"] : K["kRecB"] + 3] = bits(records["b"])
    packed[:, K["kRecC"] : K["kRecC"] + 3] = bits(records["c"])
    packed[:, K["kRecT"] : K["kRecT"] + 3] = np.where(
        records["top_left"], NEG_ZERO_NEIGHBOUR, 0
    )
    packed[:, K["kRecW"] : K["kRecW"] + 3] = bits(records["w_planes"])
    colour_planes = records["c_planes"]  # (n, coefficient, channel)
    for channel in range(3):
        column = K["kRecCol"] + 3 * channel
        packed[:, column : column + 3] = bits(colour_planes[:, :, channel])
    box = records["box"].astype(np.int64)
    packed[:, K["kRecX0"]] = half_bits(box[:, 0])
    packed[:, K["kRecX1"]] = half_bits(box[:, 1] + 1)
    packed[:, K["kRecY0"]] = box[:, 2].astype(np.uint32)
    packed[:, K["kRecY1"]] = box[:, 3].astype(np.uint32)
    packed[:, K["kRecTri"]] = bits(records["tri"].astype(np.float32))
    packed[:, K["kRecIndex"]] = np.arange(count, dtype=np.uint32)
    return packed


# Measured pack-thread cycles: per (triangle, tile), per row, per tile with triangles;
# an empty tile costs only its writer.
COST = {"tri": 260, "row": 60, "tile": 6800, "empty": 2000}


def tile_cost(packed, offsets, members, cost=COST):
    """Estimated cycles of each tile: its triangles, their rows in it and the tile."""
    triangles = np.diff(offsets)
    tile = np.repeat(np.arange(len(triangles)), triangles)
    top = 32 * (tile // raster.TILES_X)
    box_first_row = packed[members, K["kRecY0"]].astype(np.int64)
    box_last_row = packed[members, K["kRecY1"]].astype(np.int64)
    rows = np.minimum(box_last_row, top + 31) - np.maximum(box_first_row, top) + 1
    per_tile = np.bincount(tile, rows, minlength=len(triangles))
    busy = cost["tri"] * triangles + cost["row"] * per_tile + cost["tile"]
    return np.where(triangles > 0, busy, cost["empty"])


def as_item(x):
    """(tile, first_row, end_row): a tile number alone is the whole tile."""
    if isinstance(x, tuple):
        return x
    return (int(x), 0, 32)


def band_ranks(packed, tile_ranks, tile, first_row, end_row):
    """The members of a tile whose rows reach into its rows [first_row, end_row)."""
    top = 32 * (tile // raster.TILES_X)
    box_first_row = packed[tile_ranks, K["kRecY0"]].astype(np.int64)
    box_last_row = packed[tile_ranks, K["kRecY1"]].astype(np.int64)
    return tile_ranks[
        (box_first_row <= top + end_row - 1) & (box_last_row >= top + first_row)
    ]


def band_cost(packed, tile_ranks, tile, first_row, end_row, cost=COST):
    """Estimated cycles of a band of a tile, as `tile_cost` counts a tile."""
    top = 32 * (tile // raster.TILES_X)
    band_members = band_ranks(packed, tile_ranks, tile, first_row, end_row)
    if len(band_members) == 0:
        return cost["empty"]
    box_first_row = packed[band_members, K["kRecY0"]].astype(np.int64)
    box_last_row = packed[band_members, K["kRecY1"]].astype(np.int64)
    rows = (
        np.minimum(box_last_row, top + end_row - 1)
        - np.maximum(box_first_row, top + first_row)
        + 1
    )
    return (
        cost["tri"] * len(band_members) + cost["row"] * int(rows.sum()) + cost["tile"]
    )


def split_tiles(packed, offsets, members, tiles, cost, target, max_bands=16):
    """(items, their costs): each tile of `tiles` whole, or, when its cost is above
    `target`, as bands of even rows of about equal estimated cost."""
    items, costs = [], []
    for tile in tiles:
        bands = min(max_bands, int(np.ceil(cost[tile] / target)))
        if bands < 2:
            items.append((tile, 0, 32))
            costs.append(cost[tile])
            continue
        tile_members = members[offsets[tile] : offsets[tile + 1]]
        top = 32 * (tile // raster.TILES_X)
        box_first_row = np.clip(
            packed[tile_members, K["kRecY0"]].astype(np.int64) - top, 0, 31
        )
        box_last_row = np.clip(
            packed[tile_members, K["kRecY1"]].astype(np.int64) - top, 0, 31
        )
        # Row weights: each row's coverage, a triangle's setup on its first row.
        starts = np.bincount(box_first_row, minlength=33)
        cover = np.cumsum(starts - np.bincount(box_last_row + 1, minlength=33))[:32]
        row_weights = COST["row"] * cover + COST["tri"] * starts[:32]
        cumulative = np.cumsum(row_weights.reshape(16, 2).sum(axis=1))
        cuts = (
            np.searchsorted(cumulative, cumulative[-1] * np.arange(1, bands) / bands)
            + 1
        )
        bounds = np.unique(np.concatenate([[0], 2 * cuts, [32]]))
        for first_row, end_row in zip(bounds[:-1], bounds[1:]):
            items.append((tile, int(first_row), int(end_row)))
            costs.append(
                band_cost(packed, tile_members, tile, int(first_row), int(end_row))
            )
    return items, np.asarray(costs, dtype=np.float64)


def assign(cost, cores, mode="balanced"):
    """Tiles per core: round-robin over the tile numbers, or balanced by the tiles'
    costs (largest first to the least loaded core)."""
    cost = np.asarray(cost, dtype=np.float64).reshape(-1)
    if mode == "round-robin":
        return [list(range(core, len(cost), cores)) for core in range(cores)]
    if mode != "balanced":
        raise ValueError(f"unknown assignment {mode}")
    load = np.zeros(cores)
    per_core = [[] for _ in range(cores)]
    for tile in np.argsort(-cost, kind="stable"):
        core = int(np.argmin(load))
        per_core[core].append(int(tile))
        load[core] += cost[tile]
    return [sorted(tiles) for tiles in per_core]


def build_lists(packed, offsets, members, tiles_of):
    """List pages of all cores and one directory page per core, from packed records,
    `raster.bin_triangles`' offsets and members, and the items per core (tiles or
    bands, `as_item`): (pages (P, 512) uint32, directory (cores, 1024) uint32)."""
    cores = len(tiles_of)
    if max(len(items) for items in tiles_of) > K["kDirMaxItems"]:
        raise ValueError("a core has more tiles than its directory page holds")
    streams, first_page = [], 0
    directory = np.zeros((cores, K["kDirBytes"] // 4), dtype=np.uint32)
    for core, own_items in enumerate(tiles_of):
        parts, item_ids = [], []
        for item in own_items:
            tile, first_row, end_row = as_item(item)
            tile_ranks = members[offsets[tile] : offsets[tile + 1]]
            band = (first_row, end_row) != (0, 32)
            if band:
                tile_ranks = band_ranks(packed, tile_ranks, tile, first_row, end_row)
            rows_bits = (
                first_row << K["kItemRow0Shift"] | end_row << K["kItemRow1Shift"]
            )
            item_ids.append(tile | (rows_bits if band else 0))
            if len(tile_ranks) == 0:
                item_ids[-1] |= K["kItemEmpty"]
                continue  # the writer stores empty tiles by itself
            header = np.zeros((1, SLOT), dtype=np.uint32)
            header[0, 0] = K["kMarkTile"]
            header[0, K["kHdrTile"]] = tile
            header[0, K["kHdrCount"]] = len(tile_ranks)
            header[0, K["kHdrRow0"]] = first_row
            header[0, K["kHdrRow1"]] = end_row
            parts += [header, packed[tile_ranks]]
        end_slot = np.zeros((1, SLOT), dtype=np.uint32)
        end_slot[0, 0] = K["kMarkEnd"]
        parts.append(end_slot)
        slots = np.concatenate(parts)
        pad = -len(slots) % PAGE_SLOTS
        slots = np.concatenate([slots, np.zeros((pad, SLOT), dtype=np.uint32)])
        pages = slots.reshape(-1, PAGE_WORDS)
        directory[core, K["kDirFirst"]] = first_page
        directory[core, K["kDirPages"]] = len(pages)
        directory[core, K["kDirItems"]] = len(own_items)
        directory[core, K["kDirIds"] : K["kDirIds"] + len(own_items)] = item_ids
        streams.append(pages)
        first_page += len(pages)
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
    pixels = planes[:, :, TILE_ORDER].view(np.float32)  # (tiles, plane, 32, 32)
    pixels = pixels.reshape(raster.TILES_Y, raster.TILES_X, K["kPlanes"], 32, 32)
    image = pixels.transpose(2, 0, 3, 1, 4).reshape(
        K["kPlanes"], raster.HEIGHT, raster.WIDTH
    )
    triangle = image[4].astype(np.int64)
    if not np.array_equal(triangle.astype(np.float32), image[4]):
        raise ValueError("a triangle number in the frame is not an integer")
    return {
        "colour": np.ascontiguousarray(np.stack(image[1:4], axis=2)),
        "depth": np.ascontiguousarray(image[0]),
        "tri": triangle.astype(np.int32),
    }


def frame_of(images):
    """The frame pages the card would write for these images (inverse of decode)."""
    image = np.stack(
        [images["depth"], *np.moveaxis(images["colour"], 2, 0), images["tri"]]
    ).astype(np.float32)
    pixels = image.reshape(K["kPlanes"], raster.TILES_Y, 32, raster.TILES_X, 32)
    pixels = pixels.transpose(1, 3, 0, 2, 4).reshape(TILES, K["kPlanes"], 32, 32)
    frame = np.zeros((TILES, K["kPlanes"], 1024), dtype=np.uint32)
    frame[:, :, TILE_ORDER] = pixels.view(np.uint32)
    return frame.reshape(TILES * K["kPlanes"], 1024)


def busy_map(tiles_of, busy):
    """Per-pixel image of each core's busy cycles over the tiles it owns."""
    per_tile = np.zeros(TILES)
    for core, tiles in enumerate(tiles_of):
        for item in tiles:
            tile = as_item(item)[0]
            per_tile[tile] = max(per_tile[tile], busy[core])
    grid = per_tile.reshape(raster.TILES_Y, raster.TILES_X)
    return np.repeat(np.repeat(grid, 32, axis=0), 32, axis=1)
