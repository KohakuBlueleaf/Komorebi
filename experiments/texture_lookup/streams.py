"""The texel reads of the shaded views, per core, from the rasterizer reference's SHADE
stage (host only): an npz a view with every core's items and their words.

PYTHONPATH=src:. python experiments/texture_lookup/streams.py --out DIR [--views ...]"""

import argparse
import json
from pathlib import Path

import numpy as np

from projects.raster.reference import shade
from reference import native, public, raster, scenes

PAGE = 32


def accesses(scene, materials, layers):
    """Per texel read of the frame, in pixel order and slot order (shade.py's
    frame_accesses): pixel, slot, texture, level, x, y, level offset in texels."""
    prepared = raster.shade_setup(scene, materials)
    opaque, masked = layers["opaque"], layers["masked"]
    first = raster.nearer_layer(masked, opaque)
    _, hole = raster.shade_layer(scene, materials, prepared, masked)
    second = (~first | hole) & (opaque["tri"] >= 0)
    out = []
    for layer, where in ((masked, first), (opaque, second)):
        ys, xs = np.nonzero(where)
        t = layer["tri"][ys, xs]
        m = materials["material"][t]
        kind = materials["kind"][m]
        _, _, _, u, v, _ = raster.pixel_values(layer, prepared, ys, xs)
        texture = materials["textures"][m]
        reads = {
            raster.ALBEDO: kind != raster.FLAT,
            raster.MASK: kind == raster.MASKED,
            raster.NORMAL: (kind == raster.LIT) & prepared["frame_ok"][t],
            raster.SPECULAR: kind == raster.LIT,
        }
        pixel = ys * raster.WIDTH + xs
        for slot, wanted in reads.items():
            take = wanted & (texture[:, slot] >= 0)
            if not take.any():
                continue
            texture_index = texture[take, slot]
            level = prepared["level"][t[take], slot]
            x, y, entry = raster.texel_xy(
                materials, texture_index, level, u[take], v[take]
            )
            out.append(
                np.stack(
                    [
                        pixel[take],
                        np.full(take.sum(), slot),
                        texture_index,
                        level,
                        x,
                        y,
                        entry[:, 1],
                        m[take],
                    ],
                    axis=1,
                ).astype(np.int64)
            )
    table = np.concatenate(out)
    return table[np.argsort(table[:, 0], kind="stable")]


def page_keys(table, page):
    """A frame-unique number of each read's page of page x page texels, and the
    texel's place in it."""
    texture_index, level, x, y, width = (
        table[:, 2],
        table[:, 3],
        table[:, 4],
        table[:, 5],
        table[:, 6],
    )
    across = (width + page - 1) // page
    key = ((texture_index * public.MIP_LEVELS + level) << 24) | (
        (y // page) * across + x // page
    )
    offset = (y % page) * page + x % page
    return key, offset


def view_streams(view):
    """A view's reads cut by the rasterizer's items: words (key << 10 | offset, pages of
    32 x 32 texels), words16 (pages of 16 x 16), items (core, first, count, tile,
    rows), and per tile its distinct pages, materials and reads."""
    scene = scenes.load_scene(native.data_dir() / f"{view}.npz")
    materials, _ = public.load_materials(view, native.data_dir())
    layers = raster.rasterize_shaded(scene, materials)
    table = accesses(scene, materials, layers)
    key, offset = page_keys(table, PAGE)
    unique, compact = np.unique(key, return_inverse=True)
    words = (compact.astype(np.uint64) << 10 | offset.astype(np.uint64)).astype(
        np.uint32
    )
    key16, offset16 = page_keys(table, 16)
    _, compact16 = np.unique(key16, return_inverse=True)
    words16 = (compact16.astype(np.uint64) << 10 | offset16.astype(np.uint64)).astype(
        np.uint32
    )
    pixel = table[:, 0]
    starts = np.searchsorted(pixel, np.arange(raster.WIDTH * raster.HEIGHT + 1))
    # Items per core as the rasterizer cuts them; an item's reads row by row.
    per_core = shade.partition(scene)
    order, items = [], []
    at = 0
    for core, own in enumerate(per_core):
        for tile, first_row, end_row in own:
            ty, tx = divmod(tile, raster.TILES_X)
            pieces = []
            for y in range(ty * raster.TILE + first_row, ty * raster.TILE + end_row):
                begin = starts[y * raster.WIDTH + tx * raster.TILE]
                end = starts[y * raster.WIDTH + (tx + 1) * raster.TILE]
                pieces.append(np.arange(begin, end))
            index = np.concatenate(pieces) if pieces else np.zeros(0, np.int64)
            order.append(index)
            items.append([core, at, len(index), tile, first_row, end_row])
            at += len(index)
    order = np.concatenate(order)
    items = np.asarray(items, np.int64)
    # Per tile: distinct pages and materials, for the samples.
    tile = (pixel // raster.WIDTH // raster.TILE) * raster.TILES_X + (
        pixel % raster.WIDTH
    ) // raster.TILE
    tiles = raster.TILES_X * raster.TILES_Y
    pages = np.bincount(
        np.unique(np.stack([tile, key], 1), axis=0)[:, 0], minlength=tiles
    )
    mats = np.bincount(
        np.unique(np.stack([tile, table[:, 7]], 1), axis=0)[:, 0], minlength=tiles
    )
    reads = np.bincount(tile, minlength=tiles)
    return {
        "words": words[order],
        "words16": words16[order],
        "items": items,
        "tile_pages": pages,
        "tile_materials": mats,
        "tile_reads": reads,
        "pages_touched": np.int64(len(unique)),
    }


def main():
    """Writes <view>.npz and summary.json."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--views", nargs="*", default=shade.material_views())
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for view in args.views:
        data = view_streams(view)
        np.savez(args.out / f"{view}.npz", **data)
        items = data["items"]
        per_core = np.bincount(items[:, 0], weights=items[:, 2], minlength=120)
        summary[view] = {
            "reads": int(len(data["words"])),
            "pages_touched": int(data["pages_touched"]),
            "items": int(len(items)),
            "items_per_core_max": int(np.bincount(items[:, 0]).max()),
            "reads_per_core_max": int(per_core.max()),
            "reads_per_core_mean": float(per_core.mean()),
        }
        print(json.dumps({view: summary[view]}), flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
