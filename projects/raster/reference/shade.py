"""SHADE on the scenes with materials: numpy against C++ on every view, pictures
(vertex colours beside the shaded frame), and the numbers a shading pass on the cores is
sized by (see the README). `python -m projects.raster.reference.shade
agree | pictures --out DIR | stats --out DIR [--views ...] [--threads N]`"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from projects.raster import lists
from projects.raster.reference import harness
from reference import native, png, public, raster, scenes

CORES = 120
SPLIT = 0.33  # bands as projects/raster/run.py cuts them
PAGE = 32  # a texture page: 32 x 32 texels of one level, 4 KB
CACHE_PAGES = (8, 16, 32)


def material_views():
    """The views of every scene whose table entry has materials."""
    return [
        view
        for view, (name, _) in public.views().items()
        if "materials" in public.table()[name]
    ]


def agree(view, threads):
    """Statistics of numpy against C++ on the shaded frame of a view."""
    record = harness.run_cpu_shaded(view, threads)
    harness.run_numpy_shaded(view)
    found = harness.collect_shaded(view)
    statistics = harness.compare(found["numpy"][0], found["C++"][0])
    statistics.pop("mask")
    statistics["cpu_timings_ms"] = harness.runner.stage_medians(
        record, harness.SHADE_STAGES
    )
    return statistics


def shaded_result(view, threads):
    """The C++ shaded arrays of a view (run when missing) and their record."""
    found = harness.collect_shaded(view)
    if "C++" not in found:
        harness.run_cpu_shaded(view, threads)
        found = harness.collect_shaded(view)
    return found["C++"]


def pictures(view, folder, threads):
    """Writes <view>_vertex_shaded.png (FILL's vertex colours | SHADE) and
    <view>_shaded.png."""
    root = harness.runner.results_root("raster") / f"cpu{threads}"
    if not (root / f"{view}.npz").exists():
        harness.run_cpu(view, threads, repetitions=1)
    vertex, _ = harness.results.load_result(root, view, "raster")
    shaded, _ = shaded_result(view, threads)
    left, right = png.to_srgb8(vertex["colour"]), png.to_srgb8(shaded["colour"])
    png.write_png(
        folder / f"{view}_vertex_shaded.png", np.concatenate([left, right], 1)
    )
    png.write_png(folder / f"{view}_shaded.png", right)


def partition(scene):
    """The rasterizer's items (tile, first row, end row) per core, as run.py makes
    them for the frame's records on CORES cores."""
    records = raster.setup(scene)
    _, offsets, members = raster.bin_triangles(records)
    packed = lists.pack_records(records)
    cost = lists.tile_cost(packed, offsets, members)
    tiles = list(range(lists.TILES))
    mean = cost.sum() / CORES
    if cost.max() > mean:
        items, item_cost = lists.split_tiles(
            packed, offsets, members, tiles, cost, SPLIT * mean
        )
    else:
        items, item_cost = [(tile, 0, 32) for tile in tiles], cost
    return [[items[i] for i in own] for own in lists.assign(item_cost, CORES)]


def layer_accesses(scene, materials, prepared, layer, ys, xs):
    """(pixel, page, texel) of every texel SHADE reads at pixels (ys, xs) of a layer,
    in slot order per pixel; page and texel are numbers unique in the frame."""
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
    out = []
    pixel = ys * raster.WIDTH + xs
    for slot, wanted in reads.items():
        take = wanted & (texture[:, slot] >= 0)
        if not take.any():
            continue
        tex = texture[take, slot]
        level = prepared["level"][t[take], slot]
        x, y, entry = raster.texel_xy(materials, tex, level, u[take], v[take])
        pages_across = (entry[:, 1] + PAGE - 1) // PAGE
        level_key = tex.astype(np.int64) * public.MIP_LEVELS + level
        page = (level_key << 24) | ((y // PAGE) * pages_across + x // PAGE)
        texel = entry[:, 0] // 4 + y * entry[:, 1] + x
        out.append((pixel[take], slot, page, texel))
    return out


def frame_accesses(scene, materials, layers):
    """Per pixel the kinds SHADE runs and the texels it reads: the masked layer where
    it is nearer (and, at a hole, then the other layer), else the other layer."""
    prepared = raster.shade_setup(scene, materials)
    opaque, masked = layers["opaque"], layers["masked"]
    first = raster.nearer_layer(masked, opaque)
    _, hole = raster.shade_layer(scene, materials, prepared, masked)
    second = (~first | hole) & (opaque["tri"] >= 0)
    kinds = np.zeros((raster.HEIGHT, raster.WIDTH), dtype=np.int32)
    accesses = []
    for layer, where in ((masked, first), (opaque, second)):
        ys, xs = np.nonzero(where)
        kind = materials["kind"][materials["material"][layer["tri"][ys, xs]]]
        kinds[ys, xs] |= 1 << kind
        accesses += layer_accesses(scene, materials, prepared, layer, ys, xs)
    pixel = np.concatenate([a[0] for a in accesses])
    # Per pixel: the masked layer's reads first, then the other's, slot by slot.
    order = np.argsort(pixel, kind="stable")
    page = np.concatenate([a[2] for a in accesses])[order]
    texel = np.concatenate([a[3] for a in accesses])[order]
    return kinds, pixel[order], page, texel


def histogram(values):
    """{value: how many} of integer values."""
    counts = np.bincount(np.asarray(values, dtype=np.int64))
    return {int(k): int(n) for k, n in enumerate(counts) if n}


def per_tile_distinct(values, valid):
    """Distinct values per 32 x 32 tile (H, W) over the valid pixels."""
    tiles = raster.TILES_Y * raster.TILES_X
    y, x = np.nonzero(valid)
    tile = (y // raster.TILE) * raster.TILES_X + x // raster.TILE
    pairs = np.unique(np.stack([tile, values[y, x]], 1), axis=0)
    return np.bincount(pairs[:, 0], minlength=tiles)


def lru_hits(stream, size):
    """Hits of an LRU cache of `size` pages over a stream of page numbers."""
    cache, hits = OrderedDict(), 0
    for page in stream:
        if page in cache:
            hits += 1
            cache.move_to_end(page)
        else:
            cache[page] = True
            if len(cache) > size:
                cache.popitem(last=False)
    return hits


def stats(view, threads):
    """The numbers of one view's shaded frame (see the module's text)."""
    scene = scenes.load_scene(native.data_dir() / f"{view}.npz")
    materials, _ = public.load_materials(view, native.data_dir())
    layers = raster.rasterize_shaded(scene, materials)
    tri = layers["tri"]
    hit = tri >= 0
    material = np.where(hit, materials["material"][np.maximum(tri, 0)], -1)
    kinds, pixel, page, texel = frame_accesses(scene, materials, layers)
    pixel_kind = np.where(hit, materials["kind"][np.maximum(material, 0)], -1)
    present = {
        public.KINDS[k]: int((pixel_kind == k).sum())
        for k in range(len(public.KINDS))
        if (pixel_kind == k).any()
    }
    tile_of_pixel = (pixel // raster.WIDTH // raster.TILE) * raster.TILES_X + (
        pixel % raster.WIDTH
    ) // raster.TILE
    pages_per_tile = np.bincount(
        np.unique(np.stack([tile_of_pixel, page], 1), axis=0)[:, 0],
        minlength=raster.TILES_X * raster.TILES_Y,
    )
    kind_bits = kinds.reshape(raster.TILES_Y, raster.TILE, raster.TILES_X, raster.TILE)
    tile_bits = np.bitwise_or.reduce(np.bitwise_or.reduce(kind_bits, axis=3), axis=1)
    kinds_per_tile = np.array([bin(b).count("1") for b in tile_bits.ravel()])
    # Per core: its items in order, each item's pixels row by row.
    per_core = partition(scene)
    starts = np.searchsorted(pixel, np.arange(raster.WIDTH * raster.HEIGHT + 1))
    kinds_per_core, hits = [], dict.fromkeys(CACHE_PAGES, 0)
    reads = 0
    for items in per_core:
        bits, stream = 0, []
        for tile, first_row, end_row in items:
            ty, tx = divmod(tile, raster.TILES_X)
            rows = slice(ty * raster.TILE + first_row, ty * raster.TILE + end_row)
            columns = slice(tx * raster.TILE, (tx + 1) * raster.TILE)
            bits |= int(np.bitwise_or.reduce(kinds[rows, columns], axis=None))
            for y in range(rows.start, rows.stop):
                row_start = starts[y * raster.WIDTH + columns.start]
                row_end = starts[y * raster.WIDTH + columns.stop]
                stream.append(page[row_start:row_end])
        kinds_per_core.append(bin(bits).count("1"))
        stream = np.concatenate(stream) if stream else np.zeros(0, np.int64)
        reads += len(stream)
        # A read of the page just read always hits: only the changes are walked.
        change = np.ones(len(stream), dtype=bool)
        change[1:] = stream[1:] != stream[:-1]
        repeats = int((~change).sum())
        for size in CACHE_PAGES:
            hits[size] += repeats + lru_hits(stream[change].tolist(), size)
    _, record = shaded_result(view, threads)
    return {
        "view": view,
        "frame": [raster.WIDTH, raster.HEIGHT],
        "kinds_present_pixels": present,
        "tile_distinct_winners": histogram(per_tile_distinct(tri, hit)),
        "tile_distinct_materials": histogram(per_tile_distinct(material, hit)),
        "tile_distinct_kinds": histogram(kinds_per_tile),
        "core_kinds": histogram(kinds_per_core),
        "cores": CORES,
        "texel_reads": int(len(page)),
        "tile_distinct_pages": histogram(pages_per_tile),
        "texel_bytes_touched": int(len(np.unique(texel)) * 4),
        "pages_touched": int(len(np.unique(page))),
        "cache_hit_rate": {str(k): hits[k] / max(reads, 1) for k in CACHE_PAGES},
        "shade_reference_ms": record["timings_ms"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("agree", "pictures", "stats"))
    parser.add_argument("--views", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=16, help="FILL's threads")
    args = parser.parse_args()
    views = args.views or material_views()
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for view in views:
        if args.command == "agree":
            report[view] = agree(view, args.threads)
            s = report[view]
            print(
                f"{view:26s} numpy vs C++ differing pixels (tri/depth/colour/any) "
                f"{s['tri_differs']}/{s['depth_differs']}/{s['colour_differs']}/"
                f"{s['any_differs']} of {s['pixels']}, tile counts equal "
                f"{s['tile_counts_equal']}, C++ SHADE {s['cpu_timings_ms']} ms"
            )
        elif args.command == "pictures":
            pictures(view, args.out, args.threads)
        else:
            report[view] = stats(view, args.threads)
            print(json.dumps(report[view]))
    if args.out and report:
        (args.out / f"{args.command}.json").write_text(json.dumps(report, indent=1))
    if args.command == "pictures":
        names = {}
        for scene in sorted({public.views()[v][0] for v in views}):
            names.setdefault(public.table()[scene]["attribution"], []).append(scene)
        (args.out / "CREDITS.txt").write_text(
            "".join(f"{', '.join(n)}: {a}\n" for a, n in names.items())
        )


if __name__ == "__main__":
    main()
