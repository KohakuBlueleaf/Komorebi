"""Pictures of the ray tracing: image, normals, depth, nodes visited, treelets, diffs.

Each implementation found gets its final image and a map of where it differs from
numpy; --small writes the 256 x 256 images of the tiny and 10^4 scenes for the README.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from projects.raytrace.reference import harness
from reference import native, png, raytrace, scenes

SMALL_SCENES = ("tiny", "s10k")
F = np.float32


def normals_picture(scene, triangle_ids):
    """The unit normal of each primary hit's triangle, facing the camera, as colour."""
    positions = scene["positions"][
        scene["indices"][np.maximum(triangle_ids, 0)]
    ].astype(np.float64)
    normals = np.cross(
        positions[..., 1, :] - positions[..., 0, :],
        positions[..., 2, :] - positions[..., 0, :],
    )
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    _, directions = raytrace.camera_rays(scene["camera"], np.arange(triangle_ids.size))
    directions = directions.reshape(*triangle_ids.shape, 3)
    normals = np.where(
        (normals * directions).sum(axis=-1, keepdims=True) > 0, -normals, normals
    )
    picture = np.rint((0.5 + 0.5 * normals) * 255).astype(np.uint8)
    picture[triangle_ids < 0] = 0
    return picture


def nodes_picture(nodes):
    """Box tests per primary ray on the viridis map."""
    picture = png.colormap(nodes.astype(np.float64), 0.0, None)
    return picture, {"nodes_max": int(nodes.max()), "nodes_mean": float(nodes.mean())}


def treelet_picture(tree, cut, triangle_ids):
    """Treelet of each primary hit's leaf as a hashed colour; misses black."""
    leaf = np.empty(len(tree["leaf_tri"]), dtype=np.int64)
    leaf[tree["leaf_tri"]] = np.arange(len(leaf))
    treelet_ids = cut["node_treelet"][
        tree["leaf_node"][leaf[np.maximum(triangle_ids, 0)]]
    ]
    hashed = treelet_ids.astype(np.uint32) * np.uint32(2654435761)
    picture = np.stack(
        [(hashed >> shift) & 255 for shift in (0, 8, 16)], axis=-1
    ).astype(np.uint8)
    picture = (picture // 2 + 96).astype(np.uint8)
    picture[triangle_ids < 0] = 0
    return picture, {"treelets": int(len(cut["treelet_root"]))}


def pictures(scene_name, folder, external, numpy_max=None):
    """Writes the pictures of one scene into out; returns the numbers on them."""
    harness.ensure_results(scene_name, numpy_max)
    found = harness.collect(scene_name, external)
    reference_name = (
        harness.REFERENCE if harness.REFERENCE in found else next(iter(found))
    )
    reference = found[reference_name][0]
    data = native.data_dir()
    scene = scenes.load_scene(data / f"{scene_name}.npz")
    tree = harness.ensure_tree(scene_name)
    info = {"triangles": int(scene["meta"][0]), "reference": reference_name}
    info["implementations"] = {}
    png.write_png(
        folder / f"{scene_name}_normals.png", normals_picture(scene, reference["tri"])
    )
    hit = reference["tri"] >= 0
    inverse_distance = np.where(hit, 1.0 / np.where(hit, reference["t"], 1.0), 0.0)
    picture, info["depth"] = png.depth_picture(inverse_distance)
    png.write_png(folder / f"{scene_name}_depth.png", picture)
    nodes = reference.get("nodes")
    if nodes is None:
        nodes = next(
            arrays["nodes"] for arrays, _ in found.values() if "nodes" in arrays
        )
    picture, info["nodes"] = nodes_picture(nodes)
    png.write_png(folder / f"{scene_name}_nodes.png", picture)
    for kib in harness.TREELET_KIB:
        cut = np.load(data / f"{scene_name}_treelets_{kib}k.npz")
        picture, info[f"treelets_{kib}k"] = treelet_picture(tree, cut, reference["tri"])
        png.write_png(folder / f"{scene_name}_treelets_{kib}k.png", picture)
    for name, (arrays, record) in found.items():
        key = png.slug(name)
        png.write_png(
            folder / f"{scene_name}_{key}_image.png", png.to_srgb8(arrays["image"])
        )
        stats = harness.compare(reference, arrays)
        image = png.to_srgb8(reference["image"])
        png.write_png(
            folder / f"{scene_name}_{key}_diff.png",
            png.diff_picture(image, stats["mask"]),
        )
        info["implementations"][key] = {
            "name": name,
            "differing_pixels": stats["any_differs"],
            "seconds_per_frame": record.get("seconds_per_frame"),
            "timings_ms": record.get("timings_ms"),
            "power": record.get("power"),
        }
    return info, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=harness.SCENES)
    parser.add_argument("--small", type=Path, default=None)
    parser.add_argument("--external", nargs="*", default=[])
    parser.add_argument("--numpy-max", type=int, default=None, help="largest scene")
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    index = {}
    for name in arguments.scenes:
        index[name], reference = pictures(
            name, arguments.out, arguments.external, arguments.numpy_max
        )
        if arguments.small and name in SMALL_SCENES:
            arguments.small.mkdir(parents=True, exist_ok=True)
            small = png.downscale(png.to_srgb8(reference["image"]), 4)
            png.write_png(arguments.small / f"{name}_image_256.png", small)
    (arguments.out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
