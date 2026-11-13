"""Pictures of the ray tracing: image, normals, depth, nodes visited, treelets, diffs.

Each implementation found gets its final image and a map of where it differs from
numpy; --small writes the 256 x 256 images of the tiny and 10^4 scenes for the README.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from projects.raytrace.reference import harness
from ref import native, png, raytrace, scenes

SMALL_SCENES = ("tiny", "s10k")
F = np.float32


def normals_picture(scene, tri):
    """The unit normal of each primary hit's triangle, facing the camera, as colour."""
    pos = scene["positions"][scene["indices"][np.maximum(tri, 0)]].astype(np.float64)
    n = np.cross(pos[..., 1, :] - pos[..., 0, :], pos[..., 2, :] - pos[..., 0, :])
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    _, d = raytrace.camera_rays(scene["camera"], np.arange(tri.size))
    d = d.reshape(*tri.shape, 3)
    n = np.where((n * d).sum(axis=-1, keepdims=True) > 0, -n, n)
    rgb = np.rint((0.5 + 0.5 * n) * 255).astype(np.uint8)
    rgb[tri < 0] = 0
    return rgb


def nodes_picture(nodes):
    """Box tests per primary ray on the viridis map."""
    rgb = png.colormap(nodes.astype(np.float64), 0.0, None)
    return rgb, {"nodes_max": int(nodes.max()), "nodes_mean": float(nodes.mean())}


def treelet_picture(tree, cut, tri):
    """Treelet of each primary hit's leaf as a hashed colour; misses black."""
    leaf = np.empty(len(tree["leaf_tri"]), dtype=np.int64)
    leaf[tree["leaf_tri"]] = np.arange(len(leaf))
    ids = cut["node_treelet"][tree["leaf_node"][leaf[np.maximum(tri, 0)]]]
    h = ids.astype(np.uint32) * np.uint32(2654435761)
    rgb = np.stack([(h >> s) & 255 for s in (0, 8, 16)], axis=-1).astype(np.uint8)
    rgb = (rgb // 2 + 96).astype(np.uint8)
    rgb[tri < 0] = 0
    return rgb, {"treelets": int(len(cut["treelet_root"]))}


def pictures(scene_name, out, external, numpy_max=None):
    """Writes the pictures of one scene into out; returns the numbers on them."""
    harness.ensure_results(scene_name, numpy_max)
    found = harness.collect(scene_name, external)
    ref_name = harness.REFERENCE if harness.REFERENCE in found else next(iter(found))
    ref = found[ref_name][0]
    data = native.data_dir()
    scene = scenes.load_scene(data / f"{scene_name}.npz")
    tree = harness.ensure_tree(scene_name)
    info = {"triangles": int(scene["meta"][0]), "reference": ref_name}
    info["implementations"] = {}
    png.write_png(out / f"{scene_name}_normals.png", normals_picture(scene, ref["tri"]))
    hit = ref["tri"] >= 0
    z = np.where(hit, 1.0 / np.where(hit, ref["t"], 1.0), 0.0)
    rgb, info["depth"] = png.depth_picture(z)
    png.write_png(out / f"{scene_name}_depth.png", rgb)
    nodes = ref.get("nodes")
    if nodes is None:
        nodes = next(a["nodes"] for a, _ in found.values() if "nodes" in a)
    rgb, info["nodes"] = nodes_picture(nodes)
    png.write_png(out / f"{scene_name}_nodes.png", rgb)
    for kib in harness.TREELET_KIB:
        cut = np.load(data / f"{scene_name}_treelets_{kib}k.npz")
        rgb, info[f"treelets_{kib}k"] = treelet_picture(tree, cut, ref["tri"])
        png.write_png(out / f"{scene_name}_treelets_{kib}k.png", rgb)
    for name, (arrays, run) in found.items():
        key = png.slug(name)
        png.write_png(
            out / f"{scene_name}_{key}_image.png", png.to_srgb8(arrays["image"])
        )
        stats = harness.compare(ref, arrays)
        image = png.to_srgb8(ref["image"])
        png.write_png(
            out / f"{scene_name}_{key}_diff.png", png.diff_picture(image, stats["mask"])
        )
        info["implementations"][key] = {
            "name": name,
            "differing_pixels": stats["any_differs"],
            "seconds_per_frame": run.get("seconds_per_frame"),
            "timings_ms": run.get("timings_ms"),
            "power": run.get("power"),
        }
    return info, ref


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=harness.SCENES)
    parser.add_argument("--small", type=Path, default=None)
    parser.add_argument("--external", nargs="*", default=[])
    parser.add_argument("--numpy-max", type=int, default=None, help="largest scene")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index = {}
    for name in args.scenes:
        index[name], ref = pictures(name, args.out, args.external, args.numpy_max)
        if args.small and name in SMALL_SCENES:
            args.small.mkdir(parents=True, exist_ok=True)
            small = png.downscale(png.to_srgb8(ref["image"]), 4)
            png.write_png(args.small / f"{name}_image_256.png", small)
    (args.out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
