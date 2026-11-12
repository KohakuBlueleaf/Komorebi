"""The files an implementation hands to the comparison, picture and timing scripts.

For every implementation (a folder name, e.g. `cpu`, `v100`, or a card version) and
scene `<scene>`, the folder holds `<scene>.npz` (arrays below) and `<scene>.json`:
{"name": ..., "scene": ..., "seconds_per_frame": ..., "timings_ms": {stage: ms},
"power": {"gpu_idle_w", "gpu_mean_w", "gpu_peak_w", "gpu_joules_per_frame",
"cpu_package_joules_per_frame": ...}} (everything but "name" and "scene" is optional).
Arrays: rasterization colour (1024, 1024, 3) float32, depth (1024, 1024) float32 (the
stored 1/z, 0 where empty), tile_counts (32, 32) int32 and, optionally, tri (1024, 1024)
int32; ray tracing t (1024, 1024) float32, tri (1024, 1024) int32, image (1024, 1024, 3)
float32 and, optionally, nodes (1024, 1024) int32.
"""

import json
from pathlib import Path

import numpy as np

REQUIRED = {
    "raster": ("colour", "depth", "tile_counts"),
    "raytrace": ("t", "tri", "image"),
}
OPTIONAL = {"raster": ("tri",), "raytrace": ("nodes",)}


def save_result(folder, scene, task, arrays, info):
    """Writes <folder>/<scene>.npz and .json for one implementation."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    keys = REQUIRED[task] + OPTIONAL[task]
    np.savez(folder / f"{scene}.npz", **{k: arrays[k] for k in keys if k in arrays})
    (folder / f"{scene}.json").write_text(json.dumps(info, indent=1))


def load_result(folder, scene, task):
    """(arrays, info) of one implementation; raises if a required array is missing."""
    folder = Path(folder)
    with np.load(folder / f"{scene}.npz") as data:
        arrays = {k: data[k] for k in data.files}
    missing = [k for k in REQUIRED[task] if k not in arrays]
    if missing:
        raise ValueError(f"{folder / scene}.npz lacks {missing}")
    info = json.loads((folder / f"{scene}.json").read_text())
    return arrays, info


def find_results(roots, scene, task):
    """{implementation name: (arrays, info)} for every folder under the roots."""
    found = {}
    for root in roots:
        for folder in sorted(Path(root).glob("*/")):
            if (folder / f"{scene}.npz").exists():
                arrays, info = load_result(folder, scene, task)
                found[info.get("name", folder.name)] = (arrays, info)
    return found
