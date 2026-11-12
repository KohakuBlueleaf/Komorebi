"""Running the three rasterizers on a scene and comparing their outputs."""

import os
from pathlib import Path

import numpy as np

from ref import native, raster, scenes

HERE = Path(__file__).resolve().parent
SCENES = list(scenes.PRESETS)
KEYS = ("colour", "depth", "tri", "tile_counts")


def out_dir(scene, impl):
    """Folder of the saved outputs of one implementation on one scene."""
    path = native.data_dir() / "out" / "raster" / scene / impl
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_outputs(folder):
    """The four output arrays saved in a folder."""
    return {key: np.load(Path(folder) / f"{key}.npy") for key in KEYS}


def run_numpy(scene):
    """Runs the numpy rasterizer, saves and returns its outputs."""
    native.ensure_scene(scene)
    result = raster.rasterize(scenes.load_scene(native.data_dir() / f"{scene}.npz"))
    for key in KEYS:
        np.save(out_dir(scene, "numpy") / f"{key}.npy", result[key])
    return result


def run_cpu(scene, threads, reps=3, save=True):
    """Runs the C++ binary; returns its timing record (None without a compiler)."""
    exe = native.build(str(HERE / "cpu" / "raster_cpu.cpp"), "raster_cpu")
    if exe is None:
        return None
    folder = native.ensure_scene(scene)
    cmd = [exe, "--scene", folder, "--threads", threads, "--reps", reps]
    if save:
        cmd += ["--out", out_dir(scene, f"cpu{threads}")]
    return native.run_json([str(c) for c in cmd])


def run_gpu(scene, reps=3, save=True):
    """Runs the CUDA binary on an idle GPU; None without a toolchain or idle GPU."""
    exe = native.build(str(HERE / "gpu" / "raster_gpu.cu"), "raster_gpu", gpu=True)
    gpu = native.idle_gpu()
    if exe is None or gpu is None:
        return None
    folder = native.ensure_scene(scene)
    cmd = [exe, "--scene", folder, "--reps", reps]
    if save:
        cmd += ["--out", out_dir(scene, "gpu")]
    result = native.run_json([str(c) for c in cmd], gpu=gpu)
    result["gpu_index"] = gpu
    return result


def threads_all():
    """The number of logical CPUs the process may use."""
    return len(os.sched_getaffinity(0))


def edge_mask(tri):
    """Pixels whose 3 x 3 neighbourhood holds more than one triangle index."""
    padded = np.pad(tri, 1, mode="edge")
    edge = np.zeros(tri.shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            edge |= padded[dy : dy + tri.shape[0], dx : dx + tri.shape[1]] != tri
    return edge


def compare(a, b):
    """Statistics of how two output sets differ, with where the differences are."""
    tri = a["tri"] != b["tri"]
    depth = a["depth"] != b["depth"]
    colour = (a["colour"] != b["colour"]).any(axis=2)
    anywhere = tri | depth | colour
    dc = np.abs(a["colour"].astype(np.float64) - b["colour"])
    dd = np.abs(a["depth"].astype(np.float64) - b["depth"])
    edge = edge_mask(a["tri"]) | edge_mask(b["tri"])
    return {
        "pixels": int(tri.size),
        "tri_differs": int(tri.sum()),
        "depth_differs": int(depth.sum()),
        "colour_differs": int(colour.sum()),
        "any_differs": int(anywhere.sum()),
        "differs_on_edge": int((anywhere & edge).sum()),
        "differs_inside": int((anywhere & ~edge).sum()),
        "colour_max_abs": float(dc.max()),
        "colour_mean_abs": float(dc.mean()),
        "depth_max_abs": float(dd.max()),
        "tile_counts_equal": bool(np.array_equal(a["tile_counts"], b["tile_counts"])),
        "tile_count_abs_diff": int(np.abs(a["tile_counts"] - b["tile_counts"]).sum()),
        "mask": anywhere,
    }
