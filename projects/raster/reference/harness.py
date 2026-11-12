"""Running the rasterizers on a scene, saving their results, comparing any two."""

import json
import os
import time
from pathlib import Path

import numpy as np

from ref import native, power, raster, results, scenes

HERE = Path(__file__).resolve().parent
SCENES = list(scenes.PRESETS)
KEYS = ("colour", "depth", "tri", "tile_counts")
REFERENCE = "numpy"
STAGES = ("setup_ms", "bin_ms", "fill_ms", "total_ms")


def results_root():
    """Folder with one sub-folder of results per implementation."""
    return native.data_dir() / "results" / "raster"


def out_dir(scene, impl):
    """Scratch folder for the raw .npy output of one binary on one scene."""
    path = native.data_dir() / "tmp" / "raster" / scene / impl
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_outputs(folder):
    """The four output arrays a binary saved in a folder."""
    return {key: np.load(Path(folder) / f"{key}.npy") for key in KEYS}


def threads_all():
    """The number of logical CPUs the process may use."""
    return len(os.sched_getaffinity(0))


def stage_medians(record):
    """Median of every timed stage over the repetitions listed in a binary's record."""
    reps = record.get("reps", [])
    return {k: float(np.median([r[k] for r in reps])) for k in STAGES} if reps else {}


def save(scene, folder, name, arrays, record):
    """Stores an implementation's arrays and timing record in the results layout."""
    info = {"name": name, "scene": scene, "timings_ms": stage_medians(record)}
    info["seconds_per_frame"] = record["loop_ms"] / record["reps_n"] / 1000
    if "power" in record:
        info["power"] = record["power"]
    results.save_result(results_root() / folder, scene, "raster", arrays, info)


def attach_power(scene, folder, record):
    """Adds the power figures and frame time of a sampled run to a stored result."""
    path = results_root() / folder / f"{scene}.json"
    if path.exists():
        info = json.loads(path.read_text())
        info["seconds_per_frame"] = record["power"]["seconds_per_frame"]
        info["power"] = record["power"]
        path.write_text(json.dumps(info, indent=1))


def run_numpy(scene):
    """Runs the numpy rasterizer, saves its results, returns the arrays."""
    native.ensure_scene(scene)
    data = scenes.load_scene(native.data_dir() / f"{scene}.npz")
    start = time.monotonic()
    arrays = raster.rasterize(data)
    seconds = time.monotonic() - start
    record = {"loop_ms": seconds * 1000, "reps_n": 1}
    save(scene, "numpy", "numpy", arrays, record)
    return arrays


def run_binary(command, gpu, with_power):
    """Runs a binary; with power sampling when asked."""
    command = [str(c) for c in command]
    if with_power:
        return power.run_with_power(command, gpu=gpu)
    return native.run_json(command, gpu=gpu)


def run_cpu(scene, threads, reps=3, store=True, with_power=False):
    """Runs the C++ binary; returns its record (None without a compiler)."""
    exe = native.build(str(HERE / "cpu" / "raster_cpu.cpp"), "raster_cpu")
    if exe is None:
        return None
    cmd = [exe, "--scene", native.ensure_scene(scene), "--threads", threads]
    cmd += ["--reps", reps]
    folder = "cpu_all" if threads == threads_all() else f"cpu{threads}"
    if store:
        cmd += ["--out", out_dir(scene, folder)]
    record = run_binary(cmd, None, with_power)
    if store:
        label = "C++ all threads" if folder == "cpu_all" else f"C++ {threads} thread(s)"
        save(scene, folder, label, load_outputs(out_dir(scene, folder)), record)
    return record


def run_gpu(scene, reps=3, store=True, with_power=False):
    """Runs the CUDA binary on an idle GPU; None without a toolchain or idle GPU."""
    exe = native.build(str(HERE / "gpu" / "raster_gpu.cu"), "raster_gpu", gpu=True)
    gpu = native.idle_gpu()
    if exe is None or gpu is None:
        return None
    cmd = [exe, "--scene", native.ensure_scene(scene), "--reps", reps]
    if store:
        cmd += ["--out", out_dir(scene, "v100")]
    record = run_binary(cmd, gpu, with_power)
    record["gpu_index"] = gpu
    if store:
        save(scene, "v100", "V100", load_outputs(out_dir(scene, "v100")), record)
    return record


def collect(scene, external=()):
    """{name: (arrays, info)} of every implementation found for a scene."""
    return results.find_results([results_root(), *external], scene, "raster")


def edge_mask(tri):
    """Pixels whose 3 x 3 neighbourhood holds more than one triangle index."""
    padded = np.pad(tri, 1, mode="edge")
    edge = np.zeros(tri.shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            edge |= padded[dy : dy + tri.shape[0], dx : dx + tri.shape[1]] != tri
    return edge


def compare(a, b):
    """Statistics of how two output sets differ, with where the differences are.

    `tri` is compared only when both have it; edges come from it, else from coverage.
    """
    has_tri = "tri" in a and "tri" in b
    depth = a["depth"] != b["depth"]
    colour = (a["colour"] != b["colour"]).any(axis=2)
    tri = a["tri"] != b["tri"] if has_tri else np.zeros_like(depth)
    anywhere = tri | depth | colour
    dc = np.abs(a["colour"].astype(np.float64) - b["colour"])
    dd = np.abs(a["depth"].astype(np.float64) - b["depth"])
    shape = a["tri"] if has_tri else (a["depth"] > 0).astype(np.int32)
    other = b["tri"] if has_tri else (b["depth"] > 0).astype(np.int32)
    edge = edge_mask(shape) | edge_mask(other)
    return {
        "pixels": int(depth.size),
        "tri_differs": int(tri.sum()) if has_tri else None,
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
