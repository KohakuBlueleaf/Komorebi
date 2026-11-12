"""Running the rasterizers on a scene, saving their results, comparing any two."""

import time
from pathlib import Path

import numpy as np

from ref import native, raster, results, runner, scenes

HERE = Path(__file__).resolve().parent
SCENES = list(scenes.PRESETS)
KEYS = ("colour", "depth", "tri", "tile_counts")
threads_all = runner.threads_all
REFERENCE = "numpy"
STAGES = ("setup_ms", "bin_ms", "fill_ms", "total_ms")


def out_dir(scene, impl):
    """Scratch folder for the raw .npy output of one binary on one scene."""
    return runner.out_dir("raster", scene, impl)


def attach_power(scene, folder, record):
    """Adds the power figures and frame time of a sampled run to a stored result."""
    runner.attach_power("raster", scene, folder, record)


def load_outputs(folder):
    """The four output arrays a binary saved in a folder."""
    return {key: np.load(Path(folder) / f"{key}.npy") for key in KEYS}


def save(scene, folder, name, arrays, record):
    """Stores an implementation's arrays and timing record in the results layout."""
    runner.save_run("raster", STAGES, scene, folder, name, arrays, record)


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


def ensure_results(scene, numpy_max=None, fresh=False):
    """Makes sure numpy (if the scene is small enough), C++ and CUDA results exist.

    Results already stored are kept, with their timings and power figures, unless
    fresh is set.
    """
    root = runner.results_root("raster")

    def todo(folder):
        return fresh or not (root / folder / f"{scene}.npz").exists()

    if todo("numpy") and (numpy_max is None or scenes.PRESETS[scene] <= numpy_max):
        run_numpy(scene)
    if todo("cpu_all"):
        run_cpu(scene, runner.threads_all(), reps=1)
    if todo("v100"):
        run_gpu(scene, reps=1)


def run_cpu(scene, threads, reps=3, store=True, with_power=False):
    """Runs the C++ binary; returns its record (None without a compiler)."""
    exe = native.build(str(HERE / "cpu" / "raster_cpu.cpp"), "raster_cpu")
    if exe is None:
        return None
    cmd = [exe, "--scene", native.ensure_scene(scene), "--threads", threads]
    cmd += ["--reps", reps]
    folder = "cpu_all" if threads == runner.threads_all() else f"cpu{threads}"
    if store:
        cmd += ["--out", out_dir(scene, folder)]
    record = runner.run_binary(cmd, None, with_power)
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
    record = runner.run_binary(cmd, gpu, with_power)
    record["gpu_index"] = gpu
    if store:
        save(scene, "v100", "V100", load_outputs(out_dir(scene, "v100")), record)
    return record


def collect(scene, external=()):
    """{name: (arrays, info)} of every implementation found for a scene."""
    return results.find_results(
        [runner.results_root("raster"), *external], scene, "raster"
    )


edge_mask = runner.edge_mask


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
