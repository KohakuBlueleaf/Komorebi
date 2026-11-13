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

    if todo("numpy") and (
        numpy_max is None or native.scene_triangles(scene) <= numpy_max
    ):
        run_numpy(scene)
    if todo("cpu_all"):
        run_cpu(scene, runner.threads_all(), repetitions=1)
    if todo("v100"):
        run_gpu(scene, repetitions=1)


def run_cpu(scene, threads, repetitions=3, store=True, with_power=False):
    """Runs the C++ binary; returns its record (None without a compiler)."""
    executable = native.build(str(HERE / "cpu" / "raster_cpu.cpp"), "raster_cpu")
    if executable is None:
        return None
    command_line = [
        executable,
        "--scene",
        native.ensure_scene(scene),
        "--threads",
        threads,
    ]
    command_line += ["--reps", repetitions]
    folder = "cpu_all" if threads == runner.threads_all() else f"cpu{threads}"
    if store:
        command_line += ["--out", out_dir(scene, folder)]
    record = runner.run_binary(command_line, None, with_power)
    if store:
        label = "C++ all threads" if folder == "cpu_all" else f"C++ {threads} thread(s)"
        save(scene, folder, label, load_outputs(out_dir(scene, folder)), record)
    return record


def run_gpu(scene, repetitions=3, store=True, with_power=False):
    """Runs the CUDA binary on an idle GPU; None without a toolchain or idle GPU."""
    executable = native.build(
        str(HERE / "gpu" / "raster_gpu.cu"), "raster_gpu", gpu=True
    )
    gpu = native.idle_gpu()
    if executable is None or gpu is None:
        return None
    command_line = [
        executable,
        "--scene",
        native.ensure_scene(scene),
        "--reps",
        repetitions,
    ]
    if store:
        command_line += ["--out", out_dir(scene, "v100")]
    record = runner.run_binary(command_line, gpu, with_power)
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


def compare(first, second):
    """Statistics of how two output sets differ, with where the differences are.

    `tri` is compared only when both have it; edges come from it, else from coverage.
    """
    has_tri = "tri" in first and "tri" in second
    depth = first["depth"] != second["depth"]
    colour = (first["colour"] != second["colour"]).any(axis=2)
    tri = first["tri"] != second["tri"] if has_tri else np.zeros_like(depth)
    anywhere = tri | depth | colour
    colour_difference = np.abs(first["colour"].astype(np.float64) - second["colour"])
    depth_difference = np.abs(first["depth"].astype(np.float64) - second["depth"])
    shape = first["tri"] if has_tri else (first["depth"] > 0).astype(np.int32)
    other = second["tri"] if has_tri else (second["depth"] > 0).astype(np.int32)
    edge = edge_mask(shape) | edge_mask(other)
    return {
        "pixels": int(depth.size),
        "tri_differs": int(tri.sum()) if has_tri else None,
        "depth_differs": int(depth.sum()),
        "colour_differs": int(colour.sum()),
        "any_differs": int(anywhere.sum()),
        "differs_on_edge": int((anywhere & edge).sum()),
        "differs_inside": int((anywhere & ~edge).sum()),
        "colour_max_abs": float(colour_difference.max()),
        "colour_mean_abs": float(colour_difference.mean()),
        "depth_max_abs": float(depth_difference.max()),
        "tile_counts_equal": bool(
            np.array_equal(first["tile_counts"], second["tile_counts"])
        ),
        "tile_count_abs_diff": int(
            np.abs(first["tile_counts"] - second["tile_counts"]).sum()
        ),
        "mask": anywhere,
    }
