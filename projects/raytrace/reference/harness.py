"""Running the ray tracers on a scene, saving their results, comparing any two."""

import time
from pathlib import Path

import numpy as np

from reference import bvh, native, raytrace, results, runner, scenes

HERE = Path(__file__).resolve().parent
SCENES = list(scenes.PRESETS)
STAGES = ("primary_ms", "shadow_ms", "bounce_ms", "shadow2_ms", "shade_ms", "total_ms")
KEYS = ("image", "t", "tri", "nodes", "leaves")
TREELET_KIB = (128, 1024)
threads_all = runner.threads_all
REFERENCE = "numpy"
PIXELS = scenes.WIDTH * scenes.HEIGHT


def out_dir(scene, impl):
    """Scratch folder for the raw .npy output of one binary on one scene."""
    return runner.out_dir("raytrace", scene, impl)


def attach_power(scene, folder, record):
    """Adds the power figures and frame time of a sampled run to a stored result."""
    runner.attach_power("raytrace", scene, folder, record)


def load_outputs(folder):
    """The output arrays a binary saved in a folder."""
    return {key: np.load(Path(folder) / f"{key}.npy") for key in KEYS}


def ensure_tree(scene):
    """Builds and saves the tree and its treelet cuts if missing; returns the tree.

    Files in the data folder: <scene>_tree.npz, <scene>_tree/*.npy (for the C++ and
    CUDA code) and <scene>_treelets_<KiB>k.npz for the sizes in TREELET_KIB.
    """
    data = native.data_dir()
    native.ensure_scene(scene)
    path = data / f"{scene}_tree.npz"
    if not path.exists() or not (data / f"{scene}_tree").exists():
        tree = bvh.build(scenes.load_scene(data / f"{scene}.npz"))
        bvh.export_npy(tree, data / f"{scene}_tree")  # the .npz last: the folder whole
        bvh.save_tree(path, tree)
    tree = bvh.load_tree(path)
    for kib in TREELET_KIB:
        cuts = data / f"{scene}_treelets_{kib}k.npz"
        if not cuts.exists():
            cut = bvh.cut_treelets(tree, kib * 1024)
            scenes.save_beside(cuts, lambda name, cut=cut: np.savez(name, **cut))
    return tree


def lattice(stride):
    """Pixel numbers of every stride-th pixel in both directions, centred in blocks."""
    rows = np.arange(stride // 2, scenes.HEIGHT, stride)
    columns = np.arange(stride // 2, scenes.WIDTH, stride)
    return (rows[:, None] * scenes.WIDTH + columns[None, :]).ravel()


def save(scene, folder, name, arrays, record):
    """Stores an implementation's arrays and timing record in the results layout."""
    runner.save_run("raytrace", STAGES, scene, folder, name, arrays, record)


def run_numpy(scene, pixels=None):
    """Runs the numpy ray tracer; the full image is also saved as a result."""
    tree = ensure_tree(scene)
    data = scenes.load_scene(native.data_dir() / f"{scene}.npz")
    start = time.monotonic()
    arrays = raytrace.render(data, tree, pixels)
    seconds = time.monotonic() - start
    if pixels is None:
        arrays["colour"] = arrays["image"]
        save(scene, "numpy", "numpy", arrays, {"loop_ms": seconds * 1000, "reps_n": 1})
    return arrays


def run_cpu(scene, threads, repetitions=3, store=True, with_power=False):
    """Runs the C++ binary; returns its record (None without a compiler)."""
    executable = native.build(str(HERE / "cpu" / "raytrace_cpu.cpp"), "raytrace_cpu")
    if executable is None:
        return None
    ensure_tree(scene)
    data = native.data_dir()
    command_line = [
        executable,
        "--scene",
        data / scene,
        "--tree",
        data / f"{scene}_tree",
    ]
    command_line += ["--threads", threads, "--reps", repetitions]
    folder = "cpu_all" if threads == threads_all() else f"cpu{threads}"
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
        str(HERE / "gpu" / "raytrace_gpu.cu"), "raytrace_gpu", gpu=True
    )
    gpu_index = native.idle_gpu()
    if executable is None or gpu_index is None:
        return None
    ensure_tree(scene)
    data = native.data_dir()
    command_line = [
        executable,
        "--scene",
        data / scene,
        "--tree",
        data / f"{scene}_tree",
    ]
    command_line += ["--reps", repetitions]
    if store:
        command_line += ["--out", out_dir(scene, "v100")]
    record = runner.run_binary(command_line, gpu_index, with_power)
    record["gpu_index"] = gpu_index
    if store:
        save(scene, "v100", "V100", load_outputs(out_dir(scene, "v100")), record)
    return record


def ensure_results(scene, numpy_max=None, fresh=False):
    """Makes sure numpy (if the scene is small enough), C++ and CUDA results exist.

    Results already stored are kept, with their timings and power figures, unless
    fresh is set.
    """
    root = runner.results_root("raytrace")

    def todo(folder):
        return fresh or not (root / folder / f"{scene}.npz").exists()

    if todo("numpy") and (
        numpy_max is None or native.scene_triangles(scene) <= numpy_max
    ):
        run_numpy(scene)
    if todo("cpu_all"):
        run_cpu(scene, threads_all(), repetitions=1)
    if todo("v100"):
        run_gpu(scene, repetitions=1)


def collect(scene, external=()):
    """{name: (arrays, info)} of every implementation found for a scene."""
    roots = [runner.results_root("raytrace"), *external]
    return results.find_results(roots, scene, "raytrace")


def compare(first, second):
    """Statistics of how two outputs differ; arrays are (H, W) images or flat subsets.

    Triangle mismatches are split into exact ties (equal distance), near ties (distance
    within 1e-5 relative), one hit and one miss, and the rest.
    """
    image = first.get("image", first.get("colour"))
    other = second.get("image", second.get("colour"))
    first_hit, second_hit = first["tri"] >= 0, second["tri"] >= 0
    both = first_hit & second_hit
    triangle_differs = first["tri"] != second["tri"]
    exact_tie = triangle_differs & both & (first["t"] == second["t"])
    gap = np.abs(first["t"].astype(np.float64) - second["t"])
    near_tie = (
        triangle_differs
        & both
        & ~exact_tie
        & (gap <= 1e-5 * np.maximum(first["t"], second["t"]))
    )
    hit_vs_miss = triangle_differs & (first_hit != second_hit)
    distance_differs = both & (first["t"] != second["t"])
    error = np.abs(image.astype(np.float64) - other).max(axis=-1)
    anywhere = triangle_differs | distance_differs | (error > 0)
    stats = {
        "pixels": int(triangle_differs.size),
        "tri_differs": int(triangle_differs.sum()),
        "tri_exact_ties": int(exact_tie.sum()),
        "tri_near_ties": int(near_tie.sum()),
        "tri_hit_vs_miss": int(hit_vs_miss.sum()),
        "tri_other": int(
            (triangle_differs & ~exact_tie & ~near_tie & ~hit_vs_miss).sum()
        ),
        "t_differs": int(distance_differs.sum()),
        "t_max_rel": float(
            (gap / np.maximum(first["t"], 1e-30))[both].max(initial=0.0)
        ),
        "image_differs": int((error > 0).sum()),
        "image_over_1_255": int((error > 1 / 255).sum()),
        "image_max_abs": float(error.max()),
        "image_mean_abs": float(error.mean()),
        "image_rms": float(np.sqrt((error**2).mean())),
        "any_differs": int(anywhere.sum()),
        "mask": anywhere,
    }
    if triangle_differs.ndim == 2:
        edge = runner.edge_mask(first["tri"]) | runner.edge_mask(second["tri"])
        stats["differs_on_edge"] = int((anywhere & edge).sum())
        stats["differs_inside"] = int((anywhere & ~edge).sum())
    return stats
