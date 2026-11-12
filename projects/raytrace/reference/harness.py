"""Running the ray tracers on a scene, saving their results, comparing any two."""

import time
from pathlib import Path

import numpy as np

from ref import bvh, native, raytrace, results, runner, scenes

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
        bvh.save_tree(path, tree)
        bvh.export_npy(tree, data / f"{scene}_tree")
    tree = bvh.load_tree(path)
    for kib in TREELET_KIB:
        cuts = data / f"{scene}_treelets_{kib}k.npz"
        if not cuts.exists():
            np.savez(cuts, **bvh.cut_treelets(tree, kib * 1024))
    return tree


def lattice(stride):
    """Pixel numbers of every stride-th pixel in both directions, centred in blocks."""
    ys = np.arange(stride // 2, scenes.HEIGHT, stride)
    xs = np.arange(stride // 2, scenes.WIDTH, stride)
    return (ys[:, None] * scenes.WIDTH + xs[None, :]).ravel()


def save(scene, folder, name, arrays, record):
    """Stores an implementation's arrays and timing record in the results layout."""
    runner.save_run("raytrace", STAGES, scene, folder, name, arrays, record)


def run_numpy(scene, pixels=None):
    """Runs the numpy ray tracer; the full image is also saved as a result."""
    tree = ensure_tree(scene)
    data = scenes.load_scene(native.data_dir() / f"{scene}.npz")
    start = time.monotonic()
    out = raytrace.render(data, tree, pixels)
    seconds = time.monotonic() - start
    if pixels is None:
        out["colour"] = out["image"]
        save(scene, "numpy", "numpy", out, {"loop_ms": seconds * 1000, "reps_n": 1})
    return out


def run_cpu(scene, threads, reps=3, store=True, with_power=False):
    """Runs the C++ binary; returns its record (None without a compiler)."""
    exe = native.build(str(HERE / "cpu" / "raytrace_cpu.cpp"), "raytrace_cpu")
    if exe is None:
        return None
    ensure_tree(scene)
    data = native.data_dir()
    cmd = [exe, "--scene", data / scene, "--tree", data / f"{scene}_tree"]
    cmd += ["--threads", threads, "--reps", reps]
    folder = "cpu_all" if threads == threads_all() else f"cpu{threads}"
    if store:
        cmd += ["--out", out_dir(scene, folder)]
    record = runner.run_binary(cmd, None, with_power)
    if store:
        label = "C++ all threads" if folder == "cpu_all" else f"C++ {threads} thread(s)"
        save(scene, folder, label, load_outputs(out_dir(scene, folder)), record)
    return record


def run_gpu(scene, reps=3, store=True, with_power=False):
    """Runs the CUDA binary on an idle GPU; None without a toolchain or idle GPU."""
    exe = native.build(str(HERE / "gpu" / "raytrace_gpu.cu"), "raytrace_gpu", gpu=True)
    gpu = native.idle_gpu()
    if exe is None or gpu is None:
        return None
    ensure_tree(scene)
    data = native.data_dir()
    cmd = [exe, "--scene", data / scene, "--tree", data / f"{scene}_tree"]
    cmd += ["--reps", reps]
    if store:
        cmd += ["--out", out_dir(scene, "v100")]
    record = runner.run_binary(cmd, gpu, with_power)
    record["gpu_index"] = gpu
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

    if todo("numpy") and (numpy_max is None or scenes.PRESETS[scene] <= numpy_max):
        run_numpy(scene)
    if todo("cpu_all"):
        run_cpu(scene, threads_all(), reps=1)
    if todo("v100"):
        run_gpu(scene, reps=1)


def collect(scene, external=()):
    """{name: (arrays, info)} of every implementation found for a scene."""
    roots = [runner.results_root("raytrace"), *external]
    return results.find_results(roots, scene, "raytrace")


def compare(a, b):
    """Statistics of how two outputs differ; arrays are (H, W) images or flat subsets.

    Triangle mismatches are split into exact ties (equal distance), near ties (distance
    within 1e-5 relative), one hit and one miss, and the rest.
    """
    image = a.get("image", a.get("colour"))
    other = b.get("image", b.get("colour"))
    hit_a, hit_b = a["tri"] >= 0, b["tri"] >= 0
    both = hit_a & hit_b
    tri = a["tri"] != b["tri"]
    tie = tri & both & (a["t"] == b["t"])
    gap = np.abs(a["t"].astype(np.float64) - b["t"])
    near = tri & both & ~tie & (gap <= 1e-5 * np.maximum(a["t"], b["t"]))
    miss = tri & (hit_a != hit_b)
    t_differs = both & (a["t"] != b["t"])
    err = np.abs(image.astype(np.float64) - other).max(axis=-1)
    anywhere = tri | t_differs | (err > 0)
    stats = {
        "pixels": int(tri.size),
        "tri_differs": int(tri.sum()),
        "tri_exact_ties": int(tie.sum()),
        "tri_near_ties": int(near.sum()),
        "tri_hit_vs_miss": int(miss.sum()),
        "tri_other": int((tri & ~tie & ~near & ~miss).sum()),
        "t_differs": int(t_differs.sum()),
        "t_max_rel": float((gap / np.maximum(a["t"], 1e-30))[both].max(initial=0.0)),
        "image_differs": int((err > 0).sum()),
        "image_over_1_255": int((err > 1 / 255).sum()),
        "image_max_abs": float(err.max()),
        "image_mean_abs": float(err.mean()),
        "image_rms": float(np.sqrt((err**2).mean())),
        "any_differs": int(anywhere.sum()),
        "mask": anywhere,
    }
    if tri.ndim == 2:
        edge = runner.edge_mask(a["tri"]) | runner.edge_mask(b["tri"])
        stats["differs_on_edge"] = int((anywhere & edge).sum())
        stats["differs_inside"] = int((anywhere & ~edge).sum())
    return stats
