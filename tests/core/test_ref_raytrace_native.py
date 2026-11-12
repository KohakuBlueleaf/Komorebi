"""C++ and CUDA ray tracers against numpy, bit for bit, on sampled pixels."""

import numpy as np
import pytest

from projects.raytrace.reference import agree, harness
from ref import bvh, native, raytrace, scenes

needs_cxx = pytest.mark.skipif(native.find_cxx() is None, reason="no C++ compiler")
needs_gpu = pytest.mark.skipif(
    native.find_nvcc() is None or native.idle_gpu() is None,
    reason="no CUDA toolchain or no idle GPU",
)
STRIDE = 8


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    path = tmp_path_factory.mktemp("native")
    mp = pytest.MonkeyPatch()
    mp.setenv("KR_BUILD_DIR", str(path / "build"))
    mp.setenv("KR_DATA", str(path / "data"))
    yield path
    mp.undo()


def hand_made():
    """A wall with a blocker in front of it and a floor, lit from above and the side."""
    wall = [(-20, -10, 12), (20, -10, 12), (-20, 10, 12), (20, -10, 12)]
    wall += [(20, 10, 12), (-20, 10, 12)]
    blocker = [(-3, 1, 6), (3, 1, 6), (0, 6, 6)]
    floor = [(-30, -4, 1), (30, -4, 1), (-30, -4, 40), (30, -4, 1), (30, -4, 40)]
    floor += [(-30, -4, 40)]
    pos = np.array(wall + blocker + floor, dtype=np.float32)
    n = len(pos)
    colors = np.linspace(0.2, 0.9, 3 * n).reshape(n, 3).astype(np.float32)
    camera = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
    camera = np.concatenate([camera, [512, 512, 512, 0.05]]).astype(np.float32)
    return {
        "positions": pos,
        "colors": colors,
        "indices": np.arange(n, dtype=np.int32).reshape(-1, 3),
        "camera": camera,
        "light": np.array([4, 12, 2], dtype=np.float32),
        "eps": np.array([1e-3], dtype=np.float32),
        "meta": np.zeros(5, dtype=np.int32),
    }


def folders(root, name, scene):
    tree = bvh.build(scene)
    scenes.export_npy(scene, root / name)
    bvh.export_npy(tree, root / f"{name}_tree")
    return tree


def run_folder(kind, root, name, threads=4):
    out = root / f"out_{name}_{kind}_{threads}"
    out.mkdir(exist_ok=True)
    here = harness.HERE
    base = ["--scene", root / name, "--tree", root / f"{name}_tree", "--out", out]
    if kind == "cpu":
        exe = native.build(str(here / "cpu" / "raytrace_cpu.cpp"), "raytrace_cpu")
        cmd, gpu = [exe, *base, "--threads", threads, "--reps", 1], None
    else:
        source = str(here / "gpu" / "raytrace_gpu.cu")
        exe = native.build(source, "raytrace_gpu", gpu=True)
        cmd, gpu = [exe, *base, "--reps", 1], native.idle_gpu()
    native.run_json([str(c) for c in cmd], gpu=gpu)
    return harness.load_outputs(out)


def assert_same_on_pixels(scene, tree, outputs):
    pixels = harness.lattice(STRIDE)
    expect = raytrace.render(scene, tree, pixels)
    got = agree.restrict(
        {k: outputs[k] for k in ("image", "t", "tri", "nodes")}, pixels
    )
    for key in ("image", "t", "tri", "nodes"):
        assert np.array_equal(expect[key], got[key]), key
    assert (expect["tri"] >= 0).any() and (expect["tri"] < 0).any()


@needs_cxx
@pytest.mark.parametrize("threads", [1, 3, 16])
def test_cpu_matches_numpy_on_the_tiny_scene(root, threads):
    scene = scenes.make_preset("tiny")
    tree = folders(root, "tiny", scene)
    assert_same_on_pixels(scene, tree, run_folder("cpu", root, "tiny", threads))


@needs_cxx
def test_cpu_matches_numpy_on_hand_made_triangles(root):
    scene = hand_made()
    tree = folders(root, "hand", scene)
    assert_same_on_pixels(scene, tree, run_folder("cpu", root, "hand"))


@needs_cxx
def test_cpu_matches_numpy_on_the_ten_thousand_triangle_scene(root):
    scene = scenes.make_preset("s10k")
    tree = folders(root, "s10k", scene)
    assert_same_on_pixels(scene, tree, run_folder("cpu", root, "s10k", 8))


@needs_gpu
def test_gpu_matches_numpy_on_hand_made_triangles(root):
    scene = hand_made()
    tree = folders(root, "hand", scene)
    assert_same_on_pixels(scene, tree, run_folder("gpu", root, "hand"))


@needs_gpu
def test_gpu_matches_numpy_on_the_ten_thousand_triangle_scene(root):
    scene = scenes.make_preset("s10k")
    tree = folders(root, "s10k", scene)
    assert_same_on_pixels(scene, tree, run_folder("gpu", root, "s10k"))
