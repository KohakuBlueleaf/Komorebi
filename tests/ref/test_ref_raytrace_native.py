"""C++ and CUDA ray tracers against numpy, bit for bit, on sampled pixels."""

import os

import numpy as np
import pytest

from projects.raytrace.reference import agree, harness
from reference import bvh, native, raytrace, scenes

needs_cxx = pytest.mark.skipif(native.find_cxx() is None, reason="no C++ compiler")
needs_gpu = pytest.mark.skipif(
    os.environ.get("KMR_GPU") != "1"
    or native.find_nvcc() is None
    or native.idle_gpu() is None,
    reason="needs KMR_GPU=1, a CUDA toolchain and an idle GPU",
)
STRIDE = 8


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    path = tmp_path_factory.mktemp("native")
    patch = pytest.MonkeyPatch()
    patch.setenv("KR_BUILD_DIR", str(path / "build"))
    patch.setenv("KR_DATA", str(path / "data"))
    yield path
    patch.undo()


def hand_made():
    """A wall with a blocker in front of it and a floor, lit from above and the side."""
    wall = [(-20, -10, 12), (20, -10, 12), (-20, 10, 12), (20, -10, 12)]
    wall += [(20, 10, 12), (-20, 10, 12)]
    blocker = [(-3, 1, 6), (3, 1, 6), (0, 6, 6)]
    floor = [(-30, -4, 1), (30, -4, 1), (-30, -4, 40), (30, -4, 1), (30, -4, 40)]
    floor += [(-30, -4, 40)]
    positions = np.array(wall + blocker + floor, dtype=np.float32)
    vertex_count = len(positions)
    colors = (
        np.linspace(0.2, 0.9, 3 * vertex_count)
        .reshape(vertex_count, 3)
        .astype(np.float32)
    )
    camera = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
    camera = np.concatenate([camera, [512, 512, 512, 0.05]]).astype(np.float32)
    return {
        "positions": positions,
        "colors": colors,
        "indices": np.arange(vertex_count, dtype=np.int32).reshape(-1, 3),
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
        command, gpu = [exe, *base, "--threads", threads, "--reps", 1], None
    else:
        source = str(here / "gpu" / "raytrace_gpu.cu")
        exe = native.build(source, "raytrace_gpu", gpu=True)
        command, gpu = [exe, *base, "--reps", 1], native.idle_gpu()
    native.run_json([str(argument) for argument in command], gpu=gpu)
    return harness.load_outputs(out)


def assert_same_on_pixels(scene, tree, outputs):
    pixels = harness.lattice(STRIDE)
    expect = raytrace.render(scene, tree, pixels)
    got = agree.restrict(
        {name: outputs[name] for name in ("image", "t", "tri", "nodes")}, pixels
    )
    for key in ("image", "t", "tri", "nodes"):
        assert np.array_equal(expect[key], got[key]), key
    assert (expect["tri"] >= 0).any() and (expect["tri"] < 0).any()


@pytest.fixture(scope="module")
def tiny(root):
    scene = scenes.make_preset("tiny")
    return scene, folders(root, "tiny", scene)


@pytest.fixture(scope="module")
def hand(root):
    scene = hand_made()
    return scene, folders(root, "hand", scene)


@pytest.fixture(scope="module")
def s10k(root):
    scene = scenes.make_preset("s10k")
    return scene, folders(root, "s10k", scene)


@needs_cxx
@pytest.mark.parametrize("threads", [1, 3, 16])
def test_cpu_matches_numpy_on_the_tiny_scene(root, tiny, threads):
    assert_same_on_pixels(*tiny, run_folder("cpu", root, "tiny", threads))


@needs_cxx
def test_cpu_matches_numpy_on_hand_made_triangles(root, hand):
    assert_same_on_pixels(*hand, run_folder("cpu", root, "hand"))


@needs_cxx
def test_cpu_matches_numpy_on_the_ten_thousand_triangle_scene(root, s10k):
    assert_same_on_pixels(*s10k, run_folder("cpu", root, "s10k", 8))


@pytest.mark.gpu
@needs_gpu
def test_gpu_matches_numpy_on_hand_made_triangles(root, hand):
    assert_same_on_pixels(*hand, run_folder("gpu", root, "hand"))


@pytest.mark.gpu
@needs_gpu
def test_gpu_matches_numpy_on_the_ten_thousand_triangle_scene(root, s10k):
    assert_same_on_pixels(*s10k, run_folder("gpu", root, "s10k"))
