"""The C++ and CUDA rasterizers against the numpy one, bit for bit."""

import os

import numpy as np
import pytest

from projects.raster.reference import harness
from reference import native, raster, scenes
from tests.core.test_ref_raster import crossing_scene, screen_scene

HAND_MADE = {
    "diagonal": [(10, 10, 1), (18, 10, 1), (18, 18, 1), (10, 10, 1), (18, 18, 1)]
    + [(10, 18, 1)],
    "top and left edges": [(10, 12.5, 1), (18, 12.5, 1), (14, 20.5, 1)]
    + [(30, 12.5, 1), (34, 4.5, 1), (38, 12.5, 1)]
    + [(52.5, 10, 1), (60.5, 14, 1), (52.5, 18, 1)],
    "depth and ties": [(20, 20, 1), (60, 20, 1), (20, 60, 1)] * 2
    + [(10, 10, 4), (100, 10, 4), (10, 100, 4)],
    "dropped": [(300, 300, -1), (400, 300, -1), (300, 400, -1)]
    + [(-300, 300, 1), (-100, 300, 1), (-300, 400, 1)]
    + [(300, 300, 1), (400, 400, 1), (500, 500, 1)],
    "whole screen": [(-2000, -2000, 1), (6000, -2000, 1), (-2000, 6000, 1)],
}

needs_cxx = pytest.mark.skipif(native.find_cxx() is None, reason="no C++ compiler")
needs_gpu = pytest.mark.skipif(
    os.environ.get("KMR_GPU") != "1"
    or native.find_nvcc() is None
    or native.idle_gpu() is None,
    reason="needs KMR_GPU=1, a CUDA toolchain and an idle GPU",
)


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("native")
    patch = pytest.MonkeyPatch()
    patch.setenv("KR_BUILD_DIR", str(root / "build"))
    patch.setenv("KR_DATA", str(root / "data"))
    yield root
    patch.undo()


def run_folder(kind, folder, threads=4):
    """Runs a binary on a scene folder; returns its outputs."""
    out = folder / f"out_{kind}_{threads}"
    out.mkdir(exist_ok=True)
    here = harness.HERE
    if kind == "cpu":
        exe = native.build(str(here / "cpu" / "raster_cpu.cpp"), "raster_cpu")
        command = [
            exe,
            "--scene",
            folder,
            "--out",
            out,
            "--threads",
            threads,
            "--reps",
            1,
        ]
        native.run_json([str(argument) for argument in command])
    else:
        exe = native.build(str(here / "gpu" / "raster_gpu.cu"), "raster_gpu", gpu=True)
        command = [exe, "--scene", folder, "--out", out, "--reps", 1]
        native.run_json([str(argument) for argument in command], gpu=native.idle_gpu())
    return harness.load_outputs(out)


def assert_same(first, second):
    for key in harness.KEYS:
        assert np.array_equal(first[key], second[key]), key


def scene_folder(root, name, scene):
    folder = root / name.replace(" ", "_")
    scenes.export_npy(scene, folder)
    return folder


@needs_cxx
@pytest.mark.parametrize("name", list(HAND_MADE))
def test_cpu_matches_numpy_on_hand_made_triangles(paths, name):
    scene = screen_scene(HAND_MADE[name])
    scene |= {"light": np.zeros(3, np.float32), "eps": np.zeros(1, np.float32)}
    scene["meta"] = np.zeros(5, np.int32)
    folder = scene_folder(paths, name, scene)
    assert_same(run_folder("cpu", folder), raster.rasterize(scene))


@needs_cxx
@pytest.mark.parametrize("threads", [1, 3, 16])
def test_cpu_matches_numpy_on_the_tiny_scene(paths, threads):
    scene = scenes.make_preset("tiny")
    folder = scene_folder(paths, "tiny", scene)
    assert_same(run_folder("cpu", folder, threads), raster.rasterize(scene))


@needs_cxx
def test_cpu_matches_numpy_on_the_ten_thousand_triangle_scene(paths):
    scene = scenes.make_preset("s10k")
    folder = scene_folder(paths, "s10k", scene)
    assert_same(run_folder("cpu", folder, 8), raster.rasterize(scene))


@needs_gpu
@pytest.mark.parametrize("name", ["diagonal", "top and left edges", "depth and ties"])
def test_gpu_matches_numpy_on_hand_made_triangles(paths, name):
    scene = screen_scene(HAND_MADE[name])
    scene |= {"light": np.zeros(3, np.float32), "eps": np.zeros(1, np.float32)}
    scene["meta"] = np.zeros(5, np.int32)
    folder = scene_folder(paths, name, scene)
    assert_same(run_folder("gpu", folder), raster.rasterize(scene))


@needs_gpu
def test_gpu_matches_numpy_on_the_ten_thousand_triangle_scene(paths):
    scene = scenes.make_preset("s10k")
    folder = scene_folder(paths, "s10k", scene)
    assert_same(run_folder("gpu", folder), raster.rasterize(scene))


def with_extras(scene):
    """The arrays a scene folder needs beyond the mesh and camera."""
    scene = dict(scene)
    scene |= {"light": np.zeros(3, np.float32), "eps": np.zeros(1, np.float32)}
    scene["meta"] = np.zeros(5, np.int32)
    return scene


@needs_cxx
def test_cpu_matches_numpy_on_crossing_triangles(paths):
    scene = with_extras(crossing_scene())
    folder = scene_folder(paths, "crossing", scene)
    assert_same(run_folder("cpu", folder), raster.rasterize(scene))


@needs_gpu
def test_gpu_matches_numpy_on_crossing_triangles(paths):
    scene = with_extras(crossing_scene())
    folder = scene_folder(paths, "crossing_gpu", scene)
    assert_same(run_folder("gpu", folder), raster.rasterize(scene))
