"""Every project's program fits its session's kernel buffer with room to spare (no
card): each driver runs to its first load on a mock device, and the runtime's layout
of the five built kernels is compared with the buffer. KMR_SIZES=1 and TT_METAL_HOME
(a built tree); the native module and every program are built, minutes when cold."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from komorebi import native

pytestmark = pytest.mark.sizes
ROOT = Path(__file__).resolve().parents[2]
CHILD = Path(__file__).with_name("kernel_sizes_child.py")
DESCRIPTOR = Path("tt_metal/third_party/umd/tests/cluster_descriptor_examples")
HEADROOM = 1024  # bytes of the buffer left after the five kernels' text

# name: arguments of the child (a driver script and its arguments, or --program)
CASES = {
    "raytrace": ["projects/raytrace/framegraph.py", "--tiles", "0", "300"],
    "attention": ["projects/attention/run.py", "--shape", "t3", "--calls", "2"],
    "dense": ["projects/dense/run.py", "--m", "256", "--k", "64", "--n", "128"],
    "dense_grid": [
        *("projects/dense/run.py", "--m", "512", "--k", "64", "--n", "384"),
        *("--form", "grid", "--grid", "2d:4,3,4,4,2,2,1"),
    ],
    "dense_split": [
        *("projects/dense/run.py", "--m", "160", "--k", "64", "--n", "96"),
        *("--split", "2,2,2,1"),
    ],
    "swiglu": ["projects/dense/swiglu.py", "--shape", "sw256"],
    "raster": ["--program", "projects.raster.card:program"],
    "raster_passes": ["projects/raster/card_passes.py", "--setup"],
    "raster_frame": [
        *("projects/raster/card_thread.py", "--scene", "s10k", "--steps", "4"),
    ],
}
# A case whose kernels do not fit yet is an expected failure, and strictly: a fix
# turns it into a failure that asks for the entry's removal.
OVER = {
    "raytrace": (
        "its 80 KiB buffer holds 1.4 KiB less than its kernels take since the host "
        "channel's handlers and the 64 command slots"
    ),
}


@pytest.fixture(scope="module")
def environment(tmp_path_factory):
    """The child's environment: the tree, a mock cluster, the JIT cache."""
    try:
        tree = native.metal_tree()
    except Exception:  # noqa: BLE001 (no tree: the tests are skipped)
        pytest.skip("needs TT_METAL_HOME")
    descriptor = tree / DESCRIPTOR / "blackhole_P150.yaml"
    if not (tree / "build").is_dir() or not descriptor.is_file():
        pytest.skip("needs a built tt-metal tree with its cluster descriptors")
    return {
        **os.environ,
        "TT_METAL_HOME": str(tree),
        "TT_METAL_RUNTIME_ROOT": str(tree),
        "TT_METAL_SLOW_DISPATCH_MODE": "1",
        "TT_METAL_MOCK_CLUSTER_DESC_PATH": str(descriptor),
        "TT_METAL_CACHE": str(tmp_path_factory.mktemp("jit")),
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
    }


def build_sizes(environment, arguments):
    """The child's `SIZES` row: program, text per thread, total, buffer, fits."""
    done = subprocess.run(
        [sys.executable, str(CHILD), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    rows = [line for line in done.stdout.splitlines() if line.startswith("SIZES ")]
    assert rows, f"no build reached (exit {done.returncode}):\n{done.stderr[-2000:]}"
    return json.loads(rows[-1].removeprefix("SIZES "))


@pytest.mark.parametrize(
    "name",
    [
        (
            pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=OVER[name]))
            if name in OVER
            else name
        )
        for name in CASES
    ],
)
def test_the_kernels_fit_the_buffer_with_room_to_spare(environment, name):
    row = build_sizes(environment, CASES[name])
    text = ", ".join(f"{thread} {size}" for thread, size in row["text"].items())
    assert (
        row["fits"] and row["total"] + HEADROOM <= row["limit"]
    ), f"{name}: {row['total']} of {row['limit']} B ({text})"
