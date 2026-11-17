"""tests/projects/sizes: the environment of the child that builds a program on a mock
device, one a test module."""

import os

import pytest
from kernel_sizes_fixture import DESCRIPTOR, ROOT

from komorebi.device import native


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
