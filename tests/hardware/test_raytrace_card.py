"""The ray tracer on a card through the dispatch kernel (KMR_CARD=1): the vector unit's
arithmetic bit for bit, then the four passes on tiles of the tiny scene (tree in L1)
and of the 10^4 scene (treelets from DRAM), against numpy pass by pass."""

from pathlib import Path

import pytest
from child import run_json

pytestmark = pytest.mark.card
HERE = Path(__file__).resolve().parents[2] / "projects" / "raytrace"
TILES = ("--tiles", "0", "300", "528", "1000")


def test_slab_and_triangle_arithmetic_is_exact():
    out = run_json(HERE / "arith.py", "--n", "4096", "--scene", "tiny")
    assert out["ok"] and out["check"]["box_hit_flag_differs"] == 0


@pytest.mark.parametrize(
    "scene", [("tiny",), ("s10k", "--treelets", "128")], ids=["tree", "treelets"]
)
def test_four_passes_agree_on_tiles(scene):
    out = run_json(HERE / "run.py", "--scene", *scene, *TILES, timeout=300.0)
    assert out["ok"] and out["passes"] == ["primary", "shadow", "bounce", "shadow2"]
