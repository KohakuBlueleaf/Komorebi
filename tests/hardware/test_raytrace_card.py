"""The ray tracer on a card through the dispatcher (KMR_CARD=1): the vector unit's
arithmetic bit for bit; on tiles, the four passes (tree in L1) and the frame's stages
(treelets from DRAM, rays sorted on each core), and the frame as a graph in the kernel
and host-fed, against numpy."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
HERE = Path(__file__).resolve().parents[2] / "projects" / "raytrace"
TILES = ("--tiles", "0", "300", "528", "1000")


def test_slab_and_triangle_arithmetic_is_exact():
    out = run_json(
        HERE / "arith.py", "--n", "4096", "--scene", "tiny", "--math", "4096"
    )
    assert out["ok"] and out["check"]["box_hit_flag_differs"] == 0
    assert (
        out["math"]["divide"]["differ_bits"] == out["math"]["sqrt"]["differ_bits"] == 0
    )


def test_four_passes_agree_on_tiles():
    out = run_json(HERE / "run.py", "--scene", "tiny", *TILES)
    assert out["ok"] and out["passes"] == ["primary", "shadow", "bounce", "shadow2"]


@pytest.mark.parametrize(
    "scene",
    [("s10k", "--treelets", "128"), ("s10k", "--treelets", "128", "--sort")],
    ids=["treelets", "sorted"],
)
def test_frame_stages_on_the_card_agree_on_tiles(scene):
    out = run_json(HERE / "frame.py", "--scene", *scene, *TILES, "--stages")
    assert out["ok"] and out["image_check"]["differs"] == 0
    assert set(out["stages"].values()) == {0}


def test_frame_graph_gives_the_same_image_in_the_kernel_and_host_fed():
    out = run_json(HERE / "framegraph.py", *TILES, "--cameras", "2")
    assert out["ok"] and out["hostfed_commands"] == [8, 8]
    for row in out["image_checks"]:
        assert row["graph"] == row["hostfed"] == row["graph_vs_hostfed"] == 0
