"""A leader casts which overlay its group runs, on a card (KMR_CARD=1): down a column of
4, overlays named in turn so that every round loads, every core's data exact."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "17_leader_overlay"
    / "leader_overlay.py"
)


def test_every_core_loads_the_overlay_its_leader_casts_exact():
    args = ["--rect", "0,0,1,4", "--rounds", 2000, "--per", 500, "--timed", 1]
    out = run_json(SCRIPT, *args, "--choices", "cycle")
    assert out["ok"] and out["cores_bad"] == 0
    assert out["loads_per_core"] == {"reader": 2000, "with_compute": 1000}
    slots = {part: value[0] for part, value in out["slots"].items()}
    assert slots == {"reader": 3, "unpack": 1, "math": 1, "pack": 1}
