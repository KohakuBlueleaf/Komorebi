"""Overlays on a card (KMR_CARD=1): the overlays example on four cores, with one slot
(each load replaces the other overlay); every result exact."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples" / "12_overlays" / "overlays.py"
)


def test_two_overlays_in_turn_are_exact():
    slots = 1
    out = run_json(SCRIPT, "--cores", 4, "--rounds", 3, "--slots", slots)
    assert out["exact"] and out["commands"] == 12
    assert set(out["image_bytes"]) == {
        "1-reader",
        "2-reader",
        "2-writer",
        "2-unpack",
        "2-math",
        "2-pack",
    }
    assert {
        name: value[0] for name, value in out["slots_bytes"].items()
    } == dict.fromkeys(("reader", "writer", "unpack", "math", "pack"), slots)
    assert ("mix loading" in out["medians"]) and ("scale held" in out["medians"])
