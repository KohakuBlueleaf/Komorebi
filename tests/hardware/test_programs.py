"""Programs in the runtime's form on a card (KMR_CARD=1): the two programs example on
disjoint groups and combined, commands to both interleaved; every result exact."""

from pathlib import Path

import pytest
from child import run_json

pytestmark = pytest.mark.card
SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "16_two_programs"
    / "two_programs.py"
)


@pytest.mark.parametrize("phase", ["disjoint", "combined"])
def test_two_programs_interleaved_are_exact(phase):
    out = run_json(SCRIPT, "--phase", phase, "--cores", 10, "--rounds", 5)
    assert out["ok"] and out[phase]["wrong"] == 0 and out[phase]["commands"] == 10
    if phase == "disjoint":
        assert out[phase]["refused"]
