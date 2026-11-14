"""Programs in the runtime's form on a card (KMR_CARD=1): the two programs example on
disjoint groups, commands to both interleaved, and a group that switches program on the
live run; every result exact."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "16_two_programs"
    / "two_programs.py"
)


def test_two_programs_on_disjoint_groups_are_exact_and_refused_elsewhere():
    out = run_json(SCRIPT, "--phase", "disjoint", "--cores", 10, "--rounds", 5)
    assert out["ok"] and out["disjoint"]["wrong"] == 0
    assert out["disjoint"]["commands"] == 10 and out["disjoint"]["refused"]


def test_a_group_switches_program_on_a_live_run():
    out = run_json(SCRIPT, "--phase", "live", "--cores", 10, "--switches", 12)
    live = out["live"]
    assert out["ok"] and live["wrong"] == 0 and live["switches"] == 12
    assert live["refused"] == 6 and live["refused_other"] == 0
