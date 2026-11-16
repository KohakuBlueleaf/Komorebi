"""Inspect and give-up on a card (KMR_CARD=1), example 01's give_up.py: an inspect of
every core while the run goes on, then a command hung three ways and given up at its
deadline, each followed by exact commands on the same cores."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
SCRIPT = (
    Path(__file__).resolve().parents[3] / "examples" / "01_mapcollect" / "give_up.py"
)


def test_an_inspect_reads_every_core_while_the_run_goes_on():
    out = run_json(SCRIPT, "--modes", "none", "--rounds", 5)
    inspect = out["inspect"]
    assert out["ok"] and inspect["cores"] == inspect["read"] == 120
    assert inspect["waiting_for_a_command"] == 120 and inspect["microseconds"] > 0


def test_a_hung_command_is_given_up_and_the_next_ones_on_its_cores_are_exact():
    out = run_json(SCRIPT, "--rounds", 1)
    assert out["ok"] and out["maps_exact"]
    for name, mode in out["modes"].items():
        assert mode["given_up"] and mode["maps_done"] == 3, name
        assert mode["inspect"][mode["kernel"]] == mode["expected"], name
        assert mode["at_deadline"][mode["kernel"]] == mode["expected"], name
