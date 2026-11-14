"""An X280 thread issues the load (KMR_CARD=1): example 21 on a row, the thread moving
the boundary between two programs with loads every round, every line exact; its
command to cores without its program refused, its load to a busy core sent again."""

from pathlib import Path

import pytest
from child import run_json

pytestmark = pytest.mark.card
EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "21_thread_load"
    / "thread_load.py"
)


def test_a_thread_moves_the_boundary_between_two_programs_with_loads():
    out = run_json(EXAMPLE, "--cores", 12, "--rounds", 8, timeout=300.0)
    assert out["ok"] and out["wrong"] == 0, out
    assert out["loads"] > 0 and out["rows"]["host"]["loads"] == out["loads"]
    check = out["check"]
    assert check["refused"] & 0xFF == 13 and check["busy"] & 0xFF == 11
    assert check["load_status"] == check["back_status"] == check["after"] == 0
    assert out["table"]["job_us"][0] < out["table"]["job_us"][1]
