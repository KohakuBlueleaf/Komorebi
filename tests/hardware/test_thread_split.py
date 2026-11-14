"""A pull's word on a card (KMR_CARD=1): example 18, every core pulling ranges whose
word names the overlay to run; and the three limits of commands to a thread (a ninth
host command, kmx_parallel inside kmx_program, events nobody waited for)."""

from pathlib import Path

import pytest
from child import run_json

pytestmark = pytest.mark.card
ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "18_thread_split" / "thread_split.py"
CASE = Path(__file__).resolve().parent / "thread" / "open_points_case.py"


def test_each_range_runs_the_overlay_its_word_names_and_every_tile_is_exact():
    out = run_json(EXAMPLE, "--units", 2048, "--per", 2, "--run", 16)
    assert out["ok"], out
    passes = out["passes"]
    assert passes["host"]["loads"] <= out["cores"] <= passes["plain"]["loads"]
    assert passes["word"]["pulls"] == passes["pick"]["pulls"] == out["pull_units"]


def test_a_thread_rejects_a_ninth_command_runs_parallel_and_keeps_events():
    out = run_json(CASE)
    assert out["ok"], out
