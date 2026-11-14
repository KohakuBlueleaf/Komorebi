"""An X280 thread commands two programs of one run (KMR_CARD=1): example 19 on a row,
every unit of every round exact, the thread splitting each round and fixed splits."""

from pathlib import Path

import pytest
from child import run_json

pytestmark = pytest.mark.card
EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "19_thread_programs"
    / "thread_programs.py"
)


def test_a_thread_splits_a_row_between_two_programs_every_round():
    out = run_json(EXAMPLE, "--cores", 12, "--rounds", 8, "--splits", 2, timeout=300.0)
    assert out["ok"], out
    splits = out["rows"]["thread"]["splits"]
    assert len(splits) == 8 and all(1 <= k <= 11 for k in splits)
    assert out["table_us"]["thread split"] < out["rows"]["host fed"]["job_us"]
