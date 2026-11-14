"""Attention on a card through the dispatcher (KMR_CARD=1): a small causal call
with grouped heads and head dim 128 is within the check's bound of float64, with the
K / V broadcast on NoC 1."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
SCRIPT = Path(__file__).resolve().parents[2] / "projects" / "attention" / "run.py"


def test_a_small_causal_call_is_within_the_bound():
    out = run_json(SCRIPT, "--shape", "t3", "--calls", 2, "--noc", 1)
    assert out["run"]["kernel"] == "none" and out["run"]["gave_up"] == []
    assert out["run"]["ok"]  # and every core's three kernels ended
    assert out["check"]["ok"] and out["ok"]
