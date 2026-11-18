"""A class-language program on a card (KMR_CARD=1): a weight filled each turn of a
stream loop and used by that turn's product, and a stream loop's sum reduced over the
cores each turn; every word equal to float64's, as the loop forms' README states."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "language"
    / "loop_forms"
    / "loop_forms.py"
)


def test_a_product_and_a_reduce_in_stream_loops_equal_float64():
    out = run_json(SCRIPT, "--form", "turns")
    assert out["ok"], out
    assert out["fill_each_turn_vs_float64_differ"] == 0
    assert out["sum_each_turn_vs_float64_differ"] == 0
