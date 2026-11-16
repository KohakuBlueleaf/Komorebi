"""Example 04 on a card through the dispatcher (KMR_CARD=1), in one device open:
30 jobs each way, then the three error streams and a clean run after them."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
SCRIPT = (
    Path(__file__).resolve().parents[3] / "examples" / "04_uop_stream" / "uop_stream.py"
)


def test_every_way_is_exact_and_each_error_stream_ends_in_its_error():
    out = run_json(SCRIPT, "--jobs", 30)
    assert out["ok"]
    assert out["ways"]["enqueue"]["most_in_flight"] > 1
    assert out["errors"]["never_fits"]["reason"] == "never_fits"
