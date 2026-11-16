"""Both data movers on one stream, on a card (KMR_CARD=1): the movement example's copy
reads and writes every word exactly on four cores under patterns AB and AAAB (the
script runs pattern A first, its baseline)."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
SCRIPT = (
    Path(__file__).resolve().parents[3] / "examples" / "11_movement" / "movement.py"
)


def test_reads_and_writes_are_exact_under_each_pattern():
    args = ["--cores", 4, "--tiles", 48, "--block", 8, "--check-only"]
    out = run_json(SCRIPT, *args, "--pattern", "AB", "AAAB")
    assert out["exact"]
    assert [entry["pattern"] for entry in out["patterns"]] == ["A", "AB", "AAAB"]
