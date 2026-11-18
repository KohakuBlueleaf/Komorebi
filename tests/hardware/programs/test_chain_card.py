"""Example 29's chains on one card (KMR_CARD=1): four calls and three, each run as a
command a step and as one chain; every word of each output equals numpy's either way."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
SCRIPT = Path(__file__).resolve().parents[3] / "examples" / "29_chain" / "across.py"


def test_the_chain_gives_what_its_commands_give():
    out = run_json(SCRIPT, "--cards", 0, "--runs", 1)
    assert out["ok"] and out["kernel"] == ["none"], out
    for form, calls in (("straddle", 4), ("hops", 3)):
        assert len(out[form]["groups"]) == calls
        for way, bad in out[form]["checks"].items():
            assert not any(bad.values()), (form, way, bad)
