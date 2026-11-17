"""A class product's operand formats on a card (KMR_CARD=1): the pairs of
experiments/core/formats/class_pairs.py, each right against float64 or refused at
build by name (a float32 operand on either side)."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "core"
    / "formats"
    / "class_pairs.py"
)
PROVEN = {("bfloat16", "bfp8_b"), ("bfp8_b", "bfloat16"), ("bfp4_b", "bfp8_b")}


def test_each_pair_is_right_or_refused_by_name():
    pairs = run_json(SCRIPT)["pairs"]
    for pair in pairs:
        key = (pair["x"], pair["w"])
        if key in PROVEN:
            assert pair["result"] == "works", pair
        else:
            assert "float32" in key, pair
            side = "left" if key[0] == "float32" else "right"
            assert pair["result"] == "refused", pair
            assert pair["why"] == f"a product's {side} operand is float32", pair
    assert {(p["x"], p["w"]) for p in pairs} >= PROVEN
