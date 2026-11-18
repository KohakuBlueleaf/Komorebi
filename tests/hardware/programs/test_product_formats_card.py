"""A class product's operand formats on a card (KMR_CARD=1): the pairs of
experiments/core/formats/class_pairs.py, each right against float64 (a float32 stream
read as tf32 by the product)."""

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
PROVEN = {
    ("bfloat16", "bfp8_b"),
    ("bfp8_b", "bfloat16"),
    ("bfp4_b", "bfp8_b"),
    ("float32", "bfp8_b"),
    ("float32", "bfloat16"),
    ("bfloat16", "float32"),
    ("float32", "float32"),
}


def test_each_pair_is_right():
    pairs = run_json(SCRIPT)["pairs"]
    for pair in pairs:
        assert pair["result"] == "works", pair
    assert {(p["x"], p["w"]) for p in pairs} == PROVEN
