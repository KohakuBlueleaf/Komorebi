"""The matrix product on a card through the dispatch kernel (KMR_CARD=1): a small
shape on integer inputs is exact in both forms, W resident and W streamed."""

from pathlib import Path

import pytest
from child import run_json

pytestmark = pytest.mark.card
SCRIPT = Path(__file__).resolve().parents[2] / "projects" / "dense" / "run.py"


@pytest.mark.parametrize("form", ["resident", "stream"])
def test_a_small_product_is_exact(form):
    shape = ["--m", 256, "--k", 64, "--n", 128]
    out = run_json(SCRIPT, *shape, "--form", form, "--inputs", "int", "--calls", 4)
    assert out["plan"]["form"] == (0 if form == "resident" else 1)
    assert out["run"]["kernel"] == "none" and out["run"]["gave_up"] == []
    assert out["check"]["exact"] and out["ok"]
