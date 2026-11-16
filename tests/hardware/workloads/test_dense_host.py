"""The matrix product on a card through the dispatcher (KMR_CARD=1): small
shapes on integer inputs are exact in every form: W resident, W streamed, the grid
form (operands handed along rows and down columns), K split."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
SCRIPT = Path(__file__).resolve().parents[3] / "projects" / "dense" / "run.py"


@pytest.mark.parametrize("form", ["resident", "stream"])
def test_a_small_product_is_exact(form):
    shape = ["--m", 256, "--k", 64, "--n", 128]
    out = run_json(SCRIPT, *shape, "--form", form, "--inputs", "int", "--calls", 4)
    assert out["plan"]["form"] == (0 if form == "resident" else 1)
    assert out["run"]["kernel"] == "none" and out["run"]["gave_up"] == []
    assert out["check"]["exact"] and out["ok"]


def test_the_grid_form_is_exact():
    grid = "2d:4,3,4,4,2,2,1"  # two K blocks and two blocks of output
    shape = ["--m", 512, "--k", 64, "--n", 384, "--grid", grid]
    out = run_json(SCRIPT, *shape, "--form", "grid", "--inputs", "int", "--calls", 4)
    assert out["plan"]["layout"] == grid.split(":")[0] and out["plan"]["nk"] == 2
    assert out["run"]["kernel"] == "none" and out["run"]["gave_up"] == []
    assert out["run"]["broadcast_gave_up"] == []
    assert out["check"]["exact"] and out["ok"]


def test_the_k_split_form_is_exact():
    shape = ["--m", 160, "--k", 64, "--n", 96, "--split", "2,2,2,1"]
    out = run_json(SCRIPT, *shape, "--reduce", "tree", "--inputs", "int", "--calls", 4)
    assert out["plan"]["form"] == "split" and out["reduce"] == "tree"
    assert out["run"]["kernel"] == "none" and out["run"]["gave_up"] == []
    assert out["check"]["exact"] and out["ok"]
