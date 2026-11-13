"""The matrix product on a card through the dispatch kernel (KMR_CARD=1): small
shapes on integer inputs are exact in every form: W resident, W streamed, and the
grid form in each layout (operands handed along rows and down columns)."""

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


# two K blocks and two blocks of output in each layout
GRIDS = ["2d:4,3,4,4,2,2,1", "rows:10,1,2,12,1,6,1", "cols:1,12,16,1,8,1,1"]


@pytest.mark.parametrize("grid", GRIDS)
def test_the_grid_form_is_exact(grid):
    shape = ["--m", 512, "--k", 64, "--n", 384, "--grid", grid]
    out = run_json(SCRIPT, *shape, "--form", "grid", "--inputs", "int", "--calls", 4)
    assert out["plan"]["layout"] == grid.split(":")[0] and out["plan"]["nk"] == 2
    assert out["run"]["kernel"] == "none" and out["run"]["gave_up"] == []
    assert out["run"]["lane_gave_up"] == []
    assert out["check"]["exact"] and out["ok"]
