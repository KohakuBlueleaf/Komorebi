"""A broadcast that carries a word, on a card (KMR_CARD=1): the leader of a column of 10
casts its choices, alone and with tiles in alternate commands, and every core's every
word, end state and tile sum are exact."""

from pathlib import Path

import pytest
from child import run_json

pytestmark = pytest.mark.card
SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "15_leader" / "leader.py"


def test_words_alone_and_with_tiles_reach_every_core_of_a_column_exact():
    args = ["--rect", "0,0,1,10", "--casts", 2000, "--jobs", 4]
    out = run_json(SCRIPT, *args, "--carry", "alternate", "--window", 2)
    assert out["ok"] and out["cores_bad"] == 0
    assert out["words_checked"] == 4 * 2000 * 10
