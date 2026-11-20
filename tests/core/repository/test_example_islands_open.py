"""Examples 27 and 28 with `--islands`: one card split into islands in a line (the
graph of neighbouring pairs) opens the stand-in and runs there."""

from pathlib import Path

import pytest

from komorebi.builder import prebuild

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = [
    "examples/27_islands_gather/gather.py",
    "examples/28_islands_reduce/reduce.py",
]


@pytest.mark.tree
@pytest.mark.parametrize("script", SCRIPTS)
def test_a_card_split_into_islands_opens_the_stand_in(script):
    args = ["--islands", "2", "--pages", "8", "--rounds", "1"]
    assert prebuild.run(ROOT / script, args) == 0
