"""The tutorial's programs on a card (KMR_CARD=1): each folder once, at its smallest
size. The scripts check their own results against a host reference and exit non-zero
when one differs."""

from pathlib import Path

import pytest
from child import run_text

pytestmark = pytest.mark.card
TUTORIAL = Path(__file__).resolve().parents[2] / "tutorial"


@pytest.mark.parametrize(
    ("script", "line"),
    [
        ("01_the_card/card_facts.py", "L1 of a core: 70656 bytes for kernels"),
        ("03_program_structure/hello.py", "cores: 120, event: done"),
        ("04_first_kernel/add.py", "1000 + 2345 = 3345"),
        ("05_handlers_and_commands/commands.py", "100 commands, 0 wrong"),
        ("06_copy_tiles/copy_tiles.py", "6 tiles copied, identical: True"),
        ("06_sum_in_l1/sum_rows.py", "card 743169458, numpy 743169458"),
    ],
)
def test_a_tutorial_program_runs_and_matches_its_reference(script, line):
    out = run_text(TUTORIAL / script, timeout=180.0)
    assert line in out
