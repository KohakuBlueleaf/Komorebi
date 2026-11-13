"""The tutorial's programs on a card (KMR_CARD=1): each folder once, at its smallest
size. The scripts check their own results; the line each page shows is looked for in
what they print."""

from pathlib import Path

import pytest
from child import run_text

pytestmark = pytest.mark.card
TUTORIAL = Path(__file__).resolve().parents[2] / "tutorial"


@pytest.mark.parametrize(
    ("script", "line"),
    [
        ("01_first_look/add.py", "1000 + 2345 = 3345"),
        ("03_the_card/card_facts.py", "120 Tensix cores: 12 in a row, 10 rows"),
        ("04_hello_core/hello.py", "the command is done"),
        ("05_commands/commands.py", "command 3: done after"),
        ("06_a_result/a_result.py", "the core answered 42"),
        ("07_arguments/arguments.py", "1000 + 2345 = 3345"),
        ("08_handlers/handlers.py", "6 multiply 7 = 42"),
        ("09_many_commands/many_commands.py", "squares: [1, 4, 9, 16, 25, 36, 49, 64]"),
        ("10_when_it_goes_wrong/when_it_goes_wrong.py", "the card stopped the command"),
    ],
)
def test_a_tutorial_program_runs_and_prints_its_line(script, line):
    out = run_text(TUTORIAL / script, timeout=180.0)
    assert line in out
