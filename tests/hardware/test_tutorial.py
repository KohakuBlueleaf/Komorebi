"""Five of the tutorial's programs on a card (KMR_CARD=1), each a different mechanism: a
command to a core, three kernels on tensors, forwarded messages, an X280 thread, an
overlay. The scripts check their own results; the line each page shows is looked for
in what they print."""

from pathlib import Path

import pytest
from shared_card import run_text

pytestmark = pytest.mark.card
TUTORIAL = Path(__file__).resolve().parents[2] / "tutorial"


@pytest.mark.parametrize(
    ("script", "line"),
    [
        ("04_hello_core/hello.py", "the command is done"),
        ("18_three_kernels/three_kernels.py", "all 8 tiles of the destination"),
        ("29_forward_and_consume/forward_and_consume.py", "as the replay says"),
        ("32_hello_thread/hello_thread.py", "it returned 42 = 20 + 22"),
        ("43_overlays/overlays.py", "6 multiply 7 = 42 (overlay 2)"),
    ],
)
def test_a_tutorial_program_runs_and_prints_its_line(script, line):
    out = run_text(TUTORIAL / script)
    assert line in out
