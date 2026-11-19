"""Commands to an X280 thread on a card: example 13 (the host and a graph mixing them
with commands to core groups), a stream's commands to a thread host-fed and on the
X280, a thread serving past its own deadline, and a command it does not answer by its
deadline given up, the run going on (KMR_CARD=1)."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples" / "13_thread_commands" / "thread_commands.py"
CASE = Path(__file__).resolve().parent / "commands_case.py"
UNANSWERED = Path(__file__).resolve().parent / "unanswered_case.py"


def test_the_host_and_a_graph_chain_grid_and_thread_commands_exactly():
    out = run_json(EXAMPLE, "--rounds", 20, "--graph-rounds", 100, "--waves", 20)
    assert out["ok"], out


def test_a_stream_commands_a_thread_and_the_thread_outlives_its_deadline():
    out = run_json(CASE)
    assert out["ok"], out


def test_a_command_the_thread_does_not_answer_is_given_up_and_the_run_goes_on():
    out = run_json(UNANSWERED)
    assert out["ok"], out
