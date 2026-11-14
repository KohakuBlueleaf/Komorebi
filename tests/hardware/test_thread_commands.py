"""Commands to an X280 thread on a card: example 13 (the host and a graph mixing them
with commands to core groups), a stream's commands to a thread host-fed and on the
X280, and a thread serving past its own deadline (KMR_CARD=1)."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "13_thread_commands" / "thread_commands.py"
CASE = Path(__file__).resolve().parent / "thread" / "commands_case.py"


def test_the_host_and_a_graph_chain_grid_and_thread_commands_exactly():
    out = run_json(EXAMPLE, "--rounds", 20, "--graph-rounds", 100, "--waves", 20)
    assert out["ok"], out


def test_a_stream_commands_a_thread_and_the_thread_outlives_its_deadline():
    out = run_json(CASE)
    assert out["ok"], out
