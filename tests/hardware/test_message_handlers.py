"""A message names its handler, on a card (KMR_CARD=1): example 14's threads change
handler on forwards to other cores and to their own core on a small group, exact
against the replay;
a run whose queue positions pass 65536 laps (the lap mark wraps) stays exact."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
HANDLERS = EXAMPLES / "14_message_handlers" / "message_handlers.py"
FORWARD = EXAMPLES / "05_forward" / "forward.py"


def test_threads_change_handler_on_a_small_group():
    # 4 ranks, many own-core forwards
    out = run_json(HANDLERS, "--count", 6, "--threads", 400, "--steps", 10)
    assert out["ok"], out
    run = out["runs"][0]
    assert run["by_handler"][2] == 400 and run["own_core_forwards"] > 0


def test_the_lap_mark_wraps_in_one_run_and_every_message_arrives():
    # 118 messages of 72000 forwards each, all to rank 0: 66375 laps of its queue of 128
    args = ["--messages", 118, "--forwards", 72000, "--one-core", 0]
    out = run_json(FORWARD, *args, "--max-live", 128, "--queue-messages", 128)
    assert out["ok"], out
    assert out["runs"][0]["forwards"] == 118 * 72000
