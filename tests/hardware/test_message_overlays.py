"""Handlers as overlay parts, on a card (KMR_CARD=1): example 20's threads, exact
against the replay with a slot for each handler and with fewer slots (loads happen);
a message for a handler that is not the program's stops the run, the reason decoded."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "20_message_overlays"
    / "message_overlays.py"
)


def test_threads_are_exact_with_three_slots_and_with_two():
    for slots in (3, 2):
        out = run_json(EXAMPLE, "--threads", 400, "--steps", 10, "--slots", slots)
        assert out["ok"], out
        assert out["runs"][0]["by_handler"][2] == 400


def test_a_message_for_a_handler_not_the_programs_stops_the_run():
    out = run_json(EXAMPLE, "--count", 6, "--deadline", 1, "--stray")
    assert out["ok"], out  # the run stopped: core 0 gave up, where "handler", 9 seen
