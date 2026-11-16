"""X280 thread lifecycle on a card: spawned cores and the documented rejections, a
thread stopped at the host's timeout, a stalled ring (KMR_CARD=1)."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
CASES = Path(__file__).resolve().parent


def test_spawned_cores_run_beside_core_0_and_rejections_are_as_documented():
    out = run_json(CASES / "spawn_case.py")
    assert out["ok"], out


def test_a_thread_past_the_timeout_is_stopped_and_reports_where_it_was():
    out = run_json(CASES / "stop_case.py")
    assert out["ok"], out


def test_a_thread_that_stops_taking_events_holds_up_no_host_command():
    out = run_json(CASES / "ring_stall_case.py")
    assert out["ok"], out
