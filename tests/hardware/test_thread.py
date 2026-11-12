"""X280 threads on a card: the work-cluster loop host-fed and through the thread's ring
against numpy, a thread stopped at the host's timeout, a thread that stops taking its
events (KMR_CARD=1)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.card
ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "07_work_cluster" / "work_cluster.py"
STOP = Path(__file__).resolve().parent / "thread" / "stop_case.py"
STALL = Path(__file__).resolve().parent / "thread" / "ring_stall_case.py"


def run(script, *args, timeout=120):
    """Runs a script and returns its last JSON line."""
    done = subprocess.run(
        [sys.executable, str(script), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    return json.loads(lines[-1])


def test_the_work_cluster_loop_host_fed_ends_on_the_card_sum():
    out = run(EXAMPLE, "--paths", "cpu", "hostfed", "--limit", 100_000)
    assert out["hostfed"]["code"] == 0 and out["ok"], out


def test_the_work_cluster_loop_runs_in_the_thread_through_its_ring():
    out = run(EXAMPLE, "--paths", "thread", "--limit", 100_000)
    assert out["thread"]["code"] == 0 and out["ok"], out


def test_a_thread_past_the_timeout_is_stopped_and_reports_where_it_was():
    out = run(STOP)
    assert out["ok"], out


def test_a_thread_that_stops_taking_events_holds_up_no_host_record():
    out = run(STALL)
    assert out["ok"], out
