"""X280 threads on a card: the work-cluster example against its host fallback, and a
thread stopped at the host's timeout (KMR_CARD=1)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.card
ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "07_work_cluster" / "work_cluster.py"
STOP = Path(__file__).resolve().parent / "thread" / "stop_case.py"


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


def test_prefix_sums_on_the_card_equal_the_host_fallback():
    out = run(EXAMPLE, "--cores", 8, "--bins", 64)
    assert out["card"]["state"] == "done" and out["same_offsets"] and out["ok"]


def test_a_thread_past_the_timeout_is_stopped_and_reports_where_it_was():
    out = run(STOP)
    assert out["ok"], out
