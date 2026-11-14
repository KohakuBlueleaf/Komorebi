"""A session with a 128 KiB kernel buffer on a card (KMR_CARD=1): it reports its L1
split and runs the map/collect example exact."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
BUFFER = Path(__file__).resolve().parent / "host" / "kernel_buffer_case.py"


def test_a_session_with_a_128_kib_kernel_buffer_runs_exact():
    out = run_json(BUFFER)
    assert out["ok"] and out["jobs_bad"] == 0, out
    assert out["kernel_buffer"] == 128 << 10
    assert out["l1_free"] == 1536 * 1024 - 41088 - (128 << 10)
