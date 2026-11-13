"""The merge tree on a card through the dispatcher (KMR_CARD=1), in one device
open: a stalled member ends in the host's timeout, then the next load is exact. And a
session with a 128 KiB kernel buffer runs the map/collect example exact."""

from pathlib import Path

import pytest
from child import run_json

pytestmark = pytest.mark.card
SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "02_merge_tree" / "merge.py"
BUFFER = Path(__file__).resolve().parent / "host" / "kernel_buffer_case.py"


def test_a_stalled_rank_times_out_naming_its_ports_then_the_next_load_is_exact():
    args = ["--group-list", "0:8,0:119", "--form", "both", "--stall-rank", 3]
    out = run_json(SCRIPT, *args)
    s = out["stall"]  # job 1: by tree on cores 0 to 7; member 3 sends to member 2
    assert s["kernel"]["error_name"] == "deadline"
    assert s["timed_out"]["handle"] == 2 and s["timed_out"]["group"] == [0, 8]
    assert s["waiting_ports"] == {"2": {"0": 1}, "0": {"1": 1}}
    readers = {g["core"]: g["reader"] for g in s["gave_up"] if "reader" in g}
    assert readers[2]["where"] == "port" and readers[2]["port"] == 0
    after = out["after"]  # both forms on 0:8 and on 0:119, with their result words
    assert after["done"] == 4 and after["jobs_bad"] == 0
    assert after["pairs_checked"] == after["pairs_bit_equal"] == 1
    assert after["results_checked"] == 4 and after["results_bad"] == 0
    for run in (s, after):
        assert set(run["cores_ended"].values()) == {120}


def test_a_session_with_a_128_kib_kernel_buffer_runs_exact():
    out = run_json(BUFFER)
    assert out["ok"] and out["jobs_bad"] == 0, out
    assert out["kernel_buffer"] == 128 << 10
    assert out["l1_free"] == 1536 * 1024 - 41088 - (128 << 10)
