"""X280 threads on a card: the work-cluster loop host-fed and through the thread's ring
against numpy, on cluster 1 and on clusters 1 to 3 at once, a region shared by
clusters 2 and 3, the parallel-thread example's patterns on one hart and four, a thread
stopped at the host's timeout (its region read back), a stalled ring (KMR_CARD=1)."""

from pathlib import Path

import pytest
from child import run_json

pytestmark = pytest.mark.card
ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "07_work_cluster" / "work_cluster.py"
PARALLEL = ROOT / "examples" / "08_parallel_thread" / "parallel_thread.py"
STOP = Path(__file__).resolve().parent / "thread" / "stop_case.py"
STALL = Path(__file__).resolve().parent / "thread" / "ring_stall_case.py"
SHARED = Path(__file__).resolve().parent / "thread" / "shared_case.py"


def test_the_work_cluster_loop_host_fed_ends_on_the_card_sum():
    out = run_json(EXAMPLE, "--paths", "cpu", "hostfed", "--limit", 100_000)
    assert out["hostfed"]["1"]["code"] == 0 and out["ok"], out


def test_the_work_cluster_loop_runs_in_the_thread_through_its_ring():
    out = run_json(EXAMPLE, "--paths", "thread", "--limit", 100_000)
    assert out["thread"]["1"]["code"] == 0 and out["ok"], out


def test_the_loop_runs_on_three_work_clusters_at_once():
    out = run_json(
        EXAMPLE, "--paths", "thread", "--clusters", 1, 2, 3, "--limit", 100_000
    )
    assert [out["thread"][c]["code"] for c in "123"] == [0, 0, 0] and out["ok"], out


def test_two_clusters_hand_a_shared_region_over_in_their_bank():
    out = run_json(SHARED)
    assert out["ok"], out


def test_the_parallel_patterns_are_exact_on_one_hart_and_on_four():
    out = run_json(PARALLEL, "--words", 65536, "--repeats", 1)
    assert out["ok"] and all(out["exact"].values()), out


def test_a_thread_past_the_timeout_is_stopped_and_reports_where_it_was():
    out = run_json(STOP)
    assert out["ok"], out


def test_a_thread_that_stops_taking_events_holds_up_no_host_record():
    out = run_json(STALL)
    assert out["ok"], out
