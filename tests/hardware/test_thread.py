"""X280 threads on a card: the work-cluster loop host-fed and on three clusters at once
through the ring, a region shared by clusters 2 and 3, and the parallel patterns on 1
to 12 X280 cores (KMR_CARD=1)."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "07_work_cluster" / "work_cluster.py"
PARALLEL = ROOT / "examples" / "08_parallel_thread" / "parallel_thread.py"
SHARED = Path(__file__).resolve().parent / "thread" / "shared_case.py"


def test_the_work_cluster_loop_host_fed_ends_on_the_card_sum():
    out = run_json(EXAMPLE, "--paths", "cpu", "hostfed", "--limit", 100_000)
    assert out["hostfed"]["1"]["code"] == 0 and out["ok"], out


def test_the_loop_runs_on_three_work_clusters_at_once():
    out = run_json(
        EXAMPLE, "--paths", "thread", "--clusters", 1, 2, 3, "--limit", 100_000
    )
    assert [out["thread"][cluster]["code"] for cluster in "123"] == [0, 0, 0] and out[
        "ok"
    ], out


def test_two_clusters_hand_a_shared_region_over_in_their_bank():
    out = run_json(SHARED)
    assert out["ok"], out


def test_the_patterns_are_exact_on_one_to_twelve_cores_of_three_clusters():
    out = run_json(PARALLEL, "--twelve", "--words", 16384, "--repeats", 1)
    assert out["ok"] and all(out["exact"].values()), out
