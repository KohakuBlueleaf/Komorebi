"""The merge-tree example on a card, fed by the test feeder (KMR_CARD=1): one process,
three launches (both forms on four groups; a core that stalls; a clean launch)."""

from pathlib import Path

import pytest
from child import run_json

pytestmark = pytest.mark.card
SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples" / "02_merge_tree" / "merge_fed.py"
)


@pytest.fixture(scope="module")
def out():
    groups = "0:8,0:119,3:60,40:2"  # each run by owner, then by tree, on one input
    args = ["--group-list", groups, "--form", "both", "--stall-rank", 3]
    out = run_json(SCRIPT, *args)
    return out


def test_both_forms_are_exact_and_equal(out):
    assert out["status"] == "done" and out["done"] == 8 and out["exit_acked"]
    assert out["jobs_checked"] == 8 and out["jobs_bad"] == 0
    assert out["exact_checked"] == {"owner": 2, "tree": 2}
    assert out["pairs_checked"] == out["pairs_bit_equal"] == 2
    assert out["gave_up"] == [] and out["waiting_ports"] == {}
    assert set(out["cores_ended"].values()) == {119}


def test_a_stalled_rank_ends_in_a_timeout_naming_port_and_missing_count(out):
    stall = out["stall"]  # job 1: by tree on cores 0 to 7; rank 3 sends to rank 2
    assert stall["status"] == "timeout" and stall["timed_out"]["job"] == 1
    assert stall["waiting_ports"] == {"2": {"0": 1}, "0": {"1": 1}}
    readers = {
        entry["core"]: entry["reader"]
        for entry in stall["gave_up"]
        if "reader" in entry
    }
    assert readers[2]["where"] == "port" and readers[2]["code"] == "limit"
    assert readers[2]["port"] == 0 and readers[0]["port"] == 1
    assert set(stall["cores_ended"].values()) == {119}


def test_the_next_launch_runs(out):
    after_launch = out["after"]
    assert (
        after_launch["status"] == "done"
        and after_launch["done"] == 4
        and after_launch["jobs_bad"] == 0
    )
