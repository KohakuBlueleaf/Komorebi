"""Lane broadcast on a card through the dispatch kernel (KMR_CARD=1): a stalled
receiver ends in the host's timeout and the record behind it never runs, then the next
run is exact; back-to-back records on all 120 cores, every second casting no block, are
exact; a record of 65,600 pages per lane, then a short one, are exact."""

from pathlib import Path

import pytest
from child import run_json

pytestmark = pytest.mark.card
SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples" / "03_broadcast" / "broadcast.py"
)


def test_a_stalled_receiver_times_out_then_the_next_run_is_exact():
    args = ["--rect", "4,0,4,3", "--blocks", 6, "--pages", 2, "--window", 2]
    args += ["--jobs", 2, "--stall-job", 0, "--stall-member", 13, "--stall-block", 2]
    args += ["--wait", 0.05, "--deadline", 0.5, "--again"]
    out = run_json(SCRIPT, *args)
    stalled, clean = out["runs"]
    timed_out = stalled["timed_out"]
    assert stalled["kernel"]["error_name"] == "deadline"
    assert stalled["kernel"]["records"] == 1  # job 1, stored behind it, never issued
    assert timed_out["ticket"] == 1 and timed_out["team"] == [4, 28]
    named = {c["core"]: c for c in timed_out["not_reported"]}
    assert named[17]["reader"]["where"] == "stall"
    sender = [g for g in stalled["lane_gave_up"] if g["core"] == 16 and g["lane"] == 1]
    assert sender[0]["where"] == "lanedone" and sender[0]["block"] == 3
    assert clean["kernel"]["error_name"] == "none" and clean["core_gave_up"] == []
    assert clean["members_checked"] == 24 and clean["members_bad"] == 0
    for run in (stalled, clean):
        assert set(run["cores_ended"].values()) == {120}


def test_back_to_back_records_some_casting_no_block_are_exact():
    args = ["--rect", "0,0,12,10", "--blocks", 4, "--pages", 2, "--window", 2]
    out = run_json(SCRIPT, *args, "--jobs", 8, "--empty-every", 2)
    (run,) = out["runs"]
    assert run["kernel"]["error_name"] == "none" and run["kernel"]["records"] == 9
    assert run["core_gave_up"] == [] and run["lane_gave_up"] == []
    assert run["members_checked"] == 960 and run["members_bad"] == 0


def test_a_record_of_65600_pages_per_lane_then_a_short_one_are_exact():
    args = ["--rect", "4,0,2,2", "--blocks", 4, "--pages", 16, "--window", 2]
    out = run_json(SCRIPT, *args, "--jobs", 2, "--first-blocks", 4100)
    (run,) = out["runs"]
    assert run["kernel"]["error_name"] == "none" and run["kernel"]["records"] == 3
    assert run["core_gave_up"] == [] and run["lane_gave_up"] == []
    assert run["lane_blocks"] == 4 * (4100 + 4) * 2
    assert run["members_checked"] == 8 and run["members_bad"] == 0
