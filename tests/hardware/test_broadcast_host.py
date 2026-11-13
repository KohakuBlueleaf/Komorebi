"""Lane broadcast on a card through the dispatch kernel (KMR_CARD=1): a stalled
receiver ends in the host's timeout, then the next run is exact; a record that casts
no block finishes."""

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
    assert timed_out["ticket"] == 1 and timed_out["team"] == [4, 28]
    named = {c["core"]: c for c in timed_out["not_reported"]}
    assert named[17]["reader"]["where"] == "stall"
    sender = [g for g in stalled["lane_gave_up"] if g["core"] == 16 and g["lane"] == 1]
    assert sender[0]["where"] == "lanedone" and sender[0]["block"] == 3
    assert clean["kernel"]["error_name"] == "none" and clean["core_gave_up"] == []
    assert clean["members_checked"] == 24 and clean["members_bad"] == 0
    for run in (stalled, clean):
        assert set(run["cores_ended"].values()) == {120}


def test_a_record_that_casts_no_block_finishes_and_the_next_is_exact():
    args = ["--rect", "4,0,4,3", "--blocks", 5, "--pages", 2, "--window", 2]
    out = run_json(SCRIPT, *args, "--jobs", 3, "--empty-job", 1)
    (run,) = out["runs"]
    assert run["kernel"]["error_name"] == "none" and run["kernel"]["records"] == 4
    assert run["core_gave_up"] == [] and run["lane_gave_up"] == []
    assert run["members_checked"] == 36 and run["members_bad"] == 0
