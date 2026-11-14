"""Broadcast on a card through the dispatcher (KMR_CARD=1): a stalled receiver ends in
the host's timeout, then the next run is exact;
back-to-back commands on all 120 cores, every second multicasting no block, are exact; a
command of 65,600 pages per broadcast, then a short one, are exact."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples" / "03_broadcast" / "broadcast.py"
)


def test_a_stalled_receiver_times_out_then_the_next_run_is_exact():
    args = ["--rect", "4,0,4,3", "--blocks", 6, "--pages", 2, "--window", 2]
    args += ["--jobs", 2, "--stall-job", 0, "--stall-rank", 13, "--stall-block", 2]
    args += ["--wait", 0.05, "--deadline", 0.5, "--again"]
    out = run_json(SCRIPT, *args)
    stalled, clean = out["runs"]
    timed_out = stalled["timed_out"]
    assert stalled["kernel"]["error_name"] == "deadline"
    assert timed_out["group"] == [0, 120]  # the last command, on every core
    named = {core["core"]: core for core in timed_out["not_reported"]}
    assert named[0]["writer"]["where"] == "complete"  # the leader waits for the rest
    assert clean["kernel"]["error_name"] == "none" and clean["core_gave_up"] == []
    assert clean["ranks_checked"] == 24 and clean["ranks_bad"] == 0
    for run in (stalled, clean):
        assert set(run["cores_ended"].values()) == {120}


def test_back_to_back_commands_some_broadcasting_no_block_are_exact():
    args = ["--rect", "0,0,12,10", "--blocks", 4, "--pages", 2, "--window", 2]
    out = run_json(SCRIPT, *args, "--jobs", 8, "--empty-every", 2)
    (run,) = out["runs"]
    assert run["kernel"]["error_name"] == "none" and run["kernel"]["commands"] == 9
    assert run["core_gave_up"] == [] and run["broadcast_gave_up"] == []
    assert run["ranks_checked"] == 960 and run["ranks_bad"] == 0


def test_a_command_of_65600_pages_per_broadcast_then_a_short_one_are_exact():
    args = ["--rect", "4,0,2,2", "--blocks", 4, "--pages", 16, "--window", 2]
    out = run_json(SCRIPT, *args, "--jobs", 2, "--first-blocks", 4100)
    (run,) = out["runs"]
    assert run["kernel"]["error_name"] == "none" and run["kernel"]["commands"] == 3
    assert run["core_gave_up"] == [] and run["broadcast_gave_up"] == []
    assert run["broadcast_blocks"] == 4 * (4100 + 4) * 2
    assert run["ranks_checked"] == 8 and run["ranks_bad"] == 0
