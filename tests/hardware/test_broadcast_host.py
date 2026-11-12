"""Lane broadcast on a card through the dispatch kernel (KMR_CARD=1), in one device
open: a stalled receiver ends in the host's timeout, then the next run is exact."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.card
SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples" / "03_broadcast" / "broadcast.py"
)


def test_a_stalled_receiver_times_out_then_the_next_run_is_exact():
    args = ["--rect", "4,0,4,3", "--blocks", 6, "--pages", 2, "--window", 2]
    args += ["--jobs", 2, "--stall-job", 0, "--stall-member", 13, "--stall-block", 2]
    args += ["--wait", 0.05, "--deadline", 0.5, "--again"]
    done = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    line = [s for s in done.stdout.splitlines() if s.startswith("{")][-1]
    stalled, clean = json.loads(line)["runs"]
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
