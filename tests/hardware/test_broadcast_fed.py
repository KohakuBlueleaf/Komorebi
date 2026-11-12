"""Lane broadcast on a card, fed by the test feeder (KMR_CARD=1), in one device open."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.card
SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples" / "03_broadcast" / "broadcast.py"
)


def test_a_stalled_receiver_times_out_then_rows_and_columns_are_exact():
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
    assert stalled["feeder"]["status"] == "timeout"
    sender = [g for g in stalled["lane_gave_up"] if g["core"] == 16 and g["lane"] == 1]
    assert sender[0]["where"] == "lanedone" and sender[0]["block"] == 3
    assert clean["feeder"]["status"] == "done" and clean["core_gave_up"] == []
    assert clean["members_checked"] == 24 and clean["members_bad"] == 0
    for run in (stalled, clean):
        assert set(run["cores_ended"].values()) == {119}
