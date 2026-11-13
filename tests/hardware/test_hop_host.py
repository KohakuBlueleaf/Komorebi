"""The hop example on a card through the dispatch kernel (KMR_CARD=1), in one device
open: a stalled member ends the run, then the next load is exact (10^3 hops), then the
engine's second dynamic run waits, refused busy, for the first one's end."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.card
SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "05_hop" / "hop.py"


def test_a_stall_times_out_naming_the_core_then_runs_end_at_the_quiet_end():
    args = ["--items", 100, "--hops", 10, "--stall", 2, "--again", "--engine"]
    args += ["--deadline", 1, "--wait", 0.5]
    done = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    out = json.loads([x for x in done.stdout.splitlines() if x.startswith("{")][-1])
    stall, after, engine = out["runs"]
    assert stall["kernel"]["error_name"] == "deadline" and stall["event"] == "timeout"
    assert stall["timed_out"]["ticket"] == 1 and stall["timed_out"]["team"] == [0, 120]
    assert stall["gave_core"] == 2 and stall["dyn_gave"]["2"][0] == 24  # in page
    assert after["exact"] and after["event"] == "done" and after["hops"] == 1000
    assert after["kernel"] == {"error_name": "none", "records": 3, "events": 2}
    assert after["gave_core"] == -1 and after["checks"] > 0
    assert engine["engine"] == "end" and engine["halves_exact"] == [True, True]
    assert engine["kernel"]["records"] == 5 and engine["kernel"]["refused"] >= 1
    assert engine["runs"][1] > engine["runs"][0] + 1  # after the first one's end record
