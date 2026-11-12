"""The merge tree on a card through the dispatch kernel (KMR_CARD=1), in one device
open: a stalled member ends in the host's timeout, then the next load is exact."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.card
SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "02_merge_tree" / "merge.py"


def test_a_stalled_member_times_out_naming_its_ports_then_the_next_load_is_exact():
    args = ["--team-list", "0:8,0:119", "--form", "both", "--stall-member", 3]
    done = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    out = json.loads([x for x in done.stdout.splitlines() if x.startswith("{")][-1])
    s = out["stall"]  # job 1: by tree on cores 0 to 7; member 3 sends to member 2
    assert s["kernel"]["error_name"] == "deadline"
    assert s["timed_out"]["ticket"] == 2 and s["timed_out"]["team"] == [0, 8]
    assert s["waiting_ports"] == {"2": {"0": 1}, "0": {"1": 1}}
    readers = {g["core"]: g["reader"] for g in s["gave_up"] if "reader" in g}
    assert readers[2]["where"] == "port" and readers[2]["port"] == 0
    after = out["after"]  # both forms on 0:8 and on 0:119, with their result words
    assert after["done"] == 4 and after["jobs_bad"] == 0
    assert after["pairs_checked"] == after["pairs_bit_equal"] == 1
    assert after["results_checked"] == 4 and after["results_bad"] == 0
    for run in (s, after):
        assert set(run["cores_ended"].values()) == {120}
