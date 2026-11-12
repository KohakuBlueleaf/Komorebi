"""The host package on a card: the map/collect example under the dispatch kernel, and
the session cases it does not cover (KMR_CARD=1)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.card
ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "01_mapcollect" / "mapcollect.py"
CASES = Path(__file__).resolve().parent / "host" / "session_cases.py"


def run(script, *args, timeout=300):
    """Runs a script and returns its last JSON line."""
    done = subprocess.run(
        [sys.executable, str(script), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    return json.loads(lines[-1])


def every_kernel_returned(out, cores=120):
    ended = out["cores_ended"]
    return out["cores_magic_ok"] == cores and set(ended.values()) == {cores}


@pytest.mark.parametrize("count", [1, 8, 120])
def test_fixed_team_is_exact(count):
    out = run(EXAMPLE, "--jobs", 20, "--count", count, "--out-tensors", 2)
    assert out["kernel"]["error_name"] == "none" and out["kernel"]["events"] == 21
    assert out["jobs_checked"] == 20 and out["exact_jobs_checked"] == 10
    assert out["jobs_bad"] == 0 and out["compute_wrong"] == 0
    assert out["gave_up_cores"] == 0 and every_kernel_returned(out)


def test_random_teams_with_four_in_flight():
    out = run(EXAMPLE, "--jobs", 100, "--teams", "random", "--out-tensors", 4)
    assert out["jobs_checked"] >= 90 and out["jobs_bad"] == 0
    assert out["random_total_rel_rms_max"] < 1e-2 and every_kernel_returned(out)


def test_disjoint_teams_run_at_once():
    out = run(EXAMPLE, "--jobs", 40, "--teams", "pair", "--trace")
    assert out["jobs_bad"] == 0 and out["overlap"]["overlapping"] > 0


def test_a_stalled_member_ends_in_the_timeout_exception():
    stall = ["--stall", 3, "--stall-member", 2, "--deadline", 0.02]
    out = run(EXAMPLE, "--jobs", 10, "--count", 8, "--first", 4, *stall)
    t = out["timed_out"]
    assert out["kernel"]["error_name"] == "deadline"
    assert t["ticket"] == 4 and t["team"] == [4, 8]
    named = {c["core"]: c for c in t["not_reported"]}
    assert sorted(named) == [4, 6]
    assert (
        named[6]["reader"]["where"] == "stall" and named[6]["reader"]["seq"] == t["seq"]
    )
    assert named[4]["writer"]["where"] == "complete"
    assert every_kernel_returned(out)


def test_a_compile_error_leaves_no_image_running():
    assert run(CASES, "compile-error")["ok"]


def test_an_exclusive_record_waits_for_the_team_it_overlaps():
    assert run(CASES, "exclusive")["ok"]


@pytest.mark.skipif(os.environ.get("KMR_CARD_LONG") != "1", reason="KMR_CARD_LONG=1")
def test_thousand_random_jobs_are_exact():
    out = run(
        EXAMPLE, "--jobs", 1000, "--teams", "random", "--out-tensors", 32, timeout=600
    )
    assert out["jobs_checked"] >= 900 and out["jobs_bad"] == 0
