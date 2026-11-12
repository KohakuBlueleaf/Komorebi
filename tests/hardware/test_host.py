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


def counts_match(out):
    """Each core took and finished as many records as it is a member of."""
    return out["core_counts_bad"] == 0 and out["member_records"] > 0


def all_checked(out, jobs, stalled=0):
    """Every job not stalled was read back and checked; none overwritten, no stale
    tile could pass (the page held something else before)."""
    return (
        out["jobs_checked"] == jobs - stalled
        and out["jobs_overwritten"] == 0
        and out["tiles_same_as_before"] == 0
        and out["jobs_bad"] == 0
    )


@pytest.mark.parametrize("count", [1, 8, 120])
def test_fixed_team_is_exact(count):
    out = run(EXAMPLE, "--jobs", 20, "--count", count, "--out-tensors", 2)
    assert out["kernel"]["error_name"] == "none" and out["kernel"]["events"] == 21
    assert all_checked(out, 20) and out["exact_jobs_checked"] == 10
    assert out["compute_wrong"] == 0 and counts_match(out)
    assert out["gave_up_cores"] == 0 and every_kernel_returned(out)


def test_random_teams_with_four_in_flight():
    out = run(EXAMPLE, "--jobs", 100, "--teams", "random", "--out-tensors", 4)
    assert all_checked(out, 100) and counts_match(out)
    assert out["random_total_rel_rms_max"] < 1e-2 and every_kernel_returned(out)


def test_disjoint_teams_run_at_once():
    out = run(EXAMPLE, "--jobs", 40, "--teams", "pair", "--trace", "--out-tensors", 2)
    assert all_checked(out, 40) and counts_match(out)
    assert out["overlap"]["overlapping"] > 0


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


def test_a_start_error_carries_the_result_and_the_host_stops_the_grid():
    assert run(CASES, "start-error")["ok"]


def test_an_exclusive_record_waits_for_the_team_it_overlaps():
    assert run(CASES, "exclusive")["ok"]


@pytest.mark.skipif(os.environ.get("KMR_CARD_LONG") != "1", reason="KMR_CARD_LONG=1")
def test_thousand_random_jobs_are_exact():
    out = run(
        EXAMPLE, "--jobs", 1000, "--teams", "random", "--out-tensors", 34, timeout=900
    )
    assert all_checked(out, 1000) and counts_match(out)
