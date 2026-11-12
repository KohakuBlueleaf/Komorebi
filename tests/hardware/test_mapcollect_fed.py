"""The map/collect kernels on a card, fed by the test feeder (KMR_CARD=1)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.card
SCRIPT = Path(__file__).resolve().parent / "feeder" / "mapcollect_fed.py"


def run(*args):
    """Runs the example and returns its JSON line."""
    done = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    return json.loads(lines[-1])


def every_kernel_returned(out):
    return set(out["cores_ended"].values()) == {out["cores_magic_ok"]} == {119}


@pytest.mark.parametrize("count", [1, 8, 119])
def test_fixed_team_is_exact(count):
    out = run("--jobs", 20, "--count", count, "--out-tensors", 2)
    assert out["feeder"]["status"] == "done" and out["feeder"]["done"] == 20
    assert out["feeder"]["exit_acked"]
    assert out["jobs_checked"] == 20 and out["jobs_bad"] == 0
    assert out["compute_wrong"] == 0 and every_kernel_returned(out)


def test_random_teams_with_four_in_flight():
    out = run("--jobs", 100, "--teams", "random", "--depth", 4, "--out-tensors", 4)
    assert out["jobs_checked"] == 100 and out["jobs_bad"] == 0
    assert out["random_total_rel_rms_max"] < 1e-2
    assert out["gave_up_cores"] == 0 and every_kernel_returned(out)


@pytest.mark.parametrize("count", [60, 119])
def test_row_leader_completion(count):
    out = run("--jobs", 20, "--count", count, "--rows", "--out-tensors", 2)
    assert out["completion"] == "row leaders" and out["feeder"]["exit_acked"]
    assert out["jobs_checked"] == 20 and out["jobs_bad"] == 0


def test_disjoint_teams_run_at_once():
    out = run("--jobs", 40, "--teams", "pair", "--depth", 2, "--trace")
    assert out["jobs_bad"] == 0 and out["overlap"]["overlapping"] > 0


def test_a_stalled_member_ends_in_the_feeder_timeout():
    stall = ["--stall", 3, "--stall-member", 2, "--deadline", 0.02]
    out = run("--jobs", 10, "--count", 8, "--first", 4, *stall)
    seq = out["feeder"]["seq"]
    assert out["feeder"]["status"] == "timeout" and out["timed_out"]["seq"] == seq
    assert 6 in out["not_reported"] and every_kernel_returned(out)
    stalled = [n for n in out["named"] if n.get("reader", {}).get("where") == "stall"]
    assert [(n["core"], n["reader"]["seq"]) for n in stalled] == [(6, seq)]
