"""The host package on a card: the map/collect example under the dispatch kernel, and
the session cases it does not cover (KMR_CARD=1). Every case shares one device open."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import komorebi as km

pytest.importorskip("ttnn")

pytestmark = pytest.mark.card
CARD_IN_PROCESS = True  # opens the card in this process
ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mapcollect = load("mapcollect", ROOT / "examples" / "01_mapcollect" / "mapcollect.py")
cases = load("session_cases", Path(__file__).parent / "host" / "session_cases.py")


@pytest.fixture(scope="module")
def session():
    with km.Session(log=mapcollect.log) as s:
        yield s


def example(session, *argv):
    """The example's run on the shared session; returns its JSON dict."""
    args = mapcollect.parse([str(a) for a in argv])
    return mapcollect.run(session, args, np.random.default_rng(args.seed))


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
def test_fixed_team_is_exact(session, count):
    out = example(session, "--jobs", 20, "--count", count, "--out-tensors", 2)
    assert out["kernel"]["error_name"] == "none" and out["kernel"]["events"] == 21
    assert all_checked(out, 20) and out["exact_jobs_checked"] == 10
    assert out["compute_wrong"] == 0 and counts_match(out)
    assert out["gave_up_cores"] == 0 and every_kernel_returned(out)


def test_random_teams_with_four_in_flight(session):
    out = example(session, "--jobs", 50, "--teams", "random", "--out-tensors", 3)
    assert all_checked(out, 50) and counts_match(out)
    assert out["random_total_rel_rms_max"] < 1e-2 and every_kernel_returned(out)


def test_disjoint_teams_run_at_once(session):
    out = example(
        session, "--jobs", 40, "--teams", "pair", "--trace", "--out-tensors", 2
    )
    assert all_checked(out, 40) and counts_match(out)
    assert out["overlap"]["overlapping"] > 0


def test_an_exclusive_record_waits_for_the_team_it_overlaps(session):
    assert cases.case_exclusive(session)["ok"]


def test_tensors_read_by_banks_match_what_was_written_also_during_a_run(session):
    assert cases.case_read(session)["ok"]


def test_a_compile_error_leaves_no_image_running(session):
    assert cases.case_compile_error(session)["ok"]


def test_a_start_error_carries_the_result_and_the_host_stops_the_grid(session):
    assert cases.case_start_error(session)["ok"]


def test_a_stalled_member_ends_in_the_timeout_exception(session):
    stall = ["--stall", 3, "--stall-member", 2, "--deadline", 0.02]
    out = example(session, "--jobs", 10, "--count", 8, "--first", 4, *stall)
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
