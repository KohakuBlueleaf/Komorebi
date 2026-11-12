"""The map/collect kernels on a card, fed by the test feeder (KMR_CARD=1)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.card
SCRIPT = Path(__file__).resolve().parent / "feeder" / "mapcollect_fed.py"


def run(*args, timeout=300):
    """Runs the example and returns its JSON line."""
    done = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    return json.loads(lines[-1])


def every_kernel_returned(out):
    return set(out["cores_ended"].values()) == {out["cores_magic_ok"]} == {119}


def ran_as_members(out):
    """Each core took and finished the records its teams name, and no others."""
    return out["membership_bad"] == []


@pytest.mark.parametrize("count", [1, 8, 119])
def test_fixed_team_is_exact(count):
    out = run("--jobs", 20, "--count", count, "--out-tensors", 2)
    assert out["feeder"]["status"] == "done" and out["feeder"]["done"] == 20
    assert out["feeder"]["exit_acked"]
    assert out["jobs_checked"] == 20 and out["jobs_bad"] == 0
    assert out["compute_wrong"] == 0 and every_kernel_returned(out)
    assert ran_as_members(out)


def test_records_past_the_trace_capacity_are_counted():
    out = run("--jobs", 20, "--count", 8, "--trace", "--trace-cap", 8)
    assert out["feeder"]["status"] == "done" and out["jobs_checked"] == 20
    assert out["trace_dropped"] == {str(c): 12 for c in range(8)}  # seq 9 to 20
    assert every_kernel_returned(out) and ran_as_members(out)


def test_random_teams_with_four_in_flight():
    out = run("--jobs", 100, "--teams", "random", "--depth", 4, "--out-tensors", 4)
    assert out["jobs_checked"] == 100 and out["jobs_bad"] == 0
    assert out["random_total_rel_rms_max"] < 1e-2
    assert out["gave_up_cores"] == 0 and every_kernel_returned(out)
    assert ran_as_members(out)


@pytest.mark.parametrize("first, count", [(6, 13), (11, 14), (0, 60)])
def test_teams_across_rows_complete_through_row_leaders(first, count):
    out = run("--jobs", 20, "--first", first, "--count", count, "--out-tensors", 2)
    assert out["feeder"]["status"] == "done" and out["feeder"]["exit_acked"]
    assert out["jobs_checked"] == 20 and out["jobs_bad"] == 0
    assert ran_as_members(out)


@pytest.mark.skipif(os.environ.get("KMR_CARD_LONG") != "1", reason="KMR_CARD_LONG=1")
def test_a_thousand_jobs_on_random_teams():
    room = ["--out-pages", 2048, "--out-tensors", 32]  # no job's pages reused
    out = run("--jobs", 1000, "--teams", "random", *room, "--seed", 7, timeout=900)
    assert out["feeder"]["status"] == "done" and out["feeder"]["done"] == 1000
    assert out["jobs_checked"] == 1000 and out["jobs_bad"] == 0
    assert out["random_total_rel_rms_max"] < 1e-2 and out["compute_wrong"] == 0
    assert out["gave_up_cores"] == 0 and every_kernel_returned(out)
    assert ran_as_members(out)


def test_disjoint_teams_run_at_once():
    out = run(
        "--jobs", 40, "--teams", "pair", "--depth", 2, "--trace", "--out-tensors", 2
    )
    assert out["jobs_checked"] == 40 and out["jobs_bad"] == 0
    assert out["overlap"]["overlapping"] > 0
    assert ran_as_members(out)


def test_a_stalled_member_ends_in_the_feeder_timeout():
    stall = ["--stall", 3, "--stall-member", 2, "--deadline", 0.02]
    out = run("--jobs", 10, "--count", 8, "--first", 4, *stall)
    seq = out["feeder"]["seq"]
    assert out["feeder"]["status"] == "timeout" and out["timed_out"]["seq"] == seq
    assert out["not_reported"] == [4, 6] and every_kernel_returned(out)  # leader 4
    stalled = [n for n in out["named"] if n.get("reader", {}).get("where") == "stall"]
    assert [(n["core"], n["reader"]["seq"]) for n in stalled] == [(6, seq)]


def waiting(out, where):
    """Cores named by their state words as waiting in `where`, with the sequence."""
    return sorted(
        (n["core"], n[k]["seq"])
        for n in out["named"]
        for k in ("reader", "writer")
        if n.get(k, {}).get("where") == where
    )


# Team of cores 4 to 23 on a grid 12 wide: row leaders 4 (the first member) and 12.
@pytest.mark.parametrize(
    "member, stalled, complete",
    [(8, 12, [4]), (10, 14, [4, 12])],
    ids=["row-leader", "member"],
)
def test_a_stalled_member_across_rows_is_named(member, stalled, complete):
    stall = ["--stall", 3, "--stall-member", member, "--deadline", 0.02]
    out = run("--jobs", 10, "--count", 20, "--first", 4, *stall)
    seq = out["feeder"]["seq"]
    assert out["feeder"]["status"] == "timeout" and out["timed_out"]["seq"] == seq
    assert every_kernel_returned(out)
    assert waiting(out, "stall") == [(stalled, seq)]
    assert waiting(out, "complete") == [(c, seq) for c in complete]
    assert sorted(out["not_reported"]) == sorted([stalled, *complete])
    assert ran_as_members(out)


@pytest.mark.parametrize("excl", [True, False], ids=["excl", "plain"])
def test_excl_records_on_overlapping_teams_are_never_in_flight_together(excl):
    out = run(
        "--jobs",
        40,
        "--teams",
        "overlap",
        "--trace",
        "--out-tensors",
        2,
        *(["--excl"] if excl else []),
    )
    assert out["feeder"]["status"] == "done"
    assert out["jobs_checked"] == 40 and out["jobs_bad"] == 0
    seen = out["overlapping_teams"]
    assert seen["pairs"] > 0
    if excl:  # held by the feeder: no member takes it before the other leader pushed
        assert seen["together"] == 0 and out["held"] > 0
    else:  # the check sees overlap when nothing holds the records
        assert seen["together"] > 0 and out["held"] == 0
    assert ran_as_members(out)


def test_an_exit_record_goes_to_every_core_once_no_slot_is_busy():
    out = run("--jobs", 10, "--count", 60, "--exit-at", 5)  # EXIT with a team of 1
    assert out["feeder"]["status"] == "done" and out["feeder"]["exit_acked"]
    assert out["feeder"]["issued"] == out["feeder"]["done"] == 6
    exit_ = {"held": True, "status": 1, "after_every_ack": True, "reported_by": 119}
    assert out["exit"] == exit_
    assert out["jobs_checked"] == 5 and out["jobs_bad"] == 0
    assert every_kernel_returned(out)
    assert ran_as_members(out)


@pytest.mark.parametrize("row", [0, 12])
def test_an_exit_naming_a_partial_team_completes_as_every_core(row):
    exit_ = ["--exit-at", 5, "--exit-team", "0:2", "--raw", 5, "--row", row]
    out = run("--jobs", 10, "--count", 60, *exit_)  # the feeder keeps the team 0:2
    assert out["feeder"]["status"] == "done" and out["feeder"]["exit_acked"]
    assert out["exit"]["status"] == 1 and out["exit"]["reported_by"] == 119
    assert out["jobs_checked"] == 5 and out["jobs_bad"] == 0
    assert out["gave_up_cores"] == 0 and every_kernel_returned(out)
    assert ran_as_members(out)


@pytest.mark.parametrize("named", [20, 3])
def test_a_record_naming_another_slot_is_not_run(named):
    raw = ["--seg", "nop", "--raw", 0, "--raw-slot", named, "--deadline", 0.02]
    out = run("--jobs", 2, "--depth", 1, "--first", 10, "--count", 8, *raw)  # slot 0
    assert out["feeder"]["status"] == "timeout" and out["feeder"]["index"] == 0
    assert out["slot_wrong"] == {str(c): 1 for c in range(10, 18)}  # JSON keys
    assert out["runs"] == 0 and every_kernel_returned(out)
    assert ran_as_members(out)


def test_a_long_record_in_flight_at_the_stop_ends_on_every_tile():
    # Core 1 stalls (job 0); core 0 runs a MAP of 16384 tiles (job 1, through a buffer
    # of 16 pages) that is still running when the feeder's deadline sets the stop word.
    args = ["--team-list", "1:1,0:1", "--stall", 0, "--tiles", 16384]
    out = run(*args, "--deadline", 0.001, "--out-tensors", 1)
    assert out["feeder"]["status"] == "timeout" and out["timed_out"]["seq"] == 1
    assert every_kernel_returned(out)
    assert waiting(out, "stall") == [(1, 1)]
    busy = {g["core"]: g for g in out["gave_up_busy"]}
    assert set(busy) == {0, 1}
    core0 = busy[0]
    assert core0["reader"]["code"] == "stop" and core0["reader"]["where"] == "data"
    assert core0["compute"] == {"code": "stop", "where": "cinput", "slot": 0, "seq": 2}
    assert ran_as_members(out)


CYCLES_300MS = 405_000_000  # at 1350 MHz


def test_slow_compute_gives_up_and_acknowledges_nothing():
    # Core 0's compute holds job 0's tile 0.3 s; the writer's wait limit is 0.1 s.
    slow = ["--slow", 0, "--slow-member", 0, "--slow-cycles", CYCLES_300MS]
    out = run("--team-list", "0:1,0:1,0:1", *slow, "--wait", 0.1, "--deadline", 1.0)
    assert out["feeder"]["status"] == "timeout" and out["timed_out"]["seq"] == 1
    assert out["feeder"]["done"] == 0 and out["not_reported"] == [0]
    assert every_kernel_returned(out)
    writer = {g["core"]: g for g in out["gave_up_busy"]}[0]["writer"]
    assert (writer["code"], writer["where"]) == ("limit", "data")
    assert ran_as_members(out)


def test_a_leader_whose_member_is_late_acknowledges_nothing():
    # Member 1 reports job 0 after 0.3 s; leader 0 waits 0.1 s, gives up, stops.
    late = ["--late", 0, "--late-member", 1, "--late-cycles", CYCLES_300MS]
    out = run("--team-list", "0:2,0:2", *late, "--wait", 0.1, "--deadline", 1.0)
    assert out["feeder"]["status"] == "timeout" and out["timed_out"]["seq"] == 1
    assert out["feeder"]["done"] == 0 and out["not_reported"] == [0]
    assert every_kernel_returned(out)
    leader = {g["core"]: g for g in out["named"]}[0]["writer"]
    assert leader == {"code": "limit", "where": "complete", "slot": 0, "seq": 1}
    assert ran_as_members(out)


def test_a_reader_stopped_in_its_token_wait_hands_off_and_every_kernel_returns():
    # Core 0's compute holds job 0 for 0.3 s (past stop and grace); job 1 needs no data
    # (LATE for a non-member), so the reader waits for token room for job 2.
    slow = ["--slow", 0, "--slow-member", 0, "--slow-cycles", CYCLES_300MS]
    slow += ["--late", 1, "--late-member", 7, "--wait", 0.1, "--deadline", 0.05]
    out = run("--team-list", "0:1,0:1,0:1", *slow)
    assert out["feeder"]["status"] == "timeout" and out["timed_out"]["seq"] == 1
    assert every_kernel_returned(out)
    core0 = {g["core"]: g for g in out["gave_up_busy"]}[0]
    assert (core0["reader"]["code"], core0["reader"]["where"]) == ("stop", "token")
    assert ran_as_members(out)
