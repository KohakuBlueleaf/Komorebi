"""The map/collect kernels on a card, fed by the test feeder (KMR_CARD=1). One device
open for the module; each test is one behaviour of the runtime on the smallest run."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("KMR_CARD") != "1":
    pytest.skip("needs a Tenstorrent card (KMR_CARD=1)", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parent / "feeder"))

import mapcollect_fed as mf
import ttnn

pytestmark = pytest.mark.card
CYCLES_300MS = 405_000_000  # at 1350 MHz


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    dev.disable_and_clear_program_cache()
    yield dev
    ttnn.close_mesh_device(dev)


@pytest.fixture
def run(device):
    """Runs the example with these command-line arguments; returns its summary."""

    def go(*args):
        a = mf.parse([str(x) for x in ("--out-pages", 256, *args)])
        return mf.run(device, a, np.random.default_rng(a.seed))

    return go


def every_kernel_returned(out):
    return set(out["cores_ended"].values()) == {out["cores_magic_ok"]} == {119}


def ran_as_members(out):
    """Each core took and finished the records its teams name, and no others."""
    return out["membership_bad"] == []


def waiting(out, where):
    """Cores named by their state words as waiting in `where`, with the sequence."""
    return sorted(
        (n["core"], n[k]["seq"])
        for n in out["named"]
        for k in ("reader", "writer")
        if n.get(k, {}).get("where") == where
    )


def test_a_fixed_team_is_exact(run):
    out = run("--jobs", 8, "--count", 8)
    assert out["feeder"]["status"] == "done" and out["feeder"]["exit_acked"]
    assert out["jobs_checked"] == 8 and out["jobs_bad"] == 0
    assert out["compute_wrong"] == 0 and every_kernel_returned(out)
    assert ran_as_members(out)


def test_a_team_across_rows_completes_through_row_leaders(run):
    out = run("--jobs", 4, "--first", 11, "--count", 14, "--row", 12)
    assert out["feeder"]["status"] == "done" and out["feeder"]["exit_acked"]
    assert out["jobs_checked"] == 4 and out["jobs_bad"] == 0
    assert ran_as_members(out)


def test_records_past_the_trace_capacity_are_counted(run):
    out = run("--jobs", 12, "--count", 8, "--trace", "--trace-cap", 8)
    assert out["feeder"]["status"] == "done"
    assert out["trace_dropped"] == {c: 4 for c in range(8)}  # seq 9 to 12
    assert every_kernel_returned(out) and ran_as_members(out)


def test_a_flagged_record_is_acknowledged_with_the_leaders_result(run):
    out = run("--jobs", 3, "--first", 4, "--count", 8, "--result", 1)
    assert out["feeder"]["status"] == "done" and out["feeder"]["done"] == 3
    assert out["result"] == 0x5EED  # the leader's: a0 plus index 0
    assert every_kernel_returned(out) and ran_as_members(out)


def test_a_stalled_member_is_named_after_the_feeder_timeout(run):
    # Team of cores 4 to 23 (row leaders 4 and 12); member 10 (core 14) stalls job 3.
    stall = ["--stall", 3, "--stall-member", 10, "--deadline", 0.02]
    out = run("--jobs", 6, "--count", 20, "--first", 4, "--row", 12, *stall)
    seq = out["feeder"]["seq"]
    assert out["feeder"]["status"] == "timeout" and out["timed_out"]["seq"] == seq
    assert every_kernel_returned(out)
    assert waiting(out, "stall") == [(14, seq)]
    assert waiting(out, "complete") == [(4, seq), (12, seq)]
    assert sorted(out["not_reported"]) == [4, 12, 14]


def test_excl_records_on_overlapping_teams_are_never_in_flight_together(run):
    out = run("--jobs", 12, "--teams", "overlap", "--trace", "--excl")
    assert out["feeder"]["status"] == "done" and out["jobs_bad"] == 0
    seen = out["overlapping_teams"]
    assert seen["pairs"] > 0 and seen["together"] == 0 and out["held"] > 0
    assert ran_as_members(out)


def test_an_exit_naming_a_partial_team_completes_as_every_core(run):
    exit_ = ["--exit-at", 3, "--exit-team", "0:2", "--raw", 3]
    out = run("--jobs", 6, "--count", 60, *exit_)  # the feeder keeps the team 0:2
    assert out["feeder"]["status"] == "done" and out["feeder"]["exit_acked"]
    assert out["exit"]["status"] == 1 and out["exit"]["reported_by"] == 119
    assert out["gave_up_cores"] == 0 and every_kernel_returned(out)


def test_a_record_naming_another_slot_is_not_run(run):
    raw = ["--seg", "nop", "--raw", 0, "--raw-slot", 3, "--deadline", 0.02]
    out = run("--jobs", 2, "--depth", 1, "--first", 10, "--count", 8, *raw)  # slot 0
    assert out["feeder"]["status"] == "timeout" and out["feeder"]["index"] == 0
    assert out["slot_wrong"] == {c: 1 for c in range(10, 18)}
    assert out["runs"] == 0 and every_kernel_returned(out)


def test_slow_compute_gives_up_and_acknowledges_nothing(run):
    # Core 0's compute holds job 0's tile 0.3 s; the writer's wait limit is 0.1 s.
    slow = ["--slow", 0, "--slow-member", 0, "--slow-cycles", CYCLES_300MS]
    out = run("--team-list", "0:1,0:1", *slow, "--wait", 0.1, "--deadline", 1.0)
    assert out["feeder"]["status"] == "timeout" and out["timed_out"]["seq"] == 1
    assert out["feeder"]["done"] == 0 and out["not_reported"] == [0]
    assert every_kernel_returned(out)
    writer = {g["core"]: g for g in out["gave_up_busy"]}[0]["writer"]
    assert (writer["code"], writer["where"]) == ("limit", "data")


def test_a_leader_whose_member_is_late_acknowledges_nothing(run):
    # Member 1 reports job 0 after 0.3 s; leader 0 waits 0.1 s, gives up, stops.
    late = ["--late", 0, "--late-member", 1, "--late-cycles", CYCLES_300MS]
    out = run("--team-list", "0:2,0:2", *late, "--wait", 0.1, "--deadline", 1.0)
    assert out["feeder"]["status"] == "timeout" and out["timed_out"]["seq"] == 1
    assert out["feeder"]["done"] == 0 and out["not_reported"] == [0]
    assert every_kernel_returned(out)
    leader = {g["core"]: g for g in out["named"]}[0]["writer"]
    assert leader == {"code": "limit", "where": "complete", "slot": 0, "seq": 1}


def test_an_item_past_the_buffer_end_gives_up_before_any_access(run):
    # Job 0 (65 tiles, one output page) leaves the output buffer's pointer at page 1;
    # job 1 asks member 0 for 2 output pages from there.
    bad = ["--bad-item", 1, "--bad-side", "writer", "--tiles", 65, "--deadline", 0.05]
    out = run("--team-list", "0:1,0:1,0:1", "--depth", 1, *bad)  # job 0 acked first
    assert out["feeder"]["status"] == "timeout" and out["timed_out"]["seq"] == 2
    assert out["feeder"]["done"] == 1 and out["not_reported"] == [0]
    assert every_kernel_returned(out)
    core0 = {g["core"]: g for g in out["gave_up_busy"]}[0]
    assert core0["writer"] == {"code": "bad_item", "where": "data", "slot": 0, "seq": 0}
