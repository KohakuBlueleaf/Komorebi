"""The map/collect kernels on a card, fed by the test feeder (KMR_CARD=1): the
behaviours the dispatcher's tests do not reach, each on the smallest run."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("KMR_CARD") != "1":
    pytest.skip("needs a Tenstorrent card (KMR_CARD=1)", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mapcollect_fed

pytestmark = pytest.mark.card
CYCLES_300MS = 405_000_000  # at 1350 MHz


@pytest.fixture
def run(session):
    """Runs the example with these command-line arguments; returns its summary."""

    def go(*args):
        arguments = mapcollect_fed.parse([str(x) for x in ("--out-pages", 256, *args)])
        return mapcollect_fed.run(
            session, arguments, np.random.default_rng(arguments.seed)
        )

    return go


def every_kernel_returned(out):
    return set(out["cores_ended"].values()) == {out["cores_magic_ok"]} == {119}


def ran_as_ranks(out):
    """Each core took and finished the commands its groups name, and no others."""
    return out["ranks_bad"] == []


def test_a_group_across_rows_completes_through_row_leaders(run):
    out = run("--jobs", 4, "--first", 11, "--count", 14, "--row", 12)
    assert out["feeder"]["status"] == "done" and out["feeder"]["exit_acked"]
    assert out["jobs_checked"] == 4 and out["jobs_bad"] == 0
    assert ran_as_ranks(out)


def test_an_exit_naming_a_partial_group_completes_as_every_core(run):
    exit_ = ["--exit-at", 3, "--exit-group", "0:2", "--raw", 3]
    out = run("--jobs", 6, "--count", 60, *exit_)  # the feeder keeps the group 0:2
    assert out["feeder"]["status"] == "done" and out["feeder"]["exit_acked"]
    assert out["exit"]["status"] == 1 and out["exit"]["reported_by"] == 119
    assert out["gave_up_cores"] == 0 and every_kernel_returned(out)


def test_a_leader_whose_rank_is_late_acknowledges_nothing(run):
    # Rank 1 reports job 0 after 0.3 s; leader 0 waits 0.1 s, gives up, stops.
    late = ["--late", 0, "--late-rank", 1, "--late-cycles", CYCLES_300MS]
    out = run("--group-list", "0:2,0:2", *late, "--wait", 0.1, "--deadline", 1.0)
    assert out["feeder"]["status"] == "timeout" and out["timed_out"]["sequence"] == 1
    assert out["feeder"]["done"] == 0 and out["not_reported"] == [0]
    assert every_kernel_returned(out)
    leader = {entry["core"]: entry for entry in out["named"]}[0]["writer"]
    assert leader == {"code": "limit", "where": "complete", "slot": 0, "sequence": 1}


def test_an_item_past_the_buffer_end_gives_up_before_any_access(run):
    # Job 0 (65 tiles, one output page) leaves the output buffer's pointer at page 1;
    # job 1 asks rank 0 for 2 output pages from there.
    bad = ["--bad-item", 1, "--bad-side", "writer", "--tiles", 65, "--deadline", 0.05]
    out = run("--group-list", "0:1,0:1,0:1", "--depth", 1, *bad)  # job 0 acked first
    assert out["feeder"]["status"] == "timeout" and out["timed_out"]["sequence"] == 2
    assert out["feeder"]["done"] == 1 and out["not_reported"] == [0]
    assert every_kernel_returned(out)
    core0 = {entry["core"]: entry for entry in out["gave_up_busy"]}[0]
    assert core0["writer"] == {
        "code": "bad_item",
        "where": "data",
        "slot": 0,
        "sequence": 0,
    }
