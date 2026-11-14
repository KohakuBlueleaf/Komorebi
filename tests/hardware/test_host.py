"""The host package on a card: the map/collect example under the dispatcher, and
the session cases it does not cover (KMR_CARD=1). Every case shares one device open."""

import importlib.util
import time
from pathlib import Path

import numpy as np
import pytest

from komorebi import dram

pytestmark = pytest.mark.card
ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mapcollect = load("mapcollect", ROOT / "examples" / "01_mapcollect" / "mapcollect.py")
cases = load("session_cases", Path(__file__).parent / "host" / "session_cases.py")


def example(session, *argv):
    """The example's run on the shared session; returns its JSON dict."""
    args = mapcollect.parse([str(argument) for argument in argv])
    return mapcollect.run(session, args, np.random.default_rng(args.seed))


def every_kernel_returned(out, cores=120):
    ended = out["cores_ended"]
    return out["cores_magic_ok"] == cores and set(ended.values()) == {cores}


def counts_match(out):
    """Each core took and finished as many commands as it is in."""
    return out["core_counts_bad"] == 0 and out["rank_commands"] > 0


def all_checked(out, jobs, stalled=0):
    """Every job not stalled was read back and checked; none overwritten, no stale
    tile could pass (the page held something else before)."""
    return (
        out["jobs_checked"] == jobs - stalled
        and out["jobs_overwritten"] == 0
        and out["tiles_same_as_before"] == 0
        and out["jobs_bad"] == 0
    )


@pytest.mark.parametrize("count", [1, 120])
def test_fixed_group_is_exact(session, count):
    out = example(session, "--jobs", 20, "--count", count, "--out-tensors", 2)
    # one of the events answers the reset lists the load hands the dispatcher
    assert out["kernel"]["error_name"] == "none" and out["kernel"]["events"] == 22
    assert all_checked(out, 20) and out["exact_jobs_checked"] == 10
    assert out["compute_wrong"] == 0 and counts_match(out)
    assert out["gave_up_cores"] == 0 and every_kernel_returned(out)


def test_disjoint_groups_run_at_once(session):
    out = example(
        session, "--jobs", 40, "--groups", "pair", "--trace", "--out-tensors", 2
    )
    assert all_checked(out, 40) and counts_match(out)
    assert out["overlap"]["overlapping"] > 0


def test_an_exclusive_command_waits_for_the_group_it_overlaps(session):
    assert cases.case_exclusive(session)["ok"]


def test_an_exclusive_command_waits_for_the_commands_queued_before_it(session):
    out = cases.case_queue_order(session)
    assert out["ok"], out


def test_tensors_read_by_banks_match_what_was_written_also_during_a_run(session):
    assert cases.case_read(session)["ok"]


def test_a_page_range_and_the_chunks_equal_slices_of_the_whole_read(session):
    """A bfloat16 tile tensor of 300 pages: a range from the middle, the chunks of
    64 KB and two tiles equal the whole read; the read times go to the log."""
    x = np.random.default_rng(5).integers(-8, 9, (640, 480)).astype(np.float32)
    tensor = cases.bf16_tiles(session, x)
    started = time.monotonic()
    (whole,) = session.read(tensor)
    mapcollect.log(
        f"{whole.nbytes} bytes read whole in {time.monotonic() - started:.2f} s"
    )
    assert (dram.untilize_bf16(whole, 640, 480) == x).all()
    (middle,) = session.read(tensor, first=117, pages=45)
    assert (middle == whole[117:162]).all()
    chunks, started = [], time.monotonic()
    for first, pages in session.read_chunks(tensor, chunk_bytes=64 << 10):
        mapcollect.log(f"chunk at page {first}: {len(pages)} pages")
        chunks.append(pages)
    mapcollect.log(f"10 chunks read in {time.monotonic() - started:.2f} s")
    assert len(chunks) == 10 and (np.concatenate(chunks) == whole).all()
    tiles = session.read_tiles_bf16(tensor, 16, 2)  # tiles (1, 1) and (1, 2)
    assert (tiles[0] == x[32:64, 32:64]).all() and (tiles[1] == x[32:64, 64:96]).all()


def test_a_compile_error_leaves_no_image_running(session):
    assert cases.case_compile_error(session)["ok"]


def test_a_start_error_carries_the_result_and_the_host_stops_the_grid(session):
    assert cases.case_start_error(session)["ok"]


def test_a_stalled_rank_ends_in_the_timeout_exception(session):
    stall = ["--stall", 3, "--stall-rank", 2, "--deadline", 0.02]
    out = example(session, "--jobs", 10, "--count", 8, "--first", 4, *stall)
    timed_out = out["timed_out"]  # jobs 4 to 6 are in core 6's ring: no reset
    assert out["kernel"]["error_name"] == "deadline" and out["recoveries"] == []
    assert "core 6 was not reset: other commands in flight" in timed_out["message"]
    assert timed_out["handle"] == 5 and timed_out["group"] == [4, 8]
    named = {core_entry["core"]: core_entry for core_entry in timed_out["not_reported"]}
    assert sorted(named) == [4, 6]
    assert (
        named[6]["reader"]["where"] == "stall"
        and named[6]["reader"]["sequence"] == timed_out["sequence"]
    )
    assert named[4]["writer"]["where"] == "complete"
    assert every_kernel_returned(out)
