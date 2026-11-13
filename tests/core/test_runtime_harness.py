"""Known behaviours of the Tensix runtime, run on the CPU through the harness in
experiments/runtime_harness (built into build/ and reused; skipped without a C++
compiler). The harness's other scenarios are run by hand.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "experiments" / "runtime_harness" / "run.py"
pytestmark = pytest.mark.skipif(shutil.which("c++") is None, reason="needs c++")


@pytest.fixture(scope="module")
def scenario():
    build = [sys.executable, str(RUN), "--build-only", "--pinned"]
    program = subprocess.run(build, capture_output=True, text=True, check=True)

    def run(name):
        out = subprocess.run([program.stdout.strip(), name], capture_output=True)
        return json.loads(out.stdout)

    return run


def test_a_command_completes(scenario):
    got = scenario("smoke")
    assert got["ok"] and got["done"] == got["jobs"], got


def test_a_command_without_output_is_acked_after_its_compute(scenario):
    got = scenario("no_output_waits_for_compute")
    assert got["ok"], got


def test_the_wall_clock_reads_across_a_carry_of_its_low_word(scenario):
    got = scenario("wall_clock_across_a_carry")
    assert got["ok"], got


def test_a_flagged_command_is_acked_with_the_leaders_result(scenario):
    got = scenario("result_word_from_the_leader")
    assert got["ok"], got


def test_an_item_past_the_buffer_end_gives_up_with_its_word(scenario):
    got = scenario("wrappers_reject_straddling_items")
    assert got["ok"], got


def test_exit_reaches_every_worker(scenario):
    got = scenario("exit_partial_group")
    assert got["ok"], got


def test_an_excl_command_waits_for_an_overlapping_group(scenario):
    got = scenario("excl_waits_for_overlapping_group")
    assert got["ok"], got


def test_a_dynamic_run_ends_quiet_with_every_message_consumed(scenario):
    got = scenario("messages_smoke")
    assert got["ok"], got


def test_a_missing_queue_item_is_a_gap_give_up(scenario):
    got = scenario("messages_gap")
    assert got["ok"], got


def test_a_queue_past_its_capacity_is_an_overflow_give_up(scenario):
    got = scenario("messages_overflow")
    assert got["ok"], got


def test_idle_lines_that_flap_never_end_a_run_early(scenario):
    got = scenario("messages_idle_flapping")
    assert got["ok"], got
