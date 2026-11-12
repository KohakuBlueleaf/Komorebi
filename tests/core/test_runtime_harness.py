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


def test_a_record_completes(scenario):
    got = scenario("smoke")
    assert got["ok"] and got["done"] == got["jobs"], got


def test_a_record_without_output_is_acked_after_its_compute(scenario):
    got = scenario("no_output_waits_for_compute")
    assert got["ok"], got


def test_an_item_past_the_buffer_end_gives_up_with_its_word(scenario):
    got = scenario("wrappers_refuse_straddling_items")
    assert got["ok"], got


def test_exit_reaches_every_worker(scenario):
    got = scenario("exit_partial_team")
    assert got["ok"], got


def test_an_excl_record_waits_for_an_overlapping_team(scenario):
    got = scenario("excl_waits_for_overlapping_team")
    assert got["ok"], got
