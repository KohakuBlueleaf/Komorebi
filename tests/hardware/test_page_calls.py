"""Kernels that read and write a tensor's pages through Komorebi's page calls, on a card
(KMR_CARD=1): the examples whose other tests do not run them, each small and exact."""

import sys
from pathlib import Path

import pytest
from shared_card import run_json, run_text

pytestmark = pytest.mark.card
ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def test_the_bucket_sort_reads_its_keys_and_sorts_them():
    script = EXAMPLES / "09_bucket_sort" / "bucket_sort.py"
    sys.modules.pop("reference", None)  # a project's package of that name, imported
    out = run_json(script, "--keys", 1 << 16, "--buckets", 256, "--x280-cores", 1)
    assert out["ok"]


def test_the_scale_program_reads_and_writes_its_tiles():
    script = EXAMPLES / "19_thread_programs" / "thread_programs.py"
    run_json(script, "--cores", 12, "--rounds", 4, "--splits", 2)


def test_the_gate_streams_are_exact():
    run_json(EXAMPLES / "22_gate" / "gate.py", "--cores", 4, "--tiles", 16)


def test_the_cast_bodies_are_read_and_cast():
    script = EXAMPLES / "23_message_body" / "cast_body.py"
    run_json(script, "--casts", 100, "--jobs", 2)


def test_the_lesson_reads_a_tile_from_dram():
    script = ROOT / "tutorial" / "16_reading_from_dram" / "reading_from_dram.py"
    out = run_text(script)
    assert "tile 2, number 5: 2005" in out
