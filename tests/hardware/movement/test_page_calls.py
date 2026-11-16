"""Kernels that read and write a tensor's pages through Komorebi's page calls, on a card
(KMR_CARD=1): the examples whose other tests do not run them, each small and exact."""

import re
import sys
from pathlib import Path

import pytest
from shared_card import run_json, run_text

pytestmark = pytest.mark.card
ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples"
# The lessons whose kernels read or write a tensor's pages through the page calls.
LESSONS = [
    "17_writing_to_dram",
    "18_three_kernels",
    "19_circular_buffers",
    "20_computing_on_a_tile",
    "21_the_matrix_unit",
    "22_a_whole_map",
    "23_both_movers",
    "25_port",
    "26_reduce",
    "27_broadcast",
    "38_measuring",
    "39_finding_a_bug",
    "40_a_whole_program",
    "41_your_own_kernel",
    "45_casting_a_word",
]
DECIMAL = re.compile(r"\s*\d+\.\d+")


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


def test_the_thread_load_programs_read_their_tiles_and_write_their_lines():
    script = EXAMPLES / "21_thread_load" / "thread_load.py"
    out = run_json(script, "--cores", 12, "--rounds", 4)
    assert out["ok"]


@pytest.mark.parametrize("lesson", LESSONS)
def test_the_lesson_prints_its_output(lesson):
    folder = ROOT / "tutorial" / lesson
    (script,) = folder.glob("*.py")
    out = run_text(script)
    for line in (folder / "output.txt").read_text().splitlines():
        # a measured time may differ: any decimal number in its place
        parts = DECIMAL.split(line)
        pattern = r"\s*\d+\.\d+".join(re.escape(part) for part in parts)
        assert re.search(pattern, out), line
