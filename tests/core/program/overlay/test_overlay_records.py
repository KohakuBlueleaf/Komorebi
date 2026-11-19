"""The overlay records on the CPU (km_overlay_records.hpp): a check of the service
overlay's own slot marks its own record and nothing else, at 4, 6 and 8 slots a part
(no card)."""

import shutil
import subprocess
from pathlib import Path

import pytest

from komorebi import _env


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++ compiler")
@pytest.mark.parametrize("slots", [4, 6, 8])
def test_the_own_slot_s_check_marks_its_own_record_alone(tmp_path, slots):
    source = Path(__file__).resolve().parent / "km_overlay_records_check.cpp"
    exe = tmp_path / "overlay_records_check"
    subprocess.run(
        [
            "c++", "-std=c++17", "-Wall", "-Werror", f"-DKM_OVERLAY_SLOTS={slots}",
            f"-I{_env.include_directory()}", str(source), "-o", exe,
        ],
        check=True,
    )  # fmt: skip
    assert subprocess.run([exe], capture_output=True, text=True).stdout == "ok\n"
