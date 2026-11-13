"""The tutorial's programs: each folder has the three kernels and its handlers, and its
Python file builds a Program that the host accepts (no card is opened)."""

import ast
from pathlib import Path

import pytest

import komorebi as km

TUTORIAL = Path(__file__).resolve().parents[2] / "tutorial"
FOLDERS = sorted(p for p in TUTORIAL.iterdir() if p.is_dir())


@pytest.mark.parametrize("folder", FOLDERS, ids=lambda p: p.name)
def test_a_folder_has_a_script_that_parses_and_three_kernels(folder):
    scripts = list(folder.glob("*.py"))
    assert len(scripts) == 1
    ast.parse(scripts[0].read_text())
    kernels = folder / "kernels"
    if kernels.exists():
        assert {p.stem for p in kernels.glob("*.cpp")} == {
            "reader",
            "compute",
            "writer",
        }
        assert km.Program(folder.name, kernels, handlers=kernels / "handlers.h")
