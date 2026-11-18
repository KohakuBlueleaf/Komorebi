"""The tutorial: each program folder holds its kernels (three, or movement and compute)
and a script that builds a Program the host accepts (no card is opened), and the lesson
pages are the ones the order file lists, with the programs and outputs they include."""

import ast
import re
from pathlib import Path

import pytest

import komorebi as km

ROOT = Path(__file__).resolve().parents[3]
TUTORIAL = ROOT / "tutorial"
LESSONS = ROOT / "docs" / "tutorial"
FOLDERS = sorted(p for p in TUTORIAL.iterdir() if p.is_dir())
INCLUDE = re.compile(r'--8<-- "(tutorial/[^":]+)')


def lesson_slugs():
    lines = (LESSONS / "order.txt").read_text().split("\n")
    return [line.strip() for line in lines if line.strip() and line[0] not in "#>"]


@pytest.mark.parametrize("folder", FOLDERS, ids=lambda p: p.name)
def test_a_folder_has_a_script_that_parses_three_kernels_and_its_output(folder):
    scripts = list(folder.glob("*.py"))
    assert len(scripts) <= 1
    if scripts:
        ast.parse(scripts[0].read_text())
        assert (folder / "output.txt").is_file()
    kernels = folder / "kernels"
    if kernels.exists():
        stems = {p.stem for p in kernels.glob("*.cpp")}
        if stems == {"compute"}:  # a dynamic handler: Komorebi's own reader and writer
            assert ".dynamic(" in scripts[0].read_text()
            return
        if "parts=" in scripts[0].read_text():  # handlers as parts: Komorebi's kernels
            assert ".dynamic(" in scripts[0].read_text()
            return
        assert stems in ({"reader", "compute", "writer"}, {"movement", "compute"})
        header = kernels / "handlers.h"
        assert km.Program(
            folder.name, kernels, handlers=header if header.exists() else {}
        )


def test_the_order_file_and_the_lesson_pages_agree():
    slugs = lesson_slugs()
    assert len(slugs) == len(set(slugs))
    pages = {p.stem for p in LESSONS.glob("*.md")} - {"index"}
    assert pages == set(slugs)
    for slug in slugs:
        head = (LESSONS / f"{slug}.md").read_text().split("\n---\n")[0]
        for key in ("title:", "part:", "summary:"):
            assert key in head, f"{slug}: no {key}"


def test_every_program_a_lesson_includes_exists():
    for page in LESSONS.glob("*.md"):
        for path in INCLUDE.findall(page.read_text()):
            assert (ROOT / path).is_file(), f"{page.name}: {path}"
