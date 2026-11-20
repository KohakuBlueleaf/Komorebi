"""The Python face of the Komorebi IR (the wheel of packaging/ir/, installed): each
function on lesson T1 and on a refused case; the hashes compared with `komo hashes`.
"""

import os
import subprocess
from pathlib import Path

import pytest

import komorebi_ir as ir

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests" / "ir" / "corpus" / "golden"
T1 = (GOLDEN / "lessons" / "t01_scale.kom").read_text()
T6 = (GOLDEN / "lessons" / "t06_rows.kom").read_text()
BROKEN = (GOLDEN / "invalid" / "k001_parse.kom").read_text()
SCRIPT = (ROOT / "tests" / "ir" / "parity" / "t01_scale_group3.run").read_text()


def komo_hashes(path: Path) -> tuple[str, str, str]:
    """`komo hashes` on a file: the semantic, build and full hashes."""
    command = ["cargo", "run", "-q", "--bin", "komo", "--", "hashes", str(path)]
    done = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=True, env=os.environ
    )
    values = dict(line.split() for line in done.stdout.splitlines())
    return values["semantic"], values["build"], values["full"]


def test_format_gives_the_canonical_text_and_refuses_a_broken_one():
    assert ir.format(T1) == T1
    assert ir.format(T1.replace("tile [32, 32]", "tile   [32,32]")) == T1
    with pytest.raises(ir.IrError) as refused:
        ir.format(BROKEN)
    assert [d.code for d in ir.diagnostics(refused.value)] == ["K001"]


def test_check_gives_diagnostics_as_data():
    assert ir.check(T1) == []
    (d,) = ir.check(BROKEN)
    assert (d.code, d.line, d.severity, d.byte) == ("K001", 9, "error", None)
    assert d.column >= 1 and d.message
    hops = (GOLDEN / "invalid" / "k702_island_hops_in_control.kom").read_text()
    assert [(d.code, d.line) for d in ir.check(hops)] == [("K702", 14), ("K702", 17)]


def test_compile_gives_bytes_and_refuses_a_broken_text():
    program = ir.compile(T1)
    assert program[:4] == b"KOMC"
    assert ir.compile(T1) == program
    with pytest.raises(ir.IrError) as refused:
        ir.compile(BROKEN)
    assert ir.diagnostics(refused.value)[0].code == "K001"


def test_the_hashes_are_the_commands(tmp_path):
    text = tmp_path / "t01_scale.kom"
    text.write_text(T1)
    program = tmp_path / "t01_scale.komc"
    program.write_bytes(ir.compile(T1))
    assert tuple(ir.hashes(T1)) == komo_hashes(text)
    assert tuple(ir.hashes(program.read_bytes())) == komo_hashes(program)
    golden = (GOLDEN / "hashes.txt").read_text().splitlines()
    (line,) = [x for x in golden if x.startswith("t01_scale ")]
    assert tuple(ir.hashes(T1)) == tuple(line.split()[1:])


def test_a_cut_byte_code_file_is_refused_at_its_byte():
    with pytest.raises(ir.IrError) as refused:
        ir.hashes(ir.compile(T1)[:40])
    (d,) = ir.diagnostics(refused.value)
    assert d.code.startswith("K95") and d.line is None and d.byte is not None


def test_requirements_name_their_constructs():
    assert ir.requirements(T1) == []
    assert ir.requirements(ir.compile(T1)) == []
    found = ir.requirements(T6)
    assert [tuple(r) for r in found] == [
        ("group = grid", ["requires", "handler Rows.spread"])
    ]
    with pytest.raises(ir.IrError):
        ir.requirements(BROKEN)


def test_run_gives_the_report_of_a_script():
    report = ir.run(T1, SCRIPT)
    assert report.splitlines()[0] == "#0 Scale.run group(0, 3): completed"
    tiles = [x.strip() for x in report.splitlines() if x.startswith("  tile ")]
    assert tiles == [f"tile {i}: {2.0 * (i + 1)}*1024" for i in range(5)]
    assert ir.run(ir.compile(T1), SCRIPT) == report
    with pytest.raises(ir.IrError):
        ir.run(BROKEN, SCRIPT)
    with pytest.raises(ir.IrError, match="line 1"):
        ir.run(T1, "tensor x\n")
