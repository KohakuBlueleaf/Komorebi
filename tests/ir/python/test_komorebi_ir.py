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


# T1 as a frontend prints it for a class whose `def run` is on line 18: the handler on
# its def, each statement on its line, the loop on 21 to 22 and its push on 22.
ORIGIN = ir.Origin(
    files=["/work/t1_class.py"],
    lines=[(0, n, f"line {n}") for n in range(18, 23)],
    map=[
        (11, 24, 0, 18, 18),
        (18, 18, 0, 19, 19),
        (19, 20, 0, 20, 20),
        (21, 23, 0, 21, 22),
        (22, 22, 0, 22, 22),
    ],
)


def test_an_origin_changes_only_the_full_hash():
    program = ir.compile(T1, ORIGIN)
    assert program != ir.compile(T1)
    plain, placed = ir.hashes(T1), ir.hashes(program)
    assert (plain.semantic, plain.build) == (placed.semantic, placed.build)
    assert plain.full != placed.full
    assert ir.check(T1, ORIGIN) == []
    renamed = ORIGIN._replace(renames=[("tiles", "group"), ("src", "stream")])
    again = ir.hashes(ir.compile(T1, renamed))
    assert again[:2] == placed[:2] and again.full != placed.full


def test_a_diagnostic_is_placed_in_the_frontends_source():
    broken = T1.replace("out.push(times(xb, k))", "out.push(times(xb, q))")
    found = ir.check(broken, ORIGIN)
    assert found and all(d.line == 22 for d in found)
    assert all(d.place == ir.Place("/work/t1_class.py", 22, 22) for d in found)
    assert all(d.place is None for d in ir.check(broken))
    with pytest.raises(ir.IrError) as refused:
        ir.compile(broken, ORIGIN)
    assert ir.diagnostics(refused.value)[0].place.first == 22


def test_an_origin_is_refused_by_what_is_wrong():
    crossing = ORIGIN._replace(map=[*ORIGIN.map, (20, 21, 0, 20, 20)])
    with pytest.raises(ir.IrError, match="the origin is refused: .*crossing"):
        ir.compile(T1, crossing)
    unknown = ORIGIN._replace(renames=[("nothing", "x")])
    with pytest.raises(ir.IrError, match="`nothing` names no value"):
        ir.check(T1, unknown)


def test_emit_gives_the_tensix_output_as_data():
    out = ir.emit(T1)
    assert out == ir.emit(ir.compile(T1), "tensix")
    names = [name for name, _ in out.files]
    assert names == ["compute.cpp", "movement.cpp", "scale.h"] + [
        "tile_times.hpp",
        "tile_times.txt",
    ]
    assert all(isinstance(data, bytes) for _, data in out.files)
    (entry,) = out.entries
    assert (entry.name, entry.kind, entry.number) == ("Scale.run", "handler", 1)
    assert entry.words == [("address", 0), ("address", 1)] + [
        ("word", i) for i in (2, 3, 4)
    ]
    assert entry.bounds == [(t, ("word", 2), ("word", 3)) for t in (0, 1)]
    layouts = [("layout", 0), ("layout", 1)]
    assert [(k.source, k.processor, k.args) for k in out.kernels] == [
        ("movement.cpp", "reader", layouts),
        ("movement.cpp", "writer", layouts),
        ("compute.cpp", "compute", []),
    ]
    assert {b.format for b in out.buffers} == {"bf16"} and len(out.buffers) == 4
    pairs = [(16, 17), (17, 16), (0, 1), (1, 0)]
    assert [(b.index, b.pair) for b in out.buffers] == pairs
    assert out.kernels[2].options == [
        ("fp32_dest_acc_en", True),
        ("math_fidelity", "HiFi4"),
    ]
    assert out.steps == []


def test_emit_prints_the_origin_a_frontend_gives():
    plain = dict(ir.emit(T1).files)["scale.h"].decode()
    placed = dict(ir.emit(ir.compile(T1, ORIGIN)).files)["scale.h"].decode()
    assert plain.startswith("// Scale, made by komorebi-ir:")
    assert placed.startswith("// Scale, made by komorebi.dsl from t1_class.py:")
    assert ir.emit(ir.compile(T1, ORIGIN)).key != ir.emit(T1).key


def test_emit_gives_the_cpu_output_as_steps():
    out = ir.emit(T1, "cpu")
    assert "module.cpp" in [name for name, _ in out.files]
    assert [step.program for step in out.steps] == ["c++", "c++"]
    assert out.kernels == [] and out.buffers == []
    assert [(e.name, e.number) for e in out.entries] == [("Scale.run", 0)]
    assert out.key != ir.emit(T1).key


def test_emit_refuses_by_name():
    with pytest.raises(ir.IrError, match='no backend is named "gpu"'):
        ir.emit(T1, "gpu")
    with pytest.raises(ir.IrError) as refused:
        ir.emit(T1, "tensix", target="wormhole")
    assert [d.code for d in ir.diagnostics(refused.value)] == ["K964"]
    with pytest.raises(ir.IrError, match="K964: .*grid"):
        ir.emit(T6)
