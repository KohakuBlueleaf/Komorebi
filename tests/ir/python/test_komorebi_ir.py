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
    assert entry.bounds == [(t, ("word", 2), ("word", 3), None) for t in (0, 1)]
    assert entry.patterns == []  # no word of T1 is a pattern
    # a range of a difference is stated too: every range of a tensor is a bound
    less = T1.replace("x.tiles(first, tiles)", "x.tiles(first, tiles - first)").replace(
        "y.tiles(first, tiles)", "y.tiles(first, tiles - first)"
    )
    (entry,) = ir.emit(ir.compile(less), "tensix").entries
    count = ("sub", ("word", 3), ("word", 2))
    assert entry.bounds == [(t, ("word", 2), count, None) for t in (0, 1)]
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


def test_load_fills_the_layouts_and_names_the_formats_as_the_loader_does():
    x, y = ir.LoadTensor(2048, [2, 2048]), ir.LoadTensor(2048, [2, 2048])
    loaded = ir.load(T1, {"y": y, "x": x}, chip="blackhole")
    assert loaded.thread_name == "scale_thread.h"
    assert "#define SCALE_X_PAGES 2048u" in loaded.thread
    assert [(k.processor, k.args) for k in loaded.kernels] == [
        ("reader", [2, 2048, 2, 2048]),
        ("writer", [2, 2048, 2, 2048]),
        ("compute", []),
    ]
    assert loaded.kernels[2].options == ir.emit(T1).kernels[2].options
    assert {b.dtype for b in loaded.buffers} == {"bfloat16"}
    assert [b.pair for b in loaded.buffers] == [b.pair for b in ir.emit(T1).buffers]


def test_load_refuses_by_name():
    x = ir.LoadTensor(2048, [2, 2048])
    for tensors, why in [
        ({"x": x}, "no tensor given for `y`"),
        ({"x": x, "y": x, "z": x}, "a tensor `z`"),
    ]:
        with pytest.raises(ir.IrError, match=why):
            ir.load(T1, tensors)
    with pytest.raises(ir.IrError, match="blackhole or wormhole_b0, not `tensix`"):
        ir.load(T1, {"x": x, "y": x}, chip="tensix")


SMALL = """module small

tile [32, 32]
format bf16, f32

param K: u32 = 28 @bind(build)

func project(
    x: tile<bf16>[4],
    w: tile<bf16>[4, 1],
) -> tile<f32> @tensix.fidelity(HiFi2) {
    return matmul(x, w)
}

program Small {
    handler read(x: tensor<bf16>, w: tensor<bf16>, y: tensor<f32>) @on(group) {
        var acc: tile<f32> = zeros
        let rank = worker.rank
        let xs = stream(x.tiles[0:1, 0:K], block = 4)
        for xb, wb in zip(xs, stream(w.tiles[0:K, rank:rank + 1], block = [4, 1])) {
            acc += project(xb, wb)
        }
        store y.tiles(worker.rank, 1) <- acc
    }
}
"""


def test_a_rectangles_movement_is_named_at_compile_and_made_at_load():
    out = ir.emit(SMALL)
    assert out.at_load == ["movement.cpp"]
    assert "movement.cpp" not in [name for name, _ in out.files]
    assert ("math_fidelity", "HiFi2") in out.kernels[2].options
    assert out.entries[0].bounds[1][3] == (
        ("rank",),
        ("sub", ("add", ("rank",), ("const", 1)), ("rank",)),
    )
    assert ir.emit(T1).at_load == []
    x = ir.LoadTensor(28, [2, 2048], shape=(1, 28))
    w = ir.LoadTensor(112, [2, 2048], shape=(28, 4))
    y = ir.LoadTensor(4, [2, 4096], shape=(1, 4))
    loaded = ir.load(SMALL, {"x": x, "w": w, "y": y})
    ((name, text),) = loaded.files
    assert name == "movement.cpp"
    assert "km::Walk{((0) * 4 + (km_rank)), ((28) + 3) / 4, 16, 4, 4, 1," in text
    with pytest.raises(ir.IrError, match="the tensor `w` is given with no shape"):
        ir.load(SMALL, {"x": x, "w": w._replace(shape=None), "y": y})
    with pytest.raises(ir.IrError, match="columns 0:28 of `x`, which has 27 columns"):
        ir.load(SMALL, {"x": x._replace(shape=(1, 27)), "w": w, "y": y})


def test_load_sizes_a_reduces_arrivals_from_the_machines_islands():
    steps = (ROOT / "tests" / "ir" / "tensix" / "steps_sum.kom").read_text()
    s, t = ir.LoadTensor(768, [2, 4096]), ir.LoadTensor(1, [2, 4096])

    def machine(*sizes):
        links = [(i - 1, i) for i in range(1, len(sizes))]
        return ir.Machine(sum(sizes), 1 << 20, ir.Islands(list(sizes), links), [])

    one = ir.load(steps, {"s": s, "t": t})
    assert one.header_name == "stephandlers.h"
    assert "kArrival0 = {8, 7, 1};" in one.header
    for sizes in [(12,), (6, 6)]:
        assert ir.load(steps, {"s": s, "t": t}, machine=machine(*sizes)) == one
    two = ir.load(steps, {"s": s, "t": t}, machine=machine(120, 120))
    assert "kArrival0 = {8, 8, 1};" in two.header
    assert [b.pages for b in two.buffers if b.index == 8] == [8]
    x = ir.LoadTensor(2048, [2, 2048])
    ir.load(T1, {"x": x, "y": x}, machine=machine(6, 6))


def test_load_refuses_a_machine_whose_islands_are_not_a_graph_of_its_workers():
    x = ir.LoadTensor(2048, [2, 2048])

    def on(sizes, links=None):
        return ir.Machine(4, 1 << 20, ir.Islands(sizes, links), [])

    for machine, why in [
        (on([2, 2], []), "the islands are not connected"),
        (on([2, 1]), "the islands hold 3 workers, not the 4 loaded"),
        (on([4, 0]), "island 1 holds no worker"),
        (on([2, 2], [(0, 2)]), "a connection to island 2, which is not one"),
    ]:
        with pytest.raises(ir.IrError, match="K966") as refused:
            ir.load(T1, {"x": x, "y": x}, machine=machine)
        assert f"the machine given at load is not a machine: {why}" in str(
            refused.value
        )
    # no links: every pair connected
    ir.load(T1, {"x": x, "y": x}, machine=on([2, 2]))
