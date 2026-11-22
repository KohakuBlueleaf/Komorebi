"""The DSL's frontend on a user's path: a class written as today's DSL writes it, made
into an IR module (`komorebi.dsl.ir.built.module`) and compiled with its origin by the
komorebi-ir wheel; what the IR cannot say yet refused by name, with the class's line.
"""

import importlib.util
import inspect
import re
import shutil
import sys
from pathlib import Path

import pytest

import komorebi as km
import komorebi_ir as ir
from komorebi.chain import calls as kc
from komorebi.device.tensor import Tensor
from komorebi.dsl import program as dsl_program
from komorebi.dsl.chain.chain_calls import chain_calls
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.ir import walk, written
from komorebi.dsl.ir.built import Entry, chain, module, program

ROOT = Path(__file__).resolve().parents[3]
HAND = (ROOT / "tests" / "ir" / "tensix" / "t1_class.kom").read_text()


class Pages:
    """A stand-in buffer: its page size, its pages and an address."""

    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 2048, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor(dtype="bfloat16", page=2048):
    return Tensor(Pages(page), (2048, 32), dtype, "tile")


@km.tile_function
def times(a, k):
    return a * k


class Scale(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, first: int, tiles: int, k: float):
        src = self.partition(x.pages(first, tiles))
        out = km.Stream(y[src], write=True, block=1)
        for xb in km.Stream(x[src], block=1):
            out.push(times(xb, k))


class Grouped(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, first: int, group: int, k: float):
        src = self.partition(x.pages(first, group))
        out = km.Stream(y[src], write=True, block=1)
        for xb in km.Stream(x[src], block=1):
            out.push(times(xb, k))


class Counted(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, first: int, tiles: int, k: float):
        src = self.partition(x.pages(first, tiles))
        n = k * 2.0
        out = km.Stream(y[src], write=True, block=1)
        for xb in km.Stream(x[src], block=1):
            out.push(times(xb, k + n))


class Patterned(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, first: int, tiles: int, k: float):
        src = self.partition(x.pages(first, tiles))
        out = km.Stream(y[src], write=True, block=1)
        for xb in km.Stream(x[src], block=1, pattern="AB"):
            out.push(times(xb, k))


class RepeatedBody(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, tiles: int, n: int):
        out = km.Stream(y.pages(0, tiles), write=True)
        for a in km.Stream(x.pages(0, tiles), repeat=n):
            out.push(a)


class RepeatedZip(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, tiles: int, n: int):
        for _a, _b in zip(
            km.Stream(x.pages(0, tiles), repeat=n), km.Stream(y.pages(0, tiles))
        ):
            pass


class TwoPatterns(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, tiles: int, p: km.Pattern, q: km.Pattern):
        out = km.Stream(y.pages(0, tiles), write=True)
        for a, b in zip(
            km.Stream(x.pages(0, tiles), pattern=p),
            km.Stream(x.pages(0, tiles), pattern=q),
        ):
            out.push(plus(a, b))


class Unblocked(km.Program):
    TILES = km.constant(4)

    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, tiles: int):
        out = km.Stream(y.pages(0, tiles), write=True)
        for a in km.Stream(x.pages(0, tiles), block=self.size):
            out.push(times(a, 2.0))


class Floated(km.Program):
    SCALE = km.constant(2.0)

    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, tiles: int):
        out = km.Stream(y.pages(0, tiles), write=True)
        for a in km.Stream(x.pages(0, tiles)):
            out.push(times(a, 2.0))


class Owned(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, tiles: int):
        part = self.partition(x.pages(0, tiles))
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(x[part]):
            total += tile
        whole = km.reduce(total, form="owner")
        if self.rank == self.leader:
            y.pages(0, 1).write(whole)


class Led(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, tiles: int):
        part = self.partition(x.pages(0, tiles))
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(x[part]):
            total += tile
        whole = km.reduce(total)
        if self.rank == self.leader:
            y.pages(self.leader, 1).write(whole)


class Ranked(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, tiles: int):
        part = self.partition(x.pages(0, tiles))
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(x[part]):
            total += tile
        whole = km.reduce(total)
        if self.rank == 1:
            y.pages(0, 1).write(whole)


@km.tile_function
def plus(a, b):
    return a + b


class Thrice(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, tiles: int):
        out = km.Stream(y.pages(0, tiles), write=True)
        for a, b in zip(
            km.Stream(x.pages(0, tiles)),
            km.Stream(x.pages(0, tiles)),
            km.Stream(x.pages(0, tiles)),
        ):
            out.push(plus(a, b))


class IntoWritten(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, tiles: int):
        out = km.Stream(y.pages(0, tiles), write=True)
        for a, b in zip(km.Stream(x.pages(0, tiles)), out):
            out.push(plus(a, b))


@km.tile_function
def erred(a, b):
    return km.exp(a, error=1e-3) + b


def doubled(x):
    return x * 2.0


@km.tile_function
def unhelped(a, b):
    return doubled(a * b)


class Erring(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, tiles: int):
        out = km.Stream(y.pages(0, tiles), write=True)
        for a, b in zip(km.Stream(x.pages(0, tiles)), km.Stream(x.pages(0, tiles))):
            out.push(erred(a, b))


class Unhelping(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, tiles: int):
        out = km.Stream(y.pages(0, tiles), write=True)
        for a, b in zip(km.Stream(x.pages(0, tiles)), km.Stream(x.pages(0, tiles))):
            out.push(unhelped(a, b))


@pytest.mark.parametrize(
    "cls, function, why",
    [
        (Erring, erred, "a call's error= is not written in an IR module yet"),
        (Unhelping, unhelped, "a call of doubled on other than its parameters"),
    ],
)
def test_a_call_in_a_tile_function_it_does_not_write_is_refused_by_name(
    cls, function, why
):
    with pytest.raises(Refused, match=why) as refused:
        made(cls)
    line = inspect.getsourcelines(function.function)[1] + 2
    assert f"line {line}," in str(refused.value)


def line_of(cls, text):
    """The file's line number of the line of `cls` that holds `text`."""
    lines, first = inspect.getsourcelines(cls)
    return first + next(i for i, line in enumerate(lines) if text in line)


def made(cls, **tensors):
    return module(cls, l1=1 << 20, **({"x": tensor(), "y": tensor()} | tensors))


def test_the_class_is_the_hand_written_module_with_its_origin():
    m = made(Scale)
    assert ir.check(m.text, m.origin) == []
    program = ir.compile(m.text, m.origin)
    ours, hand = ir.hashes(program), ir.hashes(HAND)
    assert (ours.semantic, ours.build) == (hand.semantic, hand.build)
    assert m.files == [str(Path(__file__).resolve())]
    run = line_of(Scale, "def run")
    assert any(row[5:] == (run, run) for row in m.map)
    script = (
        "tensor x bf16 32 32 fill tile+1\ntensor y bf16 32 32\n"
        "invoke Scale.run x y 0:u32 1:u32 2.0:f32 on 0 1\nread y\n"
    )
    report = ir.run(program, script)
    assert report.splitlines()[0] == "#0 Scale.run group(0, 1): completed"


def test_a_name_that_is_a_word_of_the_language_is_written_as_given():
    # R50: the parameter `group` is written as the class gives it, between backticks
    # where the language has a `group` (`group.size`), and not renamed
    m = made(Grouped)
    assert m.renames == [] and "`group`: u32" in m.text
    assert "x.tiles(first, `group`)" in m.text and "group.size" in m.text
    hand = HAND.replace("module scale", "module grouped").replace("Scale", "Grouped")
    ours, plain = ir.hashes(ir.compile(m.text, m.origin)), ir.hashes(hand)
    assert (ours.semantic, ours.build) == (plain.semantic, plain.build)


class Counter(km.Program):
    @km.handler
    def run(self, x: km.Tensor, tiles: int, i: int):
        for _ in km.Stream(x.pages(0, tiles), repeat=i):
            pass


def test_a_name_the_walk_makes_is_none_the_class_uses():
    # the repeat's counter, which the class never wrote, is not its parameter `i`
    m = made(Counter)
    assert "for i_ in range(i) {" in m.text and m.renames == []
    assert ir.check(m.text, m.origin) == []


@pytest.mark.parametrize(
    "cls, text, why",
    [
        (Counted, "n = k * 2.0", "an assignment the IR module does not write yet"),
        (Patterned, "pattern=", "a stream's pattern= that is not a km.Pattern param"),
        (TwoPatterns, "for a, b", "the streams of one loop in different patterns"),
        (RepeatedBody, "for a in", "a repeat of a loop with a body"),
        (RepeatedZip, "for _a, _b", "a repeat of a loop over two streams"),
        (
            Unblocked,
            "for a in",
            "a stream's block that is not a number, a module's int",
        ),
        (Owned, "form=", r'a reduce by owner \(form="owner"\) is not written'),
        (Led, "pages(self.leader", "self.leader as a number is not written"),
        (Ranked, "self.rank == 1", "an if other than `if self.rank == self.leader:`"),
        (Thrice, "for a, b in", "names other than one a stream"),
        (IntoWritten, "for a, b", "a loop over what is not a read stream"),
    ],
)
def test_a_line_it_does_not_know_is_refused_with_the_class_line(cls, text, why):
    with pytest.raises(Refused, match=why) as refused:
        made(cls)
    assert f"line {line_of(cls, text)}," in str(refused.value)


def test_a_constant_other_than_an_int_is_refused_by_name():
    with pytest.raises(Refused, match="its constant SCALE is not an int"):
        made(Floated)


def gate_script(folder, monkeypatch, name="gate_class", folder_copy=True):
    """Program 4's script (or tests/ir/tensix/<name>.py), copied into `folder` (today's
    make writes beside it); `folder_copy` False: the script already written there."""
    script = folder / f"{name}.py"
    if folder_copy:
        shutil.copy(ROOT / "tests" / "ir" / "tensix" / f"{name}.py", script)
    spec = importlib.util.spec_from_file_location(name, script)
    gate = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, gate)
    spec.loader.exec_module(gate)
    return gate


def test_a_streams_block_of_a_modules_int_is_its_number(tmp_path, monkeypatch):
    # program 4 with its block a module's int, as a script's `CT = 4` at its top
    source = (ROOT / "tests" / "ir" / "tensix" / "gate_class.py").read_text()
    pairs = [
        (
            "    BLOCK = km.constant(8)  # tiles a block, as the hand program's kBlock",
            "",
        ),
        ("CORES, TILES = 12, 64", "BLOCK = 4  # tiles a block\nCORES, TILES = 12, 64"),
        ("block=self.BLOCK", "block=BLOCK"),
    ]
    for old, new in pairs:
        assert old in source, old
        source = source.replace(old, new)
    (tmp_path / "gate_class.py").write_text(source)
    gate = gate_script(tmp_path, monkeypatch, folder_copy=False)
    given = gate.made_with()
    m = module(gate.Gate, **given)
    assert "param " not in m.text and m.text.count("block = 4") == 3
    today = gate.Gate(log=lambda line: None, **given)
    ours = program(gate.Gate, log=lambda line: None, **given)
    assert ours.sources() == today.sources()


def test_the_groups_size_is_a_word(tmp_path, monkeypatch):
    # program 4 with each core's first page of the group's size: `group.size`
    source = (ROOT / "tests" / "ir" / "tensix" / "gate_class.py").read_text()
    old = "        first = self.rank * tiles\n"
    assert old in source
    new = "        first = (self.rank + self.size - self.size) * tiles\n"
    (tmp_path / "gate_class.py").write_text(source.replace(old, new))
    gate = gate_script(tmp_path, monkeypatch, folder_copy=False)
    given = gate.made_with()
    m = module(gate.Gate, **given)
    assert "let first = ((worker.rank + group.size) - group.size) * tiles" in m.text
    today = gate.Gate(log=lambda line: None, **given)
    ours = program(gate.Gate, log=lambda line: None, **given)
    assert ours.sources() == today.sources()


@km.tile_function
def lane_sum(row):
    return row * 0.0 + km.reduce_sum(row, exact=True).col()


class Lanes(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor):
        results = km.Stream(y.pages(0, 32), write=True, block=16)
        for row in km.Stream(x.pages(0, 32), block=8, row=16):
            results.push(lane_sum(row))


class Scaled(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, k: float):
        results = km.Stream(y.pages(0, 32), write=True, block=16)
        for row in km.Stream(x.pages(0, 32), row=16):
            results.push(times(row, k))


def test_a_streams_row_is_its_block_and_its_passes_a_hint():
    # row=16 in moves of 8: blocks of the row, read in passes of 8 tiles
    m = made(Lanes)
    assert "for row in stream(x.tiles(0, 32), block = 16) @tensix.passes(8) {" in m.text
    assert "    return (row * 0.0) + reduce_sum(row, axis = row).col()\n" in m.text
    assert ir.check(m.text, m.origin) == []
    # the row form's emitter is the other half's: refused by name until it is there
    with pytest.raises(ir.IrError, match="does not emit a call of lane_sum on a row"):
        ir.emit(ir.compile(m.text, m.origin), backend="tensix")
    # a function with no reduction along the row: today's row form has no IR text
    with pytest.raises(Refused, match="times on a row with no reduction along it"):
        made(Scaled)


ROWS_FORM = ROOT / "tests" / "ir" / "tensix" / "tile"


def corpus_functions():
    """The functions of the row form's corpus (tile_rows_form.kom): (name, kinds,
    result, its text)."""
    kom = (ROWS_FORM / "tile_rows_form.kom").read_text()
    found = re.finditer(
        r"^func t_(\w+)\((.*?)\) -> (\S+) \{\n.*?^\}\n", kom, re.S | re.M
    )
    return [
        (m[1], [p.split(": ")[1] for p in m[2].split(", ")], m[3], m[0]) for m in found
    ]


@pytest.mark.parametrize("name, kinds, result, theirs", corpus_functions())
def test_a_row_function_is_written_as_the_row_forms_corpus(
    monkeypatch, name, kinds, result, theirs
):
    # today's function of tile_rows_form.py, written by the frontend: the corpus's
    # function, whose listings the port compares with today's (tensix_tile.rs)
    monkeypatch.syspath_prepend(str(ROWS_FORM))
    corpus = importlib.import_module("tile_rows_form")
    text = walk.Text()
    walk.tile_function(text, km.tile_function(getattr(corpus, name)), kinds, {}, result)
    head = "module rows\n\ntile [32, 32]\nformat f32, bf16\n\n"
    ours = ir.hashes(ir.compile(head + "\n".join(text.lines) + "\n"))
    theirs = theirs.replace(f"func t_{name}(", f"func {name}(")
    assert ours.semantic == ir.hashes(ir.compile(head + theirs)).semantic


@km.tile_function
def lane_sum_rounded(row):
    return row * 0.0 + km.reduce_sum(row).col()


class Rounded(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor):
        results = km.Stream(y.pages(0, 32), write=True, block=16)
        for row in km.Stream(x.pages(0, 32), row=16):
            results.push(lane_sum_rounded(row))


def test_a_reduction_that_is_not_exact_is_refused_with_todays_text():
    with pytest.raises(Refused, match="the row form's reduction is reduce_sum"):
        made(Rounded)


@km.tile_function
def halved_twice(a):
    t = a * 0.5
    t = t * 0.5
    return t


class Twice(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor):
        results = km.Stream(y.pages(0, 4), write=True)
        for xt in km.Stream(x.pages(0, 4)):
            results.push(halved_twice(xt))


def test_a_name_given_a_value_twice_in_a_tile_function_is_refused_by_name():
    with pytest.raises(Refused, match="t given a value again"):
        made(Twice)


class InRow(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, r: int, k: float):
        out = km.Stream(y.tiles[r : r + 4, 0:8], write=True, block=4)
        for xb in km.Stream(x.tiles[r : r + 4, 0:8], block=4):
            out.push(times(xb, k))


class RowAfter(km.Program):
    @km.handler
    def run(self, x: km.Tensor, y: km.Tensor, r: int, k: float):
        out = km.Stream(y.tiles[r + 1 : r + 2, 0:8], write=True, block=8)
        for xb in km.Stream(x.tiles[r + 1 : r + 2, 0:8], block=8):
            out.push(times(xb, k))


@pytest.mark.parametrize(
    "cls, line",
    [
        # 4 tiles a block over 8 columns: within a row, a row walked after another
        (InRow, "for xb in stream(x.tiles[r:r + 4, 0:8], block = 4) {"),
        # `r + 1 : r + 2` is one row of tiles
        (RowAfter, "for xb in stream(x.tiles[r + 1:r + 2, 0:8], block = 8) {"),
    ],
)
def test_a_block_within_a_row_of_a_rectangle_is_its_number(cls, line):
    m = made(cls)
    assert line in m.text
    assert ir.check(m.text, m.origin) == []


@pytest.mark.parametrize("operands", [2, 3])
def test_a_call_batched_only_when_more_than_one_fits_the_dst_half_is_todays(
    tmp_path, monkeypatch, operands
):
    # a call of `operands` tiles takes that many Dst tiles: the Dst half's 4 hold two
    # of 2, one of 3; batched only when more than one fits (`functions.py`, slots > 1)
    names = ["x", "g", "h"][:operands]
    source = (ROOT / "tests" / "ir" / "tensix" / "gate_class.py").read_text()
    returned = {2: "km.silu(x) * g", 3: "x * g + h"}[operands]
    streams = [
        f"            km.Stream({t}.pages(first, tiles), block=self.BLOCK, "
        "pattern=pattern),\n"
        for t in ["a", "b", "a"]
    ]
    pairs = [
        ("def silu_gate(a, b):", f"def silu_gate({', '.join(names)}):"),
        ("    return km.silu(a) * b", f"    return {returned}"),
        ("        for x, g in zip(\n", f"        for {', '.join(names)} in zip(\n"),
        ("".join(streams[:2]), "".join(streams[:operands])),
        ("silu_gate(x, g)", f"silu_gate({', '.join(names)})"),
    ]
    for old, new in pairs:
        assert old in source, old
        source = source.replace(old, new)
    (tmp_path / "gate_class.py").write_text(source)
    gate = gate_script(tmp_path, monkeypatch, folder_copy=False)
    given = gate.made_with()
    today = gate.Gate(log=lambda line: None, **given)
    ours = program(gate.Gate, log=lambda line: None, **given)
    assert ours.sources() == today.sources()
    compute = ours.sources()["compute"].read_text()
    assert ("km_batch0" in compute) == (operands == 2)


def test_a_class_constant_is_a_module_parameter_the_make_sets(tmp_path, monkeypatch):
    assert "param " not in made(Scale).text
    gate = gate_script(tmp_path, monkeypatch)
    given = gate.made_with()
    m = module(gate.Gate, **given)
    assert "param BLOCK: u32 = 8 @bind(build)" in m.text
    assert "block = BLOCK" in m.text
    four = module(gate.Gate, BLOCK=4, **given)
    assert "param BLOCK: u32 = 4 @bind(build)" in four.text
    assert ir.check(four.text, four.origin) == []
    assert four.text.count("@tensix.pattern(pattern)") == 2


def test_what_the_make_states_beyond_the_module_is_refused_by_name():
    with pytest.raises(Refused, match="format bfp8_b"):
        made(Scale, x=tensor("bfp8_b", 1088))
    paged = tensor()
    paged.page_tiles = 2
    with pytest.raises(Refused, match="one tile a page"):
        made(Scale, x=paged)
    with pytest.raises(Refused, match="the make's rect="):
        made(Scale, rect=(0, 0, 1, 1))


def halves(pairs):
    return [(p.mover_a, p.mover_b, p.format, p.page_bytes, p.counts()) for p in pairs]


def test_the_program_built_through_the_ir_is_todays(tmp_path, monkeypatch):
    # The first lesson's script copied out of the tree: both builds write beside it.
    script = tmp_path / "t1_class.py"
    shutil.copy(ROOT / "tests" / "ir" / "tensix" / "t1_class.py", script)
    spec = importlib.util.spec_from_file_location("t1_class", script)
    t1 = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "t1_class", t1)  # inspect finds its source
    spec.loader.exec_module(t1)
    given = {"l1": 1 << 20, "x": t1.tensor(), "y": t1.tensor()}
    today = t1.Scale(log=lambda line: None, **given)
    ours = program(t1.Scale, log=lambda line: None, **given)
    assert ours.header == today.built.folder / "scale.h"
    for field in ("name", "depth", "defines", "compute", "handlers", "compile_args"):
        assert getattr(ours, field) == getattr(today, field), field
    assert ours.sources() == today.sources()
    for kernel in ("reader", "writer", "compute"):
        assert ours.kernel_compile_args(kernel) == today.kernel_compile_args(kernel)
    assert ours.buffers == today.buffers
    assert halves(ours.pairs) == halves(today.pairs)
    assert ours.args is None and today.args is None


def test_the_ir_handler_sends_todays_words_and_checks_its_ranges(tmp_path, monkeypatch):
    script = tmp_path / "t1_class.py"
    shutil.copy(ROOT / "tests" / "ir" / "tensix" / "t1_class.py", script)
    spec = importlib.util.spec_from_file_location("t1_class", script)
    t1 = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "t1_class", t1)
    spec.loader.exec_module(t1)
    x, y = t1.tensor(), t1.tensor()
    given = {"l1": 1 << 20, "x": x, "y": y}
    today = t1.Scale(log=lambda line: None, **given)
    ours = program(t1.Scale, log=lambda line: None, **given)
    run = ours.entry("run")
    assert run.program is ours
    # T1's ranges read no rank: as before, no group needed
    assert run.point.bounds == [(t, ("word", 2), ("word", 3), None) for t in (0, 1)]
    assert run.args(x, y, 0, 64, 2.0) == list(today.run.args(x, y, 0, 64, 2.0))
    assert run.args(x, y, 0, x.pages, 2.0) == [0x1000, 0x1000, 0, 2048, 0x40000000]
    # one tile past the end: today's make keeps no range for pages under a partition
    # and sends it; the IR's handler refuses it before anything is sent
    assert list(today.run.args(x, y, x.pages, 1, 2.0))[2:4] == [2048, 1]
    with pytest.raises(km.ProgramError) as refused:
        run.args(x, y, x.pages, 1, 2.0)
    assert str(refused.value) == (
        "run: x.tiles(first, tiles): items 2048 to 2048 of x at this call, and x "
        "holds 2048 items of 2048 B (2048 page(s) of 1): keep the range inside the "
        "tensor"
    )
    with pytest.raises(km.ProgramError, match="items 1 to 2048 of x"):
        run.args(x, y, 1, x.pages, 2.0)
    # a count of a difference below 0 (a wrapped word on the card) is refused too
    less = ("sub", ("word", 3), ("word", 2))
    point = run.point._replace(bounds=[(0, ("word", 2), less, None)])
    ranged = Entry(point, ours, run.names)
    assert ranged.args(x, y, 4, 4, 2.0)[2:4] == [4, 4]
    with pytest.raises(km.ProgramError, match=r"x.tiles\(first, tiles - first\)"):
        ranged.args(x, y, 5, 3, 2.0)


def test_program_1_built_through_the_ir_is_todays(tmp_path, monkeypatch):
    script = tmp_path / "scalars_line.py"
    shutil.copy(ROOT / "tests" / "ir" / "tensix" / "scalars_line.py", script)
    spec = importlib.util.spec_from_file_location("scalars_line", script)
    p1 = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "scalars_line", p1)
    spec.loader.exec_module(p1)
    given = p1.made_with()
    today = p1.Scalars(log=lambda line: None, **given)
    ours = program(p1.Scalars, log=lambda line: None, **given)
    assert ours.header == today.built.folder / "scalars.h"
    for field in ("name", "depth", "defines", "compute", "handlers", "compile_args"):
        assert getattr(ours, field) == getattr(today, field), field
    assert ours.sources() == today.sources()
    assert ours.buffers == today.buffers
    assert halves(ours.pairs) == halves(today.pairs)
    a, out = given["a"], given["out"]
    line = ours.entry("line")
    # each worker's range reads its rank: checked for every rank of the group; on
    # tensors of exactly 12 x 64 tiles, a group of 12 fits
    words = line.args(a, out, 2.0, group=(0, p1.CORES))
    assert words == list(today.line.args(a, out, 2.0))
    # one worker more: today checks nothing here either and sends it; the IR refuses
    # it at the rank that does not fit
    assert list(today.line.args(a, out, 2.0)) == words
    with pytest.raises(km.ProgramError) as refused:
        line.args(a, out, 2.0, group=(0, p1.CORES + 1))
    assert str(refused.value) == (
        "line: a.tiles(worker.rank * 64, 64): items 768 to 831 of a at rank 12 of 13 "
        "at this call, and a holds 768 items of 2048 B (768 page(s) of 1): keep the "
        "range inside the tensor"
    )
    with pytest.raises(km.ProgramError, match="reads the rank: give"):
        line.args(a, out, 2.0)


def test_program_2_built_through_the_ir_is_todays(tmp_path, monkeypatch):
    script = tmp_path / "steps_sum.py"
    shutil.copy(ROOT / "tests" / "ir" / "tensix" / "steps_sum.py", script)
    spec = importlib.util.spec_from_file_location("steps_sum", script)
    p2 = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "steps_sum", p2)
    spec.loader.exec_module(p2)
    given = p2.made_with()
    today = p2.StepHandlers(log=lambda line: None, **given)
    ours = program(p2.StepHandlers, log=lambda line: None, **given)
    assert ours.header == today.built.folder / "stephandlers.h"
    for field in ("name", "depth", "defines", "compute", "handlers", "compile_args"):
        assert getattr(ours, field) == getattr(today, field), field
    # the arrivals and the reads unpacked to Dst unrounded: today's option, from
    # the buffers' `unpack`
    assert ours.compute["unpack_to_dest_mode"] == dict.fromkeys(
        (0, 1, 8), "UnpackToDestFp32"
    )
    assert ours.sources() == today.sources()
    assert ours.buffers == today.buffers
    assert halves(ours.pairs) == halves(today.pairs)
    s, t = given["s"], given["t"]
    whole = p2.CORES * p2.TILES
    words = ours.entry("sum").args(s, t, whole)
    assert words == list(today.sum.args(s, t, whole))
    # every range is stated: s's tiles 0 .. tiles and t's one tile, both checked;
    # a tile more than s holds: today checks nothing here and sends it, the IR refuses
    assert list(today.sum.args(s, t, whole + 1)) == [*words[:2], whole + 1]
    with pytest.raises(km.ProgramError, match="items 0 to 768 of s"):
        ours.entry("sum").args(s, t, whole + 1)


def test_program_3_built_through_the_ir_is_todays(tmp_path, monkeypatch):
    script = tmp_path / "steps_add.py"
    shutil.copy(ROOT / "tests" / "ir" / "tensix" / "steps_add.py", script)
    spec = importlib.util.spec_from_file_location("steps_add", script)
    p3 = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "steps_add", p3)
    spec.loader.exec_module(p3)
    given = p3.made_with()
    today = p3.StepHandlers(log=lambda line: None, **given)
    ours = program(p3.StepHandlers, log=lambda line: None, **given)
    # the same folder: every file, the two tile files of `plus` included
    assert ours.header == today.built.folder / "stephandlers.h"
    for field in ("name", "depth", "defines", "compute", "handlers", "compile_args"):
        assert getattr(ours, field) == getattr(today, field), field
    assert ours.sources() == today.sources()
    assert ours.buffers == today.buffers
    assert halves(ours.pairs) == halves(today.pairs)
    a, b, out = given["a"], given["b"], given["out"]
    group = (0, p3.CORES)
    words = ours.entry("add").args(a, b, out, p3.WIDTH, group=group)
    assert words == list(today.add.args(a, b, out, p3.WIDTH))
    # every range is stated, each rank's: one worker more than the tensors hold is
    # sent by today unchecked, and refused by the IR at the rank it fails
    more = (0, p3.CORES + 1)
    assert list(today.add.args(a, b, out, p3.WIDTH)) == words
    with pytest.raises(km.ProgramError, match=f"at rank {p3.CORES} of {p3.CORES + 1}"):
        ours.entry("add").args(a, b, out, p3.WIDTH, group=more)


def test_program_4_built_through_the_ir_is_todays(tmp_path, monkeypatch):
    gate = gate_script(tmp_path, monkeypatch)
    given = gate.made_with()
    today = gate.Gate(log=lambda line: None, **given)
    ours = program(gate.Gate, log=lambda line: None, **given)
    assert ours.header == today.built.folder / "gate.h"
    for field in ("name", "depth", "defines", "compute", "handlers", "compile_args"):
        assert getattr(ours, field) == getattr(today, field), field
    assert ours.sources() == today.sources()
    assert ours.buffers == today.buffers
    assert halves(ours.pairs) == halves(today.pairs)
    a, b, out = given["a"], given["b"], given["out"]
    group = (0, gate.CORES)
    # the pattern's word is today's, for a km.Pattern and for its letters
    for pattern in (km.Pattern("AAAB"), km.Pattern("AB"), "B"):
        words = ours.entry("gate").args(a, b, out, gate.TILES, pattern, group=group)
        assert words == list(today.gate.args(a, b, out, gate.TILES, pattern))
    # a value that is not a valid pattern is never sent: compute would end the case
    word = km.Pattern("AAAB").word
    for value in (word, 0, "AXB", None):
        with pytest.raises(km.ProgramError, match="pattern"):
            ours.entry("gate").args(a, b, out, gate.TILES, value, group=group)


def test_program_5s_read_as_a_drain_is_todays(tmp_path, monkeypatch):
    # `km.Stream(...).drain()` is today's loop over the stream with no body
    source = (ROOT / "tests" / "ir" / "tensix" / "movement_class.py").read_text()
    loop = (
        "        for _ in km.Stream(\n"
        "            pages, block=self.BLOCK, pattern=pattern, repeat=repetitions\n"
        "        ):\n"
        "            pass\n"
    )
    drain = (
        "        km.Stream(\n"
        "            pages, block=self.BLOCK, pattern=pattern, repeat=repetitions\n"
        "        ).drain()\n"
    )
    assert loop in source
    (tmp_path / "movement_class.py").write_text(source.replace(loop, drain))
    movement = gate_script(tmp_path, monkeypatch, "movement_class", folder_copy=False)
    given = movement.made_with()
    m = module(movement.Movement, **given)
    assert (
        "        for i in range(repetitions) {\n"
        "            for _ in stream(pages, block = BLOCK) @tensix.pattern(pattern) {\n"
        "            }\n"
    ) in m.text
    today = movement.Movement(log=lambda line: None, **given)
    ours = program(movement.Movement, log=lambda line: None, **given)
    assert ours.sources() == today.sources()
    assert ours.buffers == today.buffers


def test_a_program_that_only_drains_is_todays(tmp_path, monkeypatch):
    # nothing written: compute starts up on its first read twice, as today's does
    (tmp_path / "drained.py").write_text(
        "import komorebi as km\n\n\n"
        "class Touch(km.Program):\n"
        "    @km.handler\n"
        "    def touch(self, a: km.Tensor):\n"
        "        km.Stream(a.pages(0, 1)).drain()\n"
    )
    drained = gate_script(tmp_path, monkeypatch, "drained", folder_copy=False)
    given = {"a": tensor()}
    today = drained.Touch(log=lambda line: None, **given)
    ours = program(drained.Touch, log=lambda line: None, **given)
    assert ours.sources() == today.sources()
    compute = Path(ours.sources()["compute"]).read_text()
    assert "compute_kernel_hw_startup(kRead0.mover_a, kRead0.mover_a);" in compute


def test_program_5_built_through_the_ir_is_todays(tmp_path, monkeypatch):
    movement = gate_script(tmp_path, monkeypatch, "movement_class")
    given = movement.made_with()
    today = movement.Movement(log=lambda line: None, **given)
    ours = program(movement.Movement, log=lambda line: None, **given)
    assert ours.header == today.built.folder / "movement.h"
    for field in ("name", "depth", "defines", "compute", "handlers", "compile_args"):
        assert getattr(ours, field) == getattr(today, field), field
    assert ours.sources() == today.sources()
    assert ours.buffers == today.buffers
    assert halves(ours.pairs) == halves(today.pairs)
    source, output = given["source"], given["output"]
    group, tiles = (0, movement.CORES), movement.TILES
    pattern = km.Pattern("AB")
    # each handler's number and words are today's, the repeat's among them
    for name, values in (
        ("read", (source, tiles, pattern, 3)),
        ("copy", (source, output, tiles, pattern)),
    ):
        assert int(ours.entry(name)) == int(getattr(today, name)), name
        words = ours.entry(name).args(*values, group=group)
        assert words == list(getattr(today, name).args(*values)), name


def test_program_2_built_on_islands_is_todays_on_a_session_of_them(
    tmp_path, monkeypatch
):
    script = tmp_path / "steps_sum.py"
    shutil.copy(ROOT / "tests" / "ir" / "tensix" / "steps_sum.py", script)
    spec = importlib.util.spec_from_file_location("steps_sum", script)
    p2 = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "steps_sum", p2)
    spec.loader.exec_module(p2)
    given = p2.made_with()
    # a session's islands reach today's compile by `islands_of` alone: two of 120
    monkeypatch.setattr(dsl_program, "islands_of", lambda session: (2, 120))
    today = p2.StepHandlers(log=lambda line: None, **given)
    machine = ir.Machine(240, 1 << 20, ir.Islands([120, 120], [(0, 1)]), [])
    ours = program(p2.StepHandlers, machine=machine, log=lambda line: None, **given)
    assert ours.header == today.built.folder / "stephandlers.h"
    assert ours.sources() == today.sources()
    assert ours.buffers == today.buffers
    assert "kArrival0 = {8, 8, 1};" in ours.header.read_text()


def test_program_6_built_through_the_ir_is_todays_and_checks_its_rectangles(
    tmp_path, monkeypatch
):
    script = tmp_path / "small_class.py"
    shutil.copy(ROOT / "tests" / "ir" / "tensix" / "small_class.py", script)
    spec = importlib.util.spec_from_file_location("small_class", script)
    p6 = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "small_class", p6)
    spec.loader.exec_module(p6)
    given = p6.made_with()
    today = p6.Small(log=lambda line: None, **given)
    ours = program(p6.Small, log=lambda line: None, **given)
    # the movement source is the load's (its rectangles walked in rows of the tensors'
    # widths), written into the same folder as today's
    assert ours.header == today.built.folder / "small.h"
    for field in ("name", "depth", "defines", "compute", "handlers", "compile_args"):
        assert getattr(ours, field) == getattr(today, field), field
    assert ours.compute["math_fidelity"] == "HiFi2"
    assert ours.sources() == today.sources()
    assert ours.buffers == today.buffers
    assert halves(ours.pairs) == halves(today.pairs)
    x, w, y = given["x"], given["w"], given["y"]
    read, group = ours.entry("read"), (0, 16)
    # the rectangles' bounds: rows, then the columns, each end of the rank
    rows = ("div", ("rank",), ("const", 4))
    assert read.point.bounds[0] == (
        0,
        rows,
        ("sub", ("add", rows, ("const", 1)), rows),
        (("const", 0), ("sub", ("const", 28), ("const", 0))),
    )
    # an exact rectangle at every rank: today's words
    words = read.args(x, w, y, 3, group=group)
    assert words == list(today.read.args(x, w, y, 3))
    # a row past X's 4 rows of tiles at rank 16 of a group of 20
    with pytest.raises(km.ProgramError) as refused:
        read.args(x, w, y, 0, group=(0, 20))
    assert str(refused.value) == (
        "read: x's rectangle: its rows 4 to 4 at rank 16 of 20 at this call, and x "
        "has 4 rows of tiles: keep the rectangle inside the tensor"
    )
    # a column past W's 4 columns of tiles, at rank 4 of 5 (a rectangle of the rank's
    # column)
    column = (1, ("const", 0), ("const", 28), (("rank",), ("const", 1)))
    ranged = Entry(read.point._replace(bounds=[column]), ours, read.names, read.widths)
    assert ranged.args(x, w, y, 0, group=(0, 4)) == words[:3] + [0]
    with pytest.raises(km.ProgramError, match="its columns 4 to 4 at rank 4 of 5"):
        ranged.args(x, w, y, 0, group=(0, 5))
    # a tensor of another width than the load's: the load wrote the text for 28
    wide = p6.Tensor(p6.Pages(2048, 4 * 29), (128, 928), "bfloat16", "tile")
    with pytest.raises(km.ProgramError) as refused:
        read.args(wide, w, y, 0, group=group)
    assert str(refused.value) == (
        "read: x is 29 tiles wide, and the handler's rectangles of x were built for "
        "28 (the example tensor's): make the program with a tensor of this width"
    )


def swiglu_script(folder, monkeypatch, text=None):
    """Program 7's script (tests/ir/tensix/swiglu_class.py), copied into `folder`, or
    `text` there in its place; loaded as the module `swiglu_class`."""
    script = folder / "swiglu_class.py"
    source = ROOT / "tests" / "ir" / "tensix" / "swiglu_class.py"
    script.write_text(source.read_text() if text is None else text)
    spec = importlib.util.spec_from_file_location("swiglu_class", script)
    p7 = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "swiglu_class", p7)
    spec.loader.exec_module(p7)
    return p7


def test_program_7s_handlers_built_through_the_ir_are_todays(tmp_path, monkeypatch):
    p7 = swiglu_script(tmp_path, monkeypatch)
    given = p7.made_with()
    today = p7.SwiGLU(log=lambda line: None, **given)
    ours = program(p7.SwiGLU, log=lambda line: None, **given)
    # self.h, self.g and self.out are vars of a block of one row (R48), kept in
    # today's field buffers; up stores gate(h, g), a function on one tile of them
    m = module(p7.SwiGLU, l1=1 << 20, **given)
    for name in ("h", "g", "out"):
        assert f"var {name}: tile<f32>[WIDTH] = zeros" in m.text
        assert (name, f"self.{name}") in m.renames
    assert "func gate(h: tile<f32>, g: tile<f32>) -> tile<f32>" in m.text
    assert "for xb, w1b, w3b in zip(" in m.text
    assert "<- gate(h, g)" in m.text
    assert "block = [CHUNK, WIDTH]" in m.text
    # the chain: its calls in order, down on the first four of up's twelve cores
    assert "        call up(x, w1, w3, hidden)\n" in m.text
    assert "        call down(hidden, w2, y) @on(group[0:4])\n" in m.text
    assert m.chains == {"forward": (0, 12)}
    # a plain submit sends today's words: the handler's own, no mask word after them
    for name, group in (("up", (0, 12)), ("down", (0, 4))):
        names = ours.entry(name).names
        values = [given[n] for n in names]
        words = ours.entry(name).args(*values, group=group)
        assert words == list(getattr(today, name).args(*values))
        assert len(words) == len(names)
    assert ours.header == today.built.folder / "swiglu.h"
    for field in ("name", "depth", "defines", "compute", "handlers", "compile_args"):
        assert getattr(ours, field) == getattr(today, field), field
    assert ours.sources() == today.sources()
    assert ours.buffers == today.buffers
    assert halves(ours.pairs) == halves(today.pairs)


@pytest.mark.parametrize(
    "block, written",
    [
        ("WIDTH * CHUNK", "[CHUNK, WIDTH]"),
        ("2 * WIDTH", "[2, WIDTH]"),
        ("km.constant(16)", "[2, WIDTH]"),
    ],
)
def test_program_7s_block_of_whole_rows_is_its_rows_by_its_columns(
    tmp_path, monkeypatch, block, written
):
    # W_BLOCK = 16 tiles over WIDTH = 8 columns: two whole rows, however declared
    p7 = swiglu_named(
        tmp_path, monkeypatch, ("W_BLOCK = CHUNK * WIDTH", f"W_BLOCK = {block}")
    )
    given = p7.made_with()
    m = module(p7.SwiGLU, l1=1 << 20, **given)
    assert m.text.count(f"block = {written}") == 3
    assert_todays(p7, given)


@pytest.mark.parametrize(
    "old, new, why",
    [
        # a block over the rectangle neither of whole rows (A * C) nor within a row
        (
            "W_BLOCK = CHUNK * WIDTH",
            "W_BLOCK = CHUNK * WIDTH + CHUNK",
            "a stream's block over a rectangle of several rows and columns that is "
            "neither",
        ),
        # self.out filled by one handler and read by down: today keeps it between
        # commands (`kept_field`), a state member
        (
            "        y.tiles[0:1, first : first + self.WIDTH].write(self.out)\n",
            "        y.tiles[0:1, first : first + self.WIDTH].write(self.out)\n\n"
            "    @km.handler\n"
            "    def keep(self, x: km.Tensor):\n"
            "        self.out.fill(km.Stream(x.tiles[0:1, 0 : self.WIDTH]))\n",
            "self.out is filled by keep and read by down: a field kept between "
            "commands is a state member",
        ),
    ],
)
def test_program_7s_field_or_block_it_does_not_place_is_refused_by_name(
    tmp_path, monkeypatch, old, new, why
):
    source = (ROOT / "tests" / "ir" / "tensix" / "swiglu_class.py").read_text()
    assert source.count(old) == 1
    p7 = swiglu_script(tmp_path, monkeypatch, source.replace(old, new))
    with pytest.raises(Refused, match=why):
        module(p7.SwiGLU, l1=1 << 20, **p7.made_with())


@pytest.mark.parametrize(
    "cores, calls, span",
    [
        # a span that does not start at core 0: its ranks count from its first core
        (
            ((4, 12), (6, 4)),
            ("", " @on(group[2:6])"),
            (4, 12),
        ),
        # no call over the whole span: each names its part
        (
            ((2, 12), (0, 4)),
            (" @on(group[2:14])", " @on(group[0:4])"),
            (0, 14),
        ),
    ],
)
def test_program_7s_chain_is_on_the_span_of_its_calls_cores(
    tmp_path, monkeypatch, cores, calls, span
):
    source = (ROOT / "tests" / "ir" / "tensix" / "swiglu_class.py").read_text()
    for given, (first, count) in zip(("(0, 12)", "(0, 4)"), cores, strict=True):
        assert source.count(f"cores={given}") == 1
        source = source.replace(f"cores={given}", f"cores=({first}, {count})")
    p7 = swiglu_script(tmp_path, monkeypatch, source)
    m = module(p7.SwiGLU, l1=1 << 20, **p7.made_with())
    up, down = calls
    assert f"        call up(x, w1, w3, hidden){up}\n" in m.text
    assert f"        call down(hidden, w2, y){down}\n" in m.text
    assert m.chains == {"forward": span}


# program 7's arguments in the order of its chain's parameters
FORWARD = ("x", "w1", "w3", "w2", "hidden", "y")


def cored(source, cores):
    """Program 7's source with up and down on `cores`, each (first core, count)."""
    for given, (first, count) in zip(("(0, 12)", "(0, 4)"), cores, strict=True):
        assert source.count(f"cores={given}") == 1
        source = source.replace(f"cores={given}", f"cores=({first}, {count})")
    return source


@pytest.mark.parametrize(
    "cores, span",
    [
        (((0, 12), (0, 4)), (0, 12)),
        # a span that starts at core 4: today's absolute cores in the records
        (((4, 12), (6, 4)), (4, 12)),
        # up writes the hidden's first 48 tiles, down reads all 96: built, as today
        # (the tiles the chain does not write are read as they were before it, R2)
        (((0, 6), (0, 4)), (0, 6)),
    ],
)
def test_program_7s_chain_records_are_todays_word_for_word(
    tmp_path, monkeypatch, cores, span
):
    source = (ROOT / "tests" / "ir" / "tensix" / "swiglu_class.py").read_text()
    p7 = swiglu_script(tmp_path, monkeypatch, cored(source, cores))
    given = p7.made_with()
    values = [given[name] for name in FORWARD]
    today = p7.SwiGLU(log=lambda line: None, **given)
    theirs = chain_calls(None, today.forward(*values), None)
    ours = program(p7.SwiGLU, log=lambda line: None, **given).chain("forward")
    assert ours.span == span
    calls = ours.calls(*values)
    # today's proof (check_calls, prove, check_direct), called as run.chain calls it
    assert kc.build(calls) == kc.build(theirs) == ({}, None)
    for mine, its in zip(calls, theirs, strict=True):
        for field in ("args", "group", "size", "width", "output", "input", "ranges"):
            assert getattr(mine, field) == getattr(its, field), field
        assert int(mine.handler) == int(its.handler)
    handlers = [int(call.handler) for call in theirs]
    for address in (0x20000, 0x7F000):
        mine, its = (kc.records(c, address, handlers) for c in (calls, theirs))
        assert mine.tolist() == its.tolist()


def test_program_7s_chain_refused_by_todays_records_is_refused_with_todays_text(
    tmp_path, monkeypatch
):
    # a hidden of 2048 tiles: down reads a range wider than a record's field (2047)
    source = (ROOT / "tests" / "ir" / "tensix" / "swiglu_class.py").read_text()
    p7 = swiglu_script(tmp_path, monkeypatch, re.sub(r"\b96\b", "2048", source))
    given = p7.made_with()
    ours = program(p7.SwiGLU, log=lambda line: None, **given).chain("forward")
    calls = ours.calls(*(given[name] for name in FORWARD))
    assert kc.build(calls) == ({}, None)
    with pytest.raises(kc.ChainRefused) as refused:
        kc.records(calls, 0x20000, [1, 2])
    assert str(refused.value) == (
        "Range(base=0, step=0, width=2048, root_only=False): base 0 to 4095, step 0 to "
        "255, width 0 to 2047"
    )


def test_program_7s_chain_through_run_chain_runs_its_own_entry(tmp_path, monkeypatch):
    p7 = swiglu_script(tmp_path, monkeypatch)
    given = p7.made_with()
    ours = program(p7.SwiGLU, log=lambda line: None, **given)
    forward = ours.chain("forward")

    class Run:  # a stand-in for a session's run: what built.chain asks of it
        def chain(self, calls, mode=0, flags=0, records_at=None):
            self.calls, self.given = calls, (mode, flags, records_at)
            return kc.Chain(int(calls[0].handler), (0, 12), mode, flags, None,
                            None, {}, calls)  # fmt: skip

        def program_field(self, program):
            assert program is ours
            return 0

        def field_handler(self, number, field):
            return number

    run = Run()
    made = chain(run, forward, *(given[name] for name in FORWARD), records_at=0x9000)
    assert run.calls == forward.calls(*(given[name] for name in FORWARD))
    assert run.given == (0, 0, 0x9000)
    # the command runs kHandlerKmChain, the chain's entry, not its first call's handler
    assert made.handler == forward.number == 3


@pytest.mark.parametrize(
    "old, new, why",
    [
        # cores the make does not know: the chain's group is not known
        (", cores=(0, 4))", ")", "a call without cores=\\(first, count\\)"),
        (
            "self.down(hidden, w2, y",
            "self.down(km.past(hidden), w2, y",
            "an argument other than a parameter of the chain",
        ),
        (
            "        self.down(hidden, w2, y, cores=(0, 4))\n",
            "        for _ in range(2):\n"
            "            self.down(hidden, w2, y, cores=(0, 4))\n",
            "a chain's line other than a call of a handler",
        ),
        (
            "cores=(0, 4))",
            "cores=(0, 4), group=2)",
            "group= is not written in an IR module yet",
        ),
    ],
)
def test_program_7s_chain_it_does_not_write_is_refused_with_its_line(
    tmp_path, monkeypatch, old, new, why
):
    source = (ROOT / "tests" / "ir" / "tensix" / "swiglu_class.py").read_text()
    assert source.count(old) == 1
    p7 = swiglu_script(tmp_path, monkeypatch, source.replace(old, new))
    with pytest.raises(Refused, match=why) as refused:
        module(p7.SwiGLU, l1=1 << 20, **p7.made_with())
    assert "SwiGLU.forward, line " in str(refused.value)


def swiglu_named(folder, monkeypatch, *pairs):
    """Program 7's script with names replaced, each (old, new) pair once or more."""
    source = (ROOT / "tests" / "ir" / "tensix" / "swiglu_class.py").read_text()
    for old, new in pairs:
        assert old in source, old
        source = source.replace(old, new)
    return swiglu_script(folder, monkeypatch, source)


def assert_todays(p7, given):
    """Program 7's class made through the IR and today: the same files and words."""
    today = p7.SwiGLU(log=lambda line: None, **given)
    ours = program(p7.SwiGLU, log=lambda line: None, **given)
    assert ours.header == today.built.folder / "swiglu.h"
    for field in ("name", "depth", "defines", "compute", "handlers", "compile_args"):
        assert getattr(ours, field) == getattr(today, field), field
    # a build folder is named by its files' text: the same folder, the same bytes
    assert ours.sources() == today.sources()
    assert ours.buffers == today.buffers
    assert halves(ours.pairs) == halves(today.pairs)
    return ours


# program 7 with its field `out` named `store` (a word of the text) and its handler
# `down` named `zip` (a call of the language, where no handler stands)
STORE_ZIP = [
    ("    out: km.L1", "    store: km.L1"),
    ("self.out", "self.store"),
    ("def down(", "def zip("),
    ("self.down(", "self.zip("),
]


def test_program_7s_names_that_are_words_are_written_as_given(tmp_path, monkeypatch):
    p7 = swiglu_named(tmp_path, monkeypatch, *STORE_ZIP)
    given = p7.made_with()
    m = module(p7.SwiGLU, l1=1 << 20, **given)
    assert "var `store`: tile<f32>[WIDTH] = zeros" in m.text
    assert "    handler zip(" in m.text and "        call zip(hidden, w2, y)" in m.text
    assert ("store", "self.store") in m.renames
    ours = assert_todays(p7, given)
    assert ours.entry("zip").point.name == "SwiGLU.zip"


def test_program_7s_refusal_names_its_handler_zip(tmp_path, monkeypatch):
    first = "        first = self.rank * self.WIDTH  # this core's columns of y\n"
    p7 = swiglu_named(
        tmp_path, monkeypatch, *STORE_ZIP, (first, first + "        n = 2.0 * 3.0\n")
    )
    with pytest.raises(Refused, match="an assignment") as refused:
        module(p7.SwiGLU, l1=1 << 20, **p7.made_with())
    assert "SwiGLU.zip, line " in str(refused.value)


def test_program_7s_handlers_docstrings_write_nothing(tmp_path, monkeypatch):
    up = (
        "    def up(self, x: km.Tensor, w1: km.Tensor, w3: km.Tensor, "
        "hidden: km.Tensor):\n"
    )
    down = "    def down(self, hidden: km.Tensor, w2: km.Tensor, y: km.Tensor):\n"
    p7 = swiglu_named(
        tmp_path,
        monkeypatch,
        (up, up + '        """The hidden\'s columns of this core."""\n'),
        (down, down + '        """y\'s columns of this core:\n\n        a sum."""\n'),
    )
    given = p7.made_with()
    m = module(p7.SwiGLU, l1=1 << 20, **given)
    assert "columns of this core" not in m.text
    assert_todays(p7, given)


def test_program_7s_tile_function_named_square_is_written_as_given(
    tmp_path, monkeypatch
):
    # `gate` named `square`, a call of the language where a function stands, and its
    # body calling the language's square: the module's is `square` between backticks
    p7 = swiglu_named(
        tmp_path,
        monkeypatch,
        ("def gate(h, g):", "def square(h, g):"),
        ("return km.silu(h) * g", "return km.square(km.silu(h)) * g"),
        ("write(gate(self.h", "write(square(self.h"),
    )
    given = p7.made_with()
    m = module(p7.SwiGLU, l1=1 << 20, **given)
    assert "func `square`(h: tile<f32>, g: tile<f32>) -> tile<f32> {" in m.text
    assert "    return square(silu(h)) * g" in m.text
    assert "<- `square`(h, g)" in m.text
    assert_todays(p7, given)


# Each program the frontend takes end to end (T1, programs 1 to 7) and the class whose
# tile function calls keep their own lines: (script in tests/ir/tensix, class)
PROGRAMS = [
    ("t1_class", "Scale"),
    ("scalars_line", "Scalars"),
    ("steps_sum", "StepHandlers"),
    ("steps_add", "StepHandlers"),
    ("gate_class", "Gate"),
    ("movement_class", "Movement"),
    ("small_class", "Small"),
    ("swiglu_class", "SwiGLU"),
    ("lines_class", "Lines"),
]


def kept(text):
    """A text's characters the writer's layout keeps: all but whitespace, commas and
    backticks."""
    return "".join(c for c in text if not c.isspace() and c not in ",`")


def held(text, row):
    """The text a map range holds, from its first (line, column) to its last."""
    lines = text.splitlines()
    first, first_col, last, last_col = row[:4]
    if first == last:
        return lines[first - 1][first_col - 1 : last_col]
    inner = lines[first : last - 1]
    return "\n".join(
        [lines[first - 1][first_col - 1 :], *inner, lines[last - 1][:last_col]]
    )


@pytest.mark.parametrize("script, name", PROGRAMS)
def test_the_text_is_in_the_writers_layout_its_origin_on_the_same_text(
    monkeypatch, script, name
):
    # module() lays the walk's text out through the wheel: its own layout, each range
    # holding the same text as before (whitespace and commas aside), same source place
    folder = ROOT / "tests" / "ir" / "tensix"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(script, folder / f"{script}.py")
    loaded = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, script, loaded)
    spec.loader.exec_module(loaded)
    given = (
        loaded.made_with()
        if hasattr(loaded, "made_with")
        else {"l1": 1 << 20, "x": loaded.tensor(), "y": loaded.tensor()}
    )
    cls = getattr(loaded, name)
    raw, laid = written(cls, **given), module(cls, **given)
    assert ir.format(laid.text) == laid.text
    assert len(raw.map) == len(laid.map)
    for before, after in zip(raw.map, laid.map):
        assert before[4:] == after[4:]
        was, now = held(raw.text, before), held(laid.text, after)
        assert kept(was) == kept(now), (before, was, after, now)


def test_the_ir_examples_are_the_frontends_text_today():
    # examples/ir/ cannot go stale: `ir_examples.py --write` refreshes them
    spec = importlib.util.spec_from_file_location(
        "ir_examples", Path(__file__).with_name("ir_examples.py")
    )
    examples = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(examples)
    assert examples.stale() == []
    names = sorted(p.name for p in examples.FOLDER.glob("*.kom"))
    assert names == sorted(file for file, *_ in examples.EXAMPLES)
