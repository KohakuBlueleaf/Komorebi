"""The DSL's frontend on a user's path: a class written as today's DSL writes it, made
into an IR module (`komorebi.dsl.ir.module`) and compiled with its origin by the
komorebi-ir wheel; what the IR cannot say yet refused by name, with the class's line.
"""

import importlib.util
import inspect
import shutil
import sys
from pathlib import Path

import pytest

import komorebi as km
import komorebi_ir as ir
from komorebi.device.tensor import Tensor
from komorebi.dsl import program as dsl_program
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.ir import module
from komorebi.dsl.ir.built import Entry, program

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
        for a, b, c in zip(
            km.Stream(x.pages(0, tiles)),
            km.Stream(x.pages(0, tiles)),
            km.Stream(x.pages(0, tiles)),
        ):
            out.push(plus(a, b) + c)


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
    assert any(row[3:] == (run, run) for row in m.map)
    script = (
        "tensor x bf16 32 32 fill tile+1\ntensor y bf16 32 32\n"
        "invoke Scale.run x y 0:u32 1:u32 2.0:f32 on 0 1\nread y\n"
    )
    report = ir.run(program, script)
    assert report.splitlines()[0] == "#0 Scale.run group(0, 1): completed"


def test_a_name_that_is_an_ir_word_is_renamed():
    m = made(Grouped)
    assert ("group_", "group") in m.renames and "group_: u32" in m.text
    hand = HAND.replace("module scale", "module grouped").replace("Scale", "Grouped")
    renamed, plain = ir.hashes(ir.compile(m.text, m.origin)), ir.hashes(hand)
    assert (renamed.semantic, renamed.build) == (plain.semantic, plain.build)


@pytest.mark.parametrize(
    "cls, text, why",
    [
        (Counted, "n = k * 2.0", "an assignment the IR module does not write yet"),
        (Patterned, "pattern=", "a stream's pattern= is not written"),
        (Owned, "form=", r'a reduce by owner \(form="owner"\) is not written'),
        (Led, "pages(self.leader", "self.leader as a number is not written"),
        (Ranked, "self.rank == 1", "an if other than `if self.rank == self.leader:`"),
        (Thrice, "for a, b, c", "names other than one a stream"),
        (IntoWritten, "for a, b", "a loop over what is not a read stream"),
    ],
)
def test_a_line_it_does_not_know_is_refused_with_the_class_line(cls, text, why):
    with pytest.raises(Refused, match=why) as refused:
        made(cls)
    assert f"line {line_of(cls, text)}," in str(refused.value)


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
    assert run.point.bounds == [(t, ("word", 2), ("word", 3)) for t in (0, 1)]
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
    point = run.point._replace(bounds=[(0, ("word", 2), less)])
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
