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
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.ir import module
from komorebi.dsl.ir.built import program

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
        n = tiles + 1
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
        (Counted, "n = tiles + 1", "an assignment the IR module does not write yet"),
        (Patterned, "pattern=", "a stream's pattern= is not written"),
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
