"""A tile function's result as another's operand (no card): nested, named, in a tile
expression and with a field given to an inner call, each one composed unit program a
tile; a plain call's text unchanged; the refusals (pushed and read, 5 operand tiles,
a field read otherwise, a call outside a tile line)."""

import pytest
from class_lines_fixture import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.errors import ProgramError


@km.tile_function
def larger(x, y):
    return km.where(x > y, x, y * 0.5)


@km.tile_function
def scaled(t, z):
    return t * z + 1.0


@km.tile_function
def outer(x, y):
    return scaled(larger(x, y), 2.0)  # tile functions called in a tile function


class Composed(km.Program):
    @km.handler
    def nested(self, a: km.Tensor, b: km.Tensor, c: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, y, z in zip(
            km.Stream(a.pages(0, 8), block=8),
            km.Stream(b.pages(0, 8), block=8),
            km.Stream(c.pages(0, 8), block=8),
        ):
            results.push(scaled(larger(x, y), z))

    @km.handler
    def named(self, a: km.Tensor, b: km.Tensor, c: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, y, z in zip(
            km.Stream(a.pages(0, 8), block=8),
            km.Stream(b.pages(0, 8), block=8),
            km.Stream(c.pages(0, 8), block=8),
        ):
            t = larger(x, y)
            results.push(scaled(t, z))

    @km.handler
    def expression(self, a: km.Tensor, b: km.Tensor, c: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, y, z in zip(
            km.Stream(a.pages(0, 8), block=8),
            km.Stream(b.pages(0, 8), block=8),
            km.Stream(c.pages(0, 8), block=8),
        ):
            results.push(km.silu(larger(x, y)) * z)

    @km.handler
    def plain(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(0, 8), block=8), km.Stream(b.pages(0, 8), block=8)
        ):
            t = larger(x, y)
            results.push(t)

    @km.handler
    def inside(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(0, 8), block=8), km.Stream(b.pages(0, 8), block=8)
        ):
            results.push(outer(x, y))


class Twice(km.Program):
    @km.handler
    def run(self, a: km.Tensor, b: km.Tensor, c: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        extra = km.Stream(c.pages(0, 8), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(0, 8), block=8), km.Stream(b.pages(0, 8), block=8)
        ):
            t = larger(x, y)
            extra.push(t)
            results.push(scaled(t, x))


class Five(km.Program):
    @km.handler
    def run(
        self,
        a: km.Tensor,
        b: km.Tensor,
        c: km.Tensor,
        d: km.Tensor,
        e: km.Tensor,
        out: km.Tensor,
    ):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for v, w, x, y, z in zip(
            km.Stream(a.pages(0, 8), block=8),
            km.Stream(b.pages(0, 8), block=8),
            km.Stream(c.pages(0, 8), block=8),
            km.Stream(d.pages(0, 8), block=8),
            km.Stream(e.pages(0, 8), block=8),
        ):
            results.push(scaled(larger(v, w), x) + larger(y, z))


@km.tile_function
def biased(t, b):
    return t + b.row()


@km.tile_function
def by_column(t, b):
    return t * b.col()


@km.tile_function
def whole(t, b):
    return t + b


class Fields(km.Program):
    bias: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def rows(self, a: km.Tensor, b: km.Tensor, c: km.Tensor, out: km.Tensor):
        self.bias.fill(km.Stream(c.pages(0, 1)))
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(0, 8), block=8), km.Stream(b.pages(0, 8), block=8)
        ):
            results.push(biased(larger(x, y), self.bias))

    @km.handler
    def columns(self, a: km.Tensor, b: km.Tensor, c: km.Tensor, out: km.Tensor):
        self.bias.fill(km.Stream(c.pages(0, 1)))
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(0, 8), block=8), km.Stream(b.pages(0, 8), block=8)
        ):
            t = larger(x, y)
            results.push(by_column(t, self.bias))


class FieldAdded(km.Program):
    bias: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def run(self, a: km.Tensor, b: km.Tensor, c: km.Tensor, out: km.Tensor):
        self.bias.fill(km.Stream(c.pages(0, 1)))
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(0, 8), block=8), km.Stream(b.pages(0, 8), block=8)
        ):
            results.push(larger(x, y) + self.bias)


class FieldWhole(km.Program):
    bias: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def run(self, a: km.Tensor, b: km.Tensor, c: km.Tensor, out: km.Tensor):
        self.bias.fill(km.Stream(c.pages(0, 1)))
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(0, 8), block=8), km.Stream(b.pages(0, 8), block=8)
        ):
            results.push(whole(larger(x, y), self.bias))


def built(cls):
    tensors = {
        n: Tensor(Pages(4096), (256, 32), "float32", "tile")
        for n in ("a", "b", "c", "d", "e", "out")
    }
    if cls is not Five:
        tensors = {n: t for n, t in tensors.items() if n not in "de"}
    return compile_class(cls, {}, tensors)


def words(unit):
    """The instructions a tile, as the unit's dump writes them."""
    text = unit.dump()
    text = text[text.index("  math per tile:") : text.index("  pack per tile:")]
    return [w.split()[0] for w in text.splitlines()[1:] if w.strip()]


def test_each_form_is_one_composed_unit_a_tile():
    compiled = built(Composed)
    units = {cxx: unit for cxx, (_, unit) in compiled.units.items()}
    composed = [cxx for cxx in units if cxx.startswith("tile_line_")]
    assert len(composed) == 3  # nested, named, expression
    for cxx in composed:
        assert "SFPGT" in words(units[cxx])  # larger's where, inside
    nested, named = (units[cxx] for cxx in composed[:2])
    assert words(nested) == words(named)
    assert "larger" in units  # the plain call: the function alone, as before
    assert "SFPGT" in words(units["outer"])  # traced in place in outer's body


def test_a_result_pushed_and_read_by_a_later_line_is_refused():
    with pytest.raises(Refused, match="t is pushed and read by a later tile line"):
        built(Twice)


def test_the_composed_function_takes_at_most_4_operand_tiles():
    with pytest.raises(Refused, match="at most 4"):
        built(Five)


def test_a_tile_function_called_outside_a_tile_line_names_where():
    with pytest.raises(ProgramError, match="a handler line calls it on its loop"):
        larger(1.0, 2.0)


def test_a_field_given_to_an_inner_call_is_the_composed_function_s():
    compiled = built(Fields)
    units = {cxx: unit for cxx, (_, unit) in compiled.units.items()}
    assert len(units) == 2  # one composed unit a handler


@pytest.mark.parametrize(
    "cls,why",
    [
        (FieldAdded, "self.bias in a tile expression: a field is read by"),
        (FieldWhole, r"self.bias in a call on a stream's tiles is a row"),
    ],
)
def test_a_field_read_otherwise_is_refused(cls, why):
    with pytest.raises(Refused, match=why):
        built(cls)
