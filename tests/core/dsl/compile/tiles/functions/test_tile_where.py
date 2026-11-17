"""km.where with the six comparisons and km.floor in a tile function (no card): the
lane tests each lowers to, floor's words, a pushed expression, and the refusals of a
comparison used as a value."""

import pytest

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.tile.emit import emit
from komorebi.errors import ProgramError


class Pages:
    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensors(dtype, *names):
    page = formats.tile_bytes(dtype)
    return {n: Tensor(Pages(page), (2048, 32), dtype, "tile") for n in names}


TESTS = {
    "gt": lambda x, y: km.where(x > y, x, y),
    "lt": lambda x, y: km.where(x < y, x, y),
    "ge": lambda x, y: km.where(x >= y, x, y),
    "le": lambda x, y: km.where(x <= y, x, y),
    "eq": lambda x, y: km.where(x == y, 1.0, 0.0),
    "ne": lambda x, y: km.where(x != y, 1.0, 0.0),
    "zero": lambda x, y: km.where(x == 0.0, y, x),
    "number": lambda x, y: km.where(0.5 < x, y, -2.0),  # noqa: SIM300 (reflected)
    "floor": lambda x, y: km.floor(x),
}
FUNCTION = TESTS["gt"]


class Picked(km.Program):
    @km.tile_function
    def pick(x, y):
        return FUNCTION(x, y)

    @km.handler
    def run(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(0, 8), block=8), km.Stream(b.pages(0, 8), block=8)
        ):
            results.push(self.pick(x, y))


def unit_of(test, dtype="bfloat16", cls=Picked):
    global FUNCTION
    FUNCTION = TESTS[test]
    try:
        compiled = compile_class(cls, {}, tensors(dtype, "a", "b", "out"))
    finally:
        FUNCTION = TESTS["gt"]
    ((_, unit),) = compiled.units.values()
    return unit


def words(unit, op):
    """The words of instruction `op` a tile, as the unit's dump writes them."""
    text = unit.dump()
    text = text[text.index("  math per tile:") : text.index("  pack per tile:")]
    return [w for w in text.splitlines() if w.strip().startswith(f"{op} ")]


@pytest.mark.parametrize(
    "test, tests, setcc",
    [
        ("gt", 1, 0),
        ("lt", 1, 0),
        ("ge", 1, 0),
        ("le", 1, 0),
        ("eq", 2, 0),
        ("ne", 2, 0),
        ("zero", 0, 1),
        ("number", 1, 0),
    ],
)
def test_each_comparison_is_its_lane_tests(test, tests, setcc):
    unit = unit_of(test)
    assert len(words(unit, "SFPGT")) == tests
    assert len(words(unit, "SFPSETCC")) == setcc
    assert len(words(unit, "SFPENCC")) == max(tests, setcc)


@pytest.mark.parametrize(
    "dtype",
    ["float32", "bfloat16", "bfp8_b", "bfp4_b", "bfp2_b"]
    + ["float16", "bfp8", "bfp4", "bfp2"],
)
def test_where_and_floor_build_on_each_tile_format(dtype):
    assert words(unit_of("ge", dtype), "SFPGT")
    floor = unit_of("floor", dtype)
    assert len(words(floor, "SFPGT")) == 3  # x < 0, r - x > 0, |x| < 2^23
    assert words(floor, "SFPABS")


def test_floor_s_constants_stay_out_of_the_compared_side():
    unit = unit_of("floor")
    for word in words(unit, "SFPGT"):
        vd = int(word.split("lreg_dest=")[1].split()[0])
        assert vd < 12, word  # VD 12 .. 14 would store a macro template
    assert emit(unit)


class Pushed(km.Program):
    @km.handler
    def run(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(0, 8), block=8), km.Stream(b.pages(0, 8), block=8)
        ):
            results.push(km.where(x >= y, x, y * 0.5))


def test_a_pushed_where_is_a_tile_expression():
    ((_, unit),) = compile_class(
        Pushed, {}, tensors("bfloat16", "a", "b", "out")
    ).units.values()
    assert len(words(unit, "SFPGT")) == 1


def refusal(function):
    TESTS["refused"] = function
    try:
        with pytest.raises(ProgramError) as error:
            unit_of("refused")
    finally:
        del TESTS["refused"]
    return str(error.value)


def branched(x, y):
    if x > y:
        return x
    return y


@pytest.mark.parametrize(
    "function, said",
    [
        (lambda x, y: x > y, "a comparison of tiles is a lane test"),
        (lambda x, y: (x > y) * x, "a comparison of tiles is a lane test"),
        (lambda x, y: km.maximum(x < y, x), "a comparison of tiles is a lane test"),
        (lambda x, y: km.where(x, x, y), "where's first argument is a comparison"),
        (branched, "no Python truth value"),
        (lambda x, y: km.where(x > y > 0.0, x, y), "no Python truth value"),
        (lambda x, y: x if x else y, "use km.where(a > b, x, y)"),
    ],
)
def test_a_comparison_anywhere_else_is_refused_by_name(function, said):
    text = refusal(function)
    assert said in text
    assert "km.where(a > b, x, y)" in text
