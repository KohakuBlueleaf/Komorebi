"""A handler's tile lines (no card): a tile expression pushed or assigned is the tile
function of the loop's tiles it names; x / y, sqrt, abs, minimum and reciprocal in
the unit program; four operand tiles; the refusals name the line."""

import pytest

import komorebi as km
from komorebi import formats
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.tile.emit import emit
from komorebi.errors import ProgramError
from komorebi.tensor import Tensor

SCALE = 0.5


class Pages:
    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensors(*names):
    page = formats.tile_bytes("bfloat16")
    return {n: Tensor(Pages(page), (2048, 32), "bfloat16", "tile") for n in names}


def unit_of(cls, *names):
    ((_, unit),) = compile_class(cls, {}, tensors(*names, "out")).units.values()
    return unit


class Pushed(km.Program):
    @km.handler
    def run(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(0, 8), block=8), km.Stream(b.pages(0, 8), block=8)
        ):
            results.push(km.silu(x) * y * SCALE)


class Called(km.Program):
    @km.tile_function
    def gate(x, y):
        return km.silu(x) * y * SCALE

    @km.handler
    def run(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(0, 8), block=8), km.Stream(b.pages(0, 8), block=8)
        ):
            results.push(self.gate(x, y))


def body(unit):
    """The unit's C++ less its first line (the header's name)."""
    return emit(unit).split("\n", 1)[1].replace(unit.name, "f")


def test_a_pushed_expression_is_the_tile_function_of_its_tiles():
    pushed, called = unit_of(Pushed, "a", "b"), unit_of(Called, "a", "b")
    assert pushed.operands == ["x", "y"] == called.operands
    assert body(pushed) == body(called)


class Assigned(km.Program):
    @km.handler
    def run(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(0, 8), block=8), km.Stream(b.pages(0, 8), block=8)
        ):
            q = y / x
            results.push(q)


def test_an_assigned_expression_names_its_tiles_in_their_order():
    unit = unit_of(Assigned, "a", "b")
    assert unit.operands == ["y", "x"]
    assert "SFPARECIP" in emit(unit)


class Worded(km.Program):
    @km.handler
    def run(self, a: km.Tensor, out: km.Tensor, scale: int):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x in km.Stream(a.pages(0, 8), block=8):
            results.push(x * scale)


def test_a_word_in_a_tile_expression_is_refused_at_its_line():
    with pytest.raises(ProgramError) as e:
        unit_of(Worded, "a")
    assert "`results.push(x * scale)`" in str(e.value)
    assert "`scale` (int) is none of them" in str(e.value)


def one(expression):
    """A class whose loop pushes `expression` of its tiles x, y, z, w, v (those it
    names)."""
    names = [n for n in "xyzwv" if n in expression.replace("km.", "")]
    streams = ", ".join(f"km.Stream({n}_in.pages(0, 8), block=8)" for n in names)
    loop = f"zip({streams})" if len(names) > 1 else streams
    params = ", ".join(f"{n}_in: km.Tensor" for n in names)
    source = (
        "class One(km.Program):\n"
        "    @km.handler\n"
        f"    def run(self, {params}, out: km.Tensor):\n"
        "        results = km.Stream(out.pages(0, 8), write=True, block=8)\n"
        f"        for {', '.join(names)} in {loop}:\n"
        f"            results.push({expression})\n"
    )
    return source, [f"{n}_in" for n in names]


@pytest.mark.parametrize(
    ("expression", "word"),
    [
        ("x / y", "SFPARECIP"),
        ("km.reciprocal(x)", "SFPARECIP"),
        ("km.sqrt(x)", "TTI_SFPSHFT(4095, 0, 1, 5);"),
        ("abs(x)", "TTI_SFPABS(0, 0, 0, 1)"),
        ("km.minimum(x, y)", "TTI_SFPSWAP(0, 0, 1, 1);\n            TTI_SFPSTORE(1, 0"),
        ("x * y + z * w", "TTI_SFPLOAD(3, 0, 7, 192);"),
    ],
)
def test_the_vector_functions_and_four_operand_tiles(tmp_path, expression, word):
    assert word in emit(unit_of(*made(tmp_path, expression)))


def made(tmp_path, expression):
    """The class One of `expression` from a file of its own, and its tensors."""
    source, names = one(expression)
    path = tmp_path / "one.py"
    path.write_text("import komorebi as km\n\n\n" + source)
    namespace = {"__name__": "one", "__file__": str(path)}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace["One"], *names


def test_five_operand_tiles_are_refused(tmp_path):
    with pytest.raises(ProgramError) as e:
        unit_of(*made(tmp_path, "x * y + z * w + v"))
    assert "`results.push(x * y + z * w + v)`" in str(e.value)
    assert "reads at most 4 operand tiles" in str(e.value)
