"""A float word (no card): its float32 bits; in tile arithmetic a run-time scalar the
unit loads by two run-time SFPLOADI before the rows; a C++ float in a body; an int word
in a tile expression, a float word in integer arithmetic and a division by one refused
at the line."""

import re
from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.errors import ProgramError
from komorebi.tensor import Tensor

HERE = Path(__file__).resolve().parent


def tensors():
    def made(dtype, address):
        page = formats.tile_bytes(dtype)
        return Tensor(Pages(page, 512, address), (32, 512 * 32), dtype, "tile")

    return {"a": made("bfloat16", 0x100000), "out": made("float32", 0x1000000)}


@km.tile_function
def affine(x, scale, shift):
    return x * scale + shift


class Line(km.Program):
    @km.handler
    def run(self, a: km.Tensor, out: km.Tensor, scale: float):
        first = self.rank * 8
        results = km.Stream(out.pages(first, 8), write=True, block=8)
        for x in km.Stream(a.pages(first, 8), block=8):
            results.push(x * scale)


class Called(km.Program):
    @km.handler
    def run(self, a: km.Tensor, out: km.Tensor, scale: float, shift: float):
        results = km.Stream(out.pages(0, 8), write=True)
        for x in km.Stream(a.pages(0, 8)):
            results.push(affine(x, scale, shift))


class Body(km.Program):
    @km.handler
    def run(self, a: km.Tensor, out: km.Tensor, scale: float):
        results = km.Stream(out.pages(0, 8), write=True)
        for tile in km.Stream(a.pages(0, 8)):
            made = km.cxx("// scaled by scale", takes=[tile, scale], gives=km.Tile)
            results.push(made)


class IntWord(km.Program):
    @km.handler
    def run(self, a: km.Tensor, out: km.Tensor, n: int):
        results = km.Stream(out.pages(0, 8), write=True)
        for x in km.Stream(a.pages(0, 8)):
            results.push(x * n)


class Bound(km.Program):
    @km.handler
    def run(self, a: km.Tensor, out: km.Tensor, scale: float):
        results = km.Stream(out.pages(0, scale), write=True)
        for x in km.Stream(a.pages(0, 8)):
            results.push(x)


class Divided(km.Program):
    @km.handler
    def run(self, a: km.Tensor, out: km.Tensor, scale: float):
        results = km.Stream(out.pages(0, 8), write=True)
        for x in km.Stream(a.pages(0, 8)):
            results.push(x / scale)


def folder_of(cls):
    cls(l1=1 << 20, log=quiet, **tensors())
    return sorted(
        (HERE / "build").glob(f"{cls.__name__.lower()}-*"),
        key=lambda p: p.stat().st_mtime,
    )[-1]


def test_a_float_word_is_its_bits_and_a_non_number_is_refused():
    folder_of(Line)
    words = Line.run.args(tensors()["a"], tensors()["out"], 2.0)
    assert words[2] == 0x40000000
    assert Line.run.args(tensors()["a"], tensors()["out"], -0.75)[2] == 0xBF400000
    with pytest.raises(ProgramError, match=r"run: scale is a float, got '2'"):
        Line.run.args(tensors()["a"], tensors()["out"], "2")


def test_a_tile_expression_loads_the_word_before_the_rows():
    folder = folder_of(Line)
    header = next(folder.glob("tile_tile_line_*.hpp")).read_text()
    assert "const uint32_t dst, const uint32_t scale_bits)" in header
    first = header.index("TT_SFPLOADI(")
    assert re.search(r"TT_SFPLOADI\((\d), 2, scale_bits & 0xFFFFu\);", header)
    assert re.search(r"TT_SFPLOADI\((\d), 8, scale_bits >> 16\);", header)
    assert first < header.index("for (uint32_t face")
    compute = (folder / "compute.cpp").read_text()
    assert "const float scale = __builtin_bit_cast(float, command.a2);" in compute
    assert "__builtin_bit_cast(uint32_t, scale))));" in compute


def test_a_tile_function_takes_two_words_in_its_parameters_order():
    compute = (folder_of(Called) / "compute.cpp").read_text()
    assert (
        "MATH((km_tile::affine(0, __builtin_bit_cast(uint32_t, scale), "
        "__builtin_bit_cast(uint32_t, shift))));"
    ) in compute.replace("1 * km_slot0", "0")


def test_a_body_takes_the_word_as_a_float():
    compute = (folder_of(Body) / "compute.cpp").read_text()
    at = compute.index("const float scale = __builtin_bit_cast(float, command.a2);")
    assert at < compute.index("// scaled by scale")


def test_an_int_word_in_a_tile_expression_names_the_float_annotation():
    with pytest.raises(ProgramError, match=r"annotate it `n: float`"):
        folder_of(IntWord)


def test_a_float_word_in_integer_arithmetic_is_refused_at_the_line():
    with pytest.raises(ProgramError, match=r"line \d+, .*scale is a float word"):
        folder_of(Bound)


def test_a_division_by_a_word_is_refused():
    with pytest.raises(ProgramError, match=r"a division by the run-time scalar scale"):
        folder_of(Divided)
