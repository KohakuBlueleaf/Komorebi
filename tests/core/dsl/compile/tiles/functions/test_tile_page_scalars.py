"""A tile function's run-time scalars from a page's words (no card): `read(km.fp32)`
words are float words, C++ floats of the page's bits on every kernel, handed from
unpack to math and pack as bits, taken by a call and a tile expression as a command's
float words are; `read()` words stay ints; the refusals."""

import pytest
from class_lines_fixture import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import compute_source


@km.tile_function
def clip(x, y, z, m0: float, m1: float, m2: float, m3: float):
    return x * m0 + y * m1 + z * m2 + m3


class Camera(km.Program):
    @km.handler
    def row(
        self,
        x: km.Tensor,
        y: km.Tensor,
        z: km.Tensor,
        o: km.Tensor,
        cam: km.Tensor,
        r: int,
    ):
        m0, m1, m2, m3 = cam.pages(r, 1).read(km.fp32)[0:4]
        results = km.Stream(o.pages(0, 8), write=True)
        for a, b, c in zip(
            km.Stream(x.pages(0, 8)), km.Stream(y.pages(0, 8)), km.Stream(z.pages(0, 8))
        ):
            results.push(clip(a, b, c, m0, m1, m2, m3))

    @km.handler
    def scaled(self, x: km.Tensor, o: km.Tensor, cam: km.Tensor):
        (m0,) = cam.pages(0, 1).read(km.fp32)[0:1]
        results = km.Stream(o.pages(0, 8), write=True)
        for a in km.Stream(x.pages(0, 8)):
            results.push(a * m0)

    @km.handler
    def words(self, x: km.Tensor, o: km.Tensor, cam: km.Tensor):
        first, count = cam.pages(0, 1).read()[0:2]
        results = km.Stream(o.pages(first, count), write=True)
        for a in km.Stream(x.pages(first, count)):
            results.push(a)


def built(cls, cam="float32"):
    tensors = {n: Tensor(Pages(4096), (256, 32), "float32", "tile") for n in "xyzo"}
    tensors["cam"] = Tensor(
        Pages(4096 if cam == "float32" else 2048), (3, 1024), cam, "row"
    )
    return compile_class(cls, {}, tensors)


def case_of(source, name):
    at = source.index(f"case kHandler{name.capitalize()}")
    end = source.find("case kHandler", at + 1)
    return source[at : end if end > 0 else len(source)]


def test_float_page_words_are_a_calls_scalars():
    case = case_of(compute_source(built(Camera)), "row")
    assert "[[maybe_unused]] float m0 = 0.0f;" in case
    taken = "m3 = __builtin_bit_cast(float, static_cast<uint32_t>(km_page_words[3]));"
    assert taken in case
    math = "ckernel::ThreadId::MathThreadId"
    assert f"compute.hand({math}, __builtin_bit_cast(uint32_t, m0));" in case
    unpack = "ckernel::ThreadId::UnpackThreadId"
    assert f"m0 = __builtin_bit_cast(float, compute.handed({unpack}));" in case
    bits = ", ".join(f"__builtin_bit_cast(uint32_t, m{k})" for k in range(4))
    assert f"MATH((km_tile::clip(0, {bits})));" in case


def test_a_float_page_word_in_a_tile_expression():
    case = case_of(compute_source(built(Camera)), "scaled")
    assert "slot0, __builtin_bit_cast(uint32_t, m0))));" in case  # tile_line_N's


def test_words_read_without_a_format_stay_ints():
    case = case_of(compute_source(built(Camera)), "words")
    assert "[[maybe_unused]] uint32_t first = 0;" in case
    assert "first = km_page_words[0];" in case


class Other(km.Program):
    @km.handler
    def run(self, x: km.Tensor, o: km.Tensor, cam: km.Tensor):
        m0, m1 = cam.pages(0, 1).read(km.bf16)[0:2]
        results = km.Stream(o.pages(0, 8), write=True)
        for a in km.Stream(x.pages(0, 8)):
            results.push(a * m0 + m1)


class Alone(km.Program):
    @km.handler
    def run(self, x: km.Tensor, o: km.Tensor, cam: km.Tensor):
        m0 = cam.pages(0, 1).read(km.fp32)
        results = km.Stream(o.pages(0, 8), write=True)
        for a in km.Stream(x.pages(0, 8)):
            results.push(a * m0)


class Counted(km.Program):
    @km.handler
    def run(self, x: km.Tensor, o: km.Tensor, cam: km.Tensor):
        first, count = cam.pages(0, 1).read(km.fp32)[0:2]
        results = km.Stream(o.pages(0, 8), write=True)
        for a in km.Stream(x.pages(first, count)):
            results.push(a)


@pytest.mark.parametrize(
    "cls,cam,why",
    [
        (Other, "float32", "takes km.fp32 alone"),
        (Alone, "float32", "one word read alone is mover B's int"),
        (Camera, "bfloat16", "read\\(km.fp32\\) takes a page's words as float32"),
        (Counted, "float32", "first is a float word"),
    ],
)
def test_the_refusals(cls, cam, why):
    with pytest.raises(Refused, match=why):
        built(cls, cam)
