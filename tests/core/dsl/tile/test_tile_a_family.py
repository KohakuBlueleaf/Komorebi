"""A tile function of the A family (no card): its unit program runs on a 16-bit Dst,
its loads and stores in the FP16A mode, the same words as the B family's otherwise;
the build prints its products' bits; the row form and a mix with float32 or the B
family are refused, naming the formats and the line."""

import pytest

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import placements_of
from komorebi.errors import ProgramError


class Pages:
    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor(dtype, shape=(2048, 32)):
    return Tensor(Pages(formats.tile_bytes(dtype)), shape, dtype, "tile")


class Gate(km.Program):
    @km.tile_function
    def gate(a, w1, w3):
        return km.silu(a @ w1) * (a @ w3)

    @km.handler
    def run(self, a: km.Tensor, w1: km.Tensor, w3: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, p, q in zip(
            km.Stream(a.pages(0, 8), block=8),
            km.Stream(w1.pages(0, 8), block=8),
            km.Stream(w3.pages(0, 8), block=8),
        ):
            results.push(self.gate(x, p, q))


def gate_tensors(x_format, w_format):
    out = {"a": tensor(x_format), "out": tensor("float32")}
    return out | {"w1": tensor(w_format), "w3": tensor(w_format)}


def unit_of(compiled):
    ((_, unit),) = compiled.units.values()
    return unit


def test_the_a_family_runs_on_a_16_bit_dst_with_the_same_words():
    a = unit_of(compile_class(Gate, {}, gate_tensors("float16", "bfp8")))
    b = unit_of(compile_class(Gate, {}, gate_tensors("bfloat16", "bfp8_b")))
    assert (a.dest, b.dest) == ("fp16a", "fp32")


def test_the_build_prints_the_bits_of_a_traced_a_family_product():
    tensors = gate_tensors("float16", "bfp8")
    lines = placements_of(compile_class(Gate, {}, tensors), tensors)[1:]
    assert len(lines) == 2
    assert all("left float16, right bfp8, HiFi4" in line for line in lines)
    assert all("plain (left in SrcB): left keeps 11, right 7" in line for line in lines)


class Norm(km.Program):
    weight: km.L1[km.fp16, 1, 32, 32]

    @km.tile_function
    def norm(row, weight):
        ms = km.reduce_sum(row * row, exact=True) * (1.0 / 64)
        return row * km.rsqrt(ms + 1e-6).col() * weight.row()

    @km.handler
    def rows(self, x: km.Tensor, w: km.Tensor, out: km.Tensor):
        self.weight.fill(km.Stream(w.pages(0, 1)))
        results = km.Stream(out.pages(0, 8), write=True, block=4)
        for row in km.Stream(x.pages(0, 8), block=4, row=2):
            results.push(self.norm(row, self.weight))


def test_the_row_form_is_refused_for_the_a_family():
    tensors = {name: tensor("float16") for name in ("x", "w", "out")}
    with pytest.raises(ProgramError) as refused:
        compile_class(Norm, {}, tensors)
    text = str(refused.value)
    assert "results.push(self.norm(row, self.weight))" in text
    assert "takes tiles of float16 (the A family, a 16-bit Dst)" in text
    assert "is called on a row" in text


class Mixed(km.Program):
    @km.tile_function
    def times(x, y):
        return x * y

    @km.handler
    def mul(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(0, 8), block=8), km.Stream(b.pages(0, 8), block=8)
        ):
            results.push(self.times(x, y))


def test_the_a_family_with_float32_is_refused_naming_both():
    tensors = {"a": tensor("float16"), "b": tensor("float32"), "out": tensor("float32")}
    with pytest.raises(ProgramError) as refused:
        compile_class(Mixed, {}, tensors)
    text = str(refused.value)
    assert "results.push(self.times(x, y))" in text
    assert "given float16 with float32" in text
    assert "an fp32 Dst reads the A family 2^-15 off" in text
