"""A product's operands of any two matrix formats of one family, each side its own;
float32 refused on either side (a class product of a float32 operand gave wrong words
on the card); an A-family class's compute in a 16-bit Dst (no card)."""

import pytest

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.tiles.products import product_refusal
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import compute_options
from komorebi.errors import ProgramError


class Pages:
    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor(dtype):
    return Tensor(Pages(formats.tile_bytes(dtype)), (64, 64), dtype, "tile")


@km.tile_function
def project(x, w):
    return x @ w


class Bfp8Weight(km.Program):
    weight: km.L1[km.bfp8_b, 2, 2, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        self.weight.fill(km.Stream(w.tiles[0:2, 0:2]))
        acc = km.tile.zeros(km.fp32, 2)
        for block in km.Stream(x.tiles[0:1, 0:2], block=2):
            acc += project(block, self.weight)
        y.tiles[0:1, 0:2].write(acc)


class HalfWeight(km.Program):
    weight: km.L1[km.bfp4, 2, 2, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        self.weight.fill(km.Stream(w.tiles[0:2, 0:2]))
        acc = km.tile.zeros(km.fp32, 2)
        for block in km.Stream(x.tiles[0:1, 0:2], block=2):
            acc += project(block, self.weight)
        y.tiles[0:1, 0:2].write(acc)


class Streamed(km.Program):
    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, 2)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:2], block=2), km.Stream(w.tiles[0:2, 0:2], block=4)
        ):
            acc += project(xb, wb)
        y.tiles[0:1, 0:2].write(acc)


class FieldLeft(km.Program):
    h: km.L1[km.fp32, 2, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, 2)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:2], block=2), km.Stream(w.tiles[0:2, 0:2], block=4)
        ):
            acc += project(xb, wb)
        self.h.write(acc)
        out = km.tile.zeros(km.fp32, 2)
        for wb in km.Stream(w.tiles[0:2, 0:2], block=4):
            out += project(self.h, wb)
        y.tiles[0:1, 0:2].write(out)


def test_each_side_takes_any_matrix_format_of_one_family():
    for left in ("bfloat16", "bfp8_b", "bfp4_b"):
        for right in ("bfloat16", "bfp8_b", "bfp4_b", "bfp2_b"):
            assert product_refusal(left, right) is None
    assert product_refusal("float16", "bfp2") is None
    assert "one exponent family" in product_refusal("bfloat16", "bfp8")
    assert "right operand is uint32" in product_refusal("bfloat16", "uint32")


def test_float32_is_refused_on_either_side():
    for left, right, side in (
        ("float32", "bfp8_b", "left"),
        ("bfloat16", "float32", "right"),
        ("float32", "float32", "left"),
        ("float32", "bfp8", "left"),
    ):
        why = product_refusal(left, right)
        assert f"{side} operand is float32" in why
        assert "a bfloat16 copy" in why and "km.bf16" in why


def streamed(x, w):
    return {"x": tensor(x), "w": tensor(w), "y": tensor("float32")}


def test_a_float32_x_stream_is_refused_at_build():
    with pytest.raises(ProgramError, match="left operand is float32"):
        compile_class(Streamed, {}, streamed("float32", "bfp8_b"))


def test_a_float32_w_stream_is_refused_at_build():
    with pytest.raises(ProgramError, match="right operand is float32"):
        compile_class(Streamed, {}, streamed("bfloat16", "float32"))


def test_a_float32_field_on_the_left_is_refused_at_build():
    with pytest.raises(ProgramError, match="left operand is float32"):
        compile_class(FieldLeft, {}, streamed("bfloat16", "bfloat16"))


def test_the_bf16_control_builds():
    compiled = compile_class(Streamed, {}, streamed("bfloat16", "bfp8_b"))
    assert compiled.products


def test_a_bfp8_b_weight_against_bf16_tiles_builds_with_its_own_buffers():
    tensors = {"x": tensor("bfloat16"), "w": tensor("bfp8_b"), "y": tensor("float32")}
    compiled = compile_class(Bfp8Weight, {}, tensors)
    weight = next(b for b in compiled.buffers if b.name == "kFieldWeight")
    assert (weight.dtype, weight.page_bytes) == ("bfp8_b", 1088)
    assert compute_options(compiled)["fp32_dest_acc_en"] is True


def test_an_a_family_class_computes_in_a_16_bit_dst():
    tensors = {"x": tensor("float16"), "w": tensor("bfp4"), "y": tensor("float32")}
    compiled = compile_class(HalfWeight, {}, tensors)
    weight = next(b for b in compiled.buffers if b.name == "kFieldWeight")
    assert (weight.dtype, weight.page_bytes) == ("bfp4", 576)
    assert compute_options(compiled)["fp32_dest_acc_en"] is False


def test_a_field_of_the_other_family_is_refused():
    tensors = {"x": tensor("float16"), "w": tensor("bfp8_b"), "y": tensor("float32")}
    with pytest.raises(ProgramError, match="one exponent family"):
        compile_class(Bfp8Weight, {}, tensors)
