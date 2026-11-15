"""A product's operands of any two matrix formats of one family, each side its own;
an A-family class's compute in a 16-bit Dst (no card)."""

import pytest

import komorebi as km
from komorebi import formats
from komorebi.dsl.compiler import compile_class, product_refusal
from komorebi.dsl.program import compute_options
from komorebi.errors import ProgramError
from komorebi.tensor import Tensor


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


def test_each_side_takes_any_matrix_format_of_one_family():
    for left in ("bfloat16", "float32", "bfp8_b"):
        for right in ("bfloat16", "bfp8_b", "bfp4_b", "bfp2_b", "float32"):
            assert product_refusal(left, right) is None
    assert product_refusal("float16", "bfp2") is None
    assert product_refusal("float32", "bfp8") is None
    assert "one exponent family" in product_refusal("bfloat16", "bfp8")
    assert "right operand is uint32" in product_refusal("bfloat16", "uint32")


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
