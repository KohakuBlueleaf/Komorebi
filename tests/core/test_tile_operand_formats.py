"""A tile function's operands of different formats (no card): the call sets SrcA's
format again before each unpack into it, a product's W and each copied operand, when
they are not of one format; of one format, the call emits no switch."""

import komorebi as km
from komorebi import formats
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import compute_source
from komorebi.tensor import Tensor


class Pages:
    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor(dtype):
    return Tensor(Pages(formats.tile_bytes(dtype)), (2048, 32), dtype, "tile")


class Mul(km.Program):
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


class Scaled(km.Program):
    @km.tile_function
    def scaled(a, w):
        return (a @ w) * a

    @km.handler
    def run(self, a: km.Tensor, w: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, p in zip(
            km.Stream(a.pages(0, 8), block=8), km.Stream(w.pages(0, 8), block=8)
        ):
            results.push(self.scaled(x, p))


def source(cls, **dtypes):
    tensors = {name: tensor(dtype) for name, dtype in dtypes.items()}
    return compute_source(compile_class(cls, {}, tensors))


def test_two_copied_operands_of_two_formats_switch_before_each_copy():
    for a, b in (("bfloat16", "bfp8_b"), ("float16", "bfp8")):
        text = source(Mul, a=a, b=b, out="float32")
        lines = [line.strip() for line in text.split("\n")]
        first = next(i for i, x in enumerate(lines) if x.startswith("copy_tile("))
        assert lines[first - 1] == "KM_UNPACK_FORMAT_A(kRead0.mover_a);"
        assert lines[first].startswith("copy_tile(km_half0,")
        assert lines[first + 1] == "KM_UNPACK_FORMAT_A(kRead1.mover_a);"
        assert lines[first + 2].startswith("copy_tile(km_half1,")


def test_a_product_and_a_copy_of_two_formats_switch_before_each():
    text = source(Scaled, a="bfloat16", w="bfp8_b", out="float32")
    lines = [line.strip() for line in text.split("\n")]
    unpack = next(i for i, x in enumerate(lines) if "llk_unpack_AB_matmul" in x)
    assert lines[unpack - 1] == "KM_UNPACK_FORMATS(kRead1.mover_a, kRead0.mover_a);"
    copy = next(i for i, x in enumerate(lines) if x.startswith("copy_tile(km_half0,"))
    assert lines[copy - 1] == "KM_UNPACK_FORMAT_A(kRead0.mover_a);"


def test_operands_of_one_format_emit_no_switch():
    text = source(Mul, a="bfloat16", b="bfloat16", out="float32")
    lines = [line.strip() for line in text.split("\n")]
    copies = [i for i, x in enumerate(lines) if x.startswith("copy_tile(")]
    assert len(copies) == 2
    assert not any(lines[i - 1].startswith("KM_UNPACK") for i in copies)
