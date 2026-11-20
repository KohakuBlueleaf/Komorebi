"""Operands copied whole, one float32 and one bfloat16 (no card): the float32 tile is
unpacked to Dst, the other through SrcA, so each copy has its own buffer's copy_init
(one setup for both left the SrcA copy waiting on a SrcA never valid: the chip hung);
copies of one kind keep the setup before the loop."""

from class_lines_fixture import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import compute_source

PAGE = {"float32": 4096, "bfloat16": 2048, "bfp8_b": 1088}  # a tile's bytes


@km.tile_function
def product(a, b):
    return a * b


class Mixed(km.Program):
    @km.handler
    def run(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 4), write=True)
        for x, y in zip(km.Stream(a.pages(0, 4)), km.Stream(b.pages(0, 4))):
            results.push(product(x, y))


def loop(dtypes):
    """The handler's text from its first acquire, a and b of `dtypes`."""
    tensors = {
        name: Tensor(Pages(PAGE[d]), (128, 32), d, "tile")
        for name, d in zip(("a", "b", "out"), (*dtypes, "float32"))
    }
    text = compute_source(compile_class(Mixed, {}, tensors))
    return text[text.index("tile_regs_acquire();") :]


def test_each_copy_of_two_kinds_has_its_own_setup():
    lines = [line.strip() for line in loop(("float32", "bfloat16")).split("\n")]
    copies = [k for k, line in enumerate(lines) if line.startswith("copy_tile(")]
    assert len(copies) == 2
    for k, buffer in zip(copies, ("kRead0", "kRead1")):
        assert lines[k - 2 : k] == [
            f"KM_UNPACK_FORMAT_A({buffer}.mover_a);",
            f"copy_init({buffer}.mover_a);",
        ]


def test_copies_of_one_kind_keep_the_setup_before_the_loop():
    assert "copy_init(" not in loop(("bfloat16", "bfloat16"))
    assert "copy_init(" not in loop(("float32", "float32"))
    assert "copy_init(" not in loop(("bfloat16", "bfp8_b"))
