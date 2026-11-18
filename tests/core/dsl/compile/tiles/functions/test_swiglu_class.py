"""The README's SwiGLU class (examples/31_swiglu) made and chained with no card: a tile
function of two fields of sums written as a rectangle; two products of one loop each
counted by mover B; the chain's calls on their own cores, the rectangle write
described; the refusals name the line."""

import importlib.util
import sys
from pathlib import Path

import pytest
from chain_fixture import Pages, quiet, run_of

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.tiles.products import product_setups
from komorebi.errors import ProgramError

ROOT = Path(__file__).resolve().parents[6]


def example():
    path = ROOT / "examples" / "31_swiglu" / "swiglu_class.py"
    spec = importlib.util.spec_from_file_location("swiglu_class", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["swiglu_class"] = module
    spec.loader.exec_module(module)
    return module


def tensor(dtype, shape, address):
    pages = shape[0] * shape[1] // 1024
    page = formats.tile_bytes(dtype)
    return Tensor(Pages(page, pages, address), shape, dtype, "tile")


def tensors():
    shapes = {
        "x": ("bfloat16", (32, 1024)),
        "w1": ("bfloat16", (1024, 3072)),
        "w3": ("bfloat16", (1024, 3072)),
        "w2": ("bfloat16", (3072, 1024)),
        "hidden": ("bfloat16", (32, 3072)),
        "y": ("float32", (32, 1024)),
    }
    return {
        name: tensor(dtype, shape, 0x1000000 * (index + 1))
        for index, (name, (dtype, shape)) in enumerate(shapes.items())
    }


def made(cls, given):
    return cls(l1=1 << 20, log=quiet, **given)


def test_the_chain_method_gives_its_calls_on_their_own_cores():
    given = tensors()
    program = made(example().SwiGLU, given)
    names = ("x", "w1", "w3", "w2", "hidden", "y")
    layer = run_of(program).chain(program.forward(*(given[n] for n in names)))
    up, down = layer.calls
    assert (up.group, down.group) == ((0, 12), (0, 4))
    assert (up.width, down.width) == (8, 8)
    assert down.input == up.output


def test_a_later_product_of_another_setup_sets_its_own_and_the_first_again():
    assert product_setups(None, "a;") == ("", "")
    assert product_setups("a;", "a;") == ("", "")
    assert product_setups("a;", "b;") == ("    b;\n", "    a;\n")


class Shapes(km.Program):
    h: km.L1[km.fp32, 1, 8, 32, 32]
    g: km.L1[km.fp32, 1, 4, 32, 32]

    @km.tile_function
    def project(x, w):
        return x @ w

    @km.tile_function
    def both(h, g):
        return h * g

    @km.handler
    def up(self, x: km.Tensor, w1: km.Tensor, w3: km.Tensor, hidden: km.Tensor):
        for xb, w1b, w3b in zip(
            km.Stream(x.tiles[0:1, 0:32], block=2),
            km.Stream(w1.tiles[0:32, 0:8], block=16),
            km.Stream(w3.tiles[0:32, 0:4], block=8),
        ):
            self.h += self.project(xb, w1b)
            self.g += self.project(xb, w3b)
        hidden.tiles[0:1, 0:8].write(self.both(self.h, self.g))


def test_fields_of_two_shapes_are_refused_at_the_line():
    given = tensors()
    del given["w2"], given["y"]
    with pytest.raises(ProgramError) as e:
        made(Shapes, given)
    assert "`hidden.tiles[0:1, 0:8].write(self.both(self.h, self.g))`" in str(e.value)
    assert "takes fields of sums of one shape" in str(e.value)
