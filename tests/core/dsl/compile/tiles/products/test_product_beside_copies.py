"""A tile function with a matrix product and an operand copied into DST beside it
(`x * c + (x @ p) * s`, a broadcast row too) refused by name at the build: the
product's unpack setup replaces the copy's, and the core hangs (no card). Products
alone, and the product then the element-wise rest as two handlers, still build."""

from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.errors import ProgramError


def tensors():
    def made(shape, address):
        pages = shape[0] * shape[1] // 1024
        return Tensor(
            Pages(formats.tile_bytes("bfloat16"), pages, address),
            shape,
            "bfloat16",
            "tile",
        )

    names = ("x", "p", "c", "s", "y", "out")
    return {name: made((32, 64), 0x1000000 * (k + 1)) for k, name in enumerate(names)}


@km.tile_function(fidelity="HiFi4")
def rotate(x, p, c, s):
    return x * c + (x @ p) * s


@km.tile_function(fidelity="HiFi4")
def scaled(x, p, c):
    return (x @ p) * c.row()


@km.tile_function(fidelity="HiFi4")
def gate(x, p, c):
    return km.silu(x @ p) * (x @ c)


@km.tile_function(fidelity="HiFi4")
def project(x, p):
    return x @ p


@km.tile_function
def mix(x, y, c, s):
    return x * c + y * s


class Fused(km.Program):
    @km.handler
    def run(
        self, x: km.Tensor, p: km.Tensor, c: km.Tensor, s: km.Tensor, out: km.Tensor
    ):
        o = km.Stream(out.tiles[0:1, 0:2], write=True, block=2)
        for a, w, u, v in zip(
            km.Stream(x.tiles[0:1, 0:2], block=2),
            km.Stream(p.tiles[0:1, 0:2], block=2),
            km.Stream(c.tiles[0:1, 0:2], block=2),
            km.Stream(s.tiles[0:1, 0:2], block=2),
        ):
            o.push(rotate(a, w, u, v))


class Row(km.Program):
    @km.handler
    def run(self, x: km.Tensor, p: km.Tensor, c: km.Tensor, out: km.Tensor):
        o = km.Stream(out.tiles[0:1, 0:2], write=True, block=2)
        for a, w, u in zip(
            km.Stream(x.tiles[0:1, 0:2], block=2),
            km.Stream(p.tiles[0:1, 0:2], block=2),
            km.Stream(c.tiles[0:1, 0:2], block=2),
        ):
            o.push(scaled(a, w, u))


class Gate(km.Program):
    @km.handler
    def run(self, x: km.Tensor, p: km.Tensor, c: km.Tensor, out: km.Tensor):
        o = km.Stream(out.tiles[0:1, 0:2], write=True, block=2)
        for a, w, u in zip(
            km.Stream(x.tiles[0:1, 0:2], block=2),
            km.Stream(p.tiles[0:1, 0:2], block=2),
            km.Stream(c.tiles[0:1, 0:2], block=2),
        ):
            o.push(gate(a, w, u))


class Split(km.Program):
    @km.handler
    def swap(self, x: km.Tensor, p: km.Tensor, y: km.Tensor):
        o = km.Stream(y.tiles[0:1, 0:2], write=True, block=2)
        for a, w in zip(
            km.Stream(x.tiles[0:1, 0:2], block=2),
            km.Stream(p.tiles[0:1, 0:2], block=2),
        ):
            o.push(project(a, w))

    @km.handler
    def rest(
        self, x: km.Tensor, y: km.Tensor, c: km.Tensor, s: km.Tensor, out: km.Tensor
    ):
        o = km.Stream(out.tiles[0:1, 0:2], write=True, block=2)
        for a, b, u, v in zip(
            km.Stream(x.tiles[0:1, 0:2], block=2),
            km.Stream(y.tiles[0:1, 0:2], block=2),
            km.Stream(c.tiles[0:1, 0:2], block=2),
            km.Stream(s.tiles[0:1, 0:2], block=2),
        ):
            o.push(mix(a, b, u, v))


def compute_of(cls):
    program = cls(l1=1 << 20, log=quiet, **tensors())
    return (Path(program.built.folder) / "compute.cpp").read_text()


@pytest.mark.parametrize(
    "cls, copied",
    [(Fused, "x, c, s"), (Row, "c")],
)
def test_a_product_beside_a_copied_operand_is_refused_by_name(cls, copied):
    refusal = (
        rf"line \d+, `.*`: \w+ has a matrix product \(`@`\) and copies {copied} "
        r"into DST beside it: .* \(the core hangs\); push the product"
    )
    with pytest.raises(ProgramError, match=refusal):
        compute_of(cls)


def test_products_alone_and_the_two_handler_form_still_build():
    gate_text = compute_of(Gate)
    assert "mm_no_mop_init_short(" in gate_text
    assert "copy_tile(" not in gate_text
    split = compute_of(Split)
    swap = split[split.index("case kHandlerSwap:") : split.index("case kHandlerRest:")]
    rest = split[split.index("case kHandlerRest:") :]
    assert "mm_no_mop_init_short(" in swap and "copy_tile(" not in swap
    assert "mm_no_mop_init_short(" not in rest and rest.count("copy_tile(") == 4
