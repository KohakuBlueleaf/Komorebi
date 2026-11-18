"""Two DST sums held together and a tile function of both written (SwiGLU's gate in
the handler that makes the sums, no card): one acquire, the second sums zeroed after
the first, the unit's second operand read its distance apart; refused by name: the
two wider than DST, of two widths, or one written alone."""

import pytest
from class_lines_fixture import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class

D, H = 8, 16  # x's width and the hidden's, in tiles


@km.tile_function
def gate(h, g):
    return km.silu(h) * g


@km.tile_function
def project(x, w):
    return x @ w


def given():
    def made(rows, cols):
        return Tensor(Pages(2048), (rows * 32, cols * 32), "bfloat16", "tile")

    return {
        "x": made(4, D),
        "w1": made(D, H),
        "w3": made(D, H),
        "hidden": made(4, H),
    }


class Up(km.Program):
    WH = km.constant(4)  # acc_h's tiles
    WG = km.constant(4)  # acc_g's tiles
    HB = WH * 2  # a block of w1's tiles: two K rows of acc_h's columns
    GB = WG * 2

    @km.handler
    def up(self, x: km.Tensor, w1: km.Tensor, w3: km.Tensor, hidden: km.Tensor):
        first = self.rank * self.WH
        for r in range(4):
            acc_h = km.tile.zeros(km.fp32, self.WH)
            for xb, wb in zip(
                km.Stream(x.tiles[r : r + 1, 0:D], block=2),
                km.Stream(w1.tiles[0:D, first : first + self.WH], block=self.HB),
            ):
                acc_h += project(xb, wb)
            acc_g = km.tile.zeros(km.fp32, self.WG)
            for xb, wb in zip(
                km.Stream(x.tiles[r : r + 1, 0:D], block=2),
                km.Stream(w3.tiles[0:D, first : first + self.WG], block=self.GB),
            ):
                acc_g += project(xb, wb)
            hidden.tiles[r : r + 1, first : first + self.WH].write(gate(acc_h, acc_g))


class Alone(Up):
    @km.handler
    def up(self, x: km.Tensor, w1: km.Tensor, w3: km.Tensor, hidden: km.Tensor):
        first = self.rank * self.WH
        acc_h = km.tile.zeros(km.fp32, self.WH)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:D], block=2),
            km.Stream(w1.tiles[0:D, first : first + self.WH], block=self.HB),
        ):
            acc_h += project(xb, wb)
        acc_g = km.tile.zeros(km.fp32, self.WG)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:D], block=2),
            km.Stream(w3.tiles[0:D, first : first + self.WG], block=self.GB),
        ):
            acc_g += project(xb, wb)
        hidden.tiles[0:1, first : first + self.WH].write(acc_h)


def built(width_h, width_g, alone=False):
    """Up (or Alone) compiled with acc_h and acc_g of these widths."""
    constants = {"WH": width_h, "WG": width_g, "HB": 2 * width_h, "GB": 2 * width_g}
    return compile_class(Alone if alone else Up, {}, given(), constants=constants)


def test_the_gate_of_two_sums_runs_in_dst_and_writes_the_first():
    compiled = built(4, 4)
    ((_, (key, _)),) = compiled.units.items()
    assert key[-1] == (("g", 4), ("h", 0))


@pytest.mark.parametrize(
    ("widths", "alone", "message"),
    [
        (
            (5, 4),
            False,
            r"acc_h \(5 tiles\) and acc_g \(4 tiles\) take 9 of DST's 8 fp32 tiles; "
            r"make each 4 or fewer",
        ),
        (
            (3, 4),
            False,
            r"gate takes acc_h \(3 tiles\) and acc_g \(4 tiles\): tile t of each",
        ),
        (
            (4, 4),
            True,
            r"acc_h and acc_g hold DST together, and writing acc_h alone releases "
            r"the other's sums; write a tile function of both",
        ),
    ],
)
def test_two_sums_are_refused_by_name(widths, alone, message):
    with pytest.raises(Refused, match=message):
        built(*widths, alone=alone)
