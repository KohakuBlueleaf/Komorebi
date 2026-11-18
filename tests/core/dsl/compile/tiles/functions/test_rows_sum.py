"""A sum over rows (no card): `reduce_sum(..., dim=-2)` in a tile function is refused
naming the way that builds, which builds: column sums by a product with a tile of ones
on the left, then the row form's sum along the row."""

import pytest
from class_lines_fixture import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class

J = 2  # tiles a row


@km.tile_function(tolerance=1e-5)
def down(row):
    return row * 0.0 + km.reduce_sum(row, dim=-2, exact=True).col()


@km.tile_function(tolerance=1e-5)
def lane_sum(row):
    return row * 0.0 + km.reduce_sum(row, exact=True).col()


class Down(km.Program):
    @km.handler
    def lanes(self, t: km.Tensor, out: km.Tensor):
        results = km.Stream(out.tiles[0:1, 0:J], write=True, block=J)
        for row in km.Stream(t.tiles[0:1, 0:J], block=J, row=J):
            results.push(down(row))


class Sums(km.Program):
    @km.tile_function(fidelity="HiFi4")
    def product(a, b):
        return a @ b

    @km.handler
    def column_sums(self, ones: km.Tensor, x: km.Tensor, t: km.Tensor):
        acc = km.tile.zeros(km.fp32, J)
        for ob, xb in zip(
            km.Stream(ones.tiles[0:1, 0:4], block=1),
            km.Stream(x.tiles[0:4, 0:J], block=J),
        ):
            acc += self.product(ob, xb)
        t.tiles[0:1, 0:J].write(acc)

    @km.handler
    def lanes(self, t: km.Tensor, out: km.Tensor):
        results = km.Stream(out.tiles[0:1, 0:J], write=True, block=J)
        for row in km.Stream(t.tiles[0:1, 0:J], block=J, row=J):
            results.push(lane_sum(row))


def tensor(rows, cols):
    return Tensor(Pages(4096), (32 * rows, 32 * cols), "float32", "tile")


def test_a_sum_over_rows_is_refused_naming_the_ones_product():
    with pytest.raises(Refused, match=r"dim=-2\) is not lowered .* tile of ones"):
        compile_class(Down, {}, {"t": tensor(1, J), "out": tensor(1, J)})


def test_the_ones_product_and_the_row_forms_sum_build():
    shapes = {"ones": tensor(1, 4), "x": tensor(4, J), "t": tensor(1, J)}
    compiled = compile_class(Sums, {}, {**shapes, "out": tensor(1, J)})
    assert {"column_sums", "lanes"} <= set(compiled.handlers)
