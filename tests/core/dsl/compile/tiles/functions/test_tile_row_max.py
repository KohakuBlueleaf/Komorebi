"""A row's largest and smallest value and its argmax in a tile function (no card): the
row form (rows of 1 and 2 tiles), the chunks of a wide row (an odd one too) and a row
read in parts, each lane pair combined by SFPSWAP; the argmax's reductions reading a
field's row of column numbers in every sum pass; the refusals of the row form's
reductions."""

import pytest
from class_lines_fixture import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import compute_source
from komorebi.dsl.tile import trace
from komorebi.dsl.tile.ir import PlanError

NOWHERE = 1e9  # an argmax's column where the row's largest value is not


@km.tile_function
def row_max(row):
    return km.reduce_max(row, exact=True).col()


@km.tile_function
def row_min(row):
    return km.reduce_min(row).col()


@km.tile_function
def argmax(row, cols):
    m = km.reduce_max(row, exact=True)
    return km.reduce_min(km.where(row == m.col(), cols.row(), NOWHERE)).col()


def make(width, part):
    """A class over rows of `width` tiles read in blocks of `part`: a handler a
    function."""

    class Rows(km.Program):
        WIDTH = km.constant(width)
        PART = km.constant(part)
        CHUNKS = km.constant((width + 1) // 2)
        cols: km.L1[km.fp32, (width + 1) // 2, 32, 32]

        @km.handler
        def run_max(self, x: km.Tensor, out: km.Tensor):
            results = km.Stream(
                out.pages(0, 2 * self.WIDTH), write=True, block=self.PART
            )
            for row in km.Stream(
                x.pages(0, 2 * self.WIDTH), block=self.PART, row=self.WIDTH
            ):
                results.push(row_max(row))

        @km.handler
        def run_min(self, x: km.Tensor, out: km.Tensor):
            results = km.Stream(
                out.pages(0, 2 * self.WIDTH), write=True, block=self.PART
            )
            for row in km.Stream(
                x.pages(0, 2 * self.WIDTH), block=self.PART, row=self.WIDTH
            ):
                results.push(row_min(row))

        @km.handler
        def run_argmax(self, x: km.Tensor, c: km.Tensor, out: km.Tensor):
            self.cols.fill(km.Stream(c.pages(0, self.CHUNKS)))
            results = km.Stream(
                out.pages(0, 2 * self.WIDTH), write=True, block=self.PART
            )
            for row in km.Stream(
                x.pages(0, 2 * self.WIDTH), block=self.PART, row=self.WIDTH
            ):
                results.push(argmax(row, self.cols))

    return Rows


def built(width, part):
    tensors = {
        n: Tensor(Pages(4096), (64, 32 * width), "float32", "tile")
        for n in ("x", "out")
    }
    tensors["c"] = Tensor(Pages(4096), (32 * ((width + 1) // 2), 32), "float32", "tile")
    constants = {"WIDTH": width, "PART": part, "CHUNKS": (width + 1) // 2}
    return compile_class(make(width, part), {}, tensors, constants=constants)


def case_of(source, name):
    """The compute source's case of handler run_<name>."""
    at = source.index(f"case kHandlerRun{name.capitalize()}")
    end = source.find("case kHandler", at + 1)
    return source[at : end if end > 0 else len(source)]


@pytest.mark.parametrize("width,part", [(2, 2), (4, 4), (5, 5), (32, 8)])
def test_each_form_builds_with_lane_swaps(width, part):
    compiled = built(width, part)
    units = {cxx: unit for cxx, (_, unit) in compiled.units.items()}
    assert sorted(units) == ["argmax", "row_max", "row_min"]
    for cxx, unit in units.items():
        words = [x.op for x in unit.tile["math"] if hasattr(x, "op")]
        words += [x.op for p in unit.parts.values() for x in p if hasattr(x, "op")]
        assert "SFPSWAP" in words  # the lanes and the chunks combined by max / min
        assert ("SFPGT" in words) == (cxx == "argmax")  # where(row == max, ...)


def test_the_argmax_reads_the_column_field_in_every_sum_pass():
    case = case_of(compute_source(built(32, 8)), "argmax")
    sums = case[: case.index("} else {", case.index("km_tile::argmax_sum0"))]
    assert "copy_tile(kFieldCols, km_part0 * 4 + km_chunk0, 4);" in sums
    source = compute_source(built(5, 5))  # chunks of 2 and a last of 1
    case = case_of(source, "argmax")
    assert "copy_tile(kFieldCols, km_chunk0, 4);" in case
    assert "copy_tile(kFieldCols, 2, 4);" in case
    assert "kFieldCols" not in case_of(source, "max")  # no field read: none copied


def test_the_row_form_refuses_other_reductions():
    with pytest.raises(PlanError, match="reduce_min is on the vector unit"):
        trace.reduce_min(None, exact=False)
    with pytest.raises(PlanError, match="reduce_min is along the last dim"):
        trace.reduce_min(None, dim=0)
