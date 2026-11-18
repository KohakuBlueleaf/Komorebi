"""A row's largest and smallest value and its argmax in a tile function (no card): the
row form, a wide row's chunks and a row read in parts (SFPSWAP a lane pair); a field's
columns read in every sum pass; the refusals; a constant read in two lane-flag runs
loaded in front of them; a row-form reduction of a resident row built in levels."""

import pytest
from class_lines_fixture import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import compute_source
from komorebi.dsl.tile import trace
from komorebi.dsl.tile.ir import PlanError
from komorebi.dsl.tile.lower import Body
from komorebi.dsl.tile.schedule import interleave

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


def test_a_constant_read_across_flag_runs_is_loaded_in_front_of_them():
    b = Body()
    x, y = b.load(0, "x"), b.load(2, "y")
    far = b.const(1e9)  # picked where x > y and again where x < y (==)
    out = b.op("SFPMOV", "x", lreg_c=x)
    for name, (left, right) in (("x gt y", (x, y)), ("x lt y", (y, x))):
        b.when("SFPGT", name, lreg_c=right, lreg_dest=left, instr_mod1=1)
        b.op("SFPMOV", "over", lreg_c=far, lreg_dest=out)
        b.end()
    b.store(out, 4)
    interleave(b.code, 0, len(b.code))
    ops = [insn.op for insn in b.code]
    first_test = ops.index("SFPGT")
    assert all(i < first_test for i, op in enumerate(ops) if op == "SFPLOADI")


@km.tile_function
def column_min(row, cols):
    return km.reduce_min(row * 0.0 + cols.row()).col()


class ColumnMin(km.Program):
    cols: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, c: km.Tensor, out: km.Tensor):
        self.cols.fill(km.Stream(c.pages(0, 1)))
        results = km.Stream(out.pages(0, 4), write=True, block=2)
        for row in km.Stream(x.pages(0, 4), block=2, row=2):
            results.push(column_min(row, self.cols))


def test_a_reduction_of_a_resident_row_reads_its_columns():
    tensors = {n: Tensor(Pages(4096), (64, 64), "float32", "tile") for n in "x"}
    tensors["out"] = tensors["x"]
    tensors["c"] = Tensor(Pages(4096), (32, 32), "float32", "tile")
    compiled = compile_class(ColumnMin, {}, tensors)
    ((_, unit),) = compiled.units.values()
    loads = [
        x.args["dest_reg_addr"]
        for x in unit.tile["math"]
        if getattr(x, "op", None) == "SFPLOAD"
    ]
    assert any(256 <= a < 320 for a in loads)  # the per-column tile (Dst tile 4)
