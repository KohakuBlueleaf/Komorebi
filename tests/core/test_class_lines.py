"""The class's defines, bodies at a kernel's start and end and compute's start-up: what
each lowers to and each refusal (no card)."""

from pathlib import Path

import pytest

import komorebi as km
from komorebi.dsl.compiler import Refused, compile_class
from komorebi.dsl.nodes import prune, render
from komorebi.dsl.program import compute_source, movement_source
from komorebi.errors import ProgramError
from komorebi.tensor import Tensor


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor(dtype):
    return Tensor(
        Pages(2048 if dtype == "bfloat16" else 4096), (2048, 32), dtype, "tile"
    )


def words():
    """A tensor of word pages, 1,024 words a page."""
    return Tensor(Pages(4096), (8, 1024), "uint32", "row")


def line_of(text):
    """The number of the line of this file that holds `text`."""
    lines = Path(__file__).read_text().split("\n")
    return next(n for n, line in enumerate(lines, start=1) if text in line)


def texts(cls, handler, **tensors):
    """(movement, compute) C++ of one handler of `cls`."""
    compiled = compile_class(cls, {}, tensors)
    return (
        "\n".join(render(prune(compiled.movement[handler]), 0)),
        "\n".join(render(prune(compiled.compute[handler]), 0)),
    )


class Defined(km.Program, defines={"SCALE": 3, "MODE": "fast"}):
    @km.handler
    def copy(self, source: km.Tensor):
        km.Stream(source.pages(0, 1)).drain()


def test_defines_of_the_class_and_of_the_make():
    made = Defined(source=tensor("bfloat16"), l1=1 << 20, log=lambda line: None)
    assert made.defines == {"SCALE": "3", "MODE": "fast"}
    made = Defined(
        source=tensor("bfloat16"), l1=1 << 20, log=lambda line: None,
        defines={"SCALE": 4, "CAUSAL": 1},
    )  # fmt: skip
    assert made.defines == {"SCALE": "4", "MODE": "fast", "CAUSAL": "1"}
    with pytest.raises(ProgramError, match="not one of the runtime's"):
        Defined(
            source=tensor("bfloat16"), l1=1 << 20, log=lambda line: None,
            defines={"KM_PAGE_LINKS": 1},
        )  # fmt: skip
    with pytest.raises(ProgramError, match="an int or a string"):
        Defined(
            source=tensor("bfloat16"), l1=1 << 20, log=lambda line: None,
            defines={"X": 1.5},
        )  # fmt: skip


class Started(km.Program, startup=km.Startup("ones", "ones", "ones", order="reverse")):
    ones: km.L1[km.bf16, 1, 32, 32] = km.own(push=None)

    @km.handler
    def copy(self, source: km.Tensor):
        km.cxx("build_ones(ones);", takes=[self.ones], at="start")
        km.cxx("restore();", at="end")
        km.cxx("noc_async_write_barrier();", on="mover_b", at="end")
        km.Stream(source.pages(0, 1)).drain()


def test_bodies_at_the_kernels_start_and_end_and_the_startup():
    compiled = compile_class(Started, {}, {"source": tensor("bfloat16")})
    compute = compute_source(compiled)
    startup = "compute_kernel_hw_startup<SrcOrder::Reverse>(kFieldOnes, kFieldOnes"
    assert startup in compute
    assert compute.index(startup) < compute.index("build_ones(ones);")
    assert compute.index("build_ones(ones);") < compute.index("compute.run(")
    assert compute.index("restore();") > compute.rindex("});")
    assert "const uint32_t ones = kFieldOnes;" in compute
    movement = movement_source(compiled)
    end = movement[movement.rindex("});") :]
    assert "#if !defined(COMPILE_FOR_BRISC)" in end
    assert "noc_async_write_barrier();" in end
    assert "restore();" not in movement


class StartTakes(km.Program):
    @km.handler
    def copy(self, source: km.Tensor, scale: int):
        km.cxx("use(scale);", takes=[scale], at="start")


class StartAt(km.Program):
    @km.handler
    def copy(self, source: km.Tensor):
        km.cxx("use();", at="middle")


def test_start_and_end_refusals():
    line = line_of('km.cxx("use(scale);", takes=[scale], at="start")')
    with pytest.raises(Refused, match=rf"line {line}, .*own fields alone"):
        compile_class(StartTakes, {}, {"source": tensor("bfloat16")})
    line = line_of('km.cxx("use();", at="middle")')
    with pytest.raises(Refused, match=rf"line {line}, .*the kernel's start or its"):
        compile_class(StartAt, {}, {"source": tensor("bfloat16")})
    with pytest.raises(ProgramError, match="a source order is of two inputs"):
        km.Startup("a", "out", order="reverse")
