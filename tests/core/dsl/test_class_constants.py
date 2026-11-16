"""Constants of a class (km.constant): the declaration on the class, the values on
each made program, derived constants under an override and under a subclass, the
build key, each refusal, and the class form of a handler with two programs of the
class in one run (no card)."""

from pathlib import Path

import numpy as np
import pytest

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import Refused, compile_class
from komorebi.dsl.nodes import prune, render
from komorebi.errors import ProgramError
from komorebi.session.run import Run


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def grid(rows, columns):
    """A bfloat16 tensor of tiles, rows x columns of them."""
    return Tensor(Pages(2048, rows * columns), (rows * 32, columns * 32), "bfloat16",
                  "tile")  # fmt: skip


def line_of(text):
    """The number of the line of this file that holds `text`."""
    lines = Path(__file__).read_text().split("\n")
    return next(n for n, line in enumerate(lines, start=1) if text in line)


def quiet(line):
    pass


TENSORS = {"x": grid(1, 8), "w": grid(8, 8), "y": grid(1, 8)}


def made(cls, **given):
    return cls(l1=1 << 20, log=quiet, **TENSORS, **given)


def movement_of(compiled, handler):
    return "\n".join(render(prune(compiled.movement[handler]), 0))


@km.tile_function
def project(x, w):
    return x @ w


class Linear(km.Program):
    K_TILES = km.constant(4)
    WIDTH = km.constant(2)
    CHUNK = km.constant(1)
    W_BLOCK = CHUNK * WIDTH  # derived: follows an override
    sums: km.L1[km.fp32, 1, WIDTH, 32, 32]

    @km.handler
    def split(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0 : self.K_TILES], block=self.CHUNK),
            km.Stream(w.tiles[0 : self.K_TILES, 0 : self.WIDTH], block=self.W_BLOCK),
        ):
            self.sums += project(xb, wb)
        y.tiles[0:1, 0 : self.WIDTH].write(self.sums)


class Wide(Linear):
    WIDTH = km.constant(4)


def test_a_declaration_on_the_class_and_its_value_on_a_program():
    assert isinstance(Linear.K_TILES, km.constant(1).__class__)
    assert Linear.K_TILES.name == "K_TILES" and Linear.K_TILES.default == 4
    program = made(Linear)
    assert (program.K_TILES, program.WIDTH, program.W_BLOCK) == (4, 2, 2)
    # the handler's lines take the program's values as a module constant's
    header = (program.built.folder / "linear.h").read_text()
    assert "kRead0Block = 1;" in header and "kRead1Block = 2;" in header
    compiled = compile_class(Linear, {}, TENSORS, constants=program.constant_values)
    assert "W(w), km::Walk{((0) * 8 + (0)), ((4) + 0) / 1, 8, 1, 8, 2," in (
        movement_of(compiled, "split")
    )


def test_an_override_by_keyword_or_by_the_plan():
    assert made(Linear, K_TILES=8).K_TILES == 8
    assert made(Linear, plan={"K_TILES": 8}).K_TILES == 8
    with pytest.raises(ProgramError, match="K_TILES is given twice"):
        made(Linear, K_TILES=8, plan={"K_TILES": 8})


def test_a_derived_constant_follows_an_override_and_a_subclass():
    assert made(Linear, WIDTH=4).W_BLOCK == 4
    assert made(Linear, WIDTH=4, CHUNK=2).W_BLOCK == 8
    wide = made(Wide)
    assert (wide.WIDTH, wide.W_BLOCK) == (4, 4)  # the parent's W_BLOCK, this WIDTH
    assert made(Wide, CHUNK=3).W_BLOCK == 12
    # the field's shape follows too: 1 x WIDTH tiles of fp32
    assert "float32, 1 x 4 x 32 x 32" in "\n".join(wide.built.report)


def test_two_programs_do_not_touch_each_other_or_the_class():
    small, large = made(Linear), made(Linear, K_TILES=8)
    assert (small.K_TILES, large.K_TILES) == (4, 8)
    assert Linear.K_TILES.default == 4 and Linear.W_BLOCK.left is Linear.CHUNK
    with pytest.raises(ProgramError, match="a constant is fixed when the program"):
        small.K_TILES = 16
    assert small.K_TILES == 4
    with pytest.raises(TypeError):
        small.constant_values["K_TILES"] = 16


def test_the_build_key_holds_the_constants():
    one, same, other = made(Linear), made(Linear, K_TILES=4), made(Linear, K_TILES=8)
    assert one.built.folder == same.built.folder
    assert one.built.folder != other.built.folder
    files = sorted(p.name for p in one.built.folder.iterdir())
    assert files == sorted(p.name for p in other.built.folder.iterdir())
    text = (other.built.folder / "movement.cpp").read_text()
    assert text != (one.built.folder / "movement.cpp").read_text()


def test_the_make_refuses_an_unknown_name_and_a_wrong_type():
    why = r"K_TILE=8 is not a constant of the class \(its constants: K_TILES, WIDTH, "
    with pytest.raises(ProgramError, match=why + r"CHUNK, W_BLOCK\)"):
        made(Linear, K_TILE=8)
    with pytest.raises(ProgramError, match=r"K_TILES=2.5: the constant is declared"):
        made(Linear, K_TILES=2.5)
    with pytest.raises(ProgramError, match=r"K_TILES=True: the constant is declared"):
        made(Linear, K_TILES=True)
    with pytest.raises(ProgramError, match="W_BLOCK is derived from the class's"):
        made(Linear, W_BLOCK=4)


def test_a_declaration_refuses_a_tuple_and_a_non_number():
    with pytest.raises(ProgramError, match="the compiler emits no table of constants"):
        km.constant((1, 2))
    with pytest.raises(ProgramError, match="an int, a float or a bool"):
        km.constant("8")


def test_a_parent_constant_rebound_to_a_plain_value_is_refused():
    with pytest.raises(ProgramError, match=r"Plain.WIDTH: a constant of a parent"):

        class Plain(Linear):
            WIDTH = 4


class Symbolic(km.Program):
    TILES = km.constant()  # no default: a symbol, the plan's or the build's
    ROWS = km.symbol()
    part: km.L1[km.fp32, ROWS, TILES, 32, 32]

    @km.handler
    def walk(self, x: km.Tensor):
        km.Stream(x.pages(0, self.ROWS), block=self.ROWS).drain()


def test_a_constant_without_a_default_is_a_symbol():
    program = Symbolic(l1=1 << 20, log=quiet, plan={"ROWS": 2}, x=grid(1, 8))
    assert program.ROWS == 2 and program.TILES > 1  # TILES solved by the build
    keyword = Symbolic(l1=1 << 20, log=quiet, ROWS=2, x=grid(1, 8))
    assert keyword.built.folder == program.built.folder
    report = "\n".join(program.built.report)
    assert "float32, ROWS = 2 x TILES = " in report


class Unsolved(km.Program):
    TILES = km.constant()
    part: km.L1[km.fp32, 1, TILES, 32, 32]

    @km.handler
    def walk(self, x: km.Tensor):
        km.Stream(x.pages(0, 2), block=self.TILES).drain()


def test_a_symbol_the_build_solves_is_not_known_to_the_lines():
    line = line_of("block=self.TILES).drain()")
    why = "self.TILES is a symbol the build solves after the handlers' lines"
    with pytest.raises(Refused, match=rf"Unsolved.walk, line {line}, .*{why}"):
        Unsolved(l1=1 << 20, log=quiet, x=grid(1, 8))


class Worded(km.Program):
    @km.handler
    def walk(self, x: km.Tensor, w: km.Tensor, tiles: int, columns: int):
        km.Stream(x.pages(0, tiles), block=tiles).drain()


class Cast(km.Program):
    @km.handler
    def walk(self, x: km.Tensor, w: km.Tensor, tiles: int, columns: int):
        down = km.Broadcast(self, along="column")
        if down.sender():
            ws = down.cast(x.tiles[0:1, columns : min(columns + tiles, 8)])
        else:
            ws = down.arrive()
        km.cxx("take(ws_buffer);", takes=[ws])


def test_a_word_of_the_command_where_a_constant_is_needed_names_the_tier():
    line = line_of("block=tiles).drain()")
    why = (
        "tiles is a word of the command, known only when the command runs, .*it "
        r"needs a constant of the class; declare TILES = km.constant\(n\) in the "
        "class and write self.TILES"
    )
    with pytest.raises(Refused, match=rf"Worded.walk, line {line}, .*{why}"):
        compile_class(Worded, {}, {"x": grid(1, 8), "w": grid(8, 8)})
    line = line_of("ws = down.cast(x.tiles[0:1, columns")
    with pytest.raises(Refused, match=rf"Cast.walk, line {line}, .*tiles is a word "
                       "of the command"):  # fmt: skip
        compile_class(Cast, {}, {"x": grid(1, 8), "w": grid(8, 8)})


class FakeRun:
    """What Run._handler reads of a run: its programs."""

    def __init__(self, programs):
        self.programs = programs

    _class_form_refused = Run._class_form_refused
    _program_told = Run._program_told
    program_field = Run.program_field


def test_the_class_form_with_two_programs_of_the_class_is_refused():
    small, large = made(Linear), made(Linear, K_TILES=8)
    run = FakeRun([small])
    assert Run._handler(run, Linear.split) == (int(Linear.split), 0)
    run = FakeRun([small, large])
    with pytest.raises(ValueError, match=r"Linear.split: the run holds 2 programs of "
                       r"the class \(Linear\(K_TILES=4, .*\) at field 0; "
                       r"Linear\(K_TILES=8, .*\) at field 1\), .*name it through the "
                       r"program \(program.split\)"):  # fmt: skip
        Run._handler(run, Linear.split)
    assert Run._handler(run, large.split) == (int(Linear.split), 1)
    run._holds = [np.array([True, True, False, False]), np.zeros(4, bool)]
    large.added_live = True
    with pytest.raises(ValueError, match=r"Linear\(K_TILES=4, .*\) at field 0 \(held "
                       r"by ranks 0-1\); Linear\(K_TILES=8, .*\) at field 1 \(added "
                       r"live, held by no core yet\)"):  # fmt: skip
        Run._handler(run, Linear.split)
