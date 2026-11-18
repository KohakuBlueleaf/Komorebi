"""A handler's constant parameters (p: km.constant[int] = n): not a command word;
the default built when no set is listed, each listed set a handler of its own; the
handler built for a set picked, an unlisted set refused; each refusal (no card)."""

from pathlib import Path

import pytest
from class_handler_constants_fixture import TENSORS, Gate, grid, quiet

import komorebi as km
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.first_use import constant_handler
from komorebi.errors import ProgramError
from komorebi.session.run import Run


def line_of(text):
    """The number of the line of this file that holds `text`."""
    lines = Path(__file__).read_text().split("\n")
    return next(n for n, line in enumerate(lines, start=1) if text in line)


def made(**given):
    return Gate(l1=1 << 20, log=quiet, **TENSORS, **given)


def test_a_constant_parameter_is_not_a_command_word_and_its_default_is_built():
    program = made()
    assert Gate.gate.words == [("a", "tensor"), ("out", "tensor"), ("tiles", "int")]
    assert program.handlers == {"gate": 1, "other": 2}
    assert constant_handler(program, Gate.gate, {}) == (1, Gate.gate)
    assert Gate.gate.args(TENSORS["a"], TENSORS["out"], 3)[2] == 3


def test_each_listed_set_is_a_handler_of_its_own():
    program = made(handlers={"gate": [{"size": 4}, {"size": 2}]})
    assert program.handlers == {"gate_size4": 1, "gate_size2": 3, "other": 2}
    assert program.defines["KM_RING_PADDED"] == "0x3"
    number, built = constant_handler(program, Gate.gate, {"size": 2})
    assert number == 3 and built.name == "gate_size2"
    assert built.words == [("a", "tensor"), ("out", "tensor"), ("tiles", "int")]
    assert program.built.folder != made().built.folder  # the sets are in the key


def test_a_set_not_built_is_refused_naming_the_sets_that_were():
    program = made(handlers={"gate": [{"size": 4}, {"size": 2}]})
    why = (
        r"Gate.gate with size=8: no handler was built for it \(built: size=4; "
        r"size=2\); list it at the make, handlers="
    )
    with pytest.raises(ProgramError, match=why):
        constant_handler(program, Gate.gate, {})
    with pytest.raises(ProgramError, match="width is not a constant parameter of it"):
        constant_handler(program, Gate.gate, {"width": 2})


def test_the_make_refuses_a_set_it_cannot_build():
    with pytest.raises(ProgramError, match="names 'gates', not a handler"):
        made(handlers={"gates": [{"size": 4}]})
    with pytest.raises(ProgramError, match="Gate.other: handlers= lists sets of a"):
        made(handlers={"other": [{"size": 4}]})
    with pytest.raises(ProgramError, match="width is not a constant parameter of it"):
        made(handlers={"gate": [{"width": 4}]})
    with pytest.raises(ProgramError, match="gate's size=2.5: the constant is declared"):
        made(handlers={"gate": [{"size": 2.5}]})
    with pytest.raises(ProgramError, match=r"the set \{'size': 8\} is listed twice"):
        made(handlers={"gate": [{"size": 8}, {}]})


class Undefaulted(km.Program):
    @km.handler
    def gate(self, a: km.Tensor, size: km.constant[int], tiles: int = 2):
        km.Stream(a.pages(0, tiles), block=size).drain()


def test_a_constant_parameter_takes_a_default():
    line = line_of("def gate(self, a: km.Tensor, size: km.constant[int], tiles")
    why = "size is a constant parameter: the handler is built for its default"
    with pytest.raises(Refused, match=rf"Undefaulted.gate, line {line}, .*{why}"):
        compile_class(Undefaulted, {}, {"a": grid(1, 8)})


def test_a_constant_parameter_is_of_int_float_or_bool():
    with pytest.raises(ProgramError, match="a constant parameter is of int, float or"):
        km.constant[str]


class FakeRun:
    """What Run._built_for reads of a run: its programs."""

    def __init__(self, programs):
        self.programs = programs

    _built_for = Run._built_for
    _handler = Run._handler
    _class_form_refused = Run._class_form_refused
    _program_told = Run._program_told
    program_field = Run.program_field


def test_a_submit_gives_the_values_at_their_places_or_leaves_them_out():
    program = made(handlers={"gate": [{"size": 4}, {"size": 8}]})
    run = FakeRun([program])
    a, out = TENSORS["a"], TENSORS["out"]
    built = run._built_for(Gate.gate, (a, out, 3, 4))  # size 4, at its place
    assert int(built) == 1 and built.method.name == "gate_size4"
    assert built.args(a, out, 3, 4) == built.args(a, out, 3)  # no word of it
    assert len(built.args(a, out, 3, 4)) == 3
    built = run._built_for(program.gate, (a, out, 3))  # left out: the default, 8
    assert int(built) == 3 and built.method.name == "gate_size8"
    why = (
        r"Gate.gate with size=2: no handler was built for it \(built: size=4; "
        r"size=8\)"
    )
    with pytest.raises(ProgramError, match=why):
        run._built_for(Gate.gate, (a, out, 3, 2))
    assert run._built_for(Gate.other, (a,)) is Gate.other  # no constant parameter
