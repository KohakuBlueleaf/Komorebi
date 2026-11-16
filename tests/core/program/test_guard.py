"""guard: the define it gives a program's kernels, a class's build key, what a run built
without the guard refuses or says."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from komorebi.dsl.program import sources_key
from komorebi.errors import KomorebiError, ProgramError
from komorebi.program import several as programs
from komorebi.program.description import Program, guard_says, unguarded
from komorebi.session.kernels import guard_path
from komorebi.session.run import Run

CORE = Path(__file__).parents[3] / "src/komorebi_native/include/komorebi/device"


def program(**kwargs):
    return Program(name=kwargs.pop("name", "cast"), kernels="kernels", **kwargs)


def test_on_by_default_adds_no_define():
    made = program(defines={"OTHER": 1})
    assert made.guard is True
    assert made.defines == {"OTHER": 1}  # the header's default: KM_GUARD 1


def test_off_defines_zero():
    made = program(guard=False, defines={"OTHER": 1})
    assert made.defines == {"OTHER": 1, "KM_GUARD": 0}


def test_define_against_the_keyword_refused():
    with pytest.raises(ProgramError, match="KM_GUARD=0 and guard=True"):
        program(defines={"KM_GUARD": 0})
    with pytest.raises(ProgramError, match="KM_GUARD=1 and guard=False"):
        program(guard=False, defines={"KM_GUARD": 1})
    assert program(guard=False, defines={"KM_GUARD": 0}).defines == {"KM_GUARD": 0}
    assert program(defines={"KM_GUARD": 1}).defines == {"KM_GUARD": 1}


def test_class_key_differs_off_and_keeps_its_value_on():
    made = ({"BLOCK": 4}, {}, {"W": 1000}, 4, None, None)
    assert sources_key(*made) == sources_key(*made, guard=True)
    assert sources_key(*made) != sources_key(*made, guard=False)
    assert sources_key(*made, True) != sources_key(*made, guard=False)


def test_the_header_default_and_its_one_switch_macro():
    text = (CORE / "core" / "km_core.hpp").read_text()
    assert "#if !defined(KM_GUARD)\n#define KM_GUARD 1\n#endif" in text
    assert "#define KM_GUARDED(...) __VA_ARGS__" in text


def test_programs_that_share_the_runtimes_kernels_agree():
    on, off, other = program(name="sum"), program(name="scale", guard=False), program()
    programs.same_guard([on, other])
    programs.same_guard([off])
    with pytest.raises(ProgramError, match="sum has guard=True and scale guard=False"):
        programs.same_guard([on, off])
    assert unguarded([on, off, other]) == ["scale"]


def test_the_runtimes_kernels_get_the_define_when_every_program_is_off():
    assert guard_path([program(guard=False)]) == [("KM_GUARD", "0")]
    assert guard_path([program()]) == []


def test_inspect_is_refused_naming_the_program():
    run = SimpleNamespace(programs=[program(name="sum", guard=False)])
    with pytest.raises(KomorebiError, match=r"run.inspect: sum built with guard=False"):
        Run.inspect(run)


def test_the_load_says_what_a_hang_costs():
    said = guard_says(["sum"], recover=True)
    assert "sum" in said and "not given up at a deadline" in said
    assert "card reset" in said and "run.inspect refuses" in said
    assert "reset lists" in said
    assert "reset lists" not in guard_says(["sum"], recover=False)
