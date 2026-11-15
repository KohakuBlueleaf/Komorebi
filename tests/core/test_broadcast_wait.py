"""broadcast_wait: the define it gives a program's kernels and a class's build key."""

import pytest

from komorebi.dsl.program import sources_key
from komorebi.errors import ProgramError
from komorebi.program import Program


def program(**kwargs):
    return Program(name="cast", kernels="kernels", **kwargs)


def test_on_by_default_adds_no_define():
    made = program(defines={"OTHER": 1})
    assert made.broadcast_wait is True
    assert made.defines == {"OTHER": 1}  # the header's default: KM_BROADCAST_WAIT 1


def test_off_defines_zero():
    made = program(broadcast_wait=False, defines={"OTHER": 1})
    assert made.defines == {"OTHER": 1, "KM_BROADCAST_WAIT": 0}


def test_define_against_the_keyword_refused():
    with pytest.raises(ProgramError, match="KM_BROADCAST_WAIT=1 and broadcast_wait"):
        program(broadcast_wait=False, defines={"KM_BROADCAST_WAIT": 1})
    with pytest.raises(ProgramError, match="KM_BROADCAST_WAIT=0 and broadcast_wait"):
        program(defines={"KM_BROADCAST_WAIT": 0})
    assert program(broadcast_wait=False, defines={"KM_BROADCAST_WAIT": 0}).defines == {
        "KM_BROADCAST_WAIT": 0
    }


def test_class_key_differs_off_and_keeps_its_value_on():
    made = ({"BLOCK": 4}, {}, 1000, 4, None, None)
    assert sources_key(*made) == sources_key(*made, True)
    assert sources_key(*made) != sources_key(*made, False)
