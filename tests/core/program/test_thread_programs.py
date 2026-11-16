"""A thread in a run of several programs (no card): the defines its C gets, the rule
of which program owns the threads, and host commands to the owner's thread."""

import pytest
from fake_port import FakeBackend, FakePort
from test_programs import GRID, made

from komorebi.errors import ProgramError
from komorebi.program import several as programs
from komorebi.program import thread as thread_module
from komorebi.session.run import Run


def a_thread(**options):
    return thread_module.Thread(
        sources=["t.c"], regions={"r": 64}, handlers={"big_step": 3}, **options
    )


def test_a_one_program_thread_s_defines_are_its_handlers_and_its_own():
    thread = a_thread(defines={"LIMIT": "7"})
    assert thread.all_defines() == {"kHandlerBigStep": "3u", "LIMIT": "7"}


def test_each_program_s_field_and_handlers_reach_the_thread(tmp_path):
    first, second = made(tmp_path, "first"), made(tmp_path, "two_parts")
    programs.plan([first, second], 120)
    assert a_thread().all_defines([first, second]) == {
        "kHandlerBigStep": "3u",
        "kProgramFirst": "0u",
        "kProgramFirstHandlerOne": "1u",
        "kProgramFirstHandlerTwo": "2u",
        "kProgramTwoParts": "1u",
        "kProgramTwoPartsHandlerOne": "1u",
        "kProgramTwoPartsHandlerTwo": "2u",
    }


def test_a_define_given_twice_or_not_a_c_name_is_refused(tmp_path):
    first, second = made(tmp_path, "first"), made(tmp_path, "second")
    programs.plan([first, second], 120)
    with pytest.raises(ValueError, match="kProgramFirst is given twice"):
        a_thread(defines={"kProgramFirst": "9"}).all_defines([first, second])
    dashed = made(tmp_path, "has-dash")
    programs.plan([first, dashed], 120)
    with pytest.raises(ValueError, match="not a C name"):
        thread_module.program_defines([first, dashed])


def test_any_one_program_may_own_the_threads_and_two_are_refused(tmp_path):
    first, second = made(tmp_path, "first"), made(tmp_path, "second")
    second.thread(cluster=1, sources=["t.c"], regions={"r": 64})
    plan = programs.plan([first, second], 120)
    assert plan.threads_owner() is second
    first.ports()
    with pytest.raises(ProgramError, match="belong to one program of a run"):
        programs.plan([first, second], 120)
    assert programs.plan([made(tmp_path, "third")], 120).threads_owner().name == (
        "third"
    )


def test_a_host_command_to_a_thread_goes_to_the_owner_s_thread(tmp_path):
    first, second = made(tmp_path, "first"), made(tmp_path, "second")
    second.thread(cluster=2, sources=["t.c"], regions={"r": 64}, handlers={"go": 4})
    plan = programs.plan([first, second], 120)
    port = FakePort()
    run = Run(
        FakeBackend(port),
        GRID,
        first,
        programs=plan.programs,
        holds=plan.holds(),
        thread_program=plan.threads_owner(),
    )
    run.submit("go", cluster=2)
    with pytest.raises(ValueError, match="second has no thread on cluster 1"):
        run.submit("go", cluster=1)
