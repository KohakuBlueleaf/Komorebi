"""A class's thread in a load of several programs: the header with each class
program's parts and a load function, their tensors as the thread's words, the C
naming another program's parts built at that load, not at the make (no card)."""

from pathlib import Path

import pytest
from class_thread_header_fixture import COMPILER, HERE, Plain, Sums, grid

import komorebi as km
from komorebi.errors import ProgramError


class Totals(Sums):
    counts = km.region(4096)
    thread = km.x280_thread(str(HERE / "programs.c"))


def header(program):
    thread = program.threads[min(program.threads)]
    return (Path(thread.includes[0]) / "totals_thread.h").read_text()


@COMPILER
def test_the_header_of_a_load_has_the_other_programs_parts():
    said = []
    totals = Totals(l1=1 << 20, log=said.append, thread_words=[9])
    assert any("names plain_load, plain_scale" in line for line in said)
    a, out = grid(8, 0x1000), grid(8, 0x9000)
    plain = Plain(l1=1 << 20, log=said.append, a=a, out=out)
    totals.program_field, plain.program_field = 0, 1
    thread = totals.thread
    thread.programs_header([totals, plain])
    text = header(totals)
    assert thread.words == [9, 0x1000, 0x9000]  # its word, then Plain's tensors
    assert "#define PLAIN_WORDS_OWN 1u" in text
    assert "static inline void plain_scale(" in text
    assert "    return kmx_load(PLAIN_PROGRAM, first, count, deadline, event);" in text
    assert "static inline int totals_load(" in text
    assert any("programs.c built with totals_thread.h" in line for line in said)
    thread.programs_header(())  # alone again: the make's header and words
    assert thread.words == [9] and "plain_scale" not in header(totals)


@COMPILER
def test_c_naming_this_program_s_lacks_is_still_refused_at_the_make():
    class Totals(Sums):  # the C's names
        counts = km.region(4096)
        thread = km.x280_thread(str(HERE / "lacks.c"))

    with pytest.raises(
        ProgramError, match="totals_nothing: the program has no handler"
    ):
        Totals(l1=1 << 20, log=print)
