"""Movement on the host: the pattern word, the movement form of a program's kernels, and
buffer pairs with the check of the halves' roles in the sources."""

from fractions import Fraction

import pytest

import komorebi as km
from komorebi.errors import ProgramError


def write(folder, **sources):
    for name, text in sources.items():
        (folder / f"{name}.cpp").write_text(text)
    return folder


def test_a_pattern_gives_its_word_and_the_movers_shares():
    pattern = km.Pattern("AAAB")
    assert pattern.word == 4 | 0b1000 << 8 and pattern.define() == "0x804"
    assert pattern.share("A") == Fraction(3, 4) and pattern.blocks("B", 9) == 2
    for letters in ("", "AC", "A" * 25):
        with pytest.raises(ValueError, match="letters A and B"):
            km.Pattern(letters)


def test_one_movement_source_serves_both_data_movers(tmp_path):
    folder = write(tmp_path, movement="", compute="")
    program = km.Program("p", kernels=folder, compile_args={"movement": [7]})
    sources = program.sources()
    assert sources["reader"] == sources["writer"] == folder / "movement.cpp"
    assert program.kernel_compile_args("writer") == [7]
    assert program.kernel_compile_args("compute") == []
    named = {"movement": folder / "movement.cpp", "compute": folder / "compute.cpp"}
    assert km.Program("p", kernels=named).sources()["reader"] == named["movement"]
    write(folder, reader="")
    with pytest.raises(ProgramError, match="keep one form"):
        km.Program("p", kernels=folder).sources()
    with pytest.raises(ProgramError, match="not both"):
        km.Program("p", kernels=folder, opt={"movement": "Os", "reader": "O2"})


def test_a_pair_is_two_buffers_and_a_half_keeps_one_filler_and_one_drainer(tmp_path):
    pair = km.BufferPair(3, 4, "bfloat16", 2048, (6, 2))
    program = km.Program("p", kernels=tmp_path, buffers=[pair])
    assert program.buffers == [
        km.Buffer(3, "bfloat16", 2048, 6),
        km.Buffer(4, "bfloat16", 2048, 2),
    ]
    with pytest.raises(ProgramError, match="declared twice"):
        km.Program("p", kernels=tmp_path, buffers=[pair, km.Buffer(4, "uint32", 64, 1)])
    header = "constexpr uint32_t kHalfB = 4;\n"
    (tmp_path / "pairs.h").write_text(header)
    compute = '#include "pairs.h"\ncb_wait_front(kHalfB, 1); cb_pop_front(kHalfB, 1);\n'
    write(tmp_path, movement="", compute=compute)
    assert program.sources()
    write(tmp_path, compute=compute + "cb_push_back(kHalfB, 1);\n")
    with pytest.raises(ProgramError, match="both fills and drains buffer 4"):
        program.sources()
