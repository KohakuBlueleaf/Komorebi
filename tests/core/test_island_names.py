"""tt-metal's own page calls in an island build's movers: refused by name at the load
(the file and line), passed with own_island_pages=True, never looked at off islands."""

import pytest

import komorebi as km
from komorebi import island_names
from komorebi.errors import ProgramError

KERNEL = "void kernel_main() {}\n"
READER = """\
#include "pages.h"
// noc_async_read_page(0, gen, 0): a comment
/* InterleavedAddrGen<true> in a block
   comment */
const char* said = "noc_async_write_page";
void kernel_main() {}
"""
PAGES = """\
#pragma once
inline void own(uint32_t page) {
    noc_async_read_page(page, gen, 0);
}
"""


def program(folder, own=False, table=True):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "reader.cpp").write_text(READER)
    (folder / "pages.h").write_text(PAGES)
    (folder / "writer.cpp").write_text(KERNEL)
    (folder / "compute.cpp").write_text(KERNEL)
    return km.Program(
        "names",
        kernels=folder,
        defines={"KM_TENSOR_TABLE": 1} if table else {},
        own_island_pages=own,
    )


def test_a_call_in_an_included_file_is_refused_with_its_file_and_line(tmp_path):
    with pytest.raises(ProgramError) as caught:
        island_names.check([program(tmp_path / "a")])
    text = str(caught.value)
    assert "pages.h:3: noc_async_read_page reaches this island's DRAM only" in text
    assert "km::read_pages" in text and "Program(own_island_pages=True)" in text


def test_comments_and_strings_are_not_calls(tmp_path):
    folder = tmp_path / "b"
    program(folder)
    (folder / "pages.h").write_text("#pragma once\n")
    assert island_names.found(folder / "reader.cpp") is None


def test_the_way_out_is_logged_and_one_card_is_not_looked_at(tmp_path):
    said = []
    island_names.check([program(tmp_path / "c", own=True)], said.append)
    assert len(said) == 1 and "own_island_pages=True: meant" in said[0]
    island_names.check([program(tmp_path / "d", table=False)], said.append)
    assert len(said) == 1
