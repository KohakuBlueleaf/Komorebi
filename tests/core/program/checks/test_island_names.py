"""tt-metal's own page calls in an island build's movers: refused by name at the load
(the file and line), passed on a line that says `// km: own island`, a stale marker
refused, never looked at off islands; Program(own_island_pages=True), the variant, takes
every call as marked."""

import pytest

import komorebi as km
from komorebi.errors import ProgramError
from komorebi.program.checks import page_calls as island_names

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
MARKED = PAGES.replace("gen, 0);", "gen, 0);  // km: own island")


def program(folder, pages=PAGES, table=True):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "reader.cpp").write_text(READER)
    (folder / "pages.h").write_text(pages)
    (folder / "writer.cpp").write_text(KERNEL)
    (folder / "compute.cpp").write_text(KERNEL)
    return km.Program(
        "names", kernels=folder, defines={"KM_TENSOR_TABLE": 1} if table else {}
    )


def test_an_unmarked_call_in_an_included_file_is_refused_naming_the_marker(tmp_path):
    with pytest.raises(ProgramError) as caught:
        island_names.check([program(tmp_path / "a")])
    text = str(caught.value)
    assert "pages.h:3: noc_async_read_page reaches this island's DRAM only" in text
    assert "say `// km: own island` on the line" in text and "km::read_pages" in text
    assert "(Program(own_island_pages=True): every call of the program)" in text


def test_a_marked_line_passes_and_is_logged_and_one_card_is_not_looked_at(tmp_path):
    said = []
    island_names.check([program(tmp_path / "c", MARKED)], said.append)
    assert len(said) == 1 and "1 page call(s) marked `// km: own island`" in said[0]
    assert "pages.h:3: noc_async_read_page" in said[0]
    island_names.check([program(tmp_path / "d", table=False)], said.append)
    assert len(said) == 1


def test_a_stale_marker_is_refused_naming_its_line(tmp_path):
    # the marker on its own line, above the call: the call unmarked, the marker stale
    stale = "#pragma once\n// km: own island\n" + MARKED.split("\n", 1)[1]
    with pytest.raises(ProgramError) as caught:
        island_names.check([program(tmp_path / "e", stale)])
    assert "pages.h:2: `// km: own island` on a line that names none" in str(
        caught.value
    )


def test_comments_and_strings_are_not_calls(tmp_path):
    folder = tmp_path / "b"
    program(folder)
    (folder / "pages.h").write_text("#pragma once\n")
    assert island_names.found(folder / "reader.cpp") is None


def test_the_program_keyword_passes_an_unmarked_call_and_is_logged_with_the_count(
    tmp_path,
):
    """Program(own_island_pages=True), the marker's variant: every page call of its
    movers taken as marked; a stale marker still refused."""
    said = []
    own = program(tmp_path / "f")
    own.own_island_pages = True
    island_names.check([own], said.append)
    assert len(said) == 1
    assert "names: 1 page call(s) marked `// km: own island` or " in said[0]
    assert "taken so by own_island_pages=True (the first pages.h:3" in said[0]
    assert km.Program("names", kernels=".", own_island_pages=True).own_island_pages
    stale = "#pragma once\n// km: own island\n" + PAGES.split("\n", 1)[1]
    own = program(tmp_path / "g", stale)
    own.own_island_pages = True
    with pytest.raises(ProgramError, match="pages.h:2: `// km: own island` on a line"):
        island_names.check([own])
