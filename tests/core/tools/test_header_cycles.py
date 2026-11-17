"""The header-cycle check: which include lines make a cycle."""

from komorebi.tools import header_cycles


def test_a_late_include_back_is_a_cycle():
    sources = {
        "a.hpp": '#pragma once\n#include "b.hpp"\n',
        "b.hpp": '#pragma once\n#include <stdint.h>\n#include "../links/c.hpp"\n',
        "c.hpp": '#pragma once\nint c();\n#include "a.hpp"  // late\n',
        "d.hpp": '#include "a.hpp"\n',
    }
    graph = header_cycles.includes(sources)
    assert graph["b.hpp"] == [("c.hpp", 3)]  # stdint.h is not one of the headers
    assert header_cycles.cycles(graph) == [["a.hpp", "b.hpp", "c.hpp"]]


def test_includes_in_one_direction_are_no_cycle():
    sources = {
        "core.hpp": "#pragma once\n",
        "pages.hpp": '#include "core.hpp"\n',
        "worker.hpp": '#include "pages.hpp"\n#include "core.hpp"\n',
    }
    assert header_cycles.cycles(header_cycles.includes(sources)) == []


def test_the_repository_headers_have_no_cycle():
    root = header_cycles.repo_root().resolve() / header_cycles.INCLUDE
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in root.rglob("*")
        if path.suffix in header_cycles.SUFFIXES
    }
    assert sources and header_cycles.cycles(header_cycles.includes(sources)) == []
