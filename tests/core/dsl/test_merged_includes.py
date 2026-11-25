"""A body's include= in a merge (no card): a relative path means what it means from
its class's own build folder, so a merged program's kernels, built beside the first
class's file, name the same files; a class built alone keeps its text."""

import importlib.util
import re
import sys

import komorebi as km
from komorebi.device.tensor import Tensor

FIRST = """
import komorebi as km


class First(km.Program):
    @km.handler
    def one(self, a: km.Tensor):
        for ia in km.Stream(a.items(first=0, count=8, size=256)):
            km.cxx("one_k::use(ia);", takes=[ia], on="pack", include=["../../one.hpp"])
"""
SECOND = """
import komorebi as km


class Second(km.Program):
    @km.handler
    def two(self, a: km.Tensor):
        for ia in km.Stream(a.items(first=0, count=8, size=256)):
            km.cxx(
                '#include "../../two.hpp"\\ntwo_k::use(ia);',
                takes=[ia],
                on="pack",
                include=["../../../two_more.hpp", "<stdint.h>"],
            )
"""
BOTH = """
import komorebi as km
from first import First
from second import Second


class Both(First, Second):
    pass
"""


class Pages:
    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 4096, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def words():
    return Tensor(Pages(1024), (4096, 256), "uint32", "row")


def module(path, name, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def quiet(line):
    pass


def included(built):
    """Each quoted #include of the built kernels, as a path from the build folder."""
    found = []
    for name in ("movement.cpp", "compute.cpp"):
        found += re.findall(r'^#include "(\.\.?/[^"]*)"$', built.files[name], re.M)
    return found


def test_a_merge_keeps_each_class_s_relative_includes_pointing_at_its_files(tmp_path):
    first = module(tmp_path / "a" / "first.py", "first", FIRST)
    second = module(tmp_path / "b" / "c" / "d" / "second.py", "second", SECOND)
    (tmp_path / "a" / "one.hpp").write_text("#pragma once\n")
    (tmp_path / "b" / "c" / "d" / "two.hpp").write_text("#pragma once\n")
    (tmp_path / "b" / "c" / "two_more.hpp").write_text("#pragma once\n")
    try:
        both = module(tmp_path / "both.py", "both", BOTH)
        alone = [
            cls(l1=1 << 20, log=quiet, a=words())
            for cls in (first.First, second.Second)
        ]
        for program, wanted in zip(
            alone, (["../../one.hpp"], ["../../two.hpp", "../../../two_more.hpp"])
        ):
            assert sorted(included(program.built)) == sorted(wanted)  # as written
        merged = km.merge(*alone, l1=1 << 20, log=quiet)
        derived = both.Both(l1=1 << 20, log=quiet, a=words())
        for program in (merged, derived):
            folder = program.built.folder
            paths = included(program.built)
            assert len(paths) == 3, paths
            for path in paths:
                assert (folder / path).resolve().is_file(), (folder, path)
            text = program.built.files["compute.cpp"]
            assert "#include <stdint.h>" in text
    finally:
        for name in ("first", "second", "both"):
            sys.modules.pop(name, None)
        for path in (tmp_path / "a", tmp_path / "b" / "c" / "d", tmp_path):
            if str(path) in sys.path:
                sys.path.remove(str(path))
