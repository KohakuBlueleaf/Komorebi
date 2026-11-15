"""The kernel buffer remembered for a script's next run (B): kept by script and by
program in the build cache; the open at that size; grown during a run, set at close."""

import sys
import types
from pathlib import Path

import pytest

from komorebi import remembered


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(remembered, "_path", lambda: tmp_path / remembered.FILE)
    main = types.ModuleType("__main__")
    main.__file__ = str(tmp_path / "script.py")
    monkeypatch.setitem(sys.modules, "__main__", main)
    return tmp_path


def test_nothing_remembered_opens_at_the_runtimes(cache):
    said = []
    assert remembered.Memory(said.append).size is None
    assert said == ["kernel buffer: the runtime's at the open (nothing remembered)"]


def test_a_run_grows_the_entry_and_its_close_sets_it(cache):
    first = remembered.Memory(lambda message: None)
    first.wanted(92496)  # a load refused: the next run opens right
    first.largest = 0
    first.closed()  # nothing loaded: the entry stays
    assert remembered.read()["scripts"][first.script] == 93568
    said = []
    second = remembered.Memory(said.append)
    assert second.size == 93568
    assert said == ["kernel buffer 93,568 B at the open (remembered for this script)"]
    second.largest = 70656
    second.closed()  # this run needed less: the entry shrinks
    assert remembered.read()["scripts"][second.script] == 70656


def test_a_corrupt_file_is_no_entry_logged_once(cache):
    (cache / remembered.FILE).write_text('{"scripts": {"')
    said = []
    assert remembered.Memory(said.append).size is None
    remembered.Memory(said.append)
    assert sum("unreadable" in line for line in said) == 1


def test_code_without_a_file_keeps_nothing(cache, monkeypatch):
    monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))
    said = []
    made = remembered.Memory(said.append)
    assert made.script is None and "no script file" in said[0]
    made.wanted(92496)
    assert remembered.read()["scripts"] == {}


def test_a_load_of_several_programs_keeps_the_scripts_entry(cache):
    made = remembered.Memory(lambda message: None)
    made.loaded([object(), object()], 70096)  # an island session's list
    assert remembered.read() == {"programs": {}, "scripts": {made.script: 71168}}


def test_a_runner_in_pythons_prefix_keeps_nothing(cache, monkeypatch):
    main = types.ModuleType("__main__")
    main.__file__ = str(Path(sys.prefix) / "lib" / "pytest" / "__main__.py")
    monkeypatch.setitem(sys.modules, "__main__", main)
    said = []
    assert remembered.Memory(said.append).script is None
    assert "a runner such as pytest: none kept" in said[0]
