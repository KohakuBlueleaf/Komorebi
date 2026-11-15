"""The kernel buffer remembered for a script's next run (B): kept by script and by
program in the build cache; the open at that size; grown during a run, set at close."""

import sys
import types

import pytest

from komorebi import remembered


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KMR_CACHE", str(tmp_path))
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
