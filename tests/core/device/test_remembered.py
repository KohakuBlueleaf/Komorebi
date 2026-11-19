"""The kernel buffer remembered for a script's next run (B): kept by script and by
program in the build cache; the open at that size; grown during a run and at close,
never lowered but by a session with headroom; every write under one lock."""

import fcntl
import json
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from komorebi.device import remembered


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


def test_a_run_grows_the_entry_and_its_close_never_lowers_it(cache):
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
    second.closed()  # this run needed less: the entry stays
    assert remembered.read()["scripts"][second.script] == 93568
    second.largest = 99840
    second.closed()  # more: it grows
    assert remembered.read()["scripts"][second.script] == 99840


def test_an_entry_measured_with_other_code_is_not_read(cache, monkeypatch):
    remembered.Memory(lambda message: None).wanted(92496)
    monkeypatch.setattr(remembered, "code_digest", lambda: "other")
    said = []
    assert remembered.Memory(said.append).size is None
    assert said == ["kernel buffer: the runtime's at the open (nothing remembered)"]


def test_a_prebuild_s_sizes_seed_the_card_s_entry_until_a_card_run(cache):
    own = cache / "own.json"
    own.write_text(json.dumps({"scripts": {remembered.script_key("card"): 93568}}))
    said = []
    remembered.seed(own, said.append)
    assert said == ["kmr prebuild: 1 kernel buffer size(s) seeded"]
    made = remembered.Memory(said.append)
    assert made.size == 93568
    assert (
        said[1]
        == "kernel buffer 93,568 B at the open (seeded by a prebuild for this script)"
    )
    made.largest = 93568
    made.closed()
    assert remembered.read()["prebuilt"] == []


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
    want = {"programs": {}, "scripts": {made.script: 71168}, "prebuilt": []}
    assert remembered.read() == want


def test_a_runner_in_pythons_prefix_keeps_nothing(cache, monkeypatch):
    main = types.ModuleType("__main__")
    main.__file__ = str(Path(sys.prefix) / "lib" / "pytest" / "__main__.py")
    monkeypatch.setitem(sys.modules, "__main__", main)
    said = []
    assert remembered.Memory(said.append).script is None
    assert "a runner such as pytest: none kept" in said[0]


def test_a_session_that_cannot_reopen_opens_with_headroom(cache):
    assert remembered.headroom(76288) == 9536
    assert remembered.headroom(40000) == 8192  # at least 8 KiB
    remembered.Memory(lambda message: None).wanted(75264)  # entry 76,288
    said = []
    made = remembered.Memory(said.append, room=True)
    assert (made.size, made.opened) == (76288, 85824)
    assert said == [
        "kernel buffer 85,824 B at the open (remembered for this script: 76,288 B "
        "and 9,536 B of headroom)"
    ]


def test_with_headroom_the_entry_shrinks_only_below_three_quarters(cache):
    remembered.Memory(lambda message: None).wanted(75264)  # entry 76,288
    made = remembered.Memory(lambda message: None, room=True)  # opened 85,824
    made.largest = 70656  # over three quarters of 85,824: kept
    made.closed()
    assert remembered.read()["scripts"][made.script] == 76288
    made.largest = 61056  # under: the entry is this run's
    made.closed()
    assert remembered.read()["scripts"][made.script] == 61056
    one_card = remembered.Memory(lambda message: None)  # exact: no headroom
    assert one_card.opened == 61056
    one_card.largest = 60000
    one_card.closed()  # no headroom, no shrink at all
    assert remembered.read()["scripts"][one_card.script] == 61056


def test_a_prebuild_s_smaller_size_does_not_lower_an_entry(cache):
    key = remembered.script_key("card")
    first = remembered.Memory(lambda message: None)
    first.wanted(92496)  # the card's entry 93,568
    own = cache / "own.json"
    own.write_text(json.dumps({"scripts": {key: 70656}, "programs": {"p": 4096}}))
    said = []
    remembered.seed(own, said.append)
    made = remembered.read()
    assert made["scripts"][key] == 93568 and made["prebuilt"] == ["p"]
    assert said == ["kmr prebuild: 1 kernel buffer size(s) seeded"]  # p alone
    own.write_text(json.dumps({"scripts": {key: 99840}}))
    remembered.seed(own, said.append)  # a larger one replaces and is marked
    assert remembered.read()["scripts"][key] == 99840


CHILD = """
import sys
from pathlib import Path
from komorebi.device import remembered
Path(sys.argv[1] + ".ready").touch()
while not Path(sys.argv[2]).exists():
    pass
remembered.seed(Path(sys.argv[1]), lambda message: None)
"""


def seeded_at_once(folder, sizes):
    """One process a (key, size) of `sizes`, each seeding its own entry the moment
    all have started (they wait for one file); returns the file's scripts."""
    go, env = folder / "go", dict(os.environ, KMR_CACHE=str(folder))
    runs = []
    for key, size in sizes.items():
        own = folder / f"own-{key}.json"
        own.write_text(json.dumps({"scripts": {key: size}}))
        command = [sys.executable, "-c", CHILD, str(own), str(go)]
        runs.append(subprocess.Popen(command, env=env))
    while not all(
        Path(f"{folder / f'own-{key}.json'}.ready").exists() for key in sizes
    ):
        time.sleep(0.01)
    go.touch()
    for run in runs:
        assert run.wait(120) == 0
    go.unlink()
    return remembered.read()["scripts"]


def test_eight_processes_recording_at_the_same_instant_all_survive(cache):
    for trial in range(6):
        sizes = {f"t{trial}-{n}": 4096 * (n + 1) for n in range(8)}
        kept = seeded_at_once(cache, sizes)
        assert {key: kept.get(key) for key in sizes} == sizes, trial


def test_a_seeded_entry_survives_concurrent_seeds_of_other_keys(cache):
    own = cache / "first.json"
    own.write_text(json.dumps({"scripts": {"seeded": 93568}}))
    remembered.seed(own, lambda message: None)
    for trial in range(4):
        others = {f"o{trial}-{n}": 4096 * (n + 1) for n in range(8)}
        kept = seeded_at_once(cache, others)
        assert kept["seeded"] == 93568
        assert {key: kept.get(key) for key in others} == others
    assert "seeded" in remembered.read()["prebuilt"]


def test_a_write_waits_for_the_lock_held_across_a_read_and_a_write(cache):
    made = remembered.Memory(lambda message: None)
    held = (cache / remembered.LOCK).open("a")
    fcntl.flock(held, fcntl.LOCK_EX)
    done = threading.Thread(target=made.loaded, args=([object()], 70096))
    done.start()
    done.join(0.5)
    assert done.is_alive() and remembered.read()["scripts"] == {}
    held.close()  # released
    done.join(30)
    assert not done.is_alive()
    assert remembered.read()["scripts"] == {made.script: 71168}
