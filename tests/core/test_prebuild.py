"""`kmr prebuild` with no card: the mock clusters shipped, the command's arguments, the
load's built line (ELFs found and compiled), the remembered sizes kept apart."""

import json
import os
import time
from types import SimpleNamespace

import pytest

from komorebi import cli, prebuild, remembered, session


def test_the_mock_clusters_ship_with_the_package():
    assert prebuild.descriptor([0]).name == "p150.yaml"
    assert prebuild.descriptor([0, 1]).name == "p150x2.yaml"
    for cards in ([0], [0, 1]):
        assert "board_type: p150" in prebuild.descriptor(cards).read_text()
    with pytest.raises(ValueError, match="one card or two"):
        prebuild.descriptor([0, 1, 2])


def test_the_command_takes_the_cards_and_the_script_s_own_arguments(monkeypatch):
    called = []
    monkeypatch.setattr(prebuild, "run", lambda *given: called.append(given) or 0)
    assert cli.main([prebuild.COMMAND, "--cards", "0,1", "x.py", "--n", "4"]) == 0
    ((script, args, cards, cluster),) = called
    assert (str(script), args, cards, cluster) == ("x.py", ["--n", "4"], [0, 1], None)


def test_the_built_line_counts_elves_written_since_the_build(tmp_path):
    old, new = tmp_path / "old.elf", tmp_path / "new.elf"
    old.write_bytes(b"")
    os.utime(old, (1, 1))
    started = time.time() - 1
    new.write_bytes(b"")
    rows = [("brisc", str(old)), ("ncrisc", str(new)), ("ncrisc", str(new))]
    recipes = [("reader.cpp", rows[:1]), ("writer.cpp", rows[1:])]
    assert session.built_files(recipes, started) == (1, 1)


def test_a_prebuild_keeps_its_sizes_in_its_own_copy(monkeypatch, tmp_path):
    real = tmp_path / "real.json"
    real.write_text('{"scripts": {"a": 4096}}')
    monkeypatch.setattr(remembered, "_path", lambda: prebuild.memory(real) or real)
    monkeypatch.setenv(prebuild.ACTIVE, "1")
    monkeypatch.setenv(prebuild.MEMORY, str(tmp_path / "own.json"))
    assert remembered.read()["scripts"] == {"a": 4096}  # the card's, copied
    remembered.write({"programs": {}, "scripts": {"a": 8192}})
    assert remembered.read()["scripts"] == {"a": 8192}
    assert real.read_text() == '{"scripts": {"a": 4096}}'
    monkeypatch.delenv(prebuild.MEMORY)
    assert prebuild.memory(real) is None


def test_a_compiled_kernel_is_logged_with_its_own_defines(tmp_path):
    elf = tmp_path / "reader.elf"
    elf.write_bytes(b"")
    row = ("brisc", str(elf), "", "", "", "", "", "", "-DNUM_DRAM_BANKS=8")
    row += ("-DKM_LINK_CHUNK=65536", "-DLINEAR_OWN=1")
    descriptor = SimpleNamespace(recipes=lambda: [("k/reader.cpp", [row])])
    logged = []
    made = SimpleNamespace(
        _built_before=[(1, 0)], log=logged.append, defines={"LINEAR_OWN": 1}
    )
    session.Session._log_built(made, descriptor, time.time() - 5, made)
    assert logged[0].startswith("built: 1 found, 1 compiled")
    assert logged[1] == "compiled: reader.cpp brisc (KM_LINK_CHUNK=65536 LINEAR_OWN=1)"


def test_a_mock_run_s_calls_return_at_once_with_nothing():
    before = prebuild.LOADS[0]
    run = prebuild.Run()
    assert prebuild.LOADS[0] == before + 1 and run.alive()
    event = run.wait(run.submit("twice", a=1))
    assert event.kind == "done" and run.wait_any([1]).kind == "done"
    assert event.parts == (event,)
    assert float(event.microseconds) == 0.0 and f"{event.microseconds:.2f}" == "0.00"
    assert list(event.values) == [] and event and int(event + 1) == 0
    run.stop()
    assert run.stopped and not run.alive()


def passes(monkeypatch, tmp_path, refusals):
    """converge over passes that refuse `refusals` in turn (then one that does not)."""
    monkeypatch.setenv(prebuild.MEMORY, str(tmp_path / "kernel_buffer.json"))
    left, said = list(refusals), []

    def one_pass():
        if not left:
            return SimpleNamespace(returncode=0)
        with prebuild.refusals_path().open("a") as out:
            out.write(json.dumps(left.pop(0)) + "\n")
        return SimpleNamespace(returncode=prebuild.REFUSED)

    return prebuild.converge(one_pass, None, said.append), said


def test_a_later_larger_load_takes_passes_until_every_load_fits(monkeypatch, tmp_path):
    grows = [
        {"load": f"p{n}", "line": n, "kind": "grows", "opens": size}
        for n, size in ((10, 86016), (20, 91264))
    ]
    code, said = passes(monkeypatch, tmp_path, grows)
    assert code == 0 and len(said) == 2 and "again" in said[1]


def test_loads_that_cannot_share_one_size_are_said_with_both(monkeypatch, tmp_path):
    code, said = passes(
        monkeypatch,
        tmp_path,
        [
            {"load": "wide", "line": 12, "kind": "grows", "opens": 200000},
            {"load": "deep", "line": 30, "kind": "shrinks", "opens": 120000},
        ],
    )
    assert code == 1
    assert said[-1] == (
        "kmr prebuild: the script's loads cannot share one kernel buffer: wide (line "
        "12) needs 200,000 B, deep (line 30) fits only at 120,000 B or less; give "
        "them sessions of their own"
    )
