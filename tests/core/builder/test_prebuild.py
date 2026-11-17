"""`kmr prebuild` with no card: the mock clusters shipped, the command's arguments, the
load's built line (ELFs found and compiled), the remembered sizes kept apart."""

import builtins
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import komorebi
from komorebi import cli
from komorebi.builder import prebuild
from komorebi.device import mock, remembered
from komorebi.session import kernels as session_kernels
from komorebi.session import session


def test_the_mock_clusters_ship_with_the_package():
    assert mock.descriptor([0]).name == "p150.yaml"
    assert mock.descriptor([0, 1]).name == "p150x2.yaml"
    for cards in ([0], [0, 1]):
        assert "board_type: p150" in mock.descriptor(cards).read_text()
    with pytest.raises(ValueError, match="one card or two"):
        mock.descriptor([0, 1, 2])


def test_the_command_takes_the_cards_and_the_script_s_own_arguments(monkeypatch):
    called = []
    monkeypatch.setattr(prebuild, "run", lambda *given: called.append(given) or 0)
    assert cli.main([mock.COMMAND, "--cards", "0,1", "x.py", "--n", "4"]) == 0
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
    assert session_kernels.built_files(recipes, started) == (1, 1)


def test_a_prebuild_keeps_its_sizes_in_its_own_copy(monkeypatch, tmp_path):
    real = tmp_path / "real.json"
    real.write_text('{"scripts": {"a": 4096}}')
    monkeypatch.setattr(remembered, "_path", lambda: mock.memory(real) or real)
    monkeypatch.setenv(mock.ACTIVE, "1")
    monkeypatch.setenv(mock.MEMORY, str(tmp_path / "own.json"))
    assert remembered.read()["scripts"] == {"a": 4096}  # the card's, copied
    remembered.write({"programs": {}, "scripts": {"a": 8192}})
    assert remembered.read()["scripts"] == {"a": 8192}
    assert real.read_text() == '{"scripts": {"a": 4096}}'
    monkeypatch.delenv(mock.MEMORY)
    assert mock.memory(real) is None


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


def test_a_mock_run_answers_as_a_run_in_which_every_command_succeeds():
    before = mock.LOADS[0]
    run = mock.Run()
    assert mock.LOADS[0] == before + 1 and run.alive()
    event = run.wait(run.submit("twice", a=1))
    assert event.kind == "done" and event.microseconds == 1.0
    assert event.result is None and run.wait(run.submit("h", result=True)).result == 0
    assert [e.kind for e in run.wait([run.submit(), run.submit()])] == ["done"] * 2
    assert run.wait_any([3, 2]).handle == 2
    taken = []  # a script's own checks: the card's success path
    if event.kind != "done" or event.kind == "rejected":
        taken.append("failure")
    if run.result()["error_name"] == "none" and run.result()["state_name"] == "done":
        taken.append("success")
    assert taken == ["success"]
    assert run.state() == [] and run.inspect() == [] and run.drain() == 0
    assert run.trace().command(1).shape == (0, 4)  # a traced phase goes on
    assert json.dumps({"us": run.chain().report}) == '{"us": 0.0}'  # no known shape
    run.stop()
    assert run.stopped and not run.alive()


def test_an_island_mock_run_s_wait_reads_as_one_event_of_its_parts():
    run = mock.Run(parts=[mock.Run(counted=False) for _ in range(2)])
    event = run.wait(run.submit())
    assert event.kind == "done" and len(event) == 2 and event.parts[1].kind == "done"
    assert [block["error_name"] for block in run.result()] == ["none", "none"]
    assert [block["error_name"] for block in run.stop()] == ["none", "none"]
    assert all(part.stopped for part in run.parts)


def passes(monkeypatch, tmp_path, refusals):
    """converge over passes that refuse `refusals` in turn (then one that does not)."""
    monkeypatch.setenv(mock.MEMORY, str(tmp_path / "kernel_buffer.json"))
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


def test_a_pass_that_changed_the_kept_size_is_run_again_at_it(tmp_path):
    memory, said, sizes = tmp_path / "kernel_buffer.json", [], [97600, 97600]

    def one_pass():  # an island script: opens with nothing kept, keeps its size
        memory.write_text(json.dumps({"scripts": {"s|islands 2": sizes.pop(0)}}))
        return SimpleNamespace(returncode=0)

    assert prebuild.converge(one_pass, memory, said.append) == 0
    assert sizes == [] and len(said) == 1 and "again" in said[0]


def test_a_transfer_in_a_prebuild_builds_the_host_transfer_kernel(monkeypatch):
    built = []  # the card would launch the transfer kernel: the prebuild builds it
    monkeypatch.setattr(
        session.Session, "_transfer_kernel_built", lambda s: built.append(1) or [(1, 0)]
    )
    mock = session.Session.__new__(session.Session)
    mock._channel_mapped, mock.channel, mock._mock_channel = True, None, True
    mock._built_before = []
    assert mock._transfer_way([], 0, None) is None
    assert built == [1] and mock._built_before == [(1, 0)]  # in the next load's line


def test_a_prebuild_writes_the_script_s_files_beside_them(monkeypatch, tmp_path):
    real = builtins.open
    script = tmp_path / "card.py"
    script.write_text(
        "import json, pathlib\n"
        "with open(pathlib.Path(__file__).parent / 'results.json', 'w') as f:\n"
        "    json.dump({'err': 0.0}, f)\n"
        "(pathlib.Path(__file__).parent / 'notes.txt').write_text('zeros')\n"
    )
    monkeypatch.setenv(mock.ACTIVE, "1")
    monkeypatch.delenv(mock.MEMORY, raising=False)
    monkeypatch.setattr("sys.argv", ["prebuild", str(script)])
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert prebuild.main() == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "card.py",
        "notes.txt.prebuild",
        "results.json.prebuild",
    ]
    assert builtins.open is real and io.open is real  # the real open again after


def test_a_split_card_s_mock_load_is_one_run_of_its_islands(monkeypatch):
    from komorebi.islands import (  # noqa: PLC0415 (heavy, this test's)
        session as island_session,
    )
    from komorebi.program import (  # noqa: PLC0415 (heavy, this test's)
        description as program,
    )

    class Kernels(program.Program):  # a kernel-form Program, nothing built
        def __init__(self):
            self.defines = {}

        def form(self):
            return "kernels"

    monkeypatch.setenv(mock.ACTIVE, "1")
    split = island_session.IslandSession.__new__(island_session.IslandSession)
    split.links, split.card_dram, split.graph = False, True, "graph"
    split._memory = None  # nothing remembered
    parts = [mock.Run(counted=False) for _ in range(3)]  # load_islands': one each
    split.sessions = [
        SimpleNamespace(load_islands=lambda *args, **kwargs: parts, _last_total=0)
    ]
    before = mock.LOADS[0]
    run = split._load_split(Kernels(), {})
    assert isinstance(run, mock.Run) and run.parts == parts and run is split.run
    assert mock.LOADS[0] == before + 1  # the islands' one load
    assert len(run.wait(run.submit())) == 3


SCALED = """
import komorebi as km
from komorebi.device import remembered
from komorebi.device.tensor import Tensor

BODY = "copy_init(tile_buffer);\\ncopy_tile(tile_buffer, tile_index, 0);"


class Pages:
    def __init__(self, address):
        self.page_size, self.num_pages, self.address = 4096, 8, address

    def accessor_args(self):
        return [2, self.page_size]


class Scaled(km.Program):
    @km.handler
    def scale(self, a: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 1), write=True)
        for tile in km.Stream(a.pages(0, 1)):
            scaled = km.cxx(BODY, takes=[tile], gives=km.Tile)
            results.push(scaled)


def tensor(address):
    return Tensor(Pages(address), (32, 32), "float32", "tile")


made = Scaled(l1=1 << 20, log=lambda line: None, a=tensor(0x1000), out=tensor(0x9000))
print(made.built.digest, remembered.program_key(made, "card"))
"""


def test_a_class_script_prebuilt_and_run_by_a_relative_path_has_one_files_key(
    tmp_path,
):
    """The #line of an inline body names the script's file: a prebuild runs it as the
    interpreter does, so its files key and its remembered program key are a run's."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "scaled.py").write_text(SCALED)
    env = dict(os.environ, PYTHONPATH=str(Path(komorebi.__file__).parents[1]))
    env.pop(mock.ACTIVE, None)
    run = [sys.executable, "sub/scaled.py"]
    built = [sys.executable, "-c", prebuild.CHILD, "sub/scaled.py"]
    keys = []
    for command, extra in ((run, {}), (built, {mock.ACTIVE: "1"})):
        done = subprocess.run(
            command, cwd=tmp_path, env=env | extra, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr
        keys.append(done.stdout.split())
    assert keys[0] == keys[1]
    (folder,) = (tmp_path / "sub" / "build").iterdir()
    assert (
        f'"{(tmp_path / "sub" / "scaled.py").resolve()}"'
        in (folder / "compute.cpp").read_text()
    )
