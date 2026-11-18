"""`kmr prebuild` with no card: the mock clusters shipped, the command's arguments, the
load's built line (ELFs found and compiled), the remembered sizes kept apart."""

import builtins
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import komorebi
from komorebi import abi, cli
from komorebi.builder import prebuild
from komorebi.device import mock, remembered
from komorebi.errors import KomorebiError
from komorebi.islands import graph as islands_graph
from komorebi.program import thread as thread_module
from komorebi.session import kernels as session_kernels
from komorebi.session import session
from komorebi.session import state as device_state


def test_the_mock_clusters_ship_with_the_package():
    assert mock.descriptor([0]).name == "p150.yaml"
    assert mock.descriptor([0, 1]).name == "p150x2.yaml"
    for cards in ([0], [0, 1]):
        assert "board_type: p150" in mock.descriptor(cards).read_text()
    with pytest.raises(ValueError, match="one card or two"):
        mock.descriptor([0, 1, 2])


def test_every_card_given_is_a_chip_of_its_mock_cluster():
    for cards in ([0], [1], [0, 1]):  # TT_VISIBLE_DEVICES names chips of it
        text = mock.descriptor(cards).read_text()
        chips = re.split(r"\n\S", text.split("chip_unique_ids:")[1])[0]
        ids = {int(line.split(":")[0]) for line in chips.split("\n") if ":" in line}
        assert set(cards) <= ids, (cards, ids)
    assert mock.descriptor([1]).name == "p150x2.yaml"  # card 0 there, not opened
    with pytest.raises(ValueError, match="cards 0 and 1"):
        mock.descriptor([2])


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
    assert run.inspect() == [] and run.drain() == 0
    assert run.trace().command(1).shape == (0, 4)  # a traced phase goes on
    assert json.dumps({"us": run.chain().report}) == '{"us": 0.0}'  # no known shape
    run.stop()
    assert run.stopped and not run.alive()
    assert run.state() == []  # no host run: no cores known


def test_a_mock_run_s_state_has_the_card_s_shape_after_the_stop():
    run = mock.Run(counted=False, host=SimpleNamespace(cores=120))
    with pytest.raises(
        KomorebiError,
        match=r"run\.state\(\) is valid only after run\.stop\(\): this run is still",
    ):
        run.state()  # as the card's run
    run.stop()
    states = run.state()
    assert [one["core"] for one in states] == list(range(120))
    for key in ("magic_ok", "runs", "completed", "reader", "writer", "compute_end"):
        assert key in states[0]
    assert (
        states[0]["compute_wrong"] == [0, 0, 0] and states[0]["overlay_epoch"] is None
    )
    assert {"ports", "broadcasts", "dynamic"} <= set(states[0])
    runs = np.array([one["runs"] for one in states])  # example 01's core counts
    assert runs.shape == (120,) and not runs.any()


def test_an_island_mock_run_s_state_is_each_rank_s_with_its_island():
    parts = [mock.Run(counted=False, host=SimpleNamespace(cores=n)) for n in (3, 2)]
    graph = SimpleNamespace(rank_cores=[[0, 2], [1]], first=lambda island: 2 * island)
    run = mock.Run(parts=parts, graph=graph)
    run.stop()
    states = run.state()
    assert [(s["rank"], s["island"], s["core"]) for s in states] == [
        (0, 0, 0),
        (1, 0, 2),
        (2, 1, 1),
    ]


def test_an_island_mock_run_s_wait_reads_as_one_event_of_its_parts():
    run = mock.Run(parts=[mock.Run(counted=False) for _ in range(2)])
    event = run.wait(run.submit())
    assert event.kind == "done" and len(event) == 2 and event.parts[1].kind == "done"
    assert run.result()["error_name"] == "none" and run.result()["island"] is None
    assert [block["error_name"] for block in run.stop().islands] == ["none", "none"]
    assert all(part.stopped for part in run.parts)


def test_an_island_mock_run_s_parts_queue_the_session_s_own_commands():
    parts = [mock.Run(counted=False) for _ in range(2)]
    run = mock.Run(counted=False, parts=parts)
    handles = [part._queue_command(7, (0, 1), (1, 2, 3), False, 0) for part in run.runs]
    for part in run.runs:  # as the island session's table write does
        part._hand()
    for part, handle in zip(run.runs, handles):
        event = part.wait(handle, 1.0)
        assert isinstance(handle, int) and event.kind == "done"


def test_an_island_mock_run_s_event_has_a_part_for_each_island_its_group_spans():
    def split(group):  # two islands of 118 ranks
        first, count = group
        return [i for i in (0, 1) if first < 118 * (i + 1) and first + count > 118 * i]

    host = SimpleNamespace(graph=SimpleNamespace(split=split))
    run = mock.Run(parts=[mock.Run(counted=False) for _ in range(2)], host=host)
    assert len(run.wait(run.submit("h", (0, 118)))) == 1  # island 0's part alone
    assert len(run.wait(run.submit("h", (100, 40)))) == 2
    assert len(run.wait(run.submit("h", [(0, 4), (118, 4)]))) == 2  # as before
    handles = run.submit_many("h", [0, 118, 100], [118, 118, 40], [[0]] * 3)
    assert [len(event) for event in run.wait(list(handles))] == [1, 1, 2]


def test_a_mock_run_s_trace_has_a_clock_and_no_threads_without_its_host_s():
    run = mock.Run(counted=False)
    assert run.trace().mhz == mock.MHZ
    assert run.threads is mock.NOTHING  # no host run's threads: as before


def test_a_mock_run_s_inspect_is_the_card_s_decode_of_each_core():
    run = mock.Run(counted=False, host=SimpleNamespace(cores=120))
    cores = run.inspect((4, 3))  # a group's cores, as the dispatcher reads them
    assert [core["core"] for core in cores] == [4, 5, 6]
    h = abi.headers()
    words = np.zeros((1, h["kStateWords"]), np.uint32)
    keys = set(device_state.core_state(words, 0, h)) | {"read", "live", "trace"}
    for core in cores:  # each read, with the card's live words and no trace entries
        assert set(core) == keys | {"reset"}
        assert core["read"] and core["trace"] == [] and not core["reset"]
        assert set(core["live"]) == set(device_state.live(np.zeros(64), abi.headers()))
    assert len(run.inspect()) == 120 and run.cores == 120
    assert mock.Run(counted=False).inspect() == []  # no host run: no cores known


def test_an_island_mock_run_s_inspect_names_each_rank_s_island():
    graph = islands_graph.split_card(120, 12, 2)  # one card's cores in two islands
    parts = [mock.Run(counted=False, host=SimpleNamespace(cores=120)) for _ in "ab"]
    run = mock.Run(parts=parts, graph=graph)
    edge = graph.first(1)
    got = [(c["rank"], c["island"], c["core"]) for c in run.inspect((edge - 1, 2))]
    assert got == [
        (edge - 1, 0, graph.rank_cores[0][-1]),
        (edge, 1, graph.rank_cores[1][0]),
    ]


def test_a_mock_run_s_trace_is_the_card_s_of_the_load_s_cores_and_slots():
    host = SimpleNamespace(cores=120, backend=SimpleNamespace(trace_cap=16))
    trace = mock.Run(counted=False, host=host).trace()
    assert isinstance(trace, device_state.Trace) and trace.h is not None
    assert trace.rows.shape == (120, 16, abi.headers()["kTraceWords"])
    assert trace.capacity == 16 and not trace.rows.any()
    assert trace.command(1).shape == (0, 4) and trace.phases(1) is None


def test_a_mock_run_s_overlay_epoch_is_read_when_the_load_has_the_area():
    for area, epoch in ((4096, 0), (0, None)):  # the card reads it with the area
        host = SimpleNamespace(cores=2, backend=SimpleNamespace(area_bytes=area))
        run = mock.Run(counted=False, host=host)
        run.stop()
        assert [core["overlay_epoch"] for core in run.state()] == [epoch, epoch]


def test_a_mock_run_has_runs_and_state_words_as_the_card_s():
    one = mock.Run(counted=False, host=SimpleNamespace(cores=120))
    assert not hasattr(one, "runs")  # one card's run has no island runs
    one.stop()
    h = abi.headers()
    assert one.state_words().shape == (120, h["kStateWords"])
    assert one.state_words().dtype == np.uint32 and not one.state_words().any()
    parts = [mock.Run(counted=False, host=SimpleNamespace(cores=2)) for _ in "ab"]
    assert mock.Run(parts=parts).runs == tuple(parts)


class HeldL1:
    """A host run's backend: the L1 it took for its launch until release_l1."""

    def __init__(self):
        self.held = 148224  # a launch's host transfer buffer, control and trace

    def release_l1(self):
        self.held = 0


def test_a_mock_run_s_stop_gives_its_host_s_l1_back_for_the_next_load():
    first = mock.Run(counted=False, host=SimpleNamespace(backend=HeldL1()))
    first.stop()  # the next load of the session has the L1 again
    assert first.host.backend.held == 0
    second = mock.Run(counted=False, host=SimpleNamespace(backend=HeldL1()))
    second.stop()
    assert second.host.backend.held == 0
    assert mock.Run(counted=False).stop() == mock.Run(counted=False).result()


def test_a_two_card_mock_run_s_stop_gives_back_each_card_s_host_l1():
    hosts = [SimpleNamespace(backend=HeldL1()) for _ in range(2)]
    parts = [mock.Run(counted=False, host=host) for host in hosts]
    run = mock.Run(parts=parts, host=SimpleNamespace(runs=tuple(hosts)))
    run.stop()
    assert [host.backend.held for host in hosts] == [0, 0]
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
    mock._channel_mapped, mock.channel, mock.device = True, None, object()
    mock._mock_channel = mock.device  # mapped for this device
    mock._built_before = []
    assert mock._transfer_way([], 0, None) is None
    assert built == [1] and mock._built_before == [(1, 0)]  # in the next load's line


def prebuilt(monkeypatch, tmp_path, text, sessions=0):
    """`text` run as a prebuild's script in tmp_path, `sessions` made before it."""
    script = tmp_path / "card.py"
    script.write_text(text)
    monkeypatch.setenv(mock.ACTIVE, "1")
    monkeypatch.delenv(mock.MEMORY, raising=False)
    monkeypatch.setattr("sys.argv", ["prebuild", str(script)])
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(mock, "SESSIONS", [sessions])
    monkeypatch.setattr(prebuild, "SESSIONS", mock.SESSIONS)
    assert prebuild.main() == 0
    return sorted(p.name for p in tmp_path.iterdir())


WRITES = (
    "import json, pathlib\n"
    "import numpy as np\n"
    "here = pathlib.Path(__file__).parent\n"
    "with open(here / 'results.json', 'w') as f:\n"
    "    json.dump({'err': 0.0}, f)\n"
    "(here / 'notes.txt').write_text('zeros')\n"
    "np.save(here / 'saved.npy', np.zeros(2))  # a library's write for the script\n"
)


def test_a_prebuild_writes_the_script_s_files_beside_them_after_a_session(
    monkeypatch, tmp_path
):
    real = builtins.open
    assert prebuilt(monkeypatch, tmp_path, WRITES, sessions=1) == [
        "card.py",
        "notes.txt.prebuild",
        "results.json.prebuild",
        "saved.npy.prebuild",
    ]
    assert builtins.open is real and io.open is real  # the real open again after


def test_a_write_before_any_session_keeps_its_name(monkeypatch, tmp_path):
    names = prebuilt(monkeypatch, tmp_path, WRITES)  # a generated source, inputs
    assert names == ["card.py", "notes.txt", "results.json", "saved.npy"]


def test_a_source_written_after_a_session_keeps_its_name(monkeypatch, tmp_path):
    writes = "".join(
        f"(__import__('pathlib').Path(__file__).parent / '{name}').write_text('0')\n"
        for name in ("gen.h", "part.cpp", "link.ld", "sizes.json")
    )
    names = prebuilt(monkeypatch, tmp_path, writes, sessions=1)  # a compiler reads
    assert names == ["card.py", "gen.h", "link.ld", "part.cpp", "sizes.json.prebuild"]


def test_a_module_of_the_script_writes_beside_too(monkeypatch, tmp_path):
    (tmp_path / "helper.py").write_text(
        "import pathlib\n"
        "def save(text):\n"
        "    (pathlib.Path(__file__).parent / 'out.txt').write_text(text)\n"
    )
    names = prebuilt(monkeypatch, tmp_path, "import helper\nhelper.save('0')\n", 1)
    assert "out.txt.prebuild" in names and "out.txt" not in names


def test_a_read_of_a_missing_file_takes_its_prebuild_and_a_real_one_wins(
    monkeypatch, tmp_path
):
    (tmp_path / "one.npz.prebuild").write_text("zeros")
    (tmp_path / "real.txt").write_text("card")
    (tmp_path / "real.txt.prebuild").write_text("zeros")
    reads = (
        "import pathlib\n"
        "here = pathlib.Path(__file__).parent\n"
        "assert (here / 'one.npz').read_text() == 'zeros'\n"
        "assert (here / 'real.txt').read_text() == 'card'\n"
    )
    names = prebuilt(monkeypatch, tmp_path, reads, sessions=1)
    assert "one.npz" not in names  # read, never written


def test_a_missing_file_with_a_prebuild_exists_as_its_read_finds_it(
    monkeypatch, tmp_path
):
    (tmp_path / "class.npz.prebuild").write_text("zeros")
    (tmp_path / "real.txt").write_text("card")
    tests = (
        "import os, pathlib\n"
        "here = pathlib.Path(__file__).parent\n"
        "for name in ('class.npz', 'real.txt'):\n"
        "    assert (here / name).exists() and (here / name).is_file(), name\n"
        "    assert os.path.exists(here / name) and os.path.isfile(str(here / name))\n"
        "assert not (here / 'none.npz').exists() and not os.path.isfile('none.npz')\n"
    )
    real = os.path.exists, os.path.isfile, Path.exists, Path.is_file
    prebuilt(monkeypatch, tmp_path, tests, sessions=1)
    assert (os.path.exists, os.path.isfile, Path.exists, Path.is_file) == real
    assert not (tmp_path / "class.npz").exists()  # the hooks gone with the script


def test_every_session_a_prebuild_makes_is_counted(monkeypatch):
    monkeypatch.setattr(mock, "SESSIONS", [0])
    monkeypatch.setattr(mock, "active", lambda: True)
    session.Session(card=0, device=object())  # an island's part: its device given
    session.Session(card=1, device=object())
    assert mock.SESSIONS == [2]
    monkeypatch.setattr(mock, "active", lambda: False)
    session.Session(card=0, device=object())  # on the card: not counted
    assert mock.SESSIONS == [2]


def test_a_script_that_exits_non_zero_having_built_nothing_ends_non_zero(
    monkeypatch, tmp_path
):
    script = tmp_path / "card.py"
    monkeypatch.setenv(mock.ACTIVE, "1")
    monkeypatch.delenv(mock.MEMORY, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))

    def ends(text, loads):  # the script's process: main() takes its own argv
        script.write_text(text)
        monkeypatch.setattr("sys.argv", ["prebuild", str(script)])
        monkeypatch.setattr(prebuild, "LOADS", [loads])
        return prebuild.main()

    exits = "import sys\nsys.exit(1)\n"
    assert ends(exits, 0) == prebuild.EXITED  # a refusal it caught, nothing built
    assert ends(exits, 1) == 0  # its check on zeros after its loads
    assert ends("print('a tool')\n", 0) == 0


def test_a_stop_s_message_keeps_the_compiler_s_first_error_line():
    assert prebuild.cut("ProgramError: short") == "ProgramError: short"
    line = "kernels/compute.cpp:14:10: fatal error: silu.hpp: No such file or directory"
    message = "OverlayError: g++ failed: " + "-O2 " * 200 + "\n" + line + "\nmore"
    said = prebuild.cut(message)
    assert said.startswith(message[: prebuild.CUT]) and said.endswith(line)
    assert prebuild.cut("x" * 500) == "x" * prebuild.CUT  # no error line: cut


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
        SimpleNamespace(
            load_islands=lambda *args, **kwargs: parts,
            _last_total=0,
            _kept=SimpleNamespace(stop=lambda: None),  # the card session's
        )
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


HOST_CPP = Path(komorebi.__file__).parents[1] / "komorebi_native/csrc/host/km_host.cpp"


def native_keys(function):
    """The keys of the dict the native Dispatch's `function` returns (km_host.cpp)."""
    text = HOST_CPP.read_text()
    body = text[text.index(f"nb::dict {function}(") :]
    body = body[: body.index("return out;")]
    return set(re.findall(r'out\["(\w+)"\]', body))


def test_the_mock_dispatch_has_the_native_work_calls_in_their_shapes():
    bound = set(re.findall(r'def\(\s*"(work_\w+)"', HOST_CPP.read_text()))
    assert bound == {name for name in dir(mock.Dispatch) if name.startswith("work_")}
    dispatch = mock.Dispatch.__new__(mock.Dispatch)  # no device: the work calls only
    assert set(dispatch.work_info(1)) >= native_keys("work_info") - {"lim_bytes"}
    with pytest.raises(RuntimeError, match="no image running"):
        dispatch.work_wait(1, 1.0)
    run = dispatch.work_start(1, "image", bytes(128))
    assert dispatch.work_running(1) and dispatch.work_read(1, 0, 8) == run.to_bytes(
        8, "little"
    )
    waited = dispatch.work_wait(1, 1.0)
    assert set(waited) == native_keys("work_wait") and waited["ready"]
    assert len(waited["results"]) == len(waited["stages"]) == mock.HARTS
    with pytest.raises(RuntimeError, match="whole lines"):
        dispatch.work_write(1, 4, bytes(64))
    stopped = dispatch.work_stop(1, 1.0)
    assert set(stopped) == native_keys("work_stop") and stopped["cleaned"]
    assert not dispatch.work_running(1) and dispatch.work_abort(1)


def test_a_mock_thread_s_result_block_is_a_clean_return():
    dispatch = mock.Dispatch.__new__(mock.Dispatch)
    dispatch.work_start(2, "image", bytes(64))
    at = thread_module.layout()["KMX_LIM_RESULT"]
    result = thread_module.parse_result(dispatch.work_read(2, at, 256))
    assert (result.ran, result.state, result.error, result.words) == (
        True,
        "done",
        "none",
        [],
    )


def test_a_mock_run_s_threads_are_its_host_run_s():
    class Host:
        threads = {1: "the card's ThreadRun"}
        thread = threads[1]
        backend = "backend"

        def thread_results(self, timeout):
            return {1: timeout}

    run = mock.Run(counted=False, host=Host())
    assert run.threads is Host.threads and run.thread == "the card's ThreadRun"
    assert run.backend == "backend" and run.thread_results(4.0) == {1: 4.0}
