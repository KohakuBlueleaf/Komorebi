"""A prebuild's mock run builds a chain as the card's run does: run.chain on it is the
session's Run's (descriptions, proofs, refusals, records), and only submit is mock."""

from types import SimpleNamespace

import numpy as np
import pytest
from class_chain_fixture import BlockBackend, Pages
from fake_port import FakePort

import komorebi as km
from komorebi.chain import calls as kc
from komorebi.device import mock
from komorebi.device.tensor import Tensor
from komorebi.errors import KomorebiError, ProgramError
from komorebi.islands.parts.mesh import cards_graph
from komorebi.session.run import Run


def grid(tiles_wide, address):
    """A float32 tensor of 2 tile rows, `tiles_wide` tiles wide."""
    pages = Pages(2 * tiles_wide, address)
    return Tensor(pages, (64, 32 * tiles_wide), "float32", "tile")


def row(address):
    return Tensor(Pages(16, address), (16, 1024), "float32", "row")


class Walked(km.Program):
    @km.handler
    def copy(self, x: km.Tensor, y: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(x.tiles[0:2, 0:2], block=2):
            total += tile
        y.pages(0, 1).write(total)

    @km.handler
    def scale(self, a: km.Tensor, z: km.Tensor):
        results = km.Stream(z.pages(0, 1), write=True)
        for tile in km.Stream(a.pages(0, 1)):
            results.push(tile)

    @km.chain
    def steps(self, x, y, z):
        self.copy(x, y, cores=(0, 1))
        self.scale(y, z, cores=(0, 1))


def mocked():
    program = Walked(
        l1=1 << 20,
        log=lambda line: None,
        x=grid(4, 0x1000),
        y=row(0),
        a=row(0),
        z=row(0),
    )
    host = Run(BlockBackend(FakePort()), (12, 10), program=program)
    return program, mock.Run(counted=False, host=host)


def test_a_mock_runs_chain_is_built_and_its_submit_mock():
    program, run = mocked()
    chain = run.chain(program.steps(grid(4, 0x1000), row(0x2000), row(0x3000)))
    assert [call.name for call in chain.calls] == ["Walked.copy", "Walked.scale"]
    records = np.frombuffer(run.host.backend.blocks[-1].data, np.uint32)
    assert records.reshape(-1, 16)[0][10] == 0 | 1 << 16  # the first call's cores
    assert run.wait(run.submit(chain)).kind == "done"


def test_a_chains_host_refusal_is_raised_under_the_mock():
    program, run = mocked()
    with pytest.raises(ProgramError, match="x is 8 tiles wide, .* built for 4"):
        run.chain(program.steps(grid(8, 0x1000), row(0x2000), row(0x3000)))
    listed = [
        Walked.scale(row(0), row(0x2000), cores=(0, 1)),
        kc.Call(1, (0,), (0, 1), 1, 1, "c"),
    ]
    with pytest.raises(kc.ChainRefused, match="a chain.Call among a class's"):
        run.chain(listed)


def test_a_mock_run_without_a_host_answers_nothing_as_before():
    assert mock.Run(counted=False).chain() is mock.NOTHING


def test_a_mock_runs_submit_checks_a_commands_words_as_the_card_s():
    program, run = mocked()
    assert run.wait(run.submit(Walked.copy, (0, 1), [grid(4, 0x1000), row(0)])).kind
    with pytest.raises(ProgramError, match="x is 8 tiles wide, .* built for 4"):
        run.submit(Walked.copy, (0, 1), [grid(8, 0x1000), row(0)])
    with pytest.raises(ValueError, match="not inside 120 ranks"):
        run.submit(Walked.copy, (100, 40), [grid(4, 0x1000), row(0)])
    with pytest.raises(ValueError, match="not inside 120 ranks"):
        run.submit_many("copy", [100], [40])
    assert len(run.submit_many("copy", [0, 1], [1, 1])) == 2
    assert run.host._ports == {}  # a mock's rows are never in flight


def test_a_mocked_addon_stops_the_mock_runs_checks():
    program, run = mocked()
    assert run.addon("anything") is mock.NOTHING  # nothing built: the host run differs
    assert run.wait(run.submit(Walked.copy, (0, 1), [grid(8, 0x1000), row(0)])).kind


def test_check_programs_set_on_a_mock_run_is_its_host_runs():
    program, run = mocked()
    run.check_programs = False  # a script's own: the dispatcher refuses, not the host
    run.submit(Walked.copy, (0, 1), [grid(4, 0x1000), row(0)])
    assert run.host.check_programs is False


def mocked_islands(split=1):
    """Walked's mock run over two islands of 120 ranks (a link core each: 119 ranks),
    each part's host the island's own Run, as an island prebuild's."""
    program, _ = mocked()
    graph = cards_graph([(12, 10), (12, 10)], [(0, 1)])
    hosts = [
        Run(
            BlockBackend(FakePort()),
            (12, 10),
            program=program,
            ranks=graph.ranks,
            island_first=graph.first(island),
            island_ranks=graph.count(island),
        )
        for island in range(2)
    ]
    parts = [mock.Run(counted=False, host=host) for host in hosts]
    host = SimpleNamespace(session=SimpleNamespace(split=split))
    return mock.Run(counted=False, parts=parts, host=host, graph=graph), graph


def test_an_island_mock_run_checks_each_islands_part_as_the_card_s():
    run, graph = mocked_islands()
    words = [grid(4, 0x1000), row(0)]
    assert run.submit(Walked.copy, (110, 20), words)  # a part on each island
    with pytest.raises(ProgramError, match="^island 0: .*x is 8 tiles wide"):
        run.submit(Walked.copy, (110, 20), [grid(8, 0x1000), row(0)])
    with pytest.raises(ProgramError, match="^island 1: .*x is 8 tiles wide"):
        run.submit(Walked.copy, (130, 2), [grid(8, 0x1000), row(0)])
    with pytest.raises(ValueError, match=f"of {graph.ranks} ranks"):
        run.submit(Walked.copy, (0, graph.ranks + 1), words)
    with pytest.raises(ValueError, match=f"of {graph.ranks} ranks"):
        run.submit_many("copy", [0, 200], [8, 100])
    assert len(run.submit_many("copy", [0, 118], [1, 4])) == 2
    with pytest.raises(ValueError, match="m <= 10"):  # island 1's rows alone
        run.submit_many("copy", [130], [2], np.zeros((1, 11), np.uint32))


def test_an_island_mock_run_refuses_a_queued_command_as_the_island_run():
    run, graph = mocked_islands()
    words = [grid(4, 0x1000), row(0)]
    assert run.submit(Walked.copy, 8, words, queued=True)
    with pytest.raises(ProgramError, match="^island 0: .*x is 8 tiles wide"):
        run.submit(Walked.copy, 8, [grid(8, 0x1000), row(0)], queued=True)
    with pytest.raises(ValueError, match="one island's ranks"):
        run.submit(Walked.copy, graph.count(0) + 1, words, queued=True)
    with pytest.raises(ValueError, match="cannot be exclusive"):
        run.submit(Walked.copy, 8, words, queued=True, exclusive=True)
    with pytest.raises(KomorebiError, match="split into islands"):
        mocked_islands(split=2)[0].submit(Walked.copy, 8, words, queued=True)


def test_an_island_mock_run_leaves_a_threads_command_and_a_mocked_addon_unchecked():
    run, _ = mocked_islands()
    bad = [grid(8, 0x1000), row(0)]
    assert run.submit(Walked.copy, args=bad, cluster=1)
    run.addon("anything")
    assert run.submit(Walked.copy, (110, 20), bad)
