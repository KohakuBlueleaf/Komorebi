"""Host-fed graphs against the fake dispatcher: the same commands and registers as
the simulator, the loop ending on result words, a queued command placed by the kernel.
"""

import shutil

import pytest
from fake_port import FakeBackend, FakePort

from komorebi import abi
from komorebi.engines.graph import Graph
from komorebi.session.run import Run

CONSTANTS = abi.headers().constants
pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="needs cc")


class ResultPort(FakePort):
    """Completes at once; every done event carries a0 + count as the result word."""

    def _event(self, req, kind, auxiliary=0, ticks=100):
        words = self.ring.get(req)
        if kind == CONSTANTS["KM_EVENT_DONE"] and words:
            ticks = words[3] + (words[2] >> 16)
        super()._event(req, kind, auxiliary, ticks)


def make_run():
    port = ResultPort(cores=119, depth=4)
    return Run(FakeBackend(port), (7, 17)), port


def loop():
    graph = Graph(max_runs=100)
    binding, result_register = graph.bindings("it"), graph.register()
    root = graph.command(1, group=(0, 60), args=[binding])
    left = graph.command(2, group=(0, 30), args=[binding], after=[root])
    right = graph.command(3, group=(30, 30), args=[binding], after=[root])
    merge = graph.command(
        4, group=(0, 60), args=[binding], after=[left, right], result=result_register
    )
    update = graph.cpu(after=[merge])
    update.addi(binding, binding, 1)
    update.lt(graph.again, result_register, graph.const(65))
    return graph


def test_hostfed_gives_the_simulated_commands_and_registers_and_takes_new_bindings():
    graph = loop()
    run, _ = make_run()
    for value, rounds in ((0, 6), (2, 4)):
        got = run.run_hostfed(graph, {"it": value})
        want = graph.simulate(
            {"it": value}, result=lambda command: command.args[0] + command.count
        )
        assert (got.status, got.runs, got.registers) == (
            "ended",
            rounds,
            want.registers,
        )
        key = sorted(
            (command.run, command.node, command.first, command.count, command.args)
            for command in got.commands
        )
        assert key == sorted(
            (command.run, command.node, command.first, command.count, command.args)
            for command in want.commands
        )


def test_a_queued_command_gets_the_kernels_group_after_the_groups_in_flight():
    graph = Graph()
    graph.command(1, group=(0, 50))
    graph.command(2, count=20)
    run, port = make_run()
    got = run.run_hostfed(graph)
    assert [(command.first, command.queued) for command in got.commands] == [
        (0, False),
        (None, True),
    ]
    assert [group for *_, group in port.submitted] == [(0, 50), (50, 20)]


def test_a_threads_word_feeds_the_next_command_and_its_timeout_is_a_deadline_error():
    graph = Graph()
    word = graph.register()
    first = graph.command(9, cluster=2, args=[40], result=word)
    graph.command(1, group=(0, 4), args=[word], after=[first])
    run, port = make_run()
    done = CONSTANTS["KM_EVENT_DONE"]
    port.handlers[9] = lambda req, words: port._event(req, done, 2)  # a0 + 1
    got = run.run_hostfed(graph)
    assert got.status == "ended" and got.registers[0] == 41
    assert [(command.first, command.args[0]) for command in got.commands] == [
        (0xFF02, 40),
        (0, 41),
    ]
    timeout = CONSTANTS["KM_EVENT_TIMEOUT"]
    port.handlers[9] = lambda req, words: port._event(req, timeout, 2)
    got = run.run_hostfed(graph)
    assert got.status == "error" and tuple(got.error) == (8, 0)  # the deadline, node 0
    assert run.wait(run.submit(1, (0, 1))).kind == "done"  # the run goes on
