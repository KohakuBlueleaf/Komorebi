"""Host-fed graphs against the fake dispatcher: the same commands and registers as
the simulator, the loop ending on result words, a queued command placed by the kernel.
"""

import shutil

import pytest
from fake_port import FakeBackend, FakePort

from komorebi import abi
from komorebi.engines.graph import Graph
from komorebi.run import Run

K = abi.headers().constants
pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="needs cc")


class ResultPort(FakePort):
    """Completes at once; every done event carries a0 + count as the result word."""

    def _event(self, req, kind, aux=0, ticks=100):
        words = self.ring.get(req)
        if kind == K["KM_EV_DONE"] and words:
            ticks = words[3] + (words[2] >> 16)
        super()._event(req, kind, aux, ticks)


def make_run():
    port = ResultPort(cores=119, depth=4)
    return Run(FakeBackend(port), (7, 17)), port


def loop():
    g = Graph(max_runs=100)
    it, err = g.bindings("it"), g.reg()
    a = g.command(1, group=(0, 60), args=[it])
    b = g.command(2, group=(0, 30), args=[it], after=[a])
    c = g.command(3, group=(30, 30), args=[it], after=[a])
    d = g.command(4, group=(0, 60), args=[it], after=[b, c], result=err)
    e = g.cpu(after=[d])
    e.addi(it, it, 1)
    e.lt(g.again, err, g.const(65))
    return g


def test_hostfed_gives_the_simulated_commands_and_registers_and_takes_new_bindings():
    g = loop()
    run, _ = make_run()
    for it, rounds in ((0, 6), (2, 4)):
        got = run.run_hostfed(g, {"it": it})
        want = g.simulate({"it": it}, result=lambda r: r.args[0] + r.count)
        assert (got.status, got.runs, got.regs) == ("ended", rounds, want.regs)
        key = sorted((r.run, r.node, r.first, r.count, r.args) for r in got.commands)
        assert key == sorted(
            (r.run, r.node, r.first, r.count, r.args) for r in want.commands
        )


def test_a_queued_command_gets_the_kernels_group_after_the_groups_in_flight():
    g = Graph()
    g.command(1, group=(0, 50))
    g.command(2, count=20)
    run, port = make_run()
    got = run.run_hostfed(g)
    assert [(r.first, r.queued) for r in got.commands] == [(0, False), (None, True)]
    assert [group for *_, group in port.submitted] == [(0, 50), (50, 20)]
