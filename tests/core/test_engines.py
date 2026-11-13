"""Run's engine calls against a fake dispatch kernel whose engine runs each upload in
the simulator at once: placement, refusals, the events as typed values."""

import struct

import pytest
from fake_port import FakeBackend, FakePort

from komorebi import abi
from komorebi.engines import graph as gr
from komorebi.engines import stream as st
from komorebi.engines.control import codes, offsets
from komorebi.errors import EngineError, EngineRefused, EngineTimeout
from komorebi.run import Run

K = codes()
AT = offsets()
REGION = 2048


def leader_a0(record):
    return record.args[0]


class EngineKernel(FakePort):
    """The fake kernel with an upload region, the ready block and RUN / CANCEL: a
    stream or graph that ends in the simulator posts its events at once, one that
    does not runs until cancelled; `refuse` answers the next RUN with that reason."""

    def __init__(self):
        super().__init__(event_depth=64)
        self.region = bytearray(REGION)
        self.writes = []
        self.refuse = 0
        self.running = {}  # req -> event kind of its cancel
        self.kept = {}  # req -> (offset, bytes) of a kept graph
        self.handlers = {K["KM_SEG_RUN"]: self.run, K["KM_SEG_CANCEL"]: self.cancel}

    def host_word(self, word):
        ready = {
            K["KMD_HOST_READY"]: K["KMD_READY_MARK"],
            K["KMD_HOST_READY"] + AT["ready_bytes"]: REGION,
        }
        return ready.get(word, super().host_word(word))

    def upload(self, offset, data):
        assert offset % 64 == 0 and len(data) % 64 == 0
        self.region[offset : offset + len(data)] = data
        self.writes.append((offset, len(data)))

    def publish(self):
        self.published = self.seen

    def run(self, cmd, words):
        a0, a1, form, bind, n = (words[i] for i in (3, 5, 6, 7, 9))
        if self.refuse:
            self._event(cmd, K["KM_EV_REFUSED"], aux=self.refuse)
            self.refuse = 0
            return
        if form == K["KM_GRUN_KEPT"]:
            if a0 not in self.kept:
                self._event(cmd, K["KM_EV_REFUSED"], aux=K["KMD_REFUSE_UNKNOWN"])
                return
            a0, a1 = self.kept[a0]
        data = bytes(self.region[a0 : a0 + a1])
        if form == 0:
            sim = st.simulate(data, result=leader_a0, max_passes=50)
            ticks = [sim.halt or 0] * len(sim.events)
        else:
            values = struct.unpack_from(f"<{n}I", self.region, bind) if n else ()
            sim = gr.simulate(data, bindings=values, result=leader_a0, max_passes=50)
            ticks = [e.extra for e in sim.events]
            if sim.kept:
                self.kept[sim.events[0].req] = (a0, a1)
        at = AT["stream" if form == 0 else "graph"]
        req = sim.events[0].req if sim.events else struct.unpack_from("<I", data, at)[0]
        if sim.status == "stuck":
            self.running[req] = K[
                "KM_EV_STREAM_ERROR" if form == 0 else "KM_EV_GRAPH_ERROR"
            ]
            return
        for e, t in zip(sim.events, ticks):
            self._event(e.req, e.kind, aux=e.aux, ticks=t)

    def cancel(self, cmd, words):
        req = words[3]
        if req in self.running:
            self._event(req, self.running.pop(req), aux=K["KM_SERR_CANCELLED"])
        elif self.kept.pop(req, None):
            no_node = K["KM_SERR_CANCELLED"] | K["KM_GRAPH_NO_NODE"] << 8
            self._event(req, K["KM_EV_GRAPH_ERROR"], aux=no_node)
        else:
            self._event(cmd, K["KM_EV_REFUSED"], aux=K["KMD_REFUSE_UNKNOWN"])


def make_run():
    port = EngineKernel()
    return Run(FakeBackend(port), (12, 10)), port


def sum_stream(req=7, n=8):
    """n records with result words one at a time on teams of 8; HALT their sum."""
    s = st.Stream(req=req)
    i, acc, got = s.regs(3)
    top = s.label()
    s.issue(1, (0, 8), args=[i], result=True)
    s.wait(0)
    s.result(got)
    s.add(acc, acc, got)
    s.addi(i, i, 3)
    s.jlt(i, s.const(3 * n), top)
    s.halt(acc)
    return s


def two_branches(keep=False):
    """Two branches on disjoint halves, then register 0 = the result word + 5."""
    g = gr.Graph(req=9, keep=keep)
    it, r = g.bindings("it"), g.reg()
    a = g.record(1, team=(0, 60), args=[it])
    b = g.record(1, team=(0, 30), args=[it], after=[a])
    c = g.record(1, team=(30, 30), args=[it], after=[a], result=r)
    g.cpu(after=[b, c]).addi(it, r, 5)
    return g


def test_the_offsets_are_the_headers_static_asserts():
    assert AT == {"ready_bytes": 2, "stream": 24, "graph": 24, "flags": 44}


def test_uploads_sit_side_by_side_and_what_does_not_fit_is_refused_unwritten():
    run, port = make_run()
    a = run.upload(sum_stream())
    b = run.upload(sum_stream(req=8))
    assert (a.offset, b.offset) == (0, -(-a.bytes // 64) * 64)
    run.release(a)
    c = run.upload(sum_stream(req=10, n=2))
    assert c.offset == 0 and c.kind == "stream" and c.req == 10
    writes = list(port.writes)
    with pytest.raises(ValueError, match="does not fit"):
        run.upload(bytes(REGION), kind="graph")
    assert port.writes == writes


def test_a_stream_ends_with_its_halt_status_beside_host_records():
    run, port = make_run()
    s = sum_stream()
    record = run.issue(0, (0, 8))
    t = run.start(run.upload(s))
    ev = run.wait_engine(t, timeout=1.0)
    assert ev.kind == "end" and ev.value == s.simulate(result=leader_a0).halt == 84
    assert run.wait(record).kind == "done" and t.events == [ev]
    assert t.upload.state == "idle"


def test_an_error_names_the_reason_and_the_pc_or_the_node():
    run, _ = make_run()
    s = st.Stream(req=3)
    x = s.regs(1)[0]
    s.set(x, 4)
    s.mod(x, x, s.const(0))
    s.halt(x)
    with pytest.raises(EngineError) as err:
        run.wait_engine(run.start(run.upload(s)), timeout=1.0)
    assert err.value.reason_name == "mod_zero" and err.value.pc == 2
    g = gr.Graph(req=4)
    g.join()
    cpu = g.cpu(after=[g.nodes[0]])
    r = g.reg()
    cpu.mod(r, r, r)
    with pytest.raises(EngineError) as err:
        run.wait_engine(run.start(run.upload(g)), timeout=1.0)
    assert (err.value.reason_name, err.value.node, err.value.pc) == (
        "mod_zero",
        1,
        None,
    )


def test_a_refused_run_raises_with_the_reason_name_and_the_run_goes_on():
    run, port = make_run()
    up = run.upload(sum_stream())
    port.refuse = K["KMD_REFUSE_BUSY"]
    with pytest.raises(EngineRefused) as err:
        run.start(up)
    assert err.value.reason_name == "busy" and up.state == "idle"
    assert run.wait_engine(run.start(up), timeout=1.0).value == 84


def test_a_kept_graph_runs_again_with_other_bindings_then_is_dropped():
    run, port = make_run()
    g = two_branches(keep=True)
    up = run.upload(g)
    first = run.wait_engine(run.start(up, {"it": 2}), timeout=1.0)
    assert first.regs == (7, 2) and up.state == "kept"
    again = run.wait_engine(run.restart(9, {"it": 40}), timeout=1.0)
    want = g.simulate({"it": 40}, result=leader_a0).regs
    assert again.regs == (want[0] & 0xFFFFFF, want[1]) == (45, 40)
    run.release(up)
    assert port.kept == {} and run.upload(sum_stream()).offset == 0


def test_wait_engine_times_out_and_cancel_ends_the_stream():
    run, _ = make_run()
    s = st.Stream(req=5, max_steps=abi.M32)
    s.jmp(s.label())
    t = run.start(run.upload(s))
    with pytest.raises(EngineTimeout):
        run.wait_engine(t, timeout=0.02)
    run.cancel(t)
    with pytest.raises(EngineError) as err:
        run.wait_engine(t, timeout=1.0)
    assert err.value.reason_name == "cancelled" and t.upload.state == "idle"


def test_a_stream_host_fed_issues_the_simulated_records_and_ends_alike():
    run, port = make_run()
    port.results = {k + 1: 3 * k for k in range(8)}  # command k + 1: the leader's a0
    s = sum_stream()
    got = run.run_hostfed(s, timeout=1.0)
    want = s.simulate(result=leader_a0)
    assert got.status == "ended" and got.events[-1].extra == want.halt == 84
    teams = [(r.first, r.count, r.args) for r in want.records]
    assert [(r.first, r.count, r.args) for r in got.records] == teams
    assert [(e.kind, e.aux) for e in got.events] == [
        (e.kind, e.aux) for e in want.events
    ]
