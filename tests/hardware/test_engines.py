"""The dispatch kernel's engines through Run on a card (KMR_CARD=1), one device open:
a stream of records with result words, a graph with two branches run and run again
kept, a bad upload, a cancel; values against Stream.simulate and Graph.simulate."""

from pathlib import Path

import pytest

import komorebi as km
from komorebi.engines.graph import Graph
from komorebi.engines.stream import Stream

pytest.importorskip("ttnn")

pytestmark = pytest.mark.card
CARD_IN_PROCESS = True  # opens the card in this process
KERNELS = Path(__file__).resolve().parents[2] / "examples" / "01_mapcollect" / "kernels"
RESULT = 6  # kSegResult: with KM_F_RESULT the record's result word is the leader's a0
CORES = 120


def log(msg):
    print(msg, flush=True)


def program():
    """The map/collect kernels; only RESULT records are sent."""
    return km.Program(
        "mapcollect",
        kernels=KERNELS,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
        ],
        compile_args={"reader": [2, 2048], "writer": [2, 2048]},
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        segments=KERNELS / "mapcollect.h",
    )


def leader_a0(record):
    return record.args[0]


def records_stream(n, req=11):
    """n records one at a time, record k on cores [8k mod 120, + 8) with a0 = 7k + 3
    and a result word; HALT the sum of the results."""
    s = Stream(req=req, deadline_us=500_000)
    k, first, key, got, acc = s.regs(5)
    top = s.label()
    s.mul(first, k, s.const(8))
    s.mod(first, first, s.const(CORES))
    s.mul(key, k, s.const(7))
    s.addi(key, key, 3)
    s.issue(RESULT, (first, s.const(8)), args=[key], result=True)
    s.wait(0)
    s.result(got)
    s.add(acc, acc, got)
    s.addi(k, k, 1)
    s.jlt(k, s.const(n), top)
    s.halt(acc)
    return s


def branches_graph():
    """Two branches on disjoint halves after a record on both, the right one's result
    word into r; then register 0 = r * 3 + 1, register 1 = r."""
    g = Graph(req=12, keep=True, deadline_us=500_000)
    it, r = g.bindings("it"), g.reg()
    a = g.record(RESULT, team=(0, 60), args=[it])
    b = g.record(RESULT, team=(0, 30), args=[it], after=[a])
    c = g.record(RESULT, team=(30, 30), args=[it], after=[a], result=r)
    cpu = g.cpu(after=[b, c])
    cpu.mul(it, r, g.const(3))
    cpu.addi(it, it, 1)
    return g


def looping_stream(req=13):
    s = Stream(req=req, max_steps=0xFFFFFFFF)
    s.jmp(s.label())
    return s


@pytest.fixture(scope="module")
def run():
    with km.Session(log=log) as session:
        yield session.load(program(), deadline=2.0)


def test_a_stream_of_eight_records_ends_alike_uploaded_and_host_fed(run):
    s = records_stream(8)
    up = run.upload(s)
    ev = run.wait_engine(run.start(up), timeout=5.0)
    want = s.simulate(n_cores=CORES, result=leader_a0)
    assert ev.kind == "end" and ev.value == want.halt == 220
    run.release(up)
    fed = run.run_hostfed(s, timeout=5.0)
    assert fed.status == "ended" and fed.events[-1].extra == want.halt
    teams = [(r.first, r.count, r.args) for r in want.records]
    assert [(r.first, r.count, r.args) for r in fed.records] == teams


def test_a_graph_runs_then_runs_again_kept_with_another_binding(run):
    g = branches_graph()
    up = run.upload(g)
    for it, again in ((5, False), (1000, True)):
        t = run.restart(12, {"it": it}) if again else run.start(up, {"it": it})
        ev = run.wait_engine(t, timeout=5.0)
        want = g.simulate({"it": it}, n_cores=CORES, result=leader_a0)
        assert want.status == "ended" and len(want.records) == 3
        assert ev.regs == (want.regs[0] & 0xFFFFFF, want.regs[1]) == (3 * it + 1, it)
        assert up.state == "kept"
    run.release(up)
    assert up.state == "idle"  # the kernel dropped it


def test_a_bad_upload_ends_in_its_error_event(run):
    up = run.upload(bytes(64), kind="stream")
    with pytest.raises(km.EngineError) as err:
        run.wait_engine(run.start(up), timeout=5.0)
    assert (err.value.reason_name, err.value.pc) == ("header", 0)
    run.release(up)


def test_cancel_ends_a_running_stream_and_the_run_goes_on(run):
    up = run.upload(looping_stream())
    t = run.start(up)
    with pytest.raises(km.EngineTimeout):
        run.wait_engine(t, timeout=0.2)
    run.cancel(t)
    with pytest.raises(km.EngineError) as err:
        run.wait_engine(t, timeout=5.0)
    assert err.value.reason_name == "cancelled" and up.state == "idle"
    assert run.wait(run.issue(RESULT, (0, 8), args=[42], result=True)).result == 42


def test_the_run_stops_cleanly(run):
    assert run.stop()["error"] == 0
