"""The dispatcher's engines through Run on a card (KMR_CARD=1), one device open:
a stream of commands with result words, a graph with two branches run and run again
kept, a bad upload, a cancel; values against Stream.simulate and Graph.simulate."""

from pathlib import Path

import pytest

import komorebi as km
from komorebi.engines.graph import Graph
from komorebi.engines.stream import Stream

pytestmark = pytest.mark.card
CARD_IN_PROCESS = True  # opens the card in this process
KERNELS = Path(__file__).resolve().parents[2] / "examples" / "01_mapcollect" / "kernels"
RESULT = 6  # kHandlerResult: with KM_FLAG_RESULT the result word is the leader's a0
CORES = 120


def log(message):
    print(message, flush=True)


def program():
    """The map/collect kernels; only RESULT commands are sent."""
    return km.Program(
        "mapcollect",
        kernels=KERNELS,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
        ],
        compile_args={"reader": [2, 2048], "writer": [2, 2048]},
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        handlers=KERNELS / "mapcollect.h",
    )


def leader_a0(command):
    return command.args[0]


def commands_stream(count, req=11):
    """n commands one at a time, command k on cores [8k mod 120, + 8) with a0 = 7k + 3
    and a result word; HALT the sum of the results."""
    stream = Stream(req=req, deadline_microseconds=500_000)
    counter, first, key, got, acc = stream.registers(5)
    top = stream.label()
    stream.mul(first, counter, stream.const(8))
    stream.mod(first, first, stream.const(CORES))
    stream.mul(key, counter, stream.const(7))
    stream.addi(key, key, 3)
    stream.submit(RESULT, (first, stream.const(8)), args=[key], result=True)
    stream.wait(0)
    stream.result(got)
    stream.add(acc, acc, got)
    stream.addi(counter, counter, 1)
    stream.jlt(counter, stream.const(count), top)
    stream.halt(acc)
    return stream


def branches_graph():
    """Two branches on disjoint halves after a command on both, the right one's result
    word into r; then register 0 = r * 3 + 1, register 1 = r."""
    graph = Graph(req=12, keep=True, deadline_microseconds=500_000)
    binding, result_register = graph.bindings("it"), graph.register()
    root = graph.command(RESULT, group=(0, 60), args=[binding])
    left = graph.command(RESULT, group=(0, 30), args=[binding], after=[root])
    right = graph.command(
        RESULT, group=(30, 30), args=[binding], after=[root], result=result_register
    )
    cpu = graph.cpu(after=[left, right])
    cpu.mul(binding, result_register, graph.const(3))
    cpu.addi(binding, binding, 1)
    return graph


def looping_stream(req=13):
    stream = Stream(req=req, max_steps=0xFFFFFFFF)
    stream.jmp(stream.label())
    return stream


@pytest.fixture(scope="module")
def run():
    with km.Session(log=log) as session:
        yield session.load(program(), deadline=2.0)


def test_a_stream_of_eight_commands_ends_alike_uploaded_and_host_fed(run):
    stream = commands_stream(8)
    upload = run.upload(stream)
    event = run.wait_engine(run.start(upload), timeout=5.0)
    want = stream.simulate(core_count=CORES, result=leader_a0)
    assert event.kind == "end" and event.value == want.halt == 220
    run.release(upload)
    fed = run.run_hostfed(stream, timeout=5.0)
    assert fed.status == "ended" and fed.events[-1].extra == want.halt
    groups = [(command.first, command.count, command.args) for command in want.commands]
    assert [
        (command.first, command.count, command.args) for command in fed.commands
    ] == groups


def test_a_graph_runs_then_runs_again_kept_with_another_binding(run):
    graph = branches_graph()
    upload = run.upload(graph)
    for binding, again in ((5, False), (1000, True)):
        engine = (
            run.restart(12, {"it": binding})
            if again
            else run.start(upload, {"it": binding})
        )
        event = run.wait_engine(engine, timeout=5.0)
        want = graph.simulate({"it": binding}, core_count=CORES, result=leader_a0)
        assert want.status == "ended" and len(want.commands) == 3
        assert (
            event.registers
            == (want.registers[0] & 0xFFFFFF, want.registers[1])
            == (3 * binding + 1, binding)
        )
        assert upload.state == "kept"
    run.release(upload)
    assert upload.state == "idle"  # the kernel dropped it


def test_a_bad_upload_ends_in_its_error_event(run):
    upload = run.upload(bytes(64), kind="stream")
    with pytest.raises(km.EngineError) as raised:
        run.wait_engine(run.start(upload), timeout=5.0)
    assert (raised.value.reason_name, raised.value.pc) == ("header", 0)
    run.release(upload)


def test_cancel_ends_a_running_stream_and_the_run_goes_on(run):
    upload = run.upload(looping_stream())
    engine = run.start(upload)
    with pytest.raises(km.EngineTimeout):
        run.wait_engine(engine, timeout=0.2)
    run.cancel(engine)
    with pytest.raises(km.EngineError) as raised:
        run.wait_engine(engine, timeout=5.0)
    assert raised.value.reason_name == "cancelled" and upload.state == "idle"
    assert run.wait(run.submit(RESULT, (0, 8), args=[42], result=True)).result == 42


def test_the_run_stops_cleanly(run):
    assert run.stop()["error"] == 0
