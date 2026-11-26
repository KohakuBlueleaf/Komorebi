"""Run's engine calls against a fake dispatcher whose engine runs each upload in
the simulator at once: placement, rejections, the events as typed values."""

import struct

import pytest
from fake_port import FakeBackend, FakePort

import komorebi as km
from komorebi import abi
from komorebi.engines import graph as graph_module
from komorebi.engines import stream as stream_module
from komorebi.engines.control import codes, offsets
from komorebi.errors import EngineError, EngineRejected, EngineTimeout, StopTimeout
from komorebi.session.run import Run

CONSTANTS = codes()
HEADER_OFFSETS = offsets()
REGION = 2048


def leader_a0(command):
    return command.args[0]


class EngineKernel(FakePort):
    """The fake kernel with an upload region, the ready block and RUN / CANCEL: an
    upload that ends in the simulator posts its events at once, one that does not
    runs until cancelled; EXIT waits for the engines. Controls: `reject` (the next
    RUN), `hold` (take no command), `deaf` (take a CANCEL and do nothing)."""

    def __init__(self):
        super().__init__(event_depth=64)
        self.region = bytearray(REGION)
        self.writes = []
        self.reject = 0
        self.hold = False
        self.deaf = False
        self.running = {}  # req -> event kind of its cancel
        self.kept = {}  # req -> (offset, bytes) of a kept graph
        self.handlers = {
            CONSTANTS["KM_HANDLER_RUN"]: self.run,
            CONSTANTS["KM_HANDLER_CANCEL"]: self.cancel,
        }

    def host_word(self, word):
        ready = {
            CONSTANTS["KMD_HOST_READY"]: CONSTANTS["KMD_READY_MARK"],
            CONSTANTS["KMD_HOST_READY"] + HEADER_OFFSETS["ready_bytes"]: REGION,
        }
        return ready.get(word, super().host_word(word))

    def _take(self):
        if not self.hold:
            super()._take()

    def _exit_waits(self):
        return bool(self.busy or self.running)

    def upload(self, offset, data):
        assert offset % 64 == 0 and len(data) % 64 == 0
        self.region[offset : offset + len(data)] = data
        self.writes.append((offset, len(data)))

    def publish(self):
        self.published = self.seen

    def run(self, command, words):
        offset_or_req, length, form, bind, binding_count = (
            words[i] for i in (3, 5, 6, 7, 9)
        )
        if self.reject:
            self._event(command, CONSTANTS["KM_EVENT_REJECTED"], auxiliary=self.reject)
            self.reject = 0
            return
        if form == CONSTANTS["KM_GRAPH_RUN_KEPT"]:
            if offset_or_req not in self.kept:
                self._event(
                    command,
                    CONSTANTS["KM_EVENT_REJECTED"],
                    auxiliary=CONSTANTS["KMD_REJECT_UNKNOWN"],
                )
                return
            offset_or_req, length = self.kept[offset_or_req]
        data = bytes(self.region[offset_or_req : offset_or_req + length])
        if form == 0:
            simulation = stream_module.simulate(data, result=leader_a0, max_passes=50)
            ticks = [simulation.halt or 0] * len(simulation.events)
        else:
            values = (
                struct.unpack_from(f"<{binding_count}I", self.region, bind)
                if binding_count
                else ()
            )
            simulation = graph_module.simulate(
                data, bindings=values, result=leader_a0, max_passes=50
            )
            ticks = [event.extra for event in simulation.events]
            if simulation.kept:
                self.kept[simulation.events[0].req] = (offset_or_req, length)
        self.sequence += len(
            simulation.commands
        )  # an engine's commands take sequence numbers
        req_offset = HEADER_OFFSETS["stream" if form == 0 else "graph"]
        req = (
            simulation.events[0].req
            if simulation.events
            else struct.unpack_from("<I", data, req_offset)[0]
        )
        if simulation.status == "stuck":
            self.running[req] = CONSTANTS[
                "KM_EVENT_STREAM_ERROR" if form == 0 else "KM_EVENT_GRAPH_ERROR"
            ]
            return
        for event, tick in zip(simulation.events, ticks):
            self._event(event.req, event.kind, auxiliary=event.auxiliary, ticks=tick)

    def cancel(self, command, words):
        req = words[3]
        if self.deaf:
            return
        if req in self.running:
            self._event(
                req,
                self.running.pop(req),
                auxiliary=CONSTANTS["KM_STREAM_ERROR_CANCELLED"],
            )
        elif self.kept.pop(req, None):
            no_node = (
                CONSTANTS["KM_STREAM_ERROR_CANCELLED"]
                | CONSTANTS["KM_GRAPH_NO_NODE"] << 8
            )
            self._event(req, CONSTANTS["KM_EVENT_GRAPH_ERROR"], auxiliary=no_node)
        else:
            self._event(
                command,
                CONSTANTS["KM_EVENT_REJECTED"],
                auxiliary=CONSTANTS["KMD_REJECT_UNKNOWN"],
            )


def make_run():
    port = EngineKernel()
    return Run(FakeBackend(port), (12, 10)), port


def sum_stream(req=7, count=8):
    """`count` commands with result words one at a time on groups of 8; HALT the sum."""
    stream = stream_module.Stream(req=req)
    i, acc, got = stream.registers(3)
    top = stream.label()
    stream.submit(1, (0, 8), args=[i], result=True)
    stream.wait(0)
    stream.result(got)
    stream.add(acc, acc, got)
    stream.addi(i, i, 3)
    stream.jlt(i, stream.const(3 * count), top)
    stream.halt(acc)
    return stream


def two_branches(keep=False):
    """Two branches on disjoint halves, then register 0 = the result word + 5."""
    graph = graph_module.Graph(req=9, keep=keep)
    binding, result_register = graph.bindings("it"), graph.register()
    root = graph.command(1, group=(0, 60), args=[binding])
    low_half = graph.command(1, group=(0, 30), args=[binding], after=[root])
    high_half = graph.command(
        1, group=(30, 30), args=[binding], after=[root], result=result_register
    )
    graph.cpu(after=[low_half, high_half]).addi(binding, result_register, 5)
    return graph


def test_the_offsets_are_the_headers_static_asserts():
    assert HEADER_OFFSETS == {"ready_bytes": 2, "stream": 24, "graph": 24, "flags": 44}


def test_uploads_sit_side_by_side_and_what_does_not_fit_is_rejected_unwritten():
    run, port = make_run()
    first = run.upload(sum_stream())
    second = run.upload(sum_stream(req=8))
    assert (first.offset, second.offset) == (0, -(-first.bytes // 64) * 64)
    run.release(first)
    third = run.upload(sum_stream(req=10, count=2))
    assert third.offset == 0 and third.kind == "stream" and third.req == 10
    writes = list(port.writes)
    with pytest.raises(ValueError, match="does not fit"):
        run.upload(bytes(REGION), kind="graph")
    assert port.writes == writes


def test_a_stream_ends_with_its_halt_status_beside_host_commands():
    run, port = make_run()
    stream = sum_stream()
    command = run.submit(0, (0, 8))
    engine = run.start(run.upload(stream))
    event = run.wait_engine(engine, timeout=1.0)
    assert (
        event.kind == "end"
        and event.value == stream.simulate(result=leader_a0).halt == 84
    )
    assert run.wait(command).kind == "done" and engine.events == [event]
    assert engine.upload.state == "idle"


def test_an_error_names_the_reason_and_the_pc_or_the_node():
    run, _ = make_run()
    stream = stream_module.Stream(req=3)
    x = stream.registers(1)[0]
    stream.set(x, 4)
    stream.mod(x, x, stream.const(0))
    stream.halt(x)
    with pytest.raises(EngineError) as raised:
        run.wait_engine(run.start(run.upload(stream)), timeout=1.0)
    assert raised.value.reason_name == "mod_zero" and raised.value.pc == 2
    graph = graph_module.Graph(req=4)
    graph.join()
    cpu = graph.cpu(after=[graph.nodes[0]])
    register = graph.register()
    cpu.mod(register, register, register)
    with pytest.raises(EngineError) as raised:
        run.wait_engine(run.start(run.upload(graph)), timeout=1.0)
    assert (raised.value.reason_name, raised.value.node, raised.value.pc) == (
        "mod_zero",
        1,
        None,
    )


def test_a_rejected_run_raises_with_the_reason_name_and_the_run_goes_on():
    run, port = make_run()
    upload = run.upload(sum_stream())
    port.reject = CONSTANTS["KMD_REJECT_BUSY"]
    with pytest.raises(EngineRejected) as raised:
        run.start(upload)
    assert raised.value.reason_name == "busy" and upload.state == "idle"
    assert run.wait_engine(run.start(upload), timeout=1.0).value == 84


def test_a_kept_graph_runs_again_with_other_bindings_then_is_dropped():
    run, port = make_run()
    graph = two_branches(keep=True)
    upload = run.upload(graph)
    first = run.wait_engine(run.start(upload, {"it": 2}), timeout=1.0)
    assert first.registers == (7, 2) and upload.state == "kept"
    again = run.wait_engine(run.restart(9, {"it": 40}), timeout=1.0)
    want = graph.simulate({"it": 40}, result=leader_a0).registers
    assert again.registers == (want[0] & 0xFFFFFF, want[1]) == (45, 40)
    run.release(upload)
    assert port.kept == {} and run.upload(sum_stream()).offset == 0


def test_restart_of_a_graph_never_started_names_start():
    run, _ = make_run()
    upload = run.upload(two_branches(keep=True))
    with pytest.raises(ValueError, match=r"req 9 was uploaded and has not run yet: st"):
        run.restart(9)
    run.wait_engine(run.start(upload, {"it": 2}), timeout=1.0)
    assert run.wait_engine(run.restart(9, {"it": 40}), timeout=1.0).kind == "end"


def test_a_graphs_handler_is_settled_against_the_run_it_is_uploaded_to(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KMR_CACHE", str(tmp_path / "cache"))
    sums, scales = (programmed(tmp_path, name) for name in ("sums", "scales"))
    merged = km.merge(sums, scales, log=lambda line: None)
    graph = graph_module.Graph(req=9)
    graph.command(scales.handler("scale"), group=(0, 8))
    assert graph.templates()[0].rec[1] & 0xFF == 1  # the source's own, as given
    for programs, want in (([merged], (3, 0)), ([sums, scales], (1, 1))):
        run = Run(FakeBackend(EngineKernel()), (12, 10), programs=programs)
        word = graph_module.Graph.decode(run.upload(graph).data).templates()[0].rec[1]
        assert (word & 0xFF, word >> 24) == want


def test_a_streams_handler_is_settled_against_the_run_it_is_uploaded_to(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KMR_CACHE", str(tmp_path / "cache"))
    sums, scales = (programmed(tmp_path, name) for name in ("sums", "scales"))
    merged = km.merge(sums, scales, log=lambda line: None)
    stream = stream_module.Stream(req=7)
    stream.submit(scales.handler("scale"), (0, 8))
    stream.submit(2, (0, 8))  # a plain number: written as given
    stream.halt(stream.registers(1)[0])
    assert [t.rec[1] & 0xFF for t in stream.templates()] == [1, 2]
    for programs, want in (([merged], (3, 0)), ([sums, scales], (1, 1))):
        run = Run(FakeBackend(EngineKernel()), (12, 10), programs=programs)
        uploaded = stream_module.Stream.decode(run.upload(stream).data).templates()
        words = [template.rec[1] for template in uploaded]
        assert [(word & 0xFF, word >> 24) for word in words] == [want, (2, 0)]


def programmed(tmp_path, name):
    """A program in the runtime's form whose kernels do nothing, handlers 1 and 2."""
    folder = tmp_path / name
    folder.mkdir()
    for kernel, kind in (("reader", "Reader"), ("writer", "Writer"), ("compute", "")):
        (folder / f"{kernel}.cpp").write_text(
            f"void km_program(km::{kind or 'Compute'}& kernel, "
            "const km_command_t& command) {}\n"
        )
    return km.Program(
        name,
        kernels=folder,
        buffers=[km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16)],
        handlers={"scale" if name == "scales" else "sum": 1, "empty": 2},
    )


def test_wait_engine_times_out_and_cancel_ends_the_stream():
    run, _ = make_run()
    engine = run.start(run.upload(looping_stream()))
    with pytest.raises(EngineTimeout):
        run.wait_engine(engine, timeout=0.02)
    run.cancel(engine)
    with pytest.raises(EngineError) as raised:
        run.wait_engine(engine, timeout=1.0)
    assert raised.value.reason_name == "cancelled" and engine.upload.state == "idle"


def looping_stream(req=5):
    stream = stream_module.Stream(req=req, max_steps=abi.M32)
    stream.jmp(stream.label())
    return stream


def test_a_run_not_answered_in_time_is_unknown_until_cancel_settles_it():
    run, port = make_run()
    upload = run.upload(looping_stream())
    port.hold = True
    with pytest.raises(EngineTimeout) as raised:
        run.start(upload, timeout=0.02)
    engine = raised.value.handle
    assert engine.unknown and upload.state == "running"
    port.hold = False
    run.cancel(engine)  # the RUN started it, the CANCEL behind it ends it
    with pytest.raises(EngineError, match="cancelled"):
        run.wait_engine(engine, timeout=1.0)
    assert not engine.unknown and upload.state == "idle"
    port.hold, port.reject = True, CONSTANTS["KMD_REJECT_BUSY"]
    with pytest.raises(EngineTimeout) as raised:
        run.start(upload, timeout=0.02)
    port.hold = False
    with pytest.raises(EngineRejected, match="busy"):
        run.cancel(raised.value.handle)  # the RUN was rejected: it never ran
    assert upload.state == "idle" and not run._engine_state.active
    assert run.wait(run.submit(0, (0, 8))).kind == "done"


def test_stop_cancels_a_running_engine_before_exit():
    run, port = make_run()
    engine = run.start(run.upload(looping_stream()))
    result = run.stop(timeout=1.0)
    assert result["error"] == 0 and port.exited and not port.running
    assert engine.end.reason_name == "cancelled" and run._engine_state.active == {}


def test_stop_raises_the_timeout_of_an_unanswered_cancel_and_still_closes():
    run, port = make_run()
    engine = run.start(run.upload(looping_stream()))
    port.deaf = True
    with pytest.raises(
        StopTimeout, match=r"of its 0\.05 s in the engines' stop"
    ) as raised:
        run.stop(timeout=0.05)
    assert isinstance(raised.value.__cause__, EngineTimeout)
    assert raised.value.__cause__.handle is engine and run.stopped
    assert run.backend.calls == ["abort", "read_back"] and not port.exited


def test_host_commands_after_an_engine_have_the_kernels_sequence_numbers():
    port = EngineKernel()
    run = Run(FakeBackend(port, trace_cap=64), (12, 10))
    before = run.submit(0, (0, 8))
    run.wait_engine(run.start(run.upload(sum_stream())), timeout=1.0)
    after = run.submit(0, (8, 8))
    run.wait([before, after])
    assert port.submitted[-1][1] == 10  # 8 were the stream's
    assert run.sequence(before) == 1 and run.sequence(after) == 10
    run.stop()
    assert run.trace().command(before) is not None
    assert run.trace().command(after).shape == (8, 4)


def test_a_stream_host_fed_submits_the_simulated_commands_and_ends_alike():
    run, port = make_run()
    port.results = {
        index + 1: 3 * index for index in range(8)
    }  # command k + 1: the leader's a0
    stream = sum_stream()
    got = run.run_hostfed(stream, timeout=1.0)
    want = stream.simulate(result=leader_a0)
    assert got.status == "ended" and got.events[-1].extra == want.halt == 84
    groups = [(command.first, command.count, command.args) for command in want.commands]
    assert [
        (command.first, command.count, command.args) for command in got.commands
    ] == groups
    assert [(event.kind, event.auxiliary) for event in got.events] == [
        (event.kind, event.auxiliary) for event in want.events
    ]
