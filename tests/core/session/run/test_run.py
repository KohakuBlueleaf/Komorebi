"""Run: handles, batches, events and stopping, against a fake dispatcher."""

import time

import numpy as np
import pytest
from fake_port import FakeBackend, FakePort

from komorebi import abi
from komorebi.device.events import Event
from komorebi.errors import (
    CommandRejected,
    CommandTimeout,
    KernelFailed,
    KomorebiError,
    WaitTimeout,
)
from komorebi.program.description import Program
from komorebi.session.parts import run_events
from komorebi.session.run import Run

CONSTANTS = abi.headers().constants
GRID = (12, 10)


def make_run(port=None, batch=64, trace_cap=0, stalled_core=None, program=None):
    port = port or FakePort()
    backend = FakeBackend(port, trace_cap=trace_cap, stalled_core=stalled_core)
    return Run(backend, GRID, program=program, batch=batch), port, backend


def test_submit_only_queues_and_returns_increasing_handles():
    run, port, _ = make_run()
    handles = [run.submit(0, (i, 1)) for i in range(5)]
    assert handles == [1, 2, 3, 4, 5]
    assert port.next == 1 and port.batches == []


def test_wait_stores_in_batches_and_returns_the_events():
    port = FakePort(host_depth=64, event_depth=128)
    run, port, _ = make_run(port, batch=8)
    handles = [run.submit(5, (i % 120, 1), [i]) for i in range(30)]
    events = run.wait(handles)
    assert [event.handle for event in events] == handles
    assert all(event.kind == "done" for event in events)
    assert max(port.batches) <= 8 and sum(port.batches) == 30
    submitted = [(req, handler, group) for req, _, _, handler, group in port.submitted]
    assert submitted == [(handle, 5, (i % 120, 1)) for i, handle in enumerate(handles)]
    with pytest.raises(ValueError, match="returned or forgotten"):
        run.wait(handles[3])


def test_a_wait_sends_only_up_to_its_handle():
    run, port, _ = make_run()
    first = run.submit(0, (0, 1))
    run.submit(0, (1, 1))
    run.wait(first)
    assert port.next == 2  # the second command is still queued


def test_the_host_never_overruns_the_rings():
    port = FakePort(depth=2, host_depth=4, event_depth=8)
    port.auto = False
    run, port, _ = make_run(port)
    handles = [run.submit(0, (0, 1)) for _ in range(20)]
    with pytest.raises(WaitTimeout) as caught:
        run.wait(handles, timeout=0.05)
    assert port.next - 1 - port.consumed <= 4
    assert port.next - 1 - port.seen <= 8
    assert caught.value.pending == handles
    port.auto = True
    run.wait(handles)
    assert port.max_unread <= 8 and len(port.submitted) == 20


def test_groups_and_handlers_are_checked_before_queueing():
    run, _, _ = make_run()
    for group in [(0, 0), (-1, 2), (119, 2), (0, 121)]:
        with pytest.raises(ValueError, match="group"):
            run.submit(1, group)
    for handler in [240, 251, 255, -1]:
        with pytest.raises(ValueError, match="handler"):
            run.submit(handler, (0, 1))
    with pytest.raises(ValueError, match="arguments"):
        run.submit(1, (0, 1), list(range(11)))
    with pytest.raises(ValueError, match="32-bit"):
        run.submit(1, (0, 1), [1 << 32])
    assert run.submit(0, None) == 1 and run.group(1) == (0, 120)


def test_handlers_by_name_through_the_program(tmp_path):
    header = tmp_path / "p.h"
    header.write_text(
        "constexpr uint32_t kHandlerMap = 1;\nconstexpr uint32_t kHandlerBigStep = 7;\n"
    )
    program = Program("p", kernels=tmp_path, handlers=header)
    assert program.handlers == {"map": 1, "big_step": 7}
    run, port, _ = make_run(program=program)
    run.wait(run.submit("big_step", (0, 4)))
    assert port.submitted[0][3] == 7
    with pytest.raises(KeyError):
        run.submit("missing", (0, 1))


def test_exclusive_commands_are_stored_at_once_and_the_kernel_orders_them():
    port = FakePort(depth=4)
    port.auto = False
    run, port, _ = make_run(port)
    first = run.submit(1, (0, 10), exclusive=True)
    second = run.submit(1, (5, 10), exclusive=True)  # overlaps a
    third = run.submit(1, (40, 10), exclusive=True)  # shares no core, behind b in order
    with pytest.raises(WaitTimeout):
        run.wait(third, timeout=0.05)
    assert run.stored == 3 and [i[0] for i in port.submitted] == [first]
    port.auto = True
    run.wait([first, second, third])
    assert [i[0] for i in port.submitted] == [first, second, third]


def test_a_command_that_times_out_ends_the_run_before_the_exclusive_one_behind_it():
    port = FakePort(depth=4)
    run, port, _ = make_run(port)
    stalled = run.submit(1, (0, 10))
    behind = run.submit(1, (5, 10), exclusive=True)
    port.stall_req = stalled
    with pytest.raises(CommandTimeout) as caught:
        run.wait([stalled, behind])
    assert caught.value.handle == stalled and run.stored == 2
    assert [i[0] for i in port.submitted] == [stalled]
    assert run.event(behind) is None and run.result()["error_name"] == "deadline"


def test_the_read_count_lets_the_kernel_go_on_past_a_full_event_ring():
    port = FakePort(depth=4, host_depth=8, event_depth=8)
    run, port, _ = make_run(port)
    handles = [run.submit(0, (i % 100, 4)) for i in range(200)]
    events = run.wait(handles, timeout=1.0)
    assert all(event.kind == "done" for event in events)
    # the host's own room (unread plus in flight) is the tighter bound
    assert port.held_for_room == 0 and 0 < port.published <= port.seen


def test_a_host_that_never_publishes_its_read_count_stalls_the_kernel():
    class Silent(FakePort):
        def push(self, rows):
            published = self.published
            pushed = super().push(rows)
            self.published = published  # the batch did not carry the read count
            return pushed

        def _publish_if_held(self):
            pass  # waits without storing the read count

    run, port, _ = make_run(Silent(depth=4, host_depth=8, event_depth=8))
    handles = [run.submit(0, (0, 4)) for _ in range(20)]
    with pytest.raises(WaitTimeout):
        run.wait(handles, timeout=0.05)


def test_sequences_follow_commands_and_skip_kernel_commands():
    run, port, _ = make_run()
    handles = [run.submit(0, (0, 1)) for _ in range(3)]
    run.drain()
    assert [run.sequence(handle) for handle in handles] == [
        sequence for _, sequence, _, _, _ in port.submitted
    ]


def test_a_rejected_command_raises_for_its_handle_and_the_run_goes_on():
    run, port, _ = make_run()
    good = run.submit(1, (0, 4))
    bad = run._queue_command(1, (110, 20))  # bypasses the host's own check
    after = run.submit(1, (0, 4))
    with pytest.raises(CommandRejected, match="reason 3: group") as caught:
        run.wait([good, bad, after])
    assert caught.value.handle == bad
    assert caught.value.reason == CONSTANTS["KMD_REJECT_GROUP"]
    assert run.event(after).kind == "done"
    assert run.sequence(after) == 2 and run.sequence(bad) is None
    assert run.wait(run.submit(1, (0, 4))).kind == "done"


def test_sequences_skip_0_at_the_wrap_as_the_kernel_numbers_them():
    run, port, _ = make_run()
    run._next_sequence, port.sequence = abi.M32 - 1, abi.M32 - 2
    handles = [run.submit(0, (0, 1)) for _ in range(3)]
    run.drain()
    assert [run.sequence(handle) for handle in handles] == [abi.M32 - 1, abi.M32, 1]
    assert [sequence for _, sequence, _, _, _ in port.submitted] == [
        abi.M32 - 1,
        abi.M32,
        1,
    ]


def test_the_kernel_holds_exit_while_a_slot_is_busy_and_rejects_commands_after_it():
    run, port, _ = make_run()
    port.auto = False
    busy = run.submit(1, (0, 4))
    run._send(busy, 1.0)
    exit_ = run._queue_command(CONSTANTS["KM_HANDLER_EXIT"], None)
    run._send(exit_, 1.0)
    run.port.poll()
    assert port.held_exit > 0 and not port.exited
    port.auto = True
    run.wait([busy, exit_])
    assert port.exited and port.submitted[-1][3:] == (
        CONSTANTS["KM_HANDLER_EXIT"],
        (0, 120),
    )
    late = run._queue_command(1, (0, 4))  # past the host's own stop
    with pytest.raises(CommandRejected, match="exited"):
        run.wait(late)


def test_stop_drains_sends_exit_to_every_core_then_stop():
    run, port, backend = make_run(trace_cap=8)
    handles = [run.submit(1, (i, 2)) for i in range(3)]
    result = run.stop()
    assert all(run.event(handle).kind == "done" for handle in handles)
    exit_req, _, _, handler, group = port.submitted[-1]
    assert handler == CONSTANTS["KM_HANDLER_EXIT"] and group == (0, 120)
    assert port.state == CONSTANTS["KMD_STATE_DONE"] and result["error"] == 0
    assert backend.calls == ["finish", "read_back"]
    assert run.stopped and len(run.state()) == 120
    assert run.trace().phases(handles[1])["count"] == 2
    with pytest.raises(KomorebiError, match="stopped"):
        run.submit(1, (0, 1))
    assert run.stop() is result


@pytest.mark.parametrize(
    "answer, says",
    [
        ({"result_ready": False, "failed": "timeout waiting for the core"}, "never"),
        ({"cleaned": False}, "could not restore"),
        ({"state": CONSTANTS["KMD_STATE_READY"]}, "state ready"),
    ],
)
def test_stop_raises_unless_the_kernel_finished_cleanly(answer, says):
    run, port, backend = make_run()
    backend.answer = answer
    run.wait(run.submit(1, (0, 2)))
    with pytest.raises(KernelFailed, match=says) as caught:
        run.stop()
    assert caught.value.result is not None and run.stopped
    with pytest.raises(KernelFailed):
        run.submit(1, (0, 1))


def test_abort_raises_when_the_runtime_could_not_restore_the_tile():
    run, _, backend = make_run()
    backend.cleaned = False
    backend.close_failed = "core 0 phase=0x2 trap(mcause=0x7)"
    words, x280_cores = [0] * CONSTANTS["KMD_R_WORDS"], [0] * 16
    words[CONSTANTS["KMD_R_STATE"]], x280_cores[CONSTANTS["KMD_H_ABANDONED"]] = (
        CONSTANTS["KMD_STATE_FAILED"],
        3,
    )
    backend.aborted = {"words": words, "x280_cores": x280_cores}
    with pytest.raises(KernelFailed, match=r"mcause=0x7.*failed, 3 commands aband"):
        run.abort()
    assert backend.calls == ["abort", "read_back"] and backend.grid_stopped
    assert run.result()["abandoned"] == 3


class StopFails(FakeBackend):
    """A backend whose finish raises after it read the result block (the grid's wait),
    or before it (the close: abort keeps what the kernel left)."""

    def __init__(self, port, before):
        super().__init__(port)
        self.before, self.logged = before, []
        self.card = type("Card", (), {"log": staticmethod(self.logged.append)})

    def finish(self, timeout):
        if self.before:
            self.calls.append("finish")
            raise RuntimeError("the runtime's close failed")
        return super().finish(timeout)

    def read_back(self, stop_grid=False):
        super().read_back(stop_grid)
        raise RuntimeError("the wait for the grid's kernels failed: core 22-27")


@pytest.mark.parametrize("before", [False, True])
def test_a_kernel_that_failed_tells_its_result_block_when_the_stop_fails(before):
    port = FakePort()
    port.auto = False
    run, port, _ = make_run(port)
    backend = StopFails(port, before)
    run.backend = backend
    words = [0] * CONSTANTS["KMD_R_WORDS"]
    words[CONSTANTS["KMD_R_STATE"]] = CONSTANTS["KMD_STATE_FAILED"]
    words[CONSTANTS["KMD_R_ERROR"]] = CONSTANTS["KMD_ERROR_DEADLINE"]
    words[CONSTANTS["KMD_R_SLOT"]], words[CONSTANTS["KMD_R_SEQUENCE"]] = 3, 41
    port.result = {"error": CONSTANTS["KMD_ERROR_DEADLINE"], "slot": 3, "sequence": 41}
    port.state = CONSTANTS["KMD_STATE_FAILED"]  # it ended on its own
    backend.aborted = {"words": words, "x280_cores": [0] * 16} if before else None
    with pytest.raises(KernelFailed) as failed:
        run._fail_kernel("the dispatcher finished on its own")
    text = str(failed.value)
    assert "error deadline" in text and "slot 3, sequence 41" in text
    assert "the wait for the grid's kernels failed" in text
    assert isinstance(failed.value.__cause__, RuntimeError)
    assert backend.logged == [text] and failed.value.result["slot"] == 3


def test_a_kernel_that_failed_says_its_block_was_not_read_when_nothing_was():
    run, port, _ = make_run()
    backend = StopFails(port, before=True)
    run.backend = backend  # aborted stays None: the card gave nothing back
    with pytest.raises(
        KernelFailed, match="block not readable; then RuntimeError: the"
    ):
        run._fail_kernel("the dispatcher finished on its own")


@pytest.mark.parametrize("before", [False, True])
def test_a_timeout_tells_its_result_block_when_the_stop_fails(before):
    port = FakePort()
    port.auto = False
    run, port, _ = make_run(port)
    handle = run.submit(1, (0, 4))
    backend = StopFails(port, before)
    run.backend = backend
    words = [0] * CONSTANTS["KMD_R_WORDS"]
    words[CONSTANTS["KMD_R_STATE"]] = CONSTANTS["KMD_STATE_FAILED"]
    words[CONSTANTS["KMD_R_ERROR"]] = CONSTANTS["KMD_ERROR_DEADLINE"]
    words[CONSTANTS["KMD_R_SLOT"]], words[CONSTANTS["KMD_R_SEQUENCE"]] = 3, 41
    port.result = {"error": CONSTANTS["KMD_ERROR_DEADLINE"], "slot": 3, "sequence": 41}
    port.state = CONSTANTS["KMD_STATE_FAILED"]
    backend.aborted = {"words": words, "x280_cores": [0] * 16} if before else None
    with pytest.raises(CommandTimeout) as failed:
        run._fail_timeout(Event(handle, "timeout", 3, 0))
    text = str(failed.value)
    assert text.startswith(f"command {handle} timed out; not reported: slot 3, ")
    assert "sequence 41, group first 0 count 4" in text
    assert "the dispatcher's result block: error deadline in" in text
    assert "the cores' words not read; the dispatcher's result block" in text
    assert "then RuntimeError: the wait for the grid's kernels failed" in text
    assert isinstance(failed.value.__cause__, RuntimeError)
    assert backend.logged == [text] and failed.value.result["slot"] == 3
    assert run._failure is failed.value  # the run's later calls raise it


def test_a_timeout_says_its_block_was_not_read_when_nothing_was():
    run, port, _ = make_run()
    handle = run.submit(1, (0, 4))
    backend = StopFails(port, before=True)
    run.backend = backend  # aborted stays None: the card gave nothing back
    with pytest.raises(
        CommandTimeout, match="block not readable; then RuntimeError: the"
    ) as failed:
        run._fail_timeout(Event(handle, "timeout", 3, 0))
    assert failed.value.result is None and failed.value.cores == []


def test_a_wait_that_runs_out_says_what_the_kernel_is_doing():
    run, port, backend = make_run()
    port.auto = False
    handle = run.submit(1, (0, 1))
    with pytest.raises(WaitTimeout, match="1 of 1 .*; the fake kernel's report"):
        run.wait(handle, timeout=0.02)
    with pytest.raises(WaitTimeout, match="none of 1 .*; the fake kernel's report"):
        run.wait_any([handle], timeout=0.02)
    assert backend.calls == ["kernel_report"] * 2


def test_stop_waits_for_commands_in_flight_before_exit():
    port = FakePort(depth=4)
    run, port, _ = make_run(port)
    handles = [run.submit(1, (10 * i, 10)) for i in range(3)]
    port.auto = False
    with pytest.raises(WaitTimeout):
        run.wait(handles, timeout=0.05)
    assert len(port.busy) == 3  # stored and taken, not complete
    port.auto = True
    run.stop()
    assert port.held_exit == 0  # the host drained: the kernel had nothing to hold
    assert [run.event(handle).kind for handle in handles] == ["done"] * 3
    assert port.submitted[-1][3] == CONSTANTS["KM_HANDLER_EXIT"] and port.submitted[-1][
        4
    ] == (
        0,
        120,
    )


def stray_event(port, req):
    port.pushed_events += 1
    port.events.append([port.pushed_events, req, CONSTANTS["KM_EVENT_DONE"], 7])


@pytest.mark.parametrize("req", [9, 1])  # never stored; already filed
def test_an_event_for_a_command_not_in_flight_stops_the_run(req):
    run, port, backend = make_run()
    run.wait(run.submit(1, (0, 1)))
    stray_event(port, req)
    with pytest.raises(KernelFailed, match=f"command {req} .*not in flight"):
        run.wait(run.submit(1, (0, 1)))
    assert backend.calls == ["abort", "read_back"] and run.stopped


def test_a_handle_is_released_once_its_event_was_returned(monkeypatch):
    monkeypatch.setattr(run_events, "COMPACT_ROWS", 64)
    port = FakePort(host_depth=64, event_depth=128)
    run, port, _ = make_run(port)
    for _ in range(10):
        run.wait([run.submit(0, (i, 1)) for i in range(100)])
    assert len(run._has) < 2 * run_events.COMPACT_ROWS and run._base > 500
    assert (
        run.event(1) is None and run.sequence(1) is None and len(port.submitted) == 1000
    )


def test_forget_releases_handles_read_after_a_drain():
    run, _, _ = make_run()
    handles = [run.submit(0, (0, 1)) for _ in range(5)]
    run.drain()
    assert run.event(handles[0]).kind == "done" and run.sequence(handles[4]) == 5
    run.forget(handles)
    assert run.event(handles[0]) is None and run.sequence(handles[4]) is None
    assert run._gone[:5].all()


class SlowPort(FakePort):
    """Each command (EXIT too) completes `delay` seconds after it was submitted."""

    def __init__(self, delay):
        super().__init__()
        self.delay, self.since = delay, {}

    def _complete(self):
        now = time.monotonic()
        for slot, (sequence, req, _) in sorted(self.busy.items()):
            if now - self.since.setdefault(sequence, now) >= self.delay:
                del self.busy[slot]
                self._event(req, CONSTANTS["KM_EVENT_DONE"])


class SlowBackend(FakeBackend):
    """The kernel's result is ready `delay` seconds into finish(), if it waits."""

    def finish(self, timeout):
        time.sleep(min(self.port.delay, timeout))
        self.answer["result_ready"] = timeout >= self.port.delay
        return super().finish(timeout)


def test_stop_ends_within_its_timeout_when_every_part_is_slow():
    port = SlowPort(delay=0.06)  # drain, EXIT and finish: 0.18 s in all
    run = Run(SlowBackend(port), GRID)
    run.submit(1, (0, 2))
    started = time.monotonic()
    with pytest.raises(KernelFailed, match="never became ready"):
        run.stop(timeout=0.14)
    assert time.monotonic() - started < 0.14 + 0.05 and run.stopped


class CardThread:
    """A run's work-cluster thread on the card, as far as Run looks at it."""

    on_card, ring, running = True, object(), False

    def stop(self, timeout):
        assert 0 < timeout <= 10.0  # inside the run's stop deadline

    def grid_stopped(self):
        pass


def test_sequences_are_exact_while_a_card_thread_submits_commands_between():
    run, port, _ = make_run(trace_cap=1024)
    run.thread = CardThread()  # started at the load, before any host command
    first = run.submit(1, (0, 2))
    run.wait(first)
    port.sequence += 700  # the thread's commands, numbered by the same kernel
    second = run.submit(1, (2, 2))
    run.wait(second)
    assert run.group(second) == (2, 2)
    assert (
        [run.sequence(handle) for handle in (first, second)]
        == [1, 702]
        == [x[1] for x in port.submitted]
    )
    run.stop()
    assert run.trace().phases(second)["count"] == 2


def test_a_gap_of_other_sources_too_long_to_tell_raises():
    run, port, _ = make_run(trace_cap=8)  # keeps the sequences of waited handles
    first = run.wait(run.submit(1, (0, 2))).handle
    port.sequence += 40_000
    second = run.wait(run.submit(1, (0, 2))).handle
    assert run.sequence(first) == 1
    with pytest.raises(KomorebiError, match="other sources"):
        run.sequence(second)


def test_a_queued_command_gets_its_first_core_and_sequence_from_its_event():
    run, port, _ = make_run(trace_cap=8)
    busy = run.submit(1, (0, 4))
    queued = run.submit(1, 3, queued=True)
    run.wait([busy, queued])
    assert run.group(queued) == (port.submitted[-1][4][0], 3)
    assert run.sequence(queued) == port.submitted[-1][1] == 2
    run.stop()
    assert run.trace().command(queued).shape == (3, 4)


def test_a_traced_run_keeps_groups_and_sequences_for_the_trace():
    run, _, _ = make_run(trace_cap=8)
    handles = [run.submit(1, (i, 2)) for i in range(3)]
    run.wait(handles)
    assert run.event(handles[0]) is None and run.sequence(handles[2]) == 3
    assert run.group(handles[1]) == (1, 2)
    run.stop()
    assert run.trace().phases(handles[1])["count"] == 2


def test_state_and_trace_only_after_the_stop():
    run, _, _ = make_run()
    with pytest.raises(KomorebiError, match="after"):
        run.state()
    with pytest.raises(KomorebiError, match="after"):
        run.trace()


def test_a_timeout_names_command_slot_sequence_and_cores():
    port = FakePort()
    run, port, backend = make_run(port, stalled_core=6)
    before = [run.submit(1, (4, 8)) for _ in range(2)]
    run.wait(before)
    stalled = run.submit(2, (4, 8), [2])
    port.stall_req = stalled
    with pytest.raises(CommandTimeout) as caught:
        run.wait(stalled)
    error = caught.value
    assert error.handle == stalled and error.group == (4, 8)
    assert error.sequence == run.sequence(stalled) == 3
    assert error.slot == port.submitted[-1][2]
    assert [entry["core"] for entry in error.cores] == [4, 6]
    assert error.cores[1]["reader"]["where"] == "stall"
    assert error.cores[0]["writer"]["where"] == "complete"
    assert "core 6 reader gave up (stop) in stall" in str(error)
    assert backend.calls == ["finish", "read_back"]
    assert run.result()["error_name"] == "deadline"
    with pytest.raises(CommandTimeout):
        run.submit(1, (0, 1))


def test_a_late_command_to_a_thread_is_its_event_and_the_run_goes_on():
    run, port, _ = make_run()
    timeout = CONSTANTS["KM_EVENT_TIMEOUT"]
    port.handlers[10] = lambda req, words: port._event(req, timeout, auxiliary=2)
    late = run._queue_command(10, (CONSTANTS["KM_GROUP_CLUSTER_BASE"] + 2, 1))
    assert run.wait(late).kind == "timeout"
    assert run.wait(run.submit(1, (0, 1))).kind == "done"


def a_program_with_a_thread(tmp_path):
    program = Program("p", kernels=tmp_path, handlers={"map": 1})
    program.thread(cluster=2, sources=["t.c"], regions={"out": 64}, handlers={"sum": 5})
    return program


def test_a_command_to_a_thread_names_its_cluster_and_its_handler(tmp_path):
    run, port, _ = make_run(program=a_program_with_a_thread(tmp_path))
    done = CONSTANTS["KM_EVENT_DONE"]
    port.handlers[5] = lambda req, words: port._event(req, done, 2, ticks=0x1234)
    handle = run.submit("sum", cluster=2, args=(3, 4), result=True)
    event = run.wait(handle)
    words = port.ring[handle]
    assert words[1] == 5 | CONSTANTS["KM_FLAG_RESULT"] << 8
    assert words[2] == CONSTANTS["KM_GROUP_CLUSTER_BASE"] + 2 | 1 << 16
    assert (words[3], words[5]) == (3, 4)
    assert event.kind == "done" and event.result == 0x1234
    assert run.wait(run.submit("map", (0, 4))).kind == "done"


def test_a_command_to_a_thread_takes_no_group_and_no_core_flags(tmp_path):
    run, _, _ = make_run(program=a_program_with_a_thread(tmp_path))
    for options in [
        {"group": (0, 1)},
        {"exclusive": True},
        {"queued": True},
    ]:
        with pytest.raises(ValueError, match="takes no group"):
            run.submit(5, cluster=2, **options)
    for cluster in [0, 4, True]:
        with pytest.raises(ValueError, match="work clusters"):
            run.submit(5, cluster=cluster)
    with pytest.raises(ValueError, match="no thread on cluster 1"):
        run.submit(5, cluster=1)
    with pytest.raises(KeyError):
        run.submit("map", cluster=2)  # a grid handler, not the thread's


def test_a_full_thread_rejects_the_hosts_command_by_name():
    run, port, _ = make_run()
    reason = CONSTANTS["KMD_REJECT_THREAD"]
    rejected = CONSTANTS["KM_EVENT_REJECTED"]
    port.handlers[5] = lambda req, words: port._event(req, rejected, reason)
    with pytest.raises(CommandRejected, match=f"reason {reason}: thread"):
        run.wait(run.submit(5, cluster=1))
    assert run.wait(run.submit(1, (0, 1))).kind == "done"


def test_a_kernel_that_ends_on_its_own_raises_kernel_failed():
    run, port, _ = make_run()
    run.wait(run.submit(1, (0, 1)))
    port._finish(CONSTANTS["KMD_ERROR_IDLE"])
    with pytest.raises(KernelFailed, match="idle"):
        run.wait(run.submit(1, (0, 1)), timeout=1.0)
    assert run.stopped


def test_a_trap_in_the_kernel_names_the_core_and_the_exception():
    run, port, _ = make_run()
    run.wait(run.submit(1, (0, 1)))
    port._finish(CONSTANTS["KMD_ERROR_TRAP"], slot=5, sequence=0x8003A40, req=0)
    port.result["seen"] = 0x6C8
    want = r"trap in .*core 0 \(mcause 0x5, mepc 0x8003a40, mtval 0x6c8\)"
    with pytest.raises(KernelFailed, match=want):
        run.wait(run.submit(1, (0, 1)), timeout=1.0)
    assert run.stopped


def test_an_error_while_stopping_aborts_through_the_runtime():
    port = FakePort()
    port.auto = False
    run, port, backend = make_run(port)
    run.submit(1, (0, 1))
    with pytest.raises(WaitTimeout):
        run.stop(timeout=0.05)  # the kernel is probed before the abort ends it
    assert backend.calls == ["kernel_report", "abort", "read_back"] and run.stopped


def test_handles_map_onto_command_numbers_across_the_32_bit_wrap():
    run, port, _ = make_run(FakePort(depth=4, host_depth=8, event_depth=16))
    start = (1 << 32) - 3  # as after 2**32 - 4 commands
    run._next = start
    run._new_handles(start)
    port.next = start
    port.consumed = port.seen = port.pushed_events = start - 1
    handles = [run.submit(0, (i, 1)) for i in range(5)]
    handles += list(run.submit_many(0, np.arange(5, 25), 1))
    assert handles == list(range(start, start + 25)) and handles[-1] > abi.M32
    assert run.drain() == 25
    assert [run.event(handle).handle for handle in handles] == handles
    assert [request for request, _, _, _, _ in port.submitted] == handles
    assert [run.sequence(handle) for handle in handles] == list(range(1, 26))


def test_python_cost_counters_add_up():
    run, _, _ = make_run()
    run.wait([run.submit(0, (0, 1)) for _ in range(10)])
    assert run.submit_s > 0 and run.wait_s >= run.native_s > 0 and run.stored == 10


def test_submit_many_encodes_as_submit_does():
    one, port_one, _ = make_run()
    many, port_many, _ = make_run()
    firsts, counts = np.array([0, 5, 119]), np.array([4, 10, 1])
    args = np.array([[1, 2, 3], [4, 5, 6], [7, 8, abi.M32]], np.uint64)
    singles = [
        one.submit(7, (first, count), arguments)
        for first, count, arguments in zip(firsts, counts, args)
    ]
    handles = many.submit_many(7, firsts, counts, args)
    assert handles == range(1, 4) and singles == list(handles)
    one.drain()
    many.drain()
    assert port_one.ring == port_many.ring
    assert [many.group(handle) for handle in handles] == [(0, 4), (5, 10), (119, 1)]


def test_submit_many_broadcasts_and_takes_a_handler_per_command():
    run, port, _ = make_run()
    handles = run.submit_many(np.array([0, 3, 0, 3]), 0, 120)
    run.wait(handles)
    assert [i[3] for i in port.submitted] == [0, 3, 0, 3]
    assert {i[4] for i in port.submitted} == {(0, 120)}


def test_submit_many_checks_every_command_before_queueing_any():
    run, port, _ = make_run()
    for firsts, counts in [([0, 119], [1, 2]), ([0, -1], 1), ([0, 1], [1, 0])]:
        with pytest.raises(ValueError, match="group"):
            run.submit_many(1, np.array(firsts), np.array(counts))
    with pytest.raises(ValueError, match="handler"):
        run.submit_many(np.array([1, 240]), 0, 1)
    with pytest.raises(ValueError, match="32-bit"):
        run.submit_many(1, [0, 1], 1, np.array([[1], [1 << 32]]))
    with pytest.raises(ValueError, match="32-bit"):
        run.submit_many(1, [0, 1], 1, np.array([[1], [-1]]))
    with pytest.raises(ValueError, match="m <= 10"):
        run.submit_many(1, [0], 1, np.zeros((1, 11), np.uint32))
    with pytest.raises(ValueError, match="one row each"):
        run.submit_many(1, [0, 1], [1, 1, 1])
    with pytest.raises(ValueError, match="integers"):
        run.submit_many(1, [0.0], 1)
    assert run._next == 1 and port.queued == 0


def test_submit_many_and_submit_keep_one_order():
    run, port, _ = make_run()
    first = run.submit(0, (0, 1))
    middle = run.submit_many(0, [1, 2], 1)
    last = run.submit(0, (3, 1))
    run.drain()
    assert [i[0] for i in port.submitted] == [first, *middle, last]
    assert [i[4][0] for i in port.submitted] == [0, 1, 2, 3]
    assert [run.sequence(handle) for handle in [first, *middle, last]] == [1, 2, 3, 4]


def test_a_drain_of_many_commands_is_one_native_loop_per_slice():
    port = FakePort(depth=4, host_depth=64, event_depth=128)
    run, port, _ = make_run(port)
    handles = run.submit_many(0, np.arange(1000) % 120, 1)
    assert run.drain() == 1000
    assert all(run._has[: len(handles)]) and len(port.submitted) == 1000
    assert port.waits < 5 and max(port.batches) == 64
    assert run.event(handles[-1]).kind == "done"


def test_exclusive_commands_from_submit_many_wait_for_overlapping_groups():
    port = FakePort(depth=4)
    port.auto = False
    run, port, _ = make_run(port)
    first = run.submit(1, (0, 10))
    held = run.submit_many(1, [5, 50], 10, exclusive=True)
    with pytest.raises(WaitTimeout):
        run.wait(held[0], timeout=0.05)
    assert run.stored == 2 and [i[0] for i in port.submitted] == [
        first
    ]  # up to held[0]
    port.auto = True
    run.wait(held)
    assert [i[0] for i in port.submitted] == [first, *held]


def test_a_result_command_returns_the_leaders_word_on_its_done_event():
    run, port, _ = make_run()
    with_result, plain = run.submit(1, (0, 8), result=True), run.submit(1, (8, 8))
    port.results[with_result] = 0xBEEF
    one, other = run.wait([with_result, plain])
    assert one.result == 0xBEEF and other.result is None
    assert port.ring[with_result][1] >> 8 == CONSTANTS["KM_FLAG_RESULT"]


def test_a_queued_command_takes_its_count_and_the_kernel_picks_the_group():
    run, port, _ = make_run()
    port.auto = False
    run.submit(1, (0, 60))
    queued = run.submit(1, 30, queued=True)
    with pytest.raises(WaitTimeout):
        run.wait(queued, timeout=0.05)
    assert port.submitted[-1][4] == (60, 30) and run.group(queued) == (None, 30)
    with pytest.raises(ValueError, match="group"):
        run.submit(1, 121, queued=True)


def test_wait_any_returns_the_first_event_of_a_set_and_forgets_only_it():
    run, port, _ = make_run()
    port.auto = False
    first, second = run.submit(1, (0, 8)), run.submit(1, (8, 8))
    with pytest.raises(WaitTimeout):
        run.wait_any([second, first], timeout=0.05)
    port.auto = True
    assert run.wait_any([second, first]).handle == first
    assert run.wait_any([second]).handle == second
    with pytest.raises(ValueError, match="returned or forgotten"):
        run.wait_any([first])
