"""Run: tickets, batches, events and stopping, against a fake dispatch kernel."""

import numpy as np
import pytest
from fake_port import FakeBackend, FakePort

from komorebi import abi
from komorebi.errors import (
    KernelFailed,
    KomorebiError,
    RecordRefused,
    RecordTimeout,
    WaitTimeout,
)
from komorebi.program import Program
from komorebi.run import Run

K = abi.headers().constants
GRID = (12, 10)


def make_run(port=None, batch=64, trace_cap=0, stalled_core=None, program=None):
    port = port or FakePort()
    backend = FakeBackend(port, trace_cap=trace_cap, stalled_core=stalled_core)
    return Run(backend, GRID, program=program, batch=batch), port, backend


def test_issue_only_queues_and_returns_increasing_tickets():
    run, port, _ = make_run()
    tickets = [run.issue(0, (i, 1)) for i in range(5)]
    assert tickets == [1, 2, 3, 4, 5]
    assert port.next == 1 and port.batches == []


def test_wait_stores_in_batches_and_returns_the_events():
    port = FakePort(host_depth=64, event_depth=128)
    run, port, _ = make_run(port, batch=8)
    tickets = [run.issue(5, (i % 120, 1), [i]) for i in range(30)]
    events = run.wait(tickets)
    assert [e.ticket for e in events] == tickets
    assert all(e.kind == "done" for e in events)
    assert max(port.batches) <= 8 and sum(port.batches) == 30
    issued = [(req, seg, team) for req, _, _, seg, team in port.issued]
    assert issued == [(t, 5, (i % 120, 1)) for i, t in enumerate(tickets)]
    assert run.wait(tickets[3]).ticket == tickets[3]


def test_a_wait_sends_only_up_to_its_ticket():
    run, port, _ = make_run()
    first = run.issue(0, (0, 1))
    run.issue(0, (1, 1))
    run.wait(first)
    assert port.next == 2  # the second record is still queued


def test_the_host_never_overruns_the_rings():
    port = FakePort(depth=2, host_depth=4, event_depth=8)
    port.auto = False
    run, port, _ = make_run(port)
    tickets = [run.issue(0, (0, 1)) for _ in range(20)]
    with pytest.raises(WaitTimeout) as caught:
        run.wait(tickets, timeout=0.05)
    assert port.next - 1 - port.consumed <= 4
    assert port.next - 1 - port.seen <= 8
    assert caught.value.pending == tickets
    port.auto = True
    run.wait(tickets)
    assert port.max_unread <= 8 and len(port.issued) == 20


def test_teams_and_segments_are_checked_before_queueing():
    run, _, _ = make_run()
    for team in [(0, 0), (-1, 2), (119, 2), (0, 121)]:
        with pytest.raises(ValueError, match="team"):
            run.issue(1, team)
    for seg in [240, 251, 255, -1]:
        with pytest.raises(ValueError, match="segment"):
            run.issue(seg, (0, 1))
    with pytest.raises(ValueError, match="arguments"):
        run.issue(1, (0, 1), list(range(11)))
    with pytest.raises(ValueError, match="32-bit"):
        run.issue(1, (0, 1), [1 << 32])
    assert run.issue(0, None) == 1 and run.team(1) == (0, 120)


def test_segments_by_name_through_the_program(tmp_path):
    header = tmp_path / "p.h"
    header.write_text(
        "constexpr uint32_t kSegMap = 1;\nconstexpr uint32_t kSegBigStep = 7;\n"
    )
    program = Program("p", kernels=tmp_path, segments=header)
    assert program.segments == {"map": 1, "big_step": 7}
    run, port, _ = make_run(program=program)
    run.wait(run.issue("big_step", (0, 4)))
    assert port.issued[0][3] == 7
    with pytest.raises(KeyError):
        run.issue("missing", (0, 1))


def test_exclusive_records_wait_for_overlapping_teams():
    port = FakePort(depth=4)
    port.auto = False
    run, port, _ = make_run(port)
    a = run.issue(1, (0, 10))
    b = run.issue(1, (20, 10))
    c = run.issue(1, (5, 10), exclusive=True)  # overlaps a
    d = run.issue(1, (40, 10))  # behind c in order
    with pytest.raises(WaitTimeout):
        run.wait(d, timeout=0.05)
    assert [i[0] for i in port.issued] == [a, b]
    port.auto = True
    run.wait([a, b, c, d])
    assert [i[0] for i in port.issued] == [a, b, c, d]


def test_the_read_count_lets_the_kernel_go_on_past_a_full_event_ring():
    port = FakePort(depth=4, host_depth=8, event_depth=8)
    run, port, _ = make_run(port)
    tickets = [run.issue(0, (i % 100, 4)) for i in range(200)]
    run.wait(tickets, timeout=1.0)
    assert all(run.event(t).kind == "done" for t in tickets)
    # the host's own room (unread plus in flight) is the tighter bound
    assert port.held_for_room == 0 and port.published == port.seen


def test_a_host_that_never_publishes_its_read_count_stalls_the_kernel():
    class Silent(FakePort):
        def push(self, rows):
            published = self.published
            k = super().push(rows)
            self.published = published  # the batch did not carry the read count
            return k

        def wait_events(self, timeout_s):
            return self.poll()

    run, port, _ = make_run(Silent(depth=4, host_depth=8, event_depth=8))
    tickets = [run.issue(0, (0, 4)) for _ in range(20)]
    with pytest.raises(WaitTimeout):
        run.wait(tickets, timeout=0.05)


def test_sequences_follow_records_and_skip_kernel_commands():
    run, port, _ = make_run()
    tickets = [run.issue(0, (0, 1)) for _ in range(3)]
    run.wait(tickets)
    assert [run.seq(t) for t in tickets] == [s for _, s, _, _, _ in port.issued]


def test_a_refused_record_raises_for_its_ticket_and_the_run_goes_on():
    run, port, _ = make_run()
    good = run.issue(1, (0, 4))
    bad = run._queue_command(1, (110, 20))  # bypasses the host's own check
    after = run.issue(1, (0, 4))
    with pytest.raises(RecordRefused, match="reason 3: team") as caught:
        run.wait([good, bad, after])
    assert caught.value.ticket == bad
    assert caught.value.reason == K["KMD_REFUSE_TEAM"]
    assert run.event(after).kind == "done"
    assert run.seq(after) == 2 and run.seq(bad) is None
    assert run.wait(run.issue(1, (0, 4))).kind == "done"


def test_sequences_skip_0_at_the_wrap_as_the_kernel_numbers_them():
    run, port, _ = make_run()
    run._next_seq, port.seq = abi.M32 - 1, abi.M32 - 2
    tickets = [run.issue(0, (0, 1)) for _ in range(3)]
    run.wait(tickets)
    assert [run.seq(t) for t in tickets] == [abi.M32 - 1, abi.M32, 1]
    assert [s for _, s, _, _, _ in port.issued] == [abi.M32 - 1, abi.M32, 1]


def test_the_kernel_holds_exit_while_a_slot_is_busy_and_refuses_records_after_it():
    run, port, _ = make_run()
    port.auto = False
    busy = run.issue(1, (0, 4))
    run._flush(busy)
    exit_ = run._queue_command(K["KM_SEG_EXIT"], None)
    run._flush(exit_)
    run.port.poll()
    assert port.held_exit > 0 and not port.exited
    port.auto = True
    run.wait([busy, exit_])
    assert port.exited and port.issued[-1][3:] == (K["KM_SEG_EXIT"], (0, 120))
    late = run._queue_command(1, (0, 4))  # past the host's own stop
    with pytest.raises(RecordRefused, match="exited"):
        run.wait(late)


def test_stop_drains_sends_exit_to_every_core_then_stop():
    run, port, backend = make_run(trace_cap=8)
    tickets = [run.issue(1, (i, 2)) for i in range(3)]
    result = run.stop()
    assert all(run.event(t).kind == "done" for t in tickets)
    exit_req, _, _, seg, team = port.issued[-1]
    assert seg == K["KM_SEG_EXIT"] and team == (0, 120)
    assert port.state == K["KMD_STATE_DONE"] and result["error"] == 0
    assert backend.calls == ["finish", "read_back"]
    assert run.stopped and len(run.state()) == 120
    assert run.trace().phases(tickets[1])["count"] == 2
    with pytest.raises(KomorebiError, match="stopped"):
        run.issue(1, (0, 1))
    assert run.stop() is result


@pytest.mark.parametrize(
    "answer, says",
    [
        ({"result_ready": False, "failed": "timeout waiting for the hart"}, "never"),
        ({"cleaned": False}, "could not restore"),
        ({"state": K["KMD_STATE_READY"]}, "state ready"),
    ],
)
def test_stop_raises_unless_the_kernel_finished_cleanly(answer, says):
    run, port, backend = make_run()
    backend.answer = answer
    run.wait(run.issue(1, (0, 2)))
    with pytest.raises(KernelFailed, match=says) as caught:
        run.stop()
    assert caught.value.result is not None and run.stopped
    with pytest.raises(KernelFailed):
        run.issue(1, (0, 1))


def test_abort_raises_when_the_runtime_could_not_restore_the_tile():
    run, _, backend = make_run()
    backend.cleaned = False
    with pytest.raises(KernelFailed, match="reset"):
        run.abort()
    assert backend.calls == ["abort", "read_back"]


def test_stop_waits_for_records_in_flight_before_exit():
    port = FakePort(depth=4)
    run, port, _ = make_run(port)
    tickets = [run.issue(1, (10 * i, 10)) for i in range(3)]
    port.auto = False
    with pytest.raises(WaitTimeout):
        run.wait(tickets, timeout=0.05)
    assert len(port.busy) == 3  # stored and taken, not complete
    port.auto = True
    run.stop()
    assert port.held_exit == 0  # the host drained: the kernel had nothing to hold
    assert [run.event(t).kind for t in tickets] == ["done"] * 3
    assert port.issued[-1][3] == K["KM_SEG_EXIT"] and port.issued[-1][4] == (0, 120)


def stray_event(port, req):
    port.pushed_events += 1
    port.events.append([port.pushed_events, req, K["KM_EV_DONE"], 7])


@pytest.mark.parametrize("req", [9, 1])  # never stored; already filed
def test_an_event_for_a_command_not_in_flight_stops_the_run(req):
    run, port, backend = make_run()
    run.wait(run.issue(1, (0, 1)))
    stray_event(port, req)
    with pytest.raises(KernelFailed, match=f"command {req} .*not in flight"):
        run.wait(run.issue(1, (0, 1)))
    assert backend.calls == ["abort", "read_back"] and run.stopped


def test_state_and_trace_only_after_the_stop():
    run, _, _ = make_run()
    with pytest.raises(KomorebiError, match="after"):
        run.state()
    with pytest.raises(KomorebiError, match="after"):
        run.trace()


def test_a_timeout_names_command_slot_sequence_and_cores():
    port = FakePort()
    run, port, backend = make_run(port, stalled_core=6)
    before = [run.issue(1, (4, 8)) for _ in range(2)]
    run.wait(before)
    stalled = run.issue(2, (4, 8), [2])
    port.stall_req = stalled
    with pytest.raises(RecordTimeout) as caught:
        run.wait(stalled)
    err = caught.value
    assert err.ticket == stalled and err.team == (4, 8)
    assert err.seq == run.seq(stalled) == 3
    assert err.slot == port.issued[-1][2]
    assert [c["core"] for c in err.cores] == [4, 6]
    assert err.cores[1]["reader"]["where"] == "stall"
    assert err.cores[0]["writer"]["where"] == "complete"
    assert "core 6 reader gave up (stop) in stall" in str(err)
    assert backend.calls == ["finish", "read_back"]
    assert run.result()["error_name"] == "deadline"
    with pytest.raises(RecordTimeout):
        run.issue(1, (0, 1))


def test_a_kernel_that_ends_on_its_own_raises_kernel_failed():
    run, port, _ = make_run()
    run.wait(run.issue(1, (0, 1)))
    port._finish(K["KMD_ERR_IDLE"])
    with pytest.raises(KernelFailed, match="idle"):
        run.wait(run.issue(1, (0, 1)), timeout=1.0)
    assert run.stopped


def test_an_error_while_stopping_aborts_through_the_runtime():
    port = FakePort()
    port.auto = False
    run, port, backend = make_run(port)
    run.issue(1, (0, 1))
    with pytest.raises(WaitTimeout):
        run.stop(timeout=0.05)
    assert backend.calls == ["abort", "read_back"] and run.stopped


def test_tickets_map_onto_command_numbers_across_the_32_bit_wrap():
    run, _, _ = make_run()
    run._next = (1 << 32) - 2  # as after 2**32 - 3 commands
    tickets = [run.issue(0, (0, 1)) for _ in range(5)]
    assert tickets[-1] > abi.M32
    run._flight = {t: (0, 1) for t in tickets}
    run._queue.clear()
    rows = [[1, t & abi.M32, K["KM_EV_DONE"], 9] for t in reversed(tickets)]
    run._take(np.asarray(rows, dtype=np.uint32))
    assert [run.event(t).ticket for t in tickets] == tickets


def test_python_cost_counters_add_up():
    run, _, _ = make_run()
    run.wait([run.issue(0, (0, 1)) for _ in range(10)])
    assert run.issue_s > 0 and run.flush_s > 0 and run.stored == 10
