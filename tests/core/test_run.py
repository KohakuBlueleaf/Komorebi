"""Run: tickets, batches, events and stopping, against a fake dispatch kernel."""

import time

import numpy as np
import pytest
from fake_port import FakeBackend, FakePort

from komorebi import abi
from komorebi import run as run_mod
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
    with pytest.raises(ValueError, match="returned or forgotten"):
        run.wait(tickets[3])


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
    events = run.wait(tickets, timeout=1.0)
    assert all(ev.kind == "done" for ev in events)
    # the host's own room (unread plus in flight) is the tighter bound
    assert port.held_for_room == 0 and 0 < port.published <= port.seen


def test_a_host_that_never_publishes_its_read_count_stalls_the_kernel():
    class Silent(FakePort):
        def push(self, rows):
            published = self.published
            k = super().push(rows)
            self.published = published  # the batch did not carry the read count
            return k

        def _publish_if_held(self):
            pass  # waits without storing the read count

    run, port, _ = make_run(Silent(depth=4, host_depth=8, event_depth=8))
    tickets = [run.issue(0, (0, 4)) for _ in range(20)]
    with pytest.raises(WaitTimeout):
        run.wait(tickets, timeout=0.05)


def test_sequences_follow_records_and_skip_kernel_commands():
    run, port, _ = make_run()
    tickets = [run.issue(0, (0, 1)) for _ in range(3)]
    run.drain()
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
    run.drain()
    assert [run.seq(t) for t in tickets] == [abi.M32 - 1, abi.M32, 1]
    assert [s for _, s, _, _, _ in port.issued] == [abi.M32 - 1, abi.M32, 1]


def test_the_kernel_holds_exit_while_a_slot_is_busy_and_refuses_records_after_it():
    run, port, _ = make_run()
    port.auto = False
    busy = run.issue(1, (0, 4))
    run._send(busy, 1.0)
    exit_ = run._queue_command(K["KM_SEG_EXIT"], None)
    run._send(exit_, 1.0)
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
    backend.close_failed = "hart 0 phase=0x2 trap(mcause=0x7)"
    words, harts = [0] * K["KMD_R_WORDS"], [0] * 16
    words[K["KMD_R_STATE"]], harts[K["KMD_H_ABANDONED"]] = K["KMD_STATE_FAILED"], 3
    backend.aborted = {"words": words, "harts": harts}
    with pytest.raises(KernelFailed, match=r"mcause=0x7.*failed, 3 records aband"):
        run.abort()
    assert backend.calls == ["abort", "read_back"] and backend.grid_stopped
    assert run.result()["abandoned"] == 3


def test_a_wait_that_runs_out_says_what_the_kernel_is_doing():
    run, port, backend = make_run()
    port.auto = False
    ticket = run.issue(1, (0, 1))
    with pytest.raises(WaitTimeout, match="1 of 1 .*; the fake kernel's report"):
        run.wait(ticket, timeout=0.02)
    with pytest.raises(WaitTimeout, match="none of 1 .*; the fake kernel's report"):
        run.wait_any([ticket], timeout=0.02)
    assert backend.calls == ["kernel_report"] * 2


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


def test_a_ticket_is_released_once_its_event_was_returned(monkeypatch):
    monkeypatch.setattr(run_mod, "COMPACT_ROWS", 64)
    port = FakePort(host_depth=64, event_depth=128)
    run, port, _ = make_run(port)
    for _ in range(10):
        run.wait([run.issue(0, (i, 1)) for i in range(100)])
    assert len(run._has) < 2 * run_mod.COMPACT_ROWS and run._base > 500
    assert run.event(1) is None and run.seq(1) is None and len(port.issued) == 1000


def test_forget_releases_tickets_read_after_a_drain():
    run, _, _ = make_run()
    tickets = [run.issue(0, (0, 1)) for _ in range(5)]
    run.drain()
    assert run.event(tickets[0]).kind == "done" and run.seq(tickets[4]) == 5
    run.forget(tickets)
    assert run.event(tickets[0]) is None and run.seq(tickets[4]) is None
    assert run._gone[:5].all()


class SlowPort(FakePort):
    """Each record (EXIT too) completes `delay` seconds after it was issued."""

    def __init__(self, delay):
        super().__init__()
        self.delay, self.since = delay, {}

    def _complete(self):
        now = time.monotonic()
        for slot, (seq, req, _) in sorted(self.busy.items()):
            if now - self.since.setdefault(seq, now) >= self.delay:
                del self.busy[slot]
                self._event(req, K["KM_EV_DONE"])


class SlowBackend(FakeBackend):
    """The kernel's result is ready `delay` seconds into finish(), if it waits."""

    def finish(self, timeout):
        time.sleep(min(self.port.delay, timeout))
        self.answer["result_ready"] = timeout >= self.port.delay
        return super().finish(timeout)


def test_stop_ends_within_its_timeout_when_every_part_is_slow():
    port = SlowPort(delay=0.06)  # drain, EXIT and finish: 0.18 s in all
    run = Run(SlowBackend(port), GRID)
    run.issue(1, (0, 2))
    t0 = time.monotonic()
    with pytest.raises(KernelFailed, match="never became ready"):
        run.stop(timeout=0.14)
    assert time.monotonic() - t0 < 0.14 + 0.05 and run.stopped


class CardThread:
    """A run's work-cluster thread on the card, as far as Run looks at it."""

    on_card, ring, running = True, object(), False

    def stop(self, timeout):
        assert 0 < timeout <= 10.0  # inside the run's stop deadline

    def grid_stopped(self):
        pass


def test_sequences_are_exact_while_a_card_thread_issues_records_between():
    run, port, _ = make_run(trace_cap=1024)
    run.thread = CardThread()  # started at the load, before any host record
    a = run.issue(1, (0, 2))
    run.wait(a)
    port.seq += 700  # the thread's records, numbered by the same kernel
    b = run.issue(1, (2, 2))
    run.wait(b)
    assert run.team(b) == (2, 2)
    assert [run.seq(t) for t in (a, b)] == [1, 702] == [x[1] for x in port.issued]
    run.stop()
    assert run.trace().phases(b)["count"] == 2


def test_a_gap_of_other_sources_too_long_to_tell_raises():
    run, port, _ = make_run(trace_cap=8)  # keeps the sequences of waited tickets
    a = run.wait(run.issue(1, (0, 2))).ticket
    port.seq += 40_000
    b = run.wait(run.issue(1, (0, 2))).ticket
    assert run.seq(a) == 1
    with pytest.raises(KomorebiError, match="other sources"):
        run.seq(b)


def test_a_queued_record_gets_its_first_core_and_sequence_from_its_event():
    run, port, _ = make_run(trace_cap=8)
    busy = run.issue(1, (0, 4))
    queued = run.issue(1, 3, queued=True)
    run.wait([busy, queued])
    assert run.team(queued) == (port.issued[-1][4][0], 3)
    assert run.seq(queued) == port.issued[-1][1] == 2
    run.stop()
    assert run.trace().record(queued).shape == (3, 4)


def test_a_traced_run_keeps_teams_and_sequences_for_the_trace():
    run, _, _ = make_run(trace_cap=8)
    tickets = [run.issue(1, (i, 2)) for i in range(3)]
    run.wait(tickets)
    assert run.event(tickets[0]) is None and run.seq(tickets[2]) == 3
    assert run.team(tickets[1]) == (1, 2)
    run.stop()
    assert run.trace().phases(tickets[1])["count"] == 2


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


def test_a_trap_in_the_kernel_names_the_core_and_the_exception():
    run, port, _ = make_run()
    run.wait(run.issue(1, (0, 1)))
    port._finish(K["KMD_ERR_TRAP"], slot=5, seq=0x8003A40, req=0)
    port.result["seen"] = 0x6C8
    want = r"trap in .*core 0 \(mcause 0x5, mepc 0x8003a40, mtval 0x6c8\)"
    with pytest.raises(KernelFailed, match=want):
        run.wait(run.issue(1, (0, 1)), timeout=1.0)
    assert run.stopped


def test_an_error_while_stopping_aborts_through_the_runtime():
    port = FakePort()
    port.auto = False
    run, port, backend = make_run(port)
    run.issue(1, (0, 1))
    with pytest.raises(WaitTimeout):
        run.stop(timeout=0.05)  # the kernel is probed before the abort ends it
    assert backend.calls == ["kernel_report", "abort", "read_back"] and run.stopped


def test_tickets_map_onto_command_numbers_across_the_32_bit_wrap():
    run, port, _ = make_run(FakePort(depth=4, host_depth=8, event_depth=16))
    start = (1 << 32) - 3  # as after 2**32 - 4 commands
    run._next = start
    run._new_tickets(start)
    port.next = start
    port.consumed = port.seen = port.pushed_events = start - 1
    tickets = [run.issue(0, (i, 1)) for i in range(5)]
    tickets += list(run.issue_many(0, np.arange(5, 25), 1))
    assert tickets == list(range(start, start + 25)) and tickets[-1] > abi.M32
    assert run.drain() == 25
    assert [run.event(t).ticket for t in tickets] == tickets
    assert [r for r, _, _, _, _ in port.issued] == tickets
    assert [run.seq(t) for t in tickets] == list(range(1, 26))


def test_python_cost_counters_add_up():
    run, _, _ = make_run()
    run.wait([run.issue(0, (0, 1)) for _ in range(10)])
    assert run.issue_s > 0 and run.wait_s >= run.native_s > 0 and run.stored == 10


def test_issue_many_encodes_as_issue_does():
    one, port_one, _ = make_run()
    many, port_many, _ = make_run()
    firsts, counts = np.array([0, 5, 119]), np.array([4, 10, 1])
    args = np.array([[1, 2, 3], [4, 5, 6], [7, 8, abi.M32]], np.uint64)
    singles = [one.issue(7, (f, c), a) for f, c, a in zip(firsts, counts, args)]
    tickets = many.issue_many(7, firsts, counts, args)
    assert tickets == range(1, 4) and singles == list(tickets)
    one.drain()
    many.drain()
    assert port_one.ring == port_many.ring
    assert [many.team(t) for t in tickets] == [(0, 4), (5, 10), (119, 1)]


def test_issue_many_broadcasts_and_takes_a_segment_per_record():
    run, port, _ = make_run()
    tickets = run.issue_many(np.array([0, 3, 0, 3]), 0, 120)
    run.wait(tickets)
    assert [i[3] for i in port.issued] == [0, 3, 0, 3]
    assert {i[4] for i in port.issued} == {(0, 120)}


def test_issue_many_checks_every_record_before_queueing_any():
    run, port, _ = make_run()
    for firsts, counts in [([0, 119], [1, 2]), ([0, -1], 1), ([0, 1], [1, 0])]:
        with pytest.raises(ValueError, match="team"):
            run.issue_many(1, np.array(firsts), np.array(counts))
    with pytest.raises(ValueError, match="segment"):
        run.issue_many(np.array([1, 240]), 0, 1)
    with pytest.raises(ValueError, match="32-bit"):
        run.issue_many(1, [0, 1], 1, np.array([[1], [1 << 32]]))
    with pytest.raises(ValueError, match="32-bit"):
        run.issue_many(1, [0, 1], 1, np.array([[1], [-1]]))
    with pytest.raises(ValueError, match="m <= 10"):
        run.issue_many(1, [0], 1, np.zeros((1, 11), np.uint32))
    with pytest.raises(ValueError, match="one row each"):
        run.issue_many(1, [0, 1], [1, 1, 1])
    with pytest.raises(ValueError, match="integers"):
        run.issue_many(1, [0.0], 1)
    assert run._next == 1 and port.queued == 0


def test_issue_many_and_issue_keep_one_order():
    run, port, _ = make_run()
    a = run.issue(0, (0, 1))
    b = run.issue_many(0, [1, 2], 1)
    c = run.issue(0, (3, 1))
    run.drain()
    assert [i[0] for i in port.issued] == [a, *b, c]
    assert [i[4][0] for i in port.issued] == [0, 1, 2, 3]
    assert [run.seq(t) for t in [a, *b, c]] == [1, 2, 3, 4]


def test_a_drain_of_many_records_is_one_native_loop_per_slice():
    port = FakePort(depth=4, host_depth=64, event_depth=128)
    run, port, _ = make_run(port)
    tickets = run.issue_many(0, np.arange(1000) % 120, 1)
    assert run.drain() == 1000
    assert all(run._has[: len(tickets)]) and len(port.issued) == 1000
    assert port.waits < 5 and max(port.batches) == 64
    assert run.event(tickets[-1]).kind == "done"


def test_exclusive_records_from_issue_many_wait_for_overlapping_teams():
    port = FakePort(depth=4)
    port.auto = False
    run, port, _ = make_run(port)
    a = run.issue(1, (0, 10))
    held = run.issue_many(1, [5, 50], 10, exclusive=True)
    with pytest.raises(WaitTimeout):
        run.wait(held[0], timeout=0.05)
    assert [i[0] for i in port.issued] == [a]
    port.auto = True
    run.wait(held)
    assert [i[0] for i in port.issued] == [a, *held]


def test_a_result_record_returns_the_leaders_word_on_its_done_event():
    run, port, _ = make_run()
    with_result, plain = run.issue(1, (0, 8), result=True), run.issue(1, (8, 8))
    port.results[with_result] = 0xBEEF
    one, other = run.wait([with_result, plain])
    assert one.result == 0xBEEF and other.result is None
    assert port.ring[with_result][1] >> 8 == K["KM_F_RESULT"]


def test_a_queued_record_takes_its_count_and_the_kernel_picks_the_team():
    run, port, _ = make_run()
    port.auto = False
    run.issue(1, (0, 60))
    queued = run.issue(1, 30, queued=True)
    with pytest.raises(WaitTimeout):
        run.wait(queued, timeout=0.05)
    assert port.issued[-1][4] == (60, 30) and run.team(queued) == (None, 30)
    with pytest.raises(ValueError, match="team"):
        run.issue(1, 121, queued=True)


def test_wait_any_returns_the_first_event_of_a_set_and_forgets_only_it():
    run, port, _ = make_run()
    port.auto = False
    a, b = run.issue(1, (0, 8)), run.issue(1, (8, 8))
    with pytest.raises(WaitTimeout):
        run.wait_any([b, a], timeout=0.05)
    port.auto = True
    assert run.wait_any([b, a]).ticket == a
    assert run.wait_any([b]).ticket == b
    with pytest.raises(ValueError, match="returned or forgotten"):
        run.wait_any([a])
