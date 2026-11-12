"""A loaded program under the dispatch kernel: issue records, wait for them, stop,
read back state words and trace. The card is reached only through `backend`, so the
bookkeeping is the same against a fake transport.
"""

import collections
import contextlib
import time
from typing import NamedTuple

import numpy as np

from komorebi import abi
from komorebi import state as st
from komorebi.errors import (
    KernelFailed,
    KomorebiError,
    RecordRefused,
    RecordTimeout,
    WaitTimeout,
)

M32 = abi.M32
WAIT_SLICE_S = 0.05  # longest single native wait, so Python stays responsive


class Event(NamedTuple):
    """A finished command: its ticket, kind (done, timeout, pong, refused), the aux
    byte (slot or reason) and issue-to-completion time in X280 timer ticks."""

    ticket: int
    kind: str
    aux: int
    ticks: int

    @property
    def us(self):
        return self.ticks / 50.0  # the X280 timer runs at 50 MHz


def _seq_number(n):
    """The kernel's sequence number of the n-th record: it counts in 32 bits and skips
    0 (it also skips a slot's last value, which only a wrap can repeat)."""
    return (n - 1) % M32 + 1


class Run:
    """Issues records to a loaded program and collects their events.

    backend: `port` (push, poll, wait_events, host_word), `finish(timeout)` -> result
    words and status, `abort()`, `read_back()` -> (state words, trace rows), `mhz`.
    """

    def __init__(self, backend, grid, program=None, batch=64, h=None):
        self.h = h or abi.headers()
        self.backend = backend
        self.port = backend.port
        self.grid = grid
        self.cores = grid[0] * grid[1]
        self.program = program
        self.batch = batch
        k = self.h.constants
        self._kinds = {
            k["KM_EV_DONE"]: "done",
            k["KM_EV_TIMEOUT"]: "timeout",
            k["KM_EV_PONG"]: "pong",
            k.get("KM_EV_REFUSED", 3): "refused",
        }
        self._done = k["KM_EV_DONE"]
        self._refusals = {
            v: n[len("KMD_REFUSE_") :].lower()
            for n, v in k.items()
            if n.startswith("KMD_REFUSE_")
        }
        self._next = 1  # the next ticket; command number = ticket mod 2**32
        self._next_seq = 1  # the kernel's sequence of the next record
        self._queue = collections.deque()  # (ticket, words, team, exclusive)
        self._flight = {}  # ticket -> team (None: a command to the kernel itself)
        self._teams = {}  # ticket -> team, every record issued
        self._seqs = {}  # ticket -> records issued up to it (see _seq_number)
        self._events = {}
        self._arrived = []  # tickets whose events arrived since the last look
        self._refused = {}
        self._failure = None
        self._stopped = False
        self._result = None
        self._state = None
        self._trace = None
        self.issue_s = 0.0  # host time in issue()
        self.flush_s = 0.0  # host time storing batches
        self.stored = 0

    # Issuing --------------------------------------------------------------------

    def issue(self, seg, team=None, args=(), exclusive=False):
        """Queues a record for segment `seg` (name or number) on cores [first, first +
        count) (default every core); returns its ticket. Nothing is sent until a wait.
        """
        t0 = time.perf_counter()
        seg = self.program.seg(seg) if self.program is not None else int(seg)
        if not (seg == self.h["KM_SEG_NOP"] or 1 <= seg <= self.h["KM_SEG_USER_MAX"]):
            raise ValueError(f"segment {seg}: not NOP and not in 1..KM_SEG_USER_MAX")
        team = self._team(team)
        if len(args) > len(abi.ARG_WORDS):
            raise ValueError(f"{len(args)} arguments; a record has 10")
        if any(not 0 <= int(a) <= M32 for a in args):
            raise ValueError("arguments are 32-bit unsigned words")
        ticket = self._queue_command(seg, team, args, exclusive)
        self.issue_s += time.perf_counter() - t0
        return ticket

    def _team(self, team):
        if team is None:
            return (0, self.cores)
        first, count = int(team[0]), int(team[1])
        if count < 1 or first < 0 or first + count > self.cores:
            raise ValueError(
                f"team ({first}, {count}) is not inside {self.cores} cores"
            )
        return (first, count)

    def _queue_command(self, seg, team, args=(), exclusive=False):
        self._check_alive()
        first, count = team if team else (0, 0)
        words = abi.record(seg, first, count, args, h=self.h)
        ticket = self._next
        self._next += 1
        if team is not None:
            self._teams[ticket] = team
            self._seqs[ticket] = self._next_seq
            self._next_seq += 1
        self._queue.append((ticket, words, team, exclusive))
        return ticket

    def _overlaps(self, team):
        a0, a1 = team[0], team[0] + team[1]
        for other in self._flight.values():
            if other is not None and other[0] < a1 and a0 < other[0] + other[1]:
                return True
        return False

    def _flush(self, upto):
        """Stores queued commands up to ticket `upto` that fit, in batches of at most
        `batch` (all lines, then one fence). Returns how many were stored."""
        rows, tickets = [], []
        for ticket, words, team, exclusive in self._queue:
            if ticket > upto or len(rows) == self.batch:
                break
            if exclusive and (rows or self._overlaps(team)):
                break  # held until no team in flight overlaps it
            rows.append(words)
            tickets.append((ticket, team))
            if exclusive:
                break
        if not rows:
            return 0
        t0 = time.perf_counter()
        n = self.port.push(np.asarray(rows, dtype=np.uint32))
        self.flush_s += time.perf_counter() - t0
        for ticket, team in tickets[:n]:
            self._queue.popleft()
            self._flight[ticket] = team
        self.stored += n
        return n

    # Events -----------------------------------------------------------------------

    def _take(self, rows):
        """Files a block of events (index, req, status, ticks). The command number
        of an event is a ticket in flight, so its low 32 bits name it."""
        rows = np.asarray(rows).reshape(-1, 4)
        base = next(iter(self._flight)) if self._flight else self._next
        reqs = rows[:, 1].astype(np.int64)
        tickets = (base + ((reqs - base) & M32)).tolist()
        statuses, ticks = rows[:, 2].tolist(), rows[:, 3].tolist()
        pop, events, done = self._flight.pop, self._events, self._done
        timeout = None
        for ticket, status, tick in zip(tickets, statuses, ticks):
            team = pop(ticket, None)
            events[ticket] = (status, tick)
            if status & 0xFF == done:
                continue
            ev = self.event(ticket)
            if ev.kind == "refused":
                self._refused[ticket] = ev.aux
                if team is not None:
                    self._unsequence(ticket)
            elif ev.kind == "timeout":
                timeout = ev
        self._arrived.extend(tickets)
        if timeout is not None:
            self._fail_timeout(timeout)

    def _unsequence(self, ticket):
        """A refused record took no sequence number: later records move down one."""
        self._seqs.pop(ticket, None)
        for t in self._seqs:
            if t > ticket:
                self._seqs[t] -= 1
        self._next_seq -= 1

    def _pump(self, upto, timeout_s):
        """One round: store what fits, read events (waiting at most timeout_s if
        nothing was stored). Ends the run if the kernel finished on its own."""
        stored = self._flush(upto)
        if stored:
            rows = self.port.poll()
        else:
            rows = self.port.wait_events(max(0.0, min(timeout_s, WAIT_SLICE_S)))
        if len(rows):
            self._take(rows)
        elif not stored and self.port.host_word(0) >= self.h["KMD_STATE_DONE"]:
            self._fail_kernel("the dispatch kernel finished on its own")

    def wait(self, tickets, timeout=10.0):
        """Sends what is queued up to the tickets and waits for their events.

        Returns the events (one, or a list for a list of tickets). Raises
        RecordRefused, RecordTimeout, KernelFailed or WaitTimeout.
        """
        single = isinstance(tickets, int)
        wanted = [tickets] if single else list(tickets)
        if not wanted:
            return []
        if max(wanted) >= self._next or min(wanted) < 1:
            raise ValueError("a ticket that was never issued")
        self._wait(wanted, timeout)
        if single:
            return self.event(wanted[0])
        return [self.event(t) for t in wanted]

    def _wait(self, wanted, timeout):
        deadline = time.monotonic() + timeout
        upto = max(wanted)
        pending = {t for t in wanted if t not in self._events}
        while True:
            self._check_alive()
            pending.difference_update(self._arrived)
            self._arrived.clear()
            if not pending:
                break
            left = deadline - time.monotonic()
            if left <= 0:
                pending = sorted(pending)
                raise WaitTimeout(
                    f"{len(pending)} of {len(wanted)} tickets not done in {timeout} s "
                    f"(first {pending[0]}; {len(self._flight)} in flight, "
                    f"{len(self._queue)} queued)",
                    pending,
                )
            self._pump(upto, left)
        for t in wanted:
            if t in self._refused:
                raise RecordRefused(t, self._refused[t], self._refusals)

    def drain(self, timeout=10.0):
        """Sends everything queued and waits for every event (refusals do not raise
        here); returns how many events it waited for."""
        wanted = [t for t, team in self._flight.items() if team is not None]
        wanted += [q[0] for q in self._queue]
        if wanted:
            with contextlib.suppress(RecordRefused):
                self._wait(wanted, timeout)
        return len(wanted)

    def event(self, ticket):
        """The event of a ticket, or None if it has not arrived."""
        found = self._events.get(ticket)
        if found is None:
            return None
        status, ticks = found
        kind = self._kinds.get(status & 0xFF, str(status & 0xFF))
        return Event(ticket, kind, status >> 8, ticks)

    def seq(self, ticket):
        """The sequence number the kernel gave a record (the trace's index)."""
        n = self._seqs.get(ticket)
        return None if n is None else _seq_number(n)

    def team(self, ticket):
        return self._teams.get(ticket)

    # Stopping ---------------------------------------------------------------------

    @property
    def stopped(self):
        return self._stopped

    def _check_alive(self):
        if self._failure is not None:
            raise self._failure
        if self._stopped:
            raise KomorebiError("the run is stopped")

    def stop(self, timeout=10.0):
        """Drains, sends EXIT to every core, stops the kernel, closes the X280
        runtime, then reads back the state words and the trace."""
        if self._stopped:
            return self._result
        try:
            self.drain(timeout)
            exit_ticket = self._queue_command(self.h["KM_SEG_EXIT"], (0, self.cores))
            self.wait(exit_ticket, timeout)
            self._queue_command(self.h["KM_SEG_STOP"], None)
            self._flush(self._next)
        except KernelFailed:
            raise
        except BaseException:
            self.abort()
            raise
        self._finish(timeout)
        result = self._result
        if result["error"] != 0:
            raise self._remember(
                KernelFailed(
                    f"the dispatch kernel finished with error {result['error_name']} "
                    f"in {result['where_name']}",
                    result=result,
                )
            )
        return result

    def abort(self):
        """Stops the kernel through the runtime and reads back what the cores left;
        for failure paths. Errors while reading back are suppressed."""
        if self._stopped:
            return
        self._stopped = True
        self.backend.abort()
        self._read_back()

    def _finish(self, timeout):
        self._stopped = True
        try:
            fin = self.backend.finish(timeout)
        except BaseException:
            self.backend.abort()
            self._read_back()
            raise
        self._result = st.result_block(fin["words"], self.h)
        self._result.update({k: v for k, v in fin.items() if k != "words"})
        self._read_back()

    def _read_back(self):
        state_rows, trace_rows = self.backend.read_back()
        self._state = state_rows
        if trace_rows is not None:
            self._trace = st.Trace(
                trace_rows,
                {t: _seq_number(n) for t, n in self._seqs.items()},
                dict(self._teams),
                self.backend.mhz,
                self.h,
            )

    def _remember(self, error):
        self._failure = error
        return error

    def _fail_timeout(self, ev):
        self._finish(10.0)
        r = self._result
        team = self._teams.get(ev.ticket)
        seq = r["seq"] if r["error"] == self.h["KMD_ERR_DEADLINE"] else None
        seq = seq if seq is not None else self.seq(ev.ticket)
        cores = st.not_reported(self._state, team, ev.aux, seq, self.h) if team else []
        raise self._remember(RecordTimeout(ev.ticket, ev.aux, seq, team, cores, r))

    def _fail_kernel(self, why):
        self._finish(10.0)
        r = self._result
        raise self._remember(
            KernelFailed(
                f"{why}: error {r['error_name']} in {r['where_name']}, slot "
                f"{r['slot']}, sequence {r['seq']}, command {r['req']}",
                result=r,
            )
        )

    # After the stop ---------------------------------------------------------------

    def result(self):
        """The dispatch kernel's result block (decoded); valid after the stop."""
        self._require_stopped()
        return self._result

    def state(self):
        """Every core's state words, as dicts; valid after the stop."""
        self._require_stopped()
        return [st.core_state(self._state, c, self.h) for c in range(self.cores)]

    def state_words(self):
        self._require_stopped()
        return self._state

    def trace(self):
        """The per-record trace (a state.Trace), or None if the run was not traced."""
        self._require_stopped()
        return self._trace

    def _require_stopped(self):
        if not self._stopped or self._state is None:
            raise KomorebiError("valid only after run.stop()")
