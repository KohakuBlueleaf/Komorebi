"""A loaded program under the dispatch kernel: issue records, wait for them, stop,
read back state words and trace. The card is reached only through `backend`, so the
bookkeeping is the same against a fake transport.
"""

import contextlib
import time
from array import array
from typing import NamedTuple

import numpy as np

from komorebi import abi
from komorebi import state as st
from komorebi.engines import control, graph_hostfed, stream_hostfed
from komorebi.errors import (
    KernelFailed,
    KomorebiError,
    RecordRefused,
    RecordTimeout,
    StartError,
    WaitTimeout,
)

M32 = abi.M32
WAIT_SLICE_S = 0.05  # longest single native wait, so Python stays responsive
COMPACT_ROWS = 4096  # released tickets kept before the columns are shifted
COLUMNS = ("_first", "_count", "_flags", "_seq", "_status", "_ticks", "_has", "_gone")


class Event(NamedTuple):
    """A finished command: its ticket, kind (done, timeout, pong, refused), the aux
    byte (slot or reason), issue-to-completion time in X280 timer ticks, and for a
    record issued with `result=True` the leader's result word (sent as the ticks)."""

    ticket: int
    kind: str
    aux: int
    ticks: int
    result: int | None = None

    @property
    def us(self):
        return self.ticks / 50.0  # the X280 timer runs at 50 MHz


def _left(by):
    """Seconds left until the monotonic time `by` (0 once it passed)."""
    return max(by - time.monotonic(), 0.0)


def _seq_number(n):
    """The kernel's sequence number of the n-th record: it counts in 32 bits and skips
    0 (it also skips a slot's last value, which only a wrap can repeat)."""
    return (n - 1) % M32 + 1


class Run:
    """Issues records to a loaded program and collects their events.

    backend: `port` (enqueue, queued, wait, host_word, batch; its own loop), `finish`,
    `abort()`, `read_back()` -> (control blocks, trace), `mhz`.
    """

    def __init__(self, backend, grid, program=None, batch=64, h=None):
        self.h = h or abi.headers()
        self.backend = backend
        self.port = backend.port
        self.port.batch = batch
        self._keep = bool(getattr(backend, "trace_cap", 0))  # the trace needs teams
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
        self._f_result, self._f_queue = k["KM_F_RESULT"], k["KM_F_QUEUE"]
        self._f_excl = k["KM_F_EXCL"]
        self._refusals = {
            v: n[len("KMD_REFUSE_") :].lower()
            for n, v in k.items()
            if n.startswith("KMD_REFUSE_")
        }
        self._nop, self._user_max = self.h["KM_SEG_NOP"], self.h["KM_SEG_USER_MAX"]
        self._hdr, self._team_at, self._arg_at = self.h.positions
        self._next = 1  # the next ticket; command number = ticket mod 2**32
        self._next_seq = 1  # the kernel's sequence of the next record
        self._new_tickets(1)
        self._refused = {}
        self._failure = None
        self._stopped = False
        self._result = None
        self._state = None
        self._blocks = None
        self._trace = None
        self.thread = None  # the program's X280 thread (komorebi.thread.ThreadRun)
        self.issue_s = 0.0  # host time in issue() and issue_many()
        self.wait_s = 0.0  # host time in wait() and drain(), native loop included
        self.native_s = 0.0  # of which in the native loop (stores, reading events)

    def _new_tickets(self, base):
        """Empty ticket columns from ticket `base` on, and an empty unsent block."""
        self._base = base  # ticket of row 0 of the columns
        self._handed = base  # tickets below this are with the port
        self._oldest = base  # the oldest ticket without an event
        self._first = np.zeros(0, np.int32)
        self._count = np.zeros(0, np.int32)  # 0: a command to the kernel itself
        self._flags = np.zeros(0, np.uint8)  # the record's KM_F_* flags
        self._seq = np.zeros(0, np.int64)  # 0: no sequence
        self._status = np.zeros(0, np.uint32)
        self._ticks = np.zeros(0, np.uint32)
        self._has = np.zeros(0, bool)  # the event arrived
        self._gone = np.zeros(0, bool)  # returned by wait() or forgotten
        self._unsent()

    def _unsent(self):
        # issue() appends here; new objects, since numpy may still view the old ones
        self._words, self._firsts, self._counts = array("I"), array("i"), array("i")
        self._excl, self._fl = array("B"), array("B")

    # Issuing --------------------------------------------------------------------

    def issue(
        self, seg, team=None, args=(), exclusive=False, result=False, queued=False
    ):
        """Queues a record for segment `seg` (name or number) on cores [first, first +
        count) (default every core); returns its ticket. Nothing is sent until a wait.
        `result`: the event carries the leader's result word. `queued`: `team` is a
        count and the kernel picks the first core (team() then has first None)."""
        t0 = time.perf_counter()
        if seg.__class__ is not int:
            seg = self.program.seg(seg) if self.program is not None else int(seg)
        if not (seg == self._nop or 1 <= seg <= self._user_max):
            raise ValueError(f"segment {seg}: not NOP and not in 1..KM_SEG_USER_MAX")
        if queued:
            if exclusive:
                raise ValueError("a queued record cannot be exclusive")
            team = self._team((0, team))
        else:
            team = self._team(team)
        if len(args) > len(self._arg_at):
            raise ValueError(f"{len(args)} arguments; a record has 10")
        if any(not 0 <= int(a) <= M32 for a in args):
            raise ValueError("arguments are 32-bit unsigned words")
        flags = (self._f_result if result else 0) | (self._f_queue if queued else 0)
        ticket = self._queue_command(seg, team, args, exclusive, flags)
        self.issue_s += time.perf_counter() - t0
        return ticket

    def issue_many(self, seg, firsts, counts, args=None, exclusive=False):
        """Queues n records at once: `seg` a name, a number or n numbers, `firsts` and
        `counts` the teams (n each, or one for all), `args` an [n, m] matrix of
        32-bit words (m <= 10) or None. Returns the tickets as a range."""
        t0 = time.perf_counter()
        self._check_alive()
        if isinstance(seg, str):
            seg = self.program.seg(seg)
        cols = [np.asarray(seg), np.asarray(firsts), np.asarray(counts)]
        if args is not None:
            args = np.asarray(args)
            if args.ndim != 2 or args.shape[1] > len(self._arg_at):
                raise ValueError("arguments are an [n, m] matrix with m <= 10")
            if args.dtype.kind not in "iu":
                raise ValueError("arguments are 32-bit unsigned words")
            cols.append(np.zeros(args.shape[0], np.int64))  # its row count
        if any(c.dtype.kind not in "iu" for c in cols[:3]):
            raise ValueError("segments, firsts and counts are integers")
        try:
            n = np.broadcast_shapes(*(c.shape for c in cols))
        except ValueError:
            n = ()
        if len(n) != 1:
            raise ValueError("segments, firsts, counts and arguments: one row each")
        segs, first, count = (np.broadcast_to(c, n).astype(np.int64) for c in cols[:3])
        bad = ~((segs == self._nop) | ((segs >= 1) & (segs <= self._user_max)))
        if bad.any():
            raise ValueError(
                f"segment {int(segs[bad][0])}: not NOP and not in 1..KM_SEG_USER_MAX"
            )
        bad = (count < 1) | (first < 0) | (first + count > self.cores)
        if bad.any():
            i = int(np.argmax(bad))
            raise ValueError(
                f"team ({int(first[i])}, {int(count[i])}) is not inside {self.cores} "
                "cores"
            )
        words = np.zeros((n[0], 16), np.uint32)
        words[:, self._hdr] = segs & 0xFF
        words[:, self._team_at] = first | count << 16
        if args is not None and args.size:
            if args.min() < 0 or args.max() > M32:
                raise ValueError("arguments are 32-bit unsigned words")
            words[:, self._arg_at[: args.shape[1]]] = args
        self._hand()
        start = self._next
        self._next += n[0]
        excl = None
        if exclusive:
            excl = np.ones(n[0], np.uint8)
            words[:, self._hdr] |= self._f_excl << 8
        self._send_block(words, first, count, excl, np.zeros(n[0], np.uint8))
        self.issue_s += time.perf_counter() - t0
        return range(start, self._next)

    def _team(self, team):
        if team is None:
            return (0, self.cores)
        first, count = int(team[0]), int(team[1])
        if count < 1 or first < 0 or first + count > self.cores:
            raise ValueError(
                f"team ({first}, {count}) is not inside {self.cores} cores"
            )
        return (first, count)

    def _queue_command(self, seg, team, args=(), exclusive=False, flags=0):
        self._check_alive()
        first, count = team if team else (0, 0)
        flags |= self._f_excl if exclusive else 0  # the kernel checks the real teams
        w = [0] * 16
        w[self._hdr] = seg & 0xFF | flags << 8
        w[self._team_at] = first | count << 16
        for at, value in zip(self._arg_at, args):
            w[at] = int(value) & M32
        self._words.extend(w)
        self._firsts.append(-1 if flags & self._f_queue else first)
        self._counts.append(count)
        self._excl.append(1 if exclusive else 0)
        self._fl.append(flags)
        ticket = self._next
        self._next += 1
        return ticket

    def _hand(self):
        """Gives the records issued one by one to the port, as one block."""
        if not self._counts:
            return
        n = len(self._counts)
        words = np.frombuffer(self._words, np.uint32).reshape(n, 16)
        first = np.frombuffer(self._firsts, np.int32)
        count = np.frombuffer(self._counts, np.int32)
        excl = np.frombuffer(self._excl, np.uint8)
        flags = np.frombuffer(self._fl, np.uint8)
        self._send_block(words, first, count, excl if excl.any() else None, flags)
        self._unsent()

    def _send_block(self, words, first, count, excl, flags):
        """Notes tickets handed .. handed + n - 1 (teams, flags, sequences) and queues
        their records in the port. A queued record's first is -1 (the kernel's)."""
        n = len(words)
        lo = self._handed - self._base
        self._grow(lo + n)
        rec = count > 0
        self._first[lo : lo + n] = first
        self._count[lo : lo + n] = count
        self._flags[lo : lo + n] = flags
        self._seq[lo : lo + n] = np.where(rec, self._next_seq + np.cumsum(rec) - 1, 0)
        self._next_seq += int(rec.sum())
        self.port.enqueue(np.ascontiguousarray(words), excl)
        self._handed += n

    def _grow(self, rows):
        have = len(self._has)
        if rows <= have:
            return
        size = max(rows, 2 * have, 1024)
        for name in COLUMNS:
            old = getattr(self, name)
            new = np.zeros(size, old.dtype)
            new[:have] = old
            setattr(self, name, new)

    # Events -----------------------------------------------------------------------

    def _take(self, rows):
        """Files a block of events (index, req, status, ticks). Every event is of a
        ticket from the oldest without one on, so its low 32 bits name it."""
        rows = np.asarray(rows).reshape(-1, 4)
        if self._engine_state is not None:  # the engines' events go to their tickets
            rows = self._engine_state.take(rows)
        base = self._oldest
        tickets = base + ((rows[:, 1].astype(np.int64) - base) & M32)
        i = tickets - self._base
        stored = self._handed - self._base - self.port.queued
        stray = (i >= stored) | self._has[np.minimum(i, len(self._has) - 1)]
        if stray.any() or len(np.unique(i)) != len(i):
            t = int(tickets[np.argmax(stray)]) if stray.any() else int(tickets[0])
            self.abort()
            raise self._remember(
                KernelFailed(
                    f"an event for command {t & M32} (ticket {t}), which is not in "
                    "flight: the run is stopped"
                )
            )
        self._status[i] = rows[:, 2]
        self._ticks[i] = rows[:, 3]
        self._has[i] = True
        at, end = self._oldest - self._base, self._handed - self._base
        waiting = ~self._has[at:end]
        self._oldest += int(np.argmax(waiting)) if waiting.any() else end - at
        timeout = None
        for ticket in tickets[(rows[:, 2] & 0xFF) != self._done].tolist():
            ev = self.event(ticket)
            if ev.kind == "refused":
                self._refused[ticket] = ev.aux
                if self.team(ticket) is not None:
                    self._unsequence(ticket)
            elif ev.kind == "timeout":
                timeout = ev
        if timeout is not None:
            self._fail_timeout(timeout)

    def _unsequence(self, ticket):
        """A refused record took no sequence number: later records move down one."""
        i = ticket - self._base
        self._seq[i] = 0
        later = self._seq[i + 1 : self._handed - self._base]
        later[later > 0] -= 1
        self._next_seq -= 1

    def _arrived(self, tickets):
        return self._has[tickets - self._base]

    def wait(self, tickets, timeout=10.0):
        """Sends what is queued up to the tickets and waits for their events.

        Returns the events (one, or a list for a list of tickets). Raises
        RecordRefused, RecordTimeout, KernelFailed or WaitTimeout.
        """
        single = isinstance(tickets, int | np.integer)
        wanted = np.asarray([tickets] if single else list(tickets), np.int64)
        if not wanted.size:
            return []
        if wanted.max() >= self._next or wanted.min() < 1:
            raise ValueError("a ticket that was never issued")
        self._hand()
        gone = wanted[
            (wanted < self._base) | self._gone[np.maximum(wanted - self._base, 0)]
        ]
        if gone.size:
            raise ValueError(f"ticket {gone[0]}: its event was returned or forgotten")
        self._wait(np.unique(wanted), timeout)
        events = [self.event(t) for t in wanted.tolist()]
        self.forget(wanted)
        return events[0] if single else events

    def wait_any(self, tickets, timeout=10.0):
        """Sends what is queued and waits for the first event among the tickets;
        returns it (the lowest ticket if several arrived) and forgets that ticket.
        Raises as wait() does, for that ticket."""
        wanted = np.unique(np.asarray(list(tickets), np.int64))
        if not wanted.size or wanted.max() >= self._next or wanted.min() < 1:
            raise ValueError("no tickets, or a ticket that was never issued")
        self._hand()
        if (
            (wanted < self._base) | self._gone[np.maximum(wanted - self._base, 0)]
        ).any():
            raise ValueError("a ticket whose event was returned or forgotten")
        t0 = time.perf_counter()
        try:
            deadline = time.monotonic() + timeout
            while not self._arrived(wanted).any():
                self._check_alive()
                left = deadline - time.monotonic()
                if left <= 0:
                    raise WaitTimeout(
                        f"none of {wanted.size} tickets done in {timeout} s",
                        wanted.tolist(),
                    )
                self._step_any(wanted, left)
        finally:
            self.wait_s += time.perf_counter() - t0
        ticket = int(wanted[self._arrived(wanted)][0])
        ev = self.event(ticket)
        self.forget([ticket])
        if ev.kind == "refused":
            raise RecordRefused(ticket, ev.aux, self._refusals)
        return ev

    def forget(self, tickets):
        """Releases what the run keeps for tickets whose events arrived (the event,
        and the team and sequence unless the run is traced): for callers that read
        events with event() after drain() rather than from wait()."""
        t = np.asarray(
            [tickets] if isinstance(tickets, int) else list(tickets), np.int64
        )
        t = t[(t >= self._base) & (t < self._handed)] - self._base
        t = t[self._has[t]]
        self._gone[t] = True
        for ticket in (t + self._base).tolist():
            self._refused.pop(ticket, None)
        self._compact()

    def _compact(self):
        """Drops the leading rows of released tickets once there are many."""
        if self._keep:
            return
        n = min(self._oldest, self._handed) - self._base
        done = int(np.argmin(self._gone[:n])) if not self._gone[:n].all() else n
        if done < COMPACT_ROWS or 2 * done < len(self._has):
            return
        for name in COLUMNS:
            setattr(self, name, getattr(self, name)[done:].copy())
        self._base += done

    def _wait(self, wanted, timeout):
        """Waits for the events of `wanted` (sorted tickets)."""
        t0 = time.perf_counter()
        try:
            self._hand()
            deadline = time.monotonic() + timeout
            pending = wanted[~self._arrived(wanted)]
            while pending.size:
                self._check_alive()
                left = deadline - time.monotonic()
                if left <= 0:
                    raise WaitTimeout(
                        f"{pending.size} of {wanted.size} tickets not done in "
                        f"{timeout} s (first {int(pending[0])}; "
                        f"{self.port.queued} not stored)",
                        pending.tolist(),
                    )
                self._step(pending, int(pending[-1]), left)
                pending = pending[~self._arrived(pending)]
            if self._refused:
                for t in sorted(self._refused):
                    if np.isin(t, wanted):
                        raise RecordRefused(t, self._refused[t], self._refusals)
        finally:
            self.wait_s += time.perf_counter() - t0

    def _step(self, pending, upto, left):
        """One native round: store up to ticket `upto`, read events until those of
        `pending` arrived (or a slice ends). Ends the run if the kernel finished."""
        t = time.perf_counter()
        rows = self.port.wait(
            (pending & M32).astype(np.uint32), upto & M32, min(left, WAIT_SLICE_S)
        )
        self.native_s += time.perf_counter() - t
        if len(rows):
            self._take(rows)
        elif self.port.host_word(0) >= self.h["KMD_STATE_DONE"]:
            self._fail_kernel("the dispatch kernel finished on its own")

    def _step_any(self, wanted, left):
        """Stores one batch up to the last ticket, then reads events until any arrives
        (or a slice ends): the native loop returns on the first event."""
        t = time.perf_counter()
        rows = self.port.wait(np.zeros(0, np.uint32), int(wanted[-1]) & M32, 0.0)
        if not len(rows):
            rows = self.port.wait_events(min(left, WAIT_SLICE_S))
        self.native_s += time.perf_counter() - t
        if len(rows):
            self._take(rows)
        elif self.port.host_word(0) >= self.h["KMD_STATE_DONE"]:
            self._fail_kernel("the dispatch kernel finished on its own")

    def drain(self, timeout=10.0):
        """Sends everything queued and waits for every record's event (refusals do not
        raise here); returns how many events it waited for."""
        self._hand()
        at, end = self._oldest - self._base, self._handed - self._base
        open_ = (self._count[at:end] > 0) & ~self._has[at:end]
        wanted = np.flatnonzero(open_) + self._oldest
        if wanted.size:
            with contextlib.suppress(RecordRefused):
                self._wait(wanted, timeout)
        return int(wanted.size)

    def thread_result(self, timeout=10.0):
        """Waits (bounded) for the program's X280 thread, stops it and returns its
        result block (komorebi.thread.ThreadResult). Events go on as before."""
        if self.thread is None:
            raise KomorebiError("the program has no thread")
        return self.thread.wait(timeout)

    def event(self, ticket):
        """The event of a ticket, or None if it has not arrived (or was released)."""
        i = ticket - self._base
        if not 0 <= i < self._handed - self._base or not self._has[i] or self._gone[i]:
            return None
        status, ticks = int(self._status[i]), int(self._ticks[i])
        kind = self._kinds.get(status & 0xFF, str(status & 0xFF))
        wanted = self._flags[i] & self._f_result and kind == "done"
        return Event(ticket, kind, status >> 8, ticks, ticks if wanted else None)

    def seq(self, ticket):
        """The sequence number the kernel gave a record (the trace's index). Raises
        KomorebiError after an engine or a card thread started (not known)."""
        n = self._seq_of(ticket)
        if n is not None and self._seq_unknown(ticket):
            raise KomorebiError(f"ticket {ticket}: {st.UNKNOWN_SEQ}")
        return n

    def _unknown_from(self):
        """The first ticket whose sequence number the host does not know, or None."""
        t = self.thread
        if t is not None and t.on_card and t.ring is not None:
            return 1  # the thread starts at the load and may issue records
        return None if self._engine_from is None else self._engine_from + 1

    def _seq_unknown(self, ticket):
        u = self._unknown_from()
        return u is not None and ticket >= u

    def _seq_of(self, ticket):
        self._hand()
        i = ticket - self._base
        if not 0 <= i < self._handed - self._base or self._seq[i] == 0:
            return None
        if self._gone[i] and not self._keep:
            return None
        return _seq_number(int(self._seq[i]))

    def team(self, ticket):
        self._hand()
        i = ticket - self._base
        if not 0 <= i < self._handed - self._base or self._count[i] == 0:
            return None
        if self._gone[i] and not self._keep:
            return None
        first = int(self._first[i])
        return (first if first >= 0 else None, int(self._count[i]))

    @property
    def stored(self):
        """Commands stored into the card so far."""
        return self._handed - self._base - self.port.queued

    # Graphs -----------------------------------------------------------------------

    def run_hostfed(self, graph, bindings=None, timeout=10.0, engine=None):
        """Runs a graph or a stream host-fed: its executor on the host, its records
        through this run's host ring. Returns a HostfedResult (status, error, regs,
        runs, records, events); see `komorebi.engines.graph_hostfed`."""
        if stream_hostfed.is_stream(graph):
            return stream_hostfed.run_hostfed(self, graph, bindings, timeout)
        return graph_hostfed.run_hostfed(self, graph, bindings, timeout, engine)

    # Engines ----------------------------------------------------------------------

    _engine_state = None  # komorebi.engines.control.Engines, from the first engine call
    _engine_from = None  # ticket of the first RUN: later records' sequences are unknown

    def _engines(self):
        if self._engine_state is None:
            self._engine_state = control.Engines(self)
        return self._engine_state

    def upload(self, obj, kind=None):
        """Writes a Stream's or a Graph's upload (or encoded bytes of `kind` "stream"
        or "graph") into the dispatch kernel's upload region, first fit beside the
        others; returns its Upload. ValueError, before any write, if it does not fit."""
        return self._engines().upload(obj, kind)

    def release(self, upload, timeout=10.0):
        """Frees an upload's lines; a kept graph is dropped first (CANCEL)."""
        self._engines().release(upload, timeout)

    def start(self, upload, bindings=(), kept=None, timeout=10.0):
        """Runs an upload (KM_SEG_RUN) and returns its EngineTicket. A graph takes
        `bindings` (a dict by name, or a list of words); `kept` sets its keep bit.
        Raises EngineRefused if the kernel refuses the command."""
        return self._engines().start(upload, bindings, kept, timeout)

    def restart(self, req, bindings=(), timeout=10.0):
        """Runs the kept graph with `req` again with new bindings; an EngineTicket."""
        return self._engines().restart(req, bindings, timeout)

    def cancel(self, ticket, timeout=10.0):
        """Cancels a running engine (KM_SEG_CANCEL); its error event ends the ticket."""
        self._engines().cancel(ticket, timeout)

    def wait_engine(self, ticket, timeout=10.0):
        """Waits for an engine's end and returns its EngineEvent. Raises EngineError
        (its error event), EngineTimeout or KernelFailed."""
        return self._engines().wait(ticket, timeout)

    def _settle(self, ticket):
        """Marks a command that has an event only when refused (RUN, CANCEL) as
        answered, once an event of a later command arrived."""
        i = ticket - self._base
        if self._has[i]:
            return
        self._has[i] = True
        at, end = self._oldest - self._base, self._handed - self._base
        waiting = ~self._has[at:end]
        self._oldest += int(np.argmax(waiting)) if waiting.any() else end - at

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
        """Cancels running engines, drains, sends EXIT to every core, stops the
        kernel, closes the X280 runtime, then reads back the state words and the
        trace, all within `timeout`. On any error it still closes (abort) and raises.
        """
        if self._stopped:
            return self._result
        by = time.monotonic() + timeout
        try:
            if self.thread is not None:
                self.thread.stop()
            if self._engine_state is not None:
                self._engine_state.stop(_left(by))  # EXIT waits for running engines
            self.drain(_left(by))
            exit_ticket = self._queue_command(self.h["KM_SEG_EXIT"], (0, self.cores))
            self.wait(exit_ticket, _left(by))
            stop_ticket = self._queue_command(self.h["KM_SEG_STOP"], None)
            self._send(stop_ticket, _left(by))
        except KernelFailed:
            raise
        except BaseException:
            self.abort()
            raise
        self._finish(_left(by))
        result = self._result
        wrong = self._finish_problems(result)
        if wrong:
            raise self._remember(
                KernelFailed(f"the stop did not end cleanly: {wrong}", result=result)
            )
        return result

    def _finish_problems(self, r):
        """What is wrong with a finish, as text ("" when the result is ready, the tile
        restored, the state DONE and the error 0)."""
        wrong = []
        if not r.get("result_ready", True):
            wrong.append("the kernel's result never became ready")
        if not r.get("cleaned", True):
            wrong.append(
                "the X280 runtime could not restore cluster 0 (reset the card)"
            )
        if r["state"] != self.h["KMD_STATE_DONE"]:
            wrong.append(f"state {r['state_name']}")
        if r["error"] != 0:
            wrong.append(f"error {r['error_name']} in {r['where_name']}")
        if r.get("failed"):
            wrong.append(r["failed"])
        return "; ".join(wrong)

    def _send(self, ticket, timeout):
        """Stores queued commands up to `ticket` without waiting for events."""
        self._hand()
        deadline = time.monotonic() + timeout
        none = np.zeros(0, np.uint32)
        while self.stored < ticket - self._base + 1:
            if time.monotonic() > deadline:
                raise WaitTimeout(f"command {ticket} not stored in {timeout} s", [])
            rows = self.port.wait(none, ticket & M32, 0.0)
            if len(rows):
                self._take(rows)

    def abort(self):
        """Stops the kernel through the runtime and reads back what the cores left;
        for failure paths. Errors while reading back are suppressed."""
        if self._stopped:
            return
        if self.thread is not None:
            self.thread.running = False  # the runtime's close stops it
        # if the close raises, the run stays loaded and a later exit path
        # (Session.close, the exit hooks) closes it again
        cleaned = self.backend.abort()
        self._stopped = True
        self._read_back(stop_grid=True)
        if not cleaned:
            raise self._remember(
                KernelFailed(
                    "the X280 runtime could not restore cluster 0 after the stop; "
                    "the card may need a reset"
                )
            )

    def _fail_start(self, state, timeout=1.0):
        """The kernel did not report READY: finishes it, reads back, raises StartError
        with its result block."""
        # stopped after this, unless the runtime could not be closed: a later exit
        # path (Session.close, the exit hooks) then closes it
        self._finish(timeout, stop_grid=True)
        r = self._result
        raise self._remember(
            StartError(
                f"the dispatch kernel did not report READY (state {state}, hart "
                f"result {r.get('result')}): {self._finish_problems(r)}",
                result=r,
            )
        )

    def _finish(self, timeout, stop_grid=False):
        self._stopped = True
        try:
            fin = self.backend.finish(timeout)
        except BaseException:
            try:
                self.backend.abort()
            except BaseException:
                self._stopped = False  # the runtime is still open: abort() again later
                raise
            self._read_back(stop_grid=True)
            raise
        self._result = st.result_block(fin["words"], self.h)
        self._result.update({k: v for k, v in fin.items() if k != "words"})
        self._read_back(stop_grid)

    def _read_back(self, stop_grid=False):
        blocks, trace_rows = self.backend.read_back(stop_grid=stop_grid)
        if self.thread is not None:
            self.thread.grid_stopped()
        self._blocks = blocks
        self._state = st.state_words(blocks, 0, self.h)
        if trace_rows is not None:
            n = self._handed - self._base
            tickets = np.arange(n) + self._base
            rec = (self._seq[:n] > 0) & (self._first[:n] >= 0)  # queued: team unknown
            late = np.zeros(n, bool)  # after an engine or a card thread: not mapped
            if (u := self._unknown_from()) is not None:
                late = rec & (tickets >= u)
            rec &= ~late
            numbers = (self._seq[:n][rec] - 1) % M32 + 1  # skips 0, as the kernel
            seqs = dict(zip(tickets[rec].tolist(), numbers.tolist()))
            teams = {
                t: (f, c)
                for t, f, c in zip(
                    tickets[rec].tolist(),
                    self._first[:n][rec].tolist(),
                    self._count[:n][rec].tolist(),
                )
            }
            self._trace = st.Trace(
                trace_rows,
                seqs,
                teams,
                self.backend.mhz,
                self.h,
                tickets[late].tolist(),
            )

    def _remember(self, error):
        self._failure = error
        return error

    def _fail_timeout(self, ev):
        self._finish(10.0)
        r = self._result
        team = self.team(ev.ticket)
        seq = r["seq"] if r["error"] == self.h["KMD_ERR_DEADLINE"] else None
        if seq is None and not self._seq_unknown(ev.ticket):
            seq = self._seq_of(ev.ticket)
        known = team and team[0] is not None
        cores = st.not_reported(self._state, team, ev.aux, seq, self.h) if known else []
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
        """Every core's state words, port slots, lane areas and dynamic-run words, as
        dicts; valid after the stop."""
        self._require_stopped()
        return [
            st.core_state(self._state, c, self.h)
            | {
                "ports": st.ports(self._blocks[c], self.h),
                "lanes": st.lanes(self._blocks[c], self.h),
                "dynamic": st.dynamic(self._blocks[c], self.h),
            }
            for c in range(self.cores)
        ]

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
