"""A loaded program under the dispatcher: submit commands, wait for them, stop,
read back state words and trace. The card is reached only through `backend`, so the
bookkeeping is the same against a fake transport.
"""

import contextlib
import time
from array import array
from typing import NamedTuple

import numpy as np

# np.unique imports numpy.ma on its first call (about 8 ms): here, not in a first wait
import numpy.ma

from komorebi import abi
from komorebi import state as st
from komorebi.engines import control, graph_hostfed, stream_hostfed
from komorebi.errors import (
    CommandRejected,
    CommandTimeout,
    KernelFailed,
    KomorebiError,
    StartError,
    ThreadFailed,
    WaitTimeout,
)

M32 = abi.M32
WAIT_SLICE_S = 0.05  # longest single native wait, so Python stays responsive
THREAD_POLL_S = 0.002  # one look at a thread while several are waited for
COMPACT_ROWS = 4096  # released handles kept before the columns are shifted
COLUMNS = (
    "_first", "_count", "_flags", "_seq", "_kseq", "_status", "_ticks", "_has", "_gone"
)  # fmt: skip
SEQ_LOW = (
    1 << 16
)  # a DONE or TIMEOUT event carries the low 16 bits of the command's seq
SEQ_BACK = 2048  # how far a queued command's seq may fall behind the host's order


class Event(NamedTuple):
    """A finished command: its handle, kind (done, timeout, pong, rejected), the aux
    (done: the group's first core; timeout: the slot; rejected: the reason),
    submit-to-completion time in X280 timer ticks, and for a command submitted with
    `result=True` the leader's result word (sent as the ticks)."""

    handle: int
    kind: str
    aux: int
    ticks: int
    result: int | None = None

    @property
    def us(self):
        return self.ticks / 50.0  # the X280 timer runs at 50 MHz


def stuck_message(cluster, r):
    """What a stuck thread (ThreadResult.stuck) means, for ThreadFailed."""
    stop = (
        f"the runtime could not stop the cluster ({r.detail}): the card needs a reset"
        if r.detail
        else "the runtime stopped the cluster"
    )
    return (
        f"the thread on cluster {cluster} did not return, nor after the stop flag: a"
        " core is stuck outside the library (which alone sees the stop flag and the"
        f" deadline) or stopped by a trap; its last progress marker {r.where}; {stop}"
    )


def _left(by):
    """Seconds left until the monotonic time `by` (0 once it passed)."""
    return max(by - time.monotonic(), 0.0)


def _seq_number(n):
    """The kernel's sequence number of the n-th command: it counts in 32 bits and skips
    0 (it also skips a slot's last value, which only a wrap can repeat)."""
    return (n - 1) % M32 + 1


class Run:
    """Submits commands to a loaded program and collects their events.

    backend: `port` (enqueue, queued, wait, host_word, batch; its own loop), `finish`,
    `abort()`, `read_back()` -> (control blocks, trace), `mhz`.
    """

    def __init__(self, backend, grid, program=None, batch=64, h=None):
        self.h = h or abi.headers()
        self.backend = backend
        self.port = backend.port
        self.port.batch = batch
        self._keep = bool(getattr(backend, "trace_cap", 0))  # the trace needs groups
        self.grid = grid
        self.cores = grid[0] * grid[1]
        self.program = program
        self.batch = batch
        k = self.h.constants
        self._kinds = {
            k["KM_EV_DONE"]: "done",
            k["KM_EV_TIMEOUT"]: "timeout",
            k["KM_EV_PONG"]: "pong",
            k.get("KM_EV_REJECTED", 3): "rejected",
        }
        self._done = k["KM_EV_DONE"]
        self._numbered = (k["KM_EV_DONE"], k["KM_EV_TIMEOUT"])  # aux holds seq bits
        self._aux_low, self._seq_shift = k["KM_EV_AUX_LOW"], k["KM_EV_SEQ_SHIFT"]
        self._f_result, self._f_queue = k["KM_F_RESULT"], k["KM_F_QUEUE"]
        self._f_excl = k["KM_F_EXCL"]
        self._rejections = {
            v: n[len("KMD_REJECT_") :].lower()
            for n, v in k.items()
            if n.startswith("KMD_REJECT_")
        }
        self._nop, self._user_max = (
            self.h["KM_HANDLER_NOP"],
            self.h["KM_HANDLER_USER_MAX"],
        )
        self._hdr, self._group_at, self._arg_at = self.h.positions
        self._next = 1  # the next handle; command number = handle mod 2**32
        self._next_seq = 1  # the kernel's sequence of the next command
        self._new_handles(1)
        self._rejected = {}
        self._failure = None
        self._stopped = False
        self._result = None
        self._state = None
        self._blocks = None
        self._trace = None
        self.thread = None  # the program's first X280 thread (thread.ThreadRun)
        self.threads = {}  # every X280 thread by work cluster
        self.submit_s = 0.0  # host time in submit() and submit_many()
        self.wait_s = 0.0  # host time in wait() and drain(), native loop included
        self.native_s = 0.0  # of which in the native loop (stores, reading events)

    def _new_handles(self, base):
        """Empty handle columns from handle `base` on, and an empty unsent block."""
        self._base = base  # handle of row 0 of the columns
        self._handed = base  # handles below this are with the port
        self._oldest = base  # the oldest handle without an event
        self._first = np.zeros(0, np.int32)
        self._count = np.zeros(0, np.int32)  # 0: a command to the kernel itself
        self._flags = np.zeros(0, np.uint8)  # the command's KM_F_* flags
        self._seq = np.zeros(0, np.int64)  # host commands up to this one; 0: no command
        self._kseq = np.zeros(
            0, np.int64
        )  # the kernel's count, from the event; 0: not yet
        self._ref = (0, 0)  # (kernel count, host count) after the dropped rows
        self._status = np.zeros(0, np.uint32)
        self._ticks = np.zeros(0, np.uint32)
        self._has = np.zeros(0, bool)  # the event arrived
        self._gone = np.zeros(0, bool)  # returned by wait() or forgotten
        self._unsent()

    def _unsent(self):
        # submit() appends here; new objects, since numpy may still view the old ones
        self._words, self._firsts, self._counts = array("I"), array("i"), array("i")
        self._fl = array("B")

    # Submitting --------------------------------------------------------------------

    def submit(
        self, handler, group=None, args=(), exclusive=False, result=False, queued=False
    ):
        """Queues a command for handler `handler` (name or number) on cores [first,
        first + count) (default every core); returns its handle. Nothing is sent until a
        wait. `result`: the event carries the leader's result word. `queued`: `group` is
        a count and the kernel picks the first core (group() then has first None)."""
        t0 = time.perf_counter()
        if handler.__class__ is not int:
            handler = (
                self.program.handler(handler)
                if self.program is not None
                else int(handler)
            )
        if not (handler == self._nop or 1 <= handler <= self._user_max):
            raise ValueError(
                f"handler {handler}: not NOP and not in 1..KM_HANDLER_USER_MAX"
            )
        if queued:
            if exclusive:
                raise ValueError("a queued command cannot be exclusive")
            group = self._group((0, group))
        else:
            group = self._group(group)
        if len(args) > len(self._arg_at):
            raise ValueError(f"{len(args)} arguments; a command has 10")
        if any(not 0 <= int(a) <= M32 for a in args):
            raise ValueError("arguments are 32-bit unsigned words")
        flags = (self._f_result if result else 0) | (self._f_queue if queued else 0)
        handle = self._queue_command(handler, group, args, exclusive, flags)
        self.submit_s += time.perf_counter() - t0
        return handle

    def submit_many(self, handler, firsts, counts, args=None, exclusive=False):
        """Queues n commands at once: `handler` a name, a number or n numbers, `firsts`
        and `counts` the groups (n each, or one for all), `args` an [n, m] matrix of
        32-bit words (m <= 10) or None. Returns the handles as a range."""
        t0 = time.perf_counter()
        self._check_alive()
        if isinstance(handler, str):
            handler = self.program.handler(handler)
        cols = [np.asarray(handler), np.asarray(firsts), np.asarray(counts)]
        if args is not None:
            args = np.asarray(args)
            if args.ndim != 2 or args.shape[1] > len(self._arg_at):
                raise ValueError("arguments are an [n, m] matrix with m <= 10")
            if args.dtype.kind not in "iu":
                raise ValueError("arguments are 32-bit unsigned words")
            cols.append(np.zeros(args.shape[0], np.int64))  # its row count
        if any(c.dtype.kind not in "iu" for c in cols[:3]):
            raise ValueError("handlers, firsts and counts are integers")
        try:
            n = np.broadcast_shapes(*(c.shape for c in cols))
        except ValueError:
            n = ()
        if len(n) != 1:
            raise ValueError("handlers, firsts, counts and arguments: one row each")
        handlers, first, count = (
            np.broadcast_to(c, n).astype(np.int64) for c in cols[:3]
        )
        bad = ~(
            (handlers == self._nop) | ((handlers >= 1) & (handlers <= self._user_max))
        )
        if bad.any():
            raise ValueError(
                f"handler {int(handlers[bad][0])}: not NOP and not in"
                " 1..KM_HANDLER_USER_MAX"
            )
        bad = (count < 1) | (first < 0) | (first + count > self.cores)
        if bad.any():
            i = int(np.argmax(bad))
            raise ValueError(
                f"group ({int(first[i])}, {int(count[i])}) is not inside {self.cores} "
                "cores"
            )
        words = np.zeros((n[0], 16), np.uint32)
        words[:, self._hdr] = handlers & 0xFF
        words[:, self._group_at] = first | count << 16
        if args is not None and args.size:
            if args.min() < 0 or args.max() > M32:
                raise ValueError("arguments are 32-bit unsigned words")
            words[:, self._arg_at[: args.shape[1]]] = args
        self._hand()
        start = self._next
        self._next += n[0]
        if exclusive:
            words[:, self._hdr] |= self._f_excl << 8
        self._send_block(words, first, count, np.zeros(n[0], np.uint8))
        self.submit_s += time.perf_counter() - t0
        return range(start, self._next)

    def _group(self, group):
        if group is None:
            return (0, self.cores)
        first, count = int(group[0]), int(group[1])
        if count < 1 or first < 0 or first + count > self.cores:
            raise ValueError(
                f"group ({first}, {count}) is not inside {self.cores} cores"
            )
        return (first, count)

    def _queue_command(self, handler, group, args=(), exclusive=False, flags=0):
        self._check_alive()
        first, count = group if group else (0, 0)
        flags |= self._f_excl if exclusive else 0  # the kernel orders it
        w = [0] * 16
        w[self._hdr] = handler & 0xFF | flags << 8
        w[self._group_at] = first | count << 16
        for at, value in zip(self._arg_at, args):
            w[at] = int(value) & M32
        self._words.extend(w)
        self._firsts.append(-1 if flags & self._f_queue else first)
        self._counts.append(count)
        self._fl.append(flags)
        handle = self._next
        self._next += 1
        return handle

    def _hand(self):
        """Gives the commands submitted one by one to the port, as one block."""
        if not self._counts:
            return
        n = len(self._counts)
        words = np.frombuffer(self._words, np.uint32).reshape(n, 16)
        first = np.frombuffer(self._firsts, np.int32)
        count = np.frombuffer(self._counts, np.int32)
        flags = np.frombuffer(self._fl, np.uint8)
        self._send_block(words, first, count, flags)
        self._unsent()

    def _send_block(self, words, first, count, flags):
        """Notes handles handed .. handed + n - 1 (groups, flags, sequences) and queues
        their commands in the port. A queued command's first is -1 (the kernel's)."""
        n = len(words)
        lo = self._handed - self._base
        self._grow(lo + n)
        rec = count > 0
        self._first[lo : lo + n] = first
        self._count[lo : lo + n] = count
        self._flags[lo : lo + n] = flags
        self._seq[lo : lo + n] = np.where(rec, self._next_seq + np.cumsum(rec) - 1, 0)
        self._next_seq += int(rec.sum())
        self.port.enqueue(np.ascontiguousarray(words))
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
        handle from the oldest without one on, so its low 32 bits name it."""
        rows = np.asarray(rows).reshape(-1, 4)
        if self._engine_state is not None:  # the engines' events go to their handles
            rows = self._engine_state.take(rows)
        base = self._oldest
        handles = base + ((rows[:, 1].astype(np.int64) - base) & M32)
        i = handles - self._base
        stored = self._handed - self._base - self.port.queued
        stray = (i >= stored) | self._has[np.minimum(i, len(self._has) - 1)]
        if stray.any() or len(np.unique(i)) != len(i):
            t = int(handles[np.argmax(stray)]) if stray.any() else int(handles[0])
            self.abort()
            raise self._remember(
                KernelFailed(
                    f"an event for command {t & M32} (handle {t}), which is not in "
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
        for handle in handles[(rows[:, 2] & 0xFF) != self._done].tolist():
            ev = self.event(handle)
            if ev.kind == "rejected":
                self._rejected[handle] = ev.aux
                if self.group(handle) is not None:
                    self._unsequence(handle)
            elif ev.kind == "timeout":
                timeout = ev
        if timeout is not None:
            self._fail_timeout(timeout)

    def _unsequence(self, handle):
        """A rejected command took no sequence number: later commands move down one."""
        i = handle - self._base
        self._seq[i] = 0
        later = self._seq[i + 1 : self._handed - self._base]
        later[later > 0] -= 1
        self._next_seq -= 1

    def _arrived(self, handles):
        return self._has[handles - self._base]

    def wait(self, handles, timeout=10.0):
        """Sends what is queued up to the handles and waits for their events.

        Returns the events (one, or a list for a list of handles). Raises
        CommandRejected, CommandTimeout, KernelFailed or WaitTimeout.
        """
        single = isinstance(handles, int | np.integer)
        wanted = np.asarray([handles] if single else list(handles), np.int64)
        if not wanted.size:
            return []
        if wanted.max() >= self._next or wanted.min() < 1:
            raise ValueError("a handle that was never submitted")
        self._hand()
        gone = wanted[
            (wanted < self._base) | self._gone[np.maximum(wanted - self._base, 0)]
        ]
        if gone.size:
            raise ValueError(f"handle {gone[0]}: its event was returned or forgotten")
        self._wait(np.unique(wanted), timeout)
        events = [self.event(t) for t in wanted.tolist()]
        self.forget(wanted)
        return events[0] if single else events

    def wait_any(self, handles, timeout=10.0):
        """Sends what is queued and waits for the first event among the handles;
        returns it (the lowest handle if several arrived) and forgets that handle.
        Raises as wait() does, for that handle."""
        wanted = np.unique(np.asarray(list(handles), np.int64))
        if not wanted.size or wanted.max() >= self._next or wanted.min() < 1:
            raise ValueError("no handles, or a handle that was never submitted")
        self._hand()
        if (
            (wanted < self._base) | self._gone[np.maximum(wanted - self._base, 0)]
        ).any():
            raise ValueError("a handle whose event was returned or forgotten")
        t0 = time.perf_counter()
        try:
            deadline = time.monotonic() + timeout
            while not self._arrived(wanted).any():
                self._check_alive()
                left = deadline - time.monotonic()
                if left <= 0:
                    raise WaitTimeout(
                        f"none of {wanted.size} handles done in {timeout} s; "
                        f"{self._kernel_report()}",
                        wanted.tolist(),
                    )
                self._step_any(wanted, left)
        finally:
            self.wait_s += time.perf_counter() - t0
        handle = int(wanted[self._arrived(wanted)][0])
        ev = self.event(handle)
        self.forget([handle])
        if ev.kind == "rejected":
            raise CommandRejected(handle, ev.aux, self._rejections)
        return ev

    def forget(self, handles):
        """Releases what the run keeps for handles whose events arrived (the event,
        and the group and sequence unless the run is traced): for callers that read
        events with event() after drain() rather than from wait()."""
        t = np.asarray(
            [handles] if isinstance(handles, int) else list(handles), np.int64
        )
        t = t[(t >= self._base) & (t < self._handed)] - self._base
        t = t[self._has[t]]
        self._gone[t] = True
        for handle in (t + self._base).tolist():
            self._rejected.pop(handle, None)
        self._compact()

    def _compact(self):
        """Drops the leading rows of released handles once there are many."""
        if self._keep:
            return
        n = min(self._oldest, self._handed) - self._base
        done = int(np.argmin(self._gone[:n])) if not self._gone[:n].all() else n
        if done < COMPACT_ROWS or 2 * done < len(self._has):
            return
        if self._ref is not None:
            self._ref = self._count_after(done, self._ref)
        for name in COLUMNS:
            setattr(self, name, getattr(self, name)[done:].copy())
        self._base += done

    def _count_after(self, rows, ref):
        """`ref` moved past the first `rows` rows by their events' seq bits, each
        against the one before; None once a gap is too long to tell."""
        at = np.arange(rows)
        status = self._status[:rows].astype(np.int64)
        on = (self._seq[:rows] > 0) & self._has[:rows]
        at = at[on & np.isin(status & 0xFF, self._numbered)]
        if not at.size:
            return ref
        host = np.concatenate([[ref[1]], self._seq[at]])
        low = np.concatenate(
            [[ref[0] % SEQ_LOW], status[at] >> (8 + self._seq_shift) & (SEQ_LOW - 1)]
        )
        gap = (np.diff(low) - np.diff(host)) % SEQ_LOW
        gap = np.where(gap >= SEQ_LOW - SEQ_BACK, gap - SEQ_LOW, gap)
        if (gap >= SEQ_LOW // 2).any():
            return None
        return (ref[0] + int((np.diff(host) + gap).sum()), int(host[-1]))

    def _wait(self, wanted, timeout):
        """Waits for the events of `wanted` (sorted handles)."""
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
                        f"{pending.size} of {wanted.size} handles not done in "
                        f"{timeout} s (first {int(pending[0])}; "
                        f"{self.port.queued} not stored); {self._kernel_report()}",
                        pending.tolist(),
                    )
                self._step(pending, int(pending[-1]), left)
                pending = pending[~self._arrived(pending)]
            if self._rejected:
                for t in sorted(self._rejected):
                    if np.isin(t, wanted):
                        raise CommandRejected(t, self._rejected[t], self._rejections)
        finally:
            self.wait_s += time.perf_counter() - t0

    def _kernel_report(self):
        """What the backend tells of the kernel after a wait ran out (never raises)."""
        try:
            return self.backend.kernel_report()
        except Exception as err:  # noqa: BLE001 (a failure path: reported, not raised)
            return f"the kernel could not be probed ({type(err).__name__}: {err})"

    def _step(self, pending, upto, left):
        """One native round: store up to handle `upto`, read events until those of
        `pending` arrived (or a slice ends). Ends the run if the kernel finished."""
        t = time.perf_counter()
        rows = self.port.wait(
            (pending & M32).astype(np.uint32), upto & M32, min(left, WAIT_SLICE_S)
        )
        self.native_s += time.perf_counter() - t
        if len(rows):
            self._take(rows)
        elif self.port.host_word(0) >= self.h["KMD_STATE_DONE"]:
            self._fail_kernel("the dispatcher finished on its own")

    def _step_any(self, wanted, left):
        """Stores one batch up to the last handle, then reads events until any arrives
        (or a slice ends): the native loop returns on the first event."""
        t = time.perf_counter()
        rows = self.port.wait(np.zeros(0, np.uint32), int(wanted[-1]) & M32, 0.0)
        if not len(rows):
            rows = self.port.wait_events(min(left, WAIT_SLICE_S))
        self.native_s += time.perf_counter() - t
        if len(rows):
            self._take(rows)
        elif self.port.host_word(0) >= self.h["KMD_STATE_DONE"]:
            self._fail_kernel("the dispatcher finished on its own")

    def drain(self, timeout=10.0):
        """Sends everything queued and waits for every command's event (rejections do
        not raise here); returns how many events it waited for."""
        self._hand()
        at, end = self._oldest - self._base, self._handed - self._base
        open_ = (self._count[at:end] > 0) & ~self._has[at:end]
        wanted = np.flatnonzero(open_) + self._oldest
        if wanted.size:
            with contextlib.suppress(CommandRejected):
                self._wait(wanted, timeout)
        return int(wanted.size)

    def thread_result(self, timeout=10.0):
        """Waits (bounded) for the program's X280 thread, stops it and returns its
        result block (komorebi.thread.ThreadResult). Events go on as before."""
        if self.thread is None:
            raise KomorebiError("the program has no thread")
        r = self.thread.wait(timeout)
        if r.stuck:
            c = self.thread.thread.cluster
            raise ThreadFailed(stuck_message(c, r), c, {c: r})
        return r

    def thread_results(self, timeout=10.0):
        """Waits (bounded) for every X280 thread, stops each and returns {cluster:
        ThreadResult}. One that fails (no return in time, a library error) has the
        others asked to stop at once, and ThreadFailed carries every result."""
        pending = dict(self._thread_runs())
        if not pending:
            raise KomorebiError("the program has no thread")
        by, results, failed = time.monotonic() + timeout, {}, None
        while pending and failed is None:
            for c, t in list(pending.items()):
                if not t.on_card:
                    results[c] = t.wait(timeout)
                else:
                    left = by - time.monotonic()
                    ran = t.returned(min(THREAD_POLL_S, max(left, 0.0)))
                    if not ran and left > 0:
                        continue
                    results[c] = t.collect(ran)
                del pending[c]
                if not results[c].ran or results[c].state != "done":
                    failed = c
                    break
        for c, t in pending.items():
            results[c] = t.collect(False)
        if failed is not None:
            r = results[failed]
            how = "did not return in time" if not r.ran else f"failed ({r.error})"
            stuck = [c for c, x in results.items() if x.stuck]
            if stuck:
                c = failed if failed in stuck else stuck[0]
                what = stuck_message(c, results[c])
                if c != failed:
                    what = f"the thread on cluster {failed} {how}; {what}"
                raise ThreadFailed(what, failed, results)
            raise ThreadFailed(
                f"the thread on cluster {failed} {how}; the others were stopped",
                failed,
                results,
            )
        return results

    def _thread_runs(self):
        """{cluster: ThreadRun} of the program (the one thread when only it is set)."""
        if self.threads:
            return self.threads
        return {} if self.thread is None else {0: self.thread}

    def event(self, handle):
        """The event of a handle, or None if it has not arrived (or was released)."""
        i = handle - self._base
        if not 0 <= i < self._handed - self._base or not self._has[i] or self._gone[i]:
            return None
        status, ticks = int(self._status[i]), int(self._ticks[i])
        kind = self._kinds.get(status & 0xFF, str(status & 0xFF))
        wanted = self._flags[i] & self._f_result and kind == "done"
        aux = status >> 8
        if status & 0xFF in self._numbered:
            aux &= self._aux_low
        return Event(handle, kind, aux, ticks, ticks if wanted else None)

    def seq(self, handle):
        """The sequence number the kernel gave a command (the trace's index), from its
        done or timeout event; None before that event or for a rejected command."""
        self._hand()
        i = handle - self._base
        if not 0 <= i < self._handed - self._base or self._seq[i] == 0:
            return None
        if self._gone[i] and not self._keep:
            return None
        n = self._kernel_count(i)
        return None if n is None else _seq_number(n)

    def _kernel_count(self, i, ref=None):
        """Row i's place in the kernel's count of commands (every source), from the low
        16 bits in its event and the nearest earlier known row (or `ref`); None
        before the event. Raises if other sources submitted too many in between."""
        if self._kseq[i]:
            return int(self._kseq[i])
        status = int(self._status[i])
        if self._seq[i] == 0 or not self._has[i] or status & 0xFF not in self._numbered:
            return None
        if ref is None:
            known = np.flatnonzero(self._kseq[:i])
            j = int(known[-1]) if known.size else None
            ref = self._ref if j is None else (int(self._kseq[j]), int(self._seq[j]))
        if ref is None:
            raise KomorebiError(
                f"handle {i + self._base}: its sequence number is not known (an "
                f"earlier gap of other sources' commands was too long to tell)"
            )
        near = ref[0] + int(self._seq[i]) - ref[1]  # as if no other source submitted
        low = status >> (8 + self._seq_shift) & (SEQ_LOW - 1)
        gap = (low - _seq_number(near)) % SEQ_LOW
        gap -= SEQ_LOW if gap >= SEQ_LOW - SEQ_BACK else 0
        if gap >= SEQ_LOW // 2:
            raise KomorebiError(
                f"handle {i + self._base}: {gap} or more commands of other sources "
                f"since the host's previous command; its sequence number is not known "
                f"(an event carries its low 16 bits)"
            )
        n = near + gap
        n += next(d for d in (0, 1, -1) if _seq_number(n + d) % SEQ_LOW == low)
        self._kseq[i] = n
        return n

    def group(self, handle):
        self._hand()
        i = handle - self._base
        if not 0 <= i < self._handed - self._base or self._count[i] == 0:
            return None
        if self._gone[i] and not self._keep:
            return None
        first = int(self._first[i])
        if first < 0 and self._has[i] and self._status[i] & 0xFF == self._done:
            first = int(self._status[i]) >> 8 & self._aux_low  # the queue's choice
        return (first if first >= 0 else None, int(self._count[i]))

    @property
    def stored(self):
        """Commands stored into the card so far."""
        return self._handed - self._base - self.port.queued

    # Graphs -----------------------------------------------------------------------

    def run_hostfed(self, graph, bindings=None, timeout=10.0, engine=None):
        """Runs a graph or a stream host-fed: its executor on the host, its commands
        through this run's host ring. Returns a HostfedResult (status, error, regs,
        runs, commands, events); see `komorebi.engines.graph_hostfed`."""
        if stream_hostfed.is_stream(graph):
            return stream_hostfed.run_hostfed(self, graph, bindings, timeout)
        return graph_hostfed.run_hostfed(self, graph, bindings, timeout, engine)

    # Engines ----------------------------------------------------------------------

    _engine_state = None  # komorebi.engines.control.Engines, from the first engine call

    def _engines(self):
        if self._engine_state is None:
            self._engine_state = control.Engines(self)
        return self._engine_state

    def upload(self, obj, kind=None):
        """Writes a Stream's or a Graph's upload (or encoded bytes of `kind` "stream"
        or "graph") into the dispatcher's upload region, first fit beside the
        others; returns its Upload. ValueError, before any write, if it does not fit."""
        return self._engines().upload(obj, kind)

    def release(self, upload, timeout=10.0):
        """Frees an upload's lines; a kept graph is dropped first (CANCEL)."""
        self._engines().release(upload, timeout)

    def start(self, upload, bindings=(), kept=None, timeout=10.0):
        """Runs an upload (KM_HANDLER_RUN) and returns its EngineHandle. A graph takes
        `bindings` (a dict by name, or a list of words); `kept` sets its keep bit.
        Raises EngineRejected if the kernel rejects the command."""
        return self._engines().start(upload, bindings, kept, timeout)

    def restart(self, req, bindings=(), timeout=10.0):
        """Runs the kept graph with `req` again with new bindings; an EngineHandle."""
        return self._engines().restart(req, bindings, timeout)

    def cancel(self, handle, timeout=10.0):
        """Cancels a running engine (KM_HANDLER_CANCEL); its error event ends the
        handle."""
        self._engines().cancel(handle, timeout)

    def wait_engine(self, handle, timeout=10.0):
        """Waits for an engine's end and returns its EngineEvent. Raises EngineError
        (its error event), EngineTimeout or KernelFailed."""
        return self._engines().wait(handle, timeout)

    def _settle(self, handle):
        """Marks a command that has an event only when rejected (RUN, CANCEL) as
        answered, once an event of a later command arrived."""
        i = handle - self._base
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
            for t in self._thread_runs().values():
                t.stop(_left(by))  # each thread's stop is inside the deadline
            if self._engine_state is not None:
                self._engine_state.stop(_left(by))  # EXIT waits for running engines
            self.drain(_left(by))
            exit_handle = self._queue_command(
                self.h["KM_HANDLER_EXIT"], (0, self.cores)
            )
            self.wait(exit_handle, _left(by))
            stop_handle = self._queue_command(self.h["KM_HANDLER_STOP"], None)
            self._send(stop_handle, _left(by))
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

    def _send(self, handle, timeout):
        """Stores queued commands up to `handle` without waiting for events."""
        self._hand()
        deadline = time.monotonic() + timeout
        none = np.zeros(0, np.uint32)
        while self.stored < handle - self._base + 1:
            if time.monotonic() > deadline:
                raise WaitTimeout(f"command {handle} not stored in {timeout} s", [])
            rows = self.port.wait(none, handle & M32, 0.0)
            if len(rows):
                self._take(rows)

    def abort(self):
        """Stops the grid, then the kernel through the runtime, and reads back what the
        cores left; for failure paths. result() then holds the kernel's result block
        with `abandoned`, the commands left without their DONE event."""
        if self._stopped:
            return
        for t in self._thread_runs().values():
            t.running = False  # the runtime's close stops it
        # if the close raises, the run stays loaded and a later exit path
        # (Session.close, the exit hooks) closes it again
        cleaned = self.backend.abort()
        self._stopped = True
        left = self.backend.aborted
        if left is not None:
            x280_cores = [int(w) for w in left["x280_cores"]]
            self._result = st.result_block(left["words"], self.h) | {
                "x280_cores": x280_cores,
                "abandoned": x280_cores[self.h["KMD_H_ABANDONED"]],
            }
        self._read_back(stop_grid=True)
        if not cleaned:
            r = self._result
            kernel = (
                f"; the kernel left state {r['state_name']}, {r['abandoned']} commands "
                "abandoned"
                if r is not None
                else ""
            )
            raise self._remember(
                KernelFailed(
                    "the X280 runtime could not restore cluster 0 after the stop; "
                    f"the card may need a reset: {self.backend.close_failed}{kernel}",
                    result=r,
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
                f"the dispatcher did not report READY (state {state}, core "
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
        for t in self._thread_runs().values():
            t.grid_stopped()
        self._blocks = blocks
        self._state = st.state_words(blocks, 0, self.h)
        if trace_rows is not None:
            seqs, groups = {}, {}
            ref = self._ref
            for i in range(self._handed - self._base):  # in order: one pass
                try:
                    n = self._kernel_count(i, ref)
                except KomorebiError:  # a gap too long to tell: not in the trace
                    continue
                if n is None:
                    continue
                ref = (n, int(self._seq[i]))
                handle = i + self._base
                group = self.group(handle)
                if group is not None and group[0] is not None:
                    seqs[handle], groups[handle] = _seq_number(n), group
            self._trace = st.Trace(trace_rows, seqs, groups, self.backend.mhz, self.h)

    def _remember(self, error):
        self._failure = error
        return error

    def _fail_timeout(self, ev):
        self._finish(10.0)
        r = self._result
        group = self.group(ev.handle)
        seq = r["seq"] if r["error"] == self.h["KMD_ERR_DEADLINE"] else None
        if seq is None:
            seq = self.seq(ev.handle)
        known = group and group[0] is not None
        cores = (
            st.not_reported(self._state, group, ev.aux, seq, self.h) if known else []
        )
        raise self._remember(CommandTimeout(ev.handle, ev.aux, seq, group, cores, r))

    def _fail_kernel(self, why):
        self._finish(10.0)
        self.backend.log_state()
        r = self._result
        where = (
            f"exception on X280 core {r['req']} (mcause {r['slot']:#x}, mepc "
            f"{r['seq']:#x}, mtval {r['seen']:#x})"
            if r["error"] == self.h["KMD_ERR_TRAP"]
            else f"slot {r['slot']}, sequence {r['seq']}, command {r['req']}"
        )
        raise self._remember(
            KernelFailed(
                f"{why}: error {r['error_name']} in {r['where_name']}, {where}",
                result=r,
            )
        )

    # After the stop ---------------------------------------------------------------

    def result(self):
        """The dispatcher's result block (decoded); valid after the stop."""
        self._require_stopped()
        return self._result

    def state(self):
        """Every core's state words, port slots, broadcast areas and dynamic-run words,
        as dicts; valid after the stop."""
        self._require_stopped()
        return [
            st.core_state(self._state, c, self.h)
            | {
                "ports": st.ports(self._blocks[c], self.h),
                "broadcasts": st.broadcasts(self._blocks[c], self.h),
                "dynamic": st.dynamic(self._blocks[c], self.h),
            }
            for c in range(self.cores)
        ]

    def state_words(self):
        self._require_stopped()
        return self._state

    def trace(self):
        """The per-command trace (a state.Trace), or None if the run was not traced."""
        self._require_stopped()
        return self._trace

    def _require_stopped(self):
        if not self._stopped or self._state is None:
            raise KomorebiError("valid only after run.stop()")
