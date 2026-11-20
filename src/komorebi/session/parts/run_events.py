"""The run's events: the watcher's rows taken into the handle columns, waits for
commands and threads, the kept handles and their release (EventsMixin of Run)."""

import contextlib
import time

import numpy as np

# np.unique imports numpy.ma on its first call (about 8 ms): here, not in a first wait
import numpy.ma

from komorebi import abi
from komorebi.device.events import Event
from komorebi.errors import (
    CommandRejected,
    CommandTimeout,
    KernelFailed,
    KomorebiError,
    ThreadFailed,
    WaitTimeout,
)
from komorebi.program.description import unguarded
from komorebi.session import state as device_state

LINK_EVENTS = ("gave", "hung", "lost")  # KMD_LINK_*: what a link event says

M32 = abi.M32

WAIT_SLICE_S = 0.05  # longest single native wait, so Python stays responsive

_NO_HANDLES = np.zeros(0, np.uint32)  # a store's native round waits for no event

THREAD_POLL_S = 0.002  # one look at a thread while several are waited for

COMPACT_ROWS = 4096  # released handles kept before the columns are shifted

COLUMNS = (
    "_first", "_count", "_flags", "_sequence", "_kseq", "_status", "_ticks", "_has",
    "_gone",
)  # fmt: skip

SEQUENCE_LOW = (
    1 << 16
)  # a DONE or TIMEOUT event carries the low 16 bits of the command's sequence

SEQUENCE_BACK = (
    2048  # how far a queued command's sequence may fall behind the host's order
)


def stuck_message(cluster, result, name="cluster"):
    """What a stuck thread (ThreadResult.stuck) means, for ThreadFailed (`name`: what
    `cluster` is, thread_results's)."""
    stop = (
        f"the runtime could not stop the cluster ({result.detail}): the card needs a"
        " reset"
        if result.detail
        else "the runtime stopped the cluster"
    )
    return (
        f"the thread on {name} {cluster} did not return, nor after the stop flag: a"
        " core is stuck outside the library (which alone sees the stop flag and the"
        " deadline) or stopped by a trap; its last progress marker"
        f" {result.where}; {stop}"
    )


def _sequence_number(number):
    """The kernel's sequence number of the n-th command: it counts in 32 bits and skips
    0 (it also skips a slot's last value, which only a wrap can repeat)."""
    return (number - 1) % M32 + 1


def thread_results(threads, timeout, name="cluster"):
    """{key: ThreadResult} of the X280 threads {key: ThreadRun}, each waited for and
    stopped within `timeout` (Run.thread_results); `name` what a key is named in a
    ThreadFailed's text (a cluster on one card, a thread over islands)."""
    pending = dict(threads)
    if not pending:
        raise KomorebiError("the program has no thread")
    deadline, results, failed = time.monotonic() + timeout, {}, None
    while pending and failed is None:
        for cluster, thread_run in list(pending.items()):
            if not thread_run.on_card:
                results[cluster] = thread_run.wait(timeout)
            else:
                left = deadline - time.monotonic()
                ran = thread_run.returned(min(THREAD_POLL_S, max(left, 0.0)))
                if not ran and left > 0:
                    continue
                results[cluster] = thread_run.collect(ran)
            del pending[cluster]
            if not results[cluster].ran or results[cluster].state != "done":
                failed = cluster
                break
    for cluster, thread_run in pending.items():
        results[cluster] = thread_run.collect(False)
    if failed is not None:
        thread_failed(results, failed, name)
    return results


def thread_checked(result, key, name="cluster"):
    """`result`, thread `key`'s, when it returned `done`; else ThreadFailed as
    thread_results raises it (Run.thread_result)."""
    if result.ran and result.state == "done" and not result.stuck:
        return result
    return thread_failed({key: result}, key, name)


def thread_failed(results, failed, name):
    """Raises ThreadFailed for thread `failed` of `results` ({key: ThreadResult}): it
    did not return in time or failed, or a thread is stuck; the others were stopped."""
    result = results[failed]
    how = "did not return in time" if not result.ran else f"failed ({result.error})"
    stuck = [cluster for cluster, outcome in results.items() if outcome.stuck]
    if stuck:
        cluster = failed if failed in stuck else stuck[0]
        what = stuck_message(cluster, results[cluster], name)
        if cluster != failed:
            what = f"the thread on {name} {failed} {how}; {what}"
        raise ThreadFailed(what, failed, results)
    others = "; the others were stopped" if len(results) > 1 else ""
    raise ThreadFailed(f"the thread on {name} {failed} {how}{others}", failed, results)


class EventsMixin:
    """Run's events and waits (komorebi.session.run.Run)."""

    # Events -----------------------------------------------------------------------

    def _stalls(self, rows):
        """Keeps the watcher's stall events (no command's) in `stalls`; a stall core 0
        did not come back from stops the run. Returns the other rows."""
        stall = self.h.constants.get("KMD_EVENT_STALL")
        mine = (rows[:, 2] & 0xFF) == stall
        if stall is None or not mine.any():
            return rows
        stages = {
            self.h["KMD_STALL_SEEN"]: "stalled",
            self.h["KMD_STALL_BACK"]: "back",
            self.h["KMD_STALL_LOST"]: "lost",
        }
        for _, req, status, ticks in rows[mine].tolist():
            stage = stages.get(int(status) >> 8, str(int(status) >> 8))
            self.stalls.append({"stage": stage, "req": int(req), "ticks": int(ticks)})
            if stage == "lost":
                self.abort()
                raise self._remember(
                    KernelFailed(
                        f"the dispatcher's loop stalled (at host command {req}) and "
                        f"did not come back from the watcher's interrupt: the run is "
                        f"stopped"
                    )
                )
        return rows[~mine]

    def _links(self, rows):
        """Keeps the dispatcher's KMD_EVENT_LINK events (a give-up on link core `link`:
        what, req the host command named or 0, ticks) in `link_events`; the others."""
        kind = self.h.constants.get("KMD_EVENT_LINK")
        mine = (rows[:, 2] & 0xFF) == kind
        if kind is None or not mine.any():
            return rows
        names = {self.h[f"KMD_LINK_{name.upper()}"]: name for name in LINK_EVENTS}
        for _, req, status, ticks in rows[mine].tolist():
            what = int(status) >> 16 & 0xFF
            self.link_events.append(
                {
                    "link": int(status) >> 8 & 0xFF,
                    "what": names.get(what, str(what)),
                    "req": int(req),
                    "ticks": int(ticks),
                }
            )
        return rows[~mine]

    def links_named(self, handle):
        """The link events that name command `handle` (link, what), oldest first."""
        req = handle & M32
        return [
            (event["link"], event["what"])
            for event in self.link_events
            if req and event["req"] == req
        ]

    def _take(self, rows):
        """Files a block of events (index, req, status, ticks). Every event is of a
        handle from the oldest without one on, so its low 32 bits name it."""
        rows = np.asarray(rows).reshape(-1, 4)
        rows = self._links(self._stalls(rows))
        if not len(rows):
            return
        if self._engine_state is not None:  # the engines' events go to their handles
            rows = self._engine_state.take(rows)
        base = self._oldest
        handles = base + ((rows[:, 1].astype(np.int64) - base) & M32)
        i = handles - self._base
        stored = self._handed - self._base - self.port.queued
        stray = (i >= stored) | self._has[np.minimum(i, len(self._has) - 1)]
        if stray.any() or len(np.unique(i)) != len(i):
            stray_handle = (
                int(handles[np.argmax(stray)]) if stray.any() else int(handles[0])
            )
            self.abort()
            raise self._remember(
                KernelFailed(
                    f"an event for command {stray_handle & M32} (handle "
                    f"{stray_handle}), which is not in flight: the run is stopped"
                )
            )
        self._status[i] = rows[:, 2]
        self._ticks[i] = rows[:, 3]
        self._has[i] = True
        if self._kept:
            self._release_blocks()
        oldest_row, end = self._oldest - self._base, self._handed - self._base
        waiting = ~self._has[oldest_row:end]
        self._oldest += int(np.argmax(waiting)) if waiting.any() else end - oldest_row
        timeout = None
        for handle in handles[(rows[:, 2] & 0xFF) != self._done].tolist():
            event = self.event(handle)
            if event.kind == "rejected":
                self._rejected[handle] = event.auxiliary
                if self.group(handle) is not None:
                    self._unsequence(handle)
            elif (
                event.kind == "timeout"
                and self._first[handle - self._base] < self._cluster_base
                and not self._gave_up(event)
            ):
                timeout = event  # a thread's late command does not end the run
        if timeout is not None:
            self._fail_timeout(timeout)

    def _own_part(self, handle, group):
        """The cores command `handle` (its flat `group`) went to on this island, as
        _local; the stop's EXIT went to every core of the island, its link cores too."""
        if handle != self._exit_handle:
            return self._local(group)
        base = self.island_cores[0]
        return base, min(self.island_span, self._workers - base)

    def _local(self, group):
        """The part of a flat group on this island, (first core, count) in its own
        cores; count 0 when it has none (one island: the group)."""
        low, high = self._part(group[0], group[1])
        return int(low), int(high - low)

    def _gave_up(self, event):
        """Whether the dispatcher gave the timed-out command up, the run going on (its
        count of give-ups passed the one seen); keeps the inspect of the deadline."""
        if not hasattr(self.backend, "inspect"):
            return False
        group = self.group(event.handle)
        known = group and group[0] is not None
        count = self._own_part(event.handle, group)[1] if known else 0
        read = self.backend.inspect(count)
        tag = getattr(getattr(self.backend, "port", None), "tag", None)
        if read["given_up"] == self._given_up_seen:
            block = device_state.inspect(read["block"], self.h, tag)
            failed = block["reset_failed"] if block["kind"] == "timeout" else 0
            self._reset_failed = (failed & 0xFFFF) - 1, failed >> 16
            self._timeout_cores = block["cores"] if block["kind"] == "timeout" else []
            return False
        self._given_up_seen += 1
        self._settling()  # a core may still settle it: no runtime command yet
        block = device_state.inspect(read["block"], self.h, tag)
        ours = (
            block["complete"]
            and block["kind"] == "timeout"
            and block["request"] == event.handle & M32
        )
        cores = (block["cores"] if ours else []) + self._far_given_up(event.handle)
        self._given_up[event.handle] = (event, cores)
        error = self._given_up_error(event.handle)
        if error.recovered:
            self.recoveries.append(
                {"handle": event.handle, "cores": error.recovered, "text": str(error)}
            )
        return True

    def _far_given_up(self, handle):
        """An island run's command given up: the cores of its parts on the other
        islands, as their dispatchers read them (IslandRun._far_cores); [] otherwise."""
        far, group = self.far_cores, self.group(handle)
        exits = handle == self._exit_handle  # every core of this island alone
        if far is None or not group or group[0] is None or exits:
            return []
        return far(group)

    def _given_up_error(self, handle):
        """The CommandTimeout of a command the dispatcher gave up."""
        event, cores = self._given_up[handle]
        error = CommandTimeout(
            handle,
            event.auxiliary,
            self.sequence(handle),
            self.group(handle),
            cores,
            given_up=True,
            recovered=[
                core["core"]
                for core in cores
                if core.get("reset") and not core.get("far")
            ],
            links=self.links_named(handle),
        )
        self._name_lines(error)
        off = unguarded(self.programs)
        if off:
            error.args = (
                f"{error.args[0]}; guard=False ({', '.join(off)}): no core was asked "
                "to leave the command, a core may still be held in it, and the next "
                "command to it may not run: a card reset if it does not",
                *error.args[1:],
            )
        return error

    def _name_lines(self, error):
        """A program written as a class names, in the error's text, the handler and
        the Python line each not-reported core waited in."""
        lines = [
            line
            for program in self.programs
            if getattr(program, "built", None) is not None
            for line in program.built.lines(error.cores)
        ]
        if lines:
            error.args = (
                f"{error.args[0]}; in Python: {'; '.join(lines)}",
                *error.args[1:],
            )

    def _unsequence(self, handle):
        """A rejected command took no sequence number: later commands move down
        as many."""
        i = handle - self._base
        later = self._sequence[i + 1 : self._handed - self._base]
        after = later[later > 0]
        took = (int(after[0]) if after.size else self._next_sequence) - int(
            self._sequence[i]
        )
        self._sequence[i] = 0
        later[later > 0] -= took
        self._next_sequence -= took

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
        self._wait_kept(wanted, timeout)
        events = [self.event(handle) for handle in wanted.tolist()]
        self.forget(wanted)
        return events[0] if single else events

    def _wait_kept(self, handles, timeout):
        """wait() of `handles` (a list) that keeps their events (event() reads them,
        forget() releases them): an island run waits for every island's parts so,
        and a later island's WaitTimeout leaves the earlier ones to be waited again."""
        wanted = np.asarray(list(handles), np.int64)
        if not wanted.size:
            return
        if wanted.max() >= self._next or wanted.min() < 1:
            raise ValueError("a handle that was never submitted")
        self._hand()
        gone = wanted[
            (wanted < self._base) | self._gone[np.maximum(wanted - self._base, 0)]
        ]
        if gone.size:
            raise ValueError(f"handle {gone[0]}: its event was returned or forgotten")
        self._wait(np.unique(wanted), timeout)

    def store(self):
        """Stores what is queued on the card now, without waiting for an event: a
        command reaches the card only here or in a wait. Runs that talk to each
        other (islands joined by a link) are all started before any is waited on."""
        self._hand()
        rows = self._store_handed()
        if len(rows):
            self._take(rows)

    def _store_handed(self):
        """Stores the commands already handed to the port (one native round, no wait);
        returns the events read meanwhile, not yet filed."""
        if self.port.queued and self._handed > self._base:
            return self.port.wait(_NO_HANDLES, self._store_upto(), 0.0)
        return ()

    def _store_upto(self):
        """The number of the last command handed: what _store_handed stores up to."""
        return (self._handed - 1) & M32

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
        started = time.perf_counter()
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
            self.wait_s += time.perf_counter() - started
        handle = int(wanted[self._arrived(wanted)][0])
        event = self.event(handle)
        error = self._given_up_error(handle) if handle in self._given_up else None
        self.forget([handle])
        if event.kind == "rejected":
            raise CommandRejected(handle, event.auxiliary, self._rejections)
        if error is not None:
            raise error
        return event

    def forget(self, handles):
        """Releases what the run keeps for handles whose events arrived (the event,
        and the group and sequence unless the run is traced): for callers that read
        events with event() after drain() rather than from wait()."""
        rows = np.asarray(
            [handles] if isinstance(handles, int) else list(handles), np.int64
        )
        rows = rows[(rows >= self._base) & (rows < self._handed)] - self._base
        rows = rows[self._has[rows]]
        self._gone[rows] = True
        for handle in (rows + self._base).tolist():
            self._rejected.pop(handle, None)
            self._given_up.pop(handle, None)
        self._compact()

    def _compact(self):
        """Drops the leading rows of released handles once there are many."""
        if self._keep:
            return
        count = min(self._oldest, self._handed) - self._base
        done = (
            int(np.argmin(self._gone[:count]))
            if not self._gone[:count].all()
            else count
        )
        if done < COMPACT_ROWS or 2 * done < len(self._has):
            return
        if self._ref is not None:
            self._ref = self._count_after(done, self._ref)
        for name in COLUMNS:
            setattr(self, name, getattr(self, name)[done:].copy())
        self._base += done

    def _count_after(self, rows, ref):
        """`ref` moved past the first `rows` rows by their events' sequence bits, each
        against the one before; None once a gap is too long to tell."""
        numbered = np.arange(rows)
        status = self._status[:rows].astype(np.int64)
        sequenced = (self._sequence[:rows] > 0) & self._has[:rows]
        numbered = numbered[sequenced & np.isin(status & 0xFF, self._numbered)]
        if not numbered.size:
            return ref
        host = np.concatenate([[ref[1]], self._sequence[numbered]])
        low = np.concatenate(
            [
                [ref[0] % SEQUENCE_LOW],
                status[numbered] >> (8 + self._sequence_shift) & (SEQUENCE_LOW - 1),
            ]
        )
        gap = (np.diff(low) - np.diff(host)) % SEQUENCE_LOW
        gap = np.where(gap >= SEQUENCE_LOW - SEQUENCE_BACK, gap - SEQUENCE_LOW, gap)
        if (gap >= SEQUENCE_LOW // 2).any():
            return None
        return (ref[0] + int((np.diff(host) + gap).sum()), int(host[-1]))

    def _wait(self, wanted, timeout, given_up=True):
        """Waits for the events of `wanted` (sorted handles); raises for a rejected
        command, and (`given_up`) for one the dispatcher gave up."""
        started = time.perf_counter()
        try:
            self._hand()
            deadline = time.monotonic() + timeout
            pending = wanted[~self._arrived(wanted)]
            stepped = False  # one native round at least, also with no time left
            while pending.size:
                self._check_alive()
                left = deadline - time.monotonic()
                if left <= 0 and stepped:
                    raise WaitTimeout(
                        f"{pending.size} of {wanted.size} handles not done in "
                        f"{timeout} s (first {int(pending[0])}; "
                        f"{self.port.queued} not stored); {self._kernel_report()}",
                        pending.tolist(),
                    )
                self._step(pending, int(pending[-1]), max(left, 0.0))
                stepped = True
                pending = pending[~self._arrived(pending)]
            if self._rejected:
                for handle in sorted(self._rejected):
                    if np.isin(handle, wanted):
                        raise CommandRejected(
                            handle, self._rejected[handle], self._rejections
                        )
            for handle in sorted(self._given_up) if given_up else ():
                if np.isin(handle, wanted):
                    raise self._given_up_error(handle)
        finally:
            self.wait_s += time.perf_counter() - started

    def _kernel_report(self):
        """What the backend tells of the kernel after a wait ran out (never raises)."""
        try:
            return self.backend.kernel_report()
        except Exception as error:  # noqa: BLE001 (a failure path: reported)
            return f"the kernel could not be probed ({type(error).__name__}: {error})"

    def _step(self, pending, upto, left):
        """One native round: store up to handle `upto`, read events until those of
        `pending` arrived (or a slice ends). Ends the run if the kernel finished."""
        started = time.perf_counter()
        rows = self.port.wait(
            (pending & M32).astype(np.uint32), upto & M32, min(left, WAIT_SLICE_S)
        )
        self.native_s += time.perf_counter() - started
        if len(rows):
            self._take(rows)
        elif self.port.host_word(0) >= self.h["KMD_STATE_DONE"]:
            self._fail_kernel("the dispatcher finished on its own")

    def _step_any(self, wanted, left):
        """Stores one batch up to the last handle, then reads events until any arrives
        (or a slice ends): the native loop returns on the first event."""
        started = time.perf_counter()
        rows = self.port.wait(np.zeros(0, np.uint32), int(wanted[-1]) & M32, 0.0)
        if not len(rows):
            rows = self.port.wait_events(min(left, WAIT_SLICE_S))
        self.native_s += time.perf_counter() - started
        if len(rows):
            self._take(rows)
        elif self.port.host_word(0) >= self.h["KMD_STATE_DONE"]:
            self._fail_kernel("the dispatcher finished on its own")

    def drain(self, timeout=10.0):
        """Sends everything queued and waits for every command's event (rejections and
        commands given up do not raise here); returns how many events it waited for.
        `drain_given_up`: how many were given up (run.given_up(h) names each)."""
        self._hand()
        oldest_row, end = self._oldest - self._base, self._handed - self._base
        open_ = (self._count[oldest_row:end] > 0) & ~self._has[oldest_row:end]
        wanted = np.flatnonzero(open_) + self._oldest
        self.drain_given_up = 0
        if wanted.size:
            with contextlib.suppress(CommandRejected):
                self._wait(wanted, timeout, given_up=False)
            if self._given_up:  # each given up looked up in the sorted handles
                given = np.fromiter(self._given_up, np.int64, len(self._given_up))
                at = np.minimum(np.searchsorted(wanted, given), wanted.size - 1)
                self.drain_given_up = int(np.count_nonzero(wanted[at] == given))
        return int(wanted.size)

    def thread_result(self, timeout=10.0):
        """Waits (with a timeout) for the program's X280 thread, stops it and returns
        its result block (komorebi.program.thread.ThreadResult); ThreadFailed, as
        thread_results raises it, unless it returned done. Events go on as before."""
        if self.thread is None:
            raise KomorebiError("the program has no thread")
        return thread_checked(self.thread.wait(timeout), self.thread.thread.cluster)

    def thread_results(self, timeout=10.0):
        """Waits (with a timeout) for every X280 thread, stops each and returns
        {cluster: ThreadResult}. One that fails (no return in time, a library error) has
        the others asked to stop at once, and ThreadFailed carries every result."""
        return thread_results(self._thread_runs(), timeout)

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
        wanted = self._flags[i] & self._flag_result and kind == "done"
        auxiliary = status >> 8
        if status & 0xFF in self._numbered:
            auxiliary &= self._auxiliary_low
        result = ticks if wanted else None
        return Event(handle, kind, auxiliary, ticks, result, self._tick_mhz)

    def sequence(self, handle):
        """The sequence number the kernel gave a command (the trace's index), from its
        done or timeout event; None before that event or for a rejected command."""
        self._hand()
        i = handle - self._base
        if not 0 <= i < self._handed - self._base or self._sequence[i] == 0:
            return None
        if self._gone[i] and not self._keep:
            return None
        count = self._kernel_count(i)
        return None if count is None else _sequence_number(count)

    def _kernel_count(self, i, ref=None):
        """Row i's place in the kernel's count of commands (every source), from the low
        16 bits in its event and the nearest earlier known row (or `ref`); None
        before the event. Raises if other sources submitted too many in between."""
        if self._kseq[i]:
            return int(self._kseq[i])
        status = int(self._status[i])
        if (
            self._sequence[i] == 0
            or not self._has[i]
            or status & 0xFF not in self._numbered
        ):
            return None
        if ref is None:
            known = np.flatnonzero(self._kseq[:i])
            j = int(known[-1]) if known.size else None
            ref = (
                self._ref if j is None else (int(self._kseq[j]), int(self._sequence[j]))
            )
        if ref is None:
            raise KomorebiError(
                f"handle {i + self._base}: its sequence number is not known (an "
                f"earlier gap of other sources' commands was too long to tell)"
            )
        near = (
            ref[0] + int(self._sequence[i]) - ref[1]
        )  # as if no other source submitted
        low = status >> (8 + self._sequence_shift) & (SEQUENCE_LOW - 1)
        gap = (low - _sequence_number(near)) % SEQUENCE_LOW
        gap -= SEQUENCE_LOW if gap >= SEQUENCE_LOW - SEQUENCE_BACK else 0
        if gap >= SEQUENCE_LOW // 2:
            raise KomorebiError(
                f"handle {i + self._base}: {gap} or more commands of other sources "
                f"since the host's previous command; its sequence number is not known "
                f"(an event carries its low 16 bits)"
            )
        count = near + gap
        count += next(
            adjust
            for adjust in (0, 1, -1)
            if _sequence_number(count + adjust) % SEQUENCE_LOW == low
        )
        self._kseq[i] = count
        return count

    def group(self, handle):
        self._hand()
        i = handle - self._base
        if not 0 <= i < self._handed - self._base or self._count[i] == 0:
            return None
        if self._gone[i] and not self._keep:
            return None
        first = int(self._first[i])
        if first < 0 and self._has[i] and self._status[i] & 0xFF == self._done:
            first = (
                int(self._status[i]) >> 8 & self._auxiliary_low
            )  # the queue's choice
        return (first if first >= 0 else None, int(self._count[i]))

    @property
    def stored(self):
        """Commands stored into the card so far."""
        return self._handed - self._base - self.port.queued

    def dispatcher_given_up(self):
        """The dispatcher's count of the host's commands it gave up, read now from its
        memory through the runtime (no command)."""
        return int(self.backend.inspect(0)["given_up"])

    @property
    def blocks_held(self):
        """Command blocks the run still holds: each is freed at its command's event."""
        return len(self._kept)
