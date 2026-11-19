"""The run's stop: the cores' end read back, the failures it names, the state words
and trace after it (StopMixin of Run)."""

import time

import numpy as np

# np.unique imports numpy.ma on its first call (about 8 ms): here, not in a first wait
import numpy.ma

from komorebi.errors import (
    CommandTimeout,
    KernelFailed,
    KomorebiError,
    MessagesGaveUp,
    StartError,
    WaitTimeout,
    describe_core,
)
from komorebi.program import overlay
from komorebi.session import messages_gave
from komorebi.session import state as device_state
from komorebi.session.parts.run_events import M32, _sequence_number

# KMD_RESET_*, from 1
RESET_STATUS = ("BAD_LIST", "NO_WINDOW", "NO_FIRMWARE", "NO_LIST", "HELD")


def unit_not_ready(core, status):
    """A Wormhole core whose compute unit is not ready, as text: `status` its
    INSTRN_BUF_STATUS (0x77 when ready; bit t clear: thread t's instruction FIFO full).
    """
    full = [str(thread) for thread in range(3) if not status >> thread & 1]
    fifos = (
        f": the instruction FIFOs of threads {', '.join(full)} are full" if full else ""
    )
    return (
        f"core {core}'s compute unit is not ready (status {status:#x}{fifos}); on "
        "Wormhole no reset by software clears this; the card needs a reset"
    )


def _left(deadline):
    """Seconds left until the monotonic time `deadline` (0 once it passed)."""
    return max(deadline - time.monotonic(), 0.0)


class StopMixin:
    """Run's stop and what it reads back (komorebi.session.run.Run)."""

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
        """Cancels engines, drains, EXIT to every core, stops the kernel, closes the
        X280 runtime, reads back state and trace: its waits within `timeout`, the
        runtime's cleanups outside it (up to 5 s each, at most 25 s: host.md). On any
        error it still closes (abort) and raises."""
        if self._stopped:
            return self._result
        deadline = time.monotonic() + timeout
        try:
            for thread_run in self._thread_runs().values():
                thread_run.stop(
                    _left(deadline)
                )  # each thread's stop is inside the deadline
            if self._engine_state is not None:
                self._engine_state.stop(
                    _left(deadline)
                )  # EXIT waits for running engines
            self.drain(_left(deadline))
            exit_handle = self._queue_command(
                self.h["KM_HANDLER_EXIT"], (0, self.cores)
            )
            self.wait(exit_handle, _left(deadline))
            stop_handle = self._queue_command(self.h["KM_HANDLER_STOP"], None)
            self._send(stop_handle, _left(deadline))
        except KernelFailed:
            raise
        except BaseException:
            self.abort()
            raise
        result, stop_failed = self._finish_kept(_left(deadline))
        if stop_failed is not None:  # its own error, the block told in it
            stop_failed.args = (
                f"{stop_failed.args[0]}; {self._told(result)}",
                *stop_failed.args[1:],
            )
            self._log(stop_failed.args[0])
            raise stop_failed
        wrong = self._finish_problems(result)
        if wrong:
            raise self._remember(
                KernelFailed(f"the stop did not end cleanly: {wrong}", result=result)
            )
        return result

    def _finish_problems(self, result):
        """What is wrong with a finish, as text ("" when the result is ready, the tile
        restored, the state DONE and the error 0)."""
        wrong = []
        named = result["error"] == self.h["KMD_ERROR_ENDED"] and self._ended_named()
        if named:  # cores a give-up already named ended there: no reset on a split card
            result["ended"] = self._ended_text(result)
        if not result.get("result_ready", True):
            wrong.append("the kernel's result never became ready")
        if not result.get("cleaned", True):
            wrong.append(
                "the X280 runtime could not restore cluster 0 (reset the card)"
            )
        if self._close_failed():
            wrong.append(self._close_failed())
        if result["state"] != self.h["KMD_STATE_DONE"] and not named:
            wrong.append(f"state {result['state_name']}")
        if result["error"] == self.h["KMD_ERROR_ENDED"]:
            wrong += [] if named else [self._ended_text(result)]
        elif result["error"] != 0:
            wrong.append(f"error {result['error_name']} in {result['where_name']}")
        if result.get("failed"):
            wrong.append(result["failed"])
        return "; ".join(wrong)

    def _close_failed(self):
        """What a finish's close of the X280 runtime could not restore, as text ("" when
        it restored every cluster): every error a failed finish raises names it."""
        failed = getattr(self.backend, "close_failed", "")
        return (
            "the X280 runtime could not restore its clusters at the close; the card "
            f"may need a reset: {failed}"
            if failed
            else ""
        )

    def _ended_cores(self):
        """This run's cores that ended by their own give-up."""
        out = []
        for core in range(self.cores):
            report = device_state.core_state(self._state, core, self.h)
            if self.h["kEndGaveUp"] in (report["reader_end"], report["writer_end"]):
                out.append(core)
        return out

    def _ended_named(self):
        """Whether every core that ended by its own give-up was named by a give-up
        the run already reported (an island's run, IslandRun._named_cores)."""
        ended = self._ended_cores() if self.named_cores is not None else []
        return bool(ended) and set(ended) <= set(self.named_cores())

    def _ended_text(self, result):
        """The stop's EXIT left done because cores had ended by their own give-up
        (KMD_ERROR_ENDED): which cores, and why each ended."""
        gave = self.h["kEndGaveUp"]
        named = []
        for core in range(self.cores):
            report = device_state.core_state(self._state, core, self.h)
            if gave in (report["reader_end"], report["writer_end"]):
                named.append(
                    describe_core(
                        {
                            "core": core,
                            **{r: report[r] for r in ("reader", "writer", "compute")},
                        }
                    )
                )
        return (
            f"the stop's EXIT was left done: {result['seen']} core(s) had ended by "
            f"their own give-up and never report: {'; '.join(named) or 'none named'}"
        )

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
        """Stops the grid, then the kernel through the runtime, reads back what the
        cores left and gives back its L1 tensors as a stop does (kept if the runtime
        could not be closed); for failure paths. result() then holds the kernel's
        result block with `abandoned`, the commands left without their DONE event."""
        if self._stopped:
            return
        for thread_run in self._thread_runs().values():
            thread_run.running = False  # the runtime's close stops it
        # if the close raises, the run stays loaded and a later exit path
        # (Session.close, the exit hooks) closes it again
        cleaned = self.backend.abort()
        self._stopped = True
        left = self.backend.aborted
        if left is not None:
            x280_cores = [int(word) for word in left["x280_cores"]]
            self._result = device_state.result_block(left["words"], self.h) | {
                "x280_cores": x280_cores,
                "abandoned": x280_cores[self.h["KMD_H_ABANDONED"]],
            }
        self._read_back(stop_grid=True)
        if not cleaned:
            result = self._result
            kernel = (
                f"; the kernel left state {result['state_name']}, "
                f"{result['abandoned']} commands abandoned"
                if result is not None
                else ""
            )
            raise self._remember(
                KernelFailed(
                    "the X280 runtime could not restore cluster 0 after the stop; "
                    f"the card may need a reset: {self.backend.close_failed}{kernel}",
                    result=result,
                )
            )
        release = getattr(self.backend, "release_l1", None)
        if release is not None:  # as _finish: the next load of the session has its L1
            release()

    def _fail_start(self, state, timeout=1.0):
        """The kernel did not report READY: finishes it, reads back, raises StartError
        with its result block."""
        # stopped after this, unless the runtime could not be closed: a later exit
        # path (Session.close, the exit hooks) then closes it
        result, stop_failed = self._finish_kept(timeout, stop_grid=True)
        said = (
            f"): {self._told(result)}"  # the cores not read: the block only
            if stop_failed is not None
            else f", core result {result.get('result')}): "
            f"{self._finish_problems(result)}"
        )
        text = f"the dispatcher did not report READY (state {state}{said}"
        if result is not None and result["where"] == self.h["KMD_WHERE_UNIT"]:
            seen = result["seen"]  # the core, its status above
            text = f"{unit_not_ready(seen & 0xFFFF, seen >> 16)}; {text}"
        error = StartError(text, result=result)
        self._raise_after_stop(error, stop_failed)

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
        self._result = device_state.result_block(fin["words"], self.h)
        self._result.update(
            {key: value for key, value in fin.items() if key != "words"}
        )
        self._read_back(stop_grid)
        release = getattr(self.backend, "release_l1", None)
        if release is not None:  # the next load of the session has its L1
            release()

    def _read_back(self, stop_grid=False):
        blocks, trace_rows = self.backend.read_back(stop_grid=stop_grid)
        for thread_run in self._thread_runs().values():
            thread_run.grid_stopped()
        self._blocks = blocks
        self._state = device_state.state_words(blocks, 0, self.h)
        self._epochs = getattr(self.backend, "overlay_epochs", None)
        if trace_rows is not None:
            seqs, groups = {}, {}
            ref = self._ref
            for i in range(self._handed - self._base):  # in order: one pass
                try:
                    count = self._kernel_count(i, ref)
                except KomorebiError:  # a gap too long to tell: not in the trace
                    continue
                if count is None:
                    continue
                ref = (count, int(self._sequence[i]))
                handle = i + self._base
                group = self.group(handle)
                if group is not None and group[0] is not None:
                    # the trace's rows are the card's cores: the group's first row
                    first = int(group[0]) - self.island_first + self.island_cores[0]
                    row = (first, int(group[1]))
                    seqs[handle], groups[handle] = _sequence_number(count), row
            self._trace = device_state.Trace(
                trace_rows, seqs, groups, self.backend.mhz, self.h, self.own_commands
            )

    def _remember(self, error):
        self._failure = error
        return error

    def _fail_timeout(self, event):
        seen = self._timeout_seen()  # before the stop: a grid that does not return
        result, stop_failed = self._finish_kept(10.0)
        group = self.group(event.handle)
        sequence = (
            result["sequence"]
            if result is not None and result["error"] == self.h["KMD_ERROR_DEADLINE"]
            else None
        )
        if sequence is None:
            sequence = self.sequence(event.handle)
        known = group and group[0] is not None and self._state is not None
        cores = (
            device_state.not_reported(
                self._state, self._local(group), event.auxiliary, sequence, self.h
            )
            if known
            else []
        )
        gave = (  # the cores' words, unless the stop failed before it read them
            messages_gave.named(result, self._blocks, self.h, self._state)
            if result is not None and self._blocks is not None
            else None
        )
        error = (
            MessagesGaveUp(
                event.handle, event.auxiliary, sequence, group, cores, result, gave=gave
            )
            if gave is not None
            else CommandTimeout(
                event.handle, event.auxiliary, sequence, group, cores, result
            )
        )
        self._name_lines(error)
        if seen:
            error.args = (f"{error.args[0]}; {seen}", *error.args[1:])
        if self._close_failed():
            error.args = (f"{error.args[0]}; {self._close_failed()}", *error.args[1:])
        if stop_failed is not None:  # the block told; the cores, if they were not read
            unread = "the cores' words not read; " if self._state is None else ""
            told = f"{unread}{self._told(result)}"
            error.args = (f"{error.args[0]}; {told}", *error.args[1:])
        self._raise_after_stop(error, stop_failed)

    def _timeout_seen(self):
        """What the deadline's inspect read, as text: the cores where a kernel waited
        or gave up, and the dispatcher's failed reset of a core that did not leave
        ("" when it read nothing)."""
        named = [describe_core(core) for core in self._timeout_cores]
        waited = [text for text in named if not text.endswith("no give-up recorded")]
        parts = [f"at the deadline: {'; '.join(waited)}"] if waited else []
        if len(waited) < len(named):
            parts.append(f"{len(named) - len(waited)} cores read at it named no wait")
        core, status = self._reset_failed
        if core >= 0:  # a give-up's reset that failed left the command in its core
            names = {self.h[f"KMD_RESET_{name}"]: name for name in RESET_STATUS}
            why = f"the dispatcher's reset of core {core} failed"
            why += f" ({names.get(status, status)})"
            if status & 0xFF == self.h["KMD_RESET_UNIT"]:  # its status above
                why = f"after its reset {unit_not_ready(core, status >> 8)}"
            if status == self.h["KMD_RESET_HELD"]:
                why = (
                    f"core {core} was not reset: other commands in flight are in "
                    "its ring, which a reset would clear"
                )
            parts.append(why)
        return "; ".join(parts)

    def _fail_kernel(self, why):
        result, stop_failed = self._finish_kept(10.0)
        self.backend.log_state()
        told = self._told(result, "")
        parts = [f"{why}: {told}", self._timeout_seen(), self._close_failed()]
        text = "; ".join(part for part in parts if part)
        self._raise_after_stop(KernelFailed(text, result=result), stop_failed, log=True)

    def _finish_kept(self, timeout, stop_grid=False):
        """_finish: (the result block, None), or when it raised (what it read or the
        abort kept, _failed_result, and its error): every caller raises with it."""
        try:
            self._finish(timeout, stop_grid)
        except Exception as error:  # noqa: BLE001 (raised by the caller, chained)
            return self._failed_result(), error
        return self._result, None

    def _told(self, result, label="the dispatcher's result block: "):
        """A result block as text: its error, where, slot, sequence, command (a trap's
        core and registers) and state; that it was not readable for None."""
        if result is None:
            return "the dispatcher's result block not readable"
        where = (
            f"exception on X280 core {result['req']} (mcause {result['slot']:#x}, "
            f"mepc {result['sequence']:#x}, mtval {result['seen']:#x})"
            if result["error"] == self.h["KMD_ERROR_TRAP"]
            else f"slot {result['slot']}, sequence {result['sequence']}, "
            f"command {result['req']}"
        )
        return (
            f"{label}error {result['error_name']} in {result['where_name']}, {where} "
            f"(state {result['state_name']})"
        )

    def _log(self, text):
        card = getattr(self.backend, "card", None)  # the session's log, through it
        if card is not None:
            card.log(text)

    def _raise_after_stop(self, error, stop_failed, log=False):
        """Raises `error` (remembered) from the stop's error, which its text ends with;
        logged when the stop failed (or `log`)."""
        if stop_failed is not None:
            name = type(stop_failed).__name__
            error.args = (
                f"{error.args[0]}; then {name}: {stop_failed}",
                *error.args[1:],
            )
        if log or stop_failed is not None:
            self._log(error.args[0])
        raise self._remember(error) from stop_failed

    def _failed_result(self):
        """The result block _fail_kernel tells: _finish's, else what the abort of a
        _finish that raised kept (backend.aborted); None if neither was read. Reads
        nothing from the card: a dead card cannot hold the first error back."""
        if self._result is not None:
            return self._result
        left = getattr(self.backend, "aborted", None)
        if left is None:
            return None
        return device_state.result_block(left["words"], self.h)

    # After the stop ---------------------------------------------------------------

    def result(self):
        """The dispatcher's result block (decoded); valid after the stop."""
        self._require_stopped("result")
        return self._result

    def state(self):
        """Every core's state words, port slots, broadcast areas, dynamic-run words
        and overlay epoch, as dicts (the program's cores: a Tensix dispatcher's left
        out); valid after the stop."""
        self._require_stopped("state")
        epochs = getattr(self, "_epochs", None)
        mask = overlay.constants()["kOverlayEpochMask"]
        return [
            device_state.core_state(self._state, core, self.h)
            | {
                "ports": device_state.ports(self._blocks[core], self.h),
                "broadcasts": device_state.broadcasts(self._blocks[core], self.h),
                "dynamic": device_state.dynamic(self._blocks[core], self.h),
                "overlay_epoch": None if epochs is None else int(epochs[core]) & mask,
            }
            for core in range(getattr(self, "_workers", self.cores))
        ]

    def state_words(self):
        self._require_stopped("state_words")
        return self._state

    def trace(self):
        """The per-command trace (a state.Trace), or None if the run was not traced."""
        self._require_stopped("trace")
        return self._trace

    def _require_stopped(self, call):
        if not self._stopped:
            raise KomorebiError(
                f"run.{call}() is valid only after run.stop(): this run is still "
                f"loaded; call run.stop(), then run.{call}()"
            )
        if self._state is None:
            raise KomorebiError(
                f"run.{call}() is valid only after run.stop() reads the cores: this "
                "run's stop ended before it read them (its error says why)"
            )
