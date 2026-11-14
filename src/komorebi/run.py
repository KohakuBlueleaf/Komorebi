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
from komorebi import state as device_state
from komorebi.engines import control, graph_hostfed, stream_hostfed
from komorebi.engines import stream as stream_module
from komorebi.errors import (
    CommandRejected,
    CommandTimeout,
    KernelFailed,
    KomorebiError,
    ProgramError,
    StartError,
    ThreadFailed,
    WaitTimeout,
)
from komorebi.program import Handler

M32 = abi.M32
WAIT_SLICE_S = 0.05  # longest single native wait, so Python stays responsive
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


class Event(NamedTuple):
    """A finished command: its handle, kind (done, timeout, pong, rejected), the
    auxiliary (done: the group's first core; timeout: the slot; rejected: the reason),
    submit-to-completion time in X280 timer ticks, and for a command submitted with
    `result=True` the leader's result word (sent as the ticks)."""

    handle: int
    kind: str
    auxiliary: int
    ticks: int
    result: int | None = None

    @property
    def microseconds(self):
        return self.ticks / 50.0  # the X280 timer runs at 50 MHz


def stuck_message(cluster, result):
    """What a stuck thread (ThreadResult.stuck) means, for ThreadFailed."""
    stop = (
        f"the runtime could not stop the cluster ({result.detail}): the card needs a"
        " reset"
        if result.detail
        else "the runtime stopped the cluster"
    )
    return (
        f"the thread on cluster {cluster} did not return, nor after the stop flag: a"
        " core is stuck outside the library (which alone sees the stop flag and the"
        " deadline) or stopped by a trap; its last progress marker"
        f" {result.where}; {stop}"
    )


def _left(deadline):
    """Seconds left until the monotonic time `deadline` (0 once it passed)."""
    return max(deadline - time.monotonic(), 0.0)


def _sequence_number(number):
    """The kernel's sequence number of the n-th command: it counts in 32 bits and skips
    0 (it also skips a slot's last value, which only a wrap can repeat)."""
    return (number - 1) % M32 + 1


class Run:
    """Submits commands to a loaded program and collects their events.

    backend: `port` (enqueue, queued, wait, host_word, batch; its own loop), `finish`,
    `abort()`, `read_back()` -> (control blocks, trace), `mhz`.
    """

    def __init__(
        self,
        backend,
        grid,
        program=None,
        batch=64,
        abi_headers=None,
        programs=None,
        holds=None,
        thread_program=None,
        switch=None,
    ):
        self.h = abi_headers or abi.headers()
        self.backend = backend
        self.port = backend.port
        self.port.batch = batch
        self._keep = bool(getattr(backend, "trace_cap", 0))  # the trace needs groups
        self.grid = grid
        self.cores = grid[0] * grid[1]
        self.program = program
        # the run's programs by program field; holds[field]: the cores holding it
        self.programs = list(programs or ([program] if program is not None else []))
        self._holds = holds
        self.thread_program = thread_program or program  # owns the X280 threads
        self._switch = switch  # {"records": {field: DRAM address}, "refused": why}
        self.check_programs = True  # False: the dispatcher alone refuses (reason 13)
        self.batch = batch
        constants = self.h.constants
        self._kinds = {
            constants["KM_EVENT_DONE"]: "done",
            constants["KM_EVENT_TIMEOUT"]: "timeout",
            constants["KM_EVENT_PONG"]: "pong",
            constants.get("KM_EVENT_REJECTED", 3): "rejected",
        }
        self._done = constants["KM_EVENT_DONE"]
        self._numbered = (
            constants["KM_EVENT_DONE"],
            constants["KM_EVENT_TIMEOUT"],
        )  # auxiliary holds sequence bits
        self._auxiliary_low, self._sequence_shift = (
            constants["KM_EVENT_AUXILIARY_LOW"],
            constants["KM_EVENT_SEQUENCE_SHIFT"],
        )
        self._flag_result, self._flag_queue = (
            constants["KM_FLAG_RESULT"],
            constants["KM_FLAG_QUEUE"],
        )
        self._flag_exclusive = constants["KM_FLAG_EXCLUSIVE"]
        # a group from this first core on names a work cluster's thread
        self._cluster_base = constants["KM_GROUP_CLUSTER_BASE"]
        self._work_clusters = constants["KMD_WORK_CLUSTERS"]
        self._rejections = {
            value: name[len("KMD_REJECT_") :].lower()
            for name, value in constants.items()
            if name.startswith("KMD_REJECT_")
        }
        self._nop, self._user_max = (
            self.h["KM_HANDLER_NOP"],
            self.h["KM_HANDLER_USER_MAX"],
        )
        self._hdr, self._group_at, self._arg_at = self.h.positions
        self._next = 1  # the next handle; command number = handle mod 2**32
        self._next_sequence = 1  # the kernel's sequence of the next command
        self._new_handles(1)
        self._kept = {}  # handle -> the DRAM tensor of its block, until its event
        self._unfenced = False  # a block written since the last fence
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
        self._sequence = np.zeros(
            0, np.int64
        )  # host commands up to this one; 0: no command
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
        self,
        handler,
        group=None,
        args=(),
        exclusive=False,
        result=False,
        queued=False,
        cluster=None,
        block=None,
    ):
        """Queues a command for handler `handler` on cores [first, first + count)
        (default every core), or on work cluster `cluster`'s X280 thread; returns its
        handle. `result`: the event carries the result word; `queued`: the kernel picks
        the cores; `block`: bytes or an array its handlers read (a7, a8)."""
        started = time.perf_counter()
        program_field = 0
        if cluster is not None:
            handler, group = self._to_thread(handler, cluster, group, exclusive, queued)
        else:
            handler, program_field = self._handler(handler)
        if not (handler == self._nop or 1 <= handler <= self._user_max):
            raise ValueError(
                f"handler {handler}: not NOP and not in 1..KM_HANDLER_USER_MAX"
            )
        if cluster is None and queued:
            if exclusive:
                raise ValueError("a queued command cannot be exclusive")
            group = self._group((0, group))
            self._check_holds(program_field, handler, 0, self.cores, "queued")
        elif cluster is None:
            group = self._group(group)
            self._check_holds(program_field, handler, *group)
        if len(args) > len(self._arg_at):
            raise ValueError(f"{len(args)} arguments; a command has 10")
        if any(not 0 <= int(word) <= M32 for word in args):
            raise ValueError("arguments are 32-bit unsigned words")
        kept = None
        if block is not None:
            args, kept = self._block_words(block, args, cluster)
        flags = (self._flag_result if result else 0) | (
            self._flag_queue if queued else 0
        )
        handle = self._queue_command(
            handler, group, args, exclusive, flags, program_field
        )
        if kept is not None:
            self._kept[handle] = kept
        self.submit_s += time.perf_counter() - started
        return handle

    def _block_words(self, block, args, cluster):
        """The arguments with a block's address and bytes in a7 and a8, and the block
        written to DRAM (kept until the command's event)."""
        at = self.h["KM_BLOCK_ADDRESS_WORD"]
        if cluster is not None:
            raise ValueError("a block goes to Tensix cores, not to an X280 thread")
        if len(args) > at:
            raise ValueError(
                f"{len(args)} arguments: a command with a block has {at}, "
                f"a{at} and a{at + 1} carry the block"
            )
        if isinstance(block, bytes | bytearray | memoryview):
            data = np.frombuffer(block, np.uint8)
        else:
            data = np.ascontiguousarray(block).view(np.uint8).reshape(-1)
        if not 1 <= len(data) <= self.h["KM_BLOCK_BYTES"]:
            raise ValueError(
                f"a block of {len(data)} bytes: 1 to {self.h['KM_BLOCK_BYTES']}"
            )
        kept = self.backend.block(data)
        self._unfenced = True
        args = [*args, *[0] * (at - len(args)), kept.address, len(data)]
        return args, kept

    def _release_blocks(self):
        """Frees the blocks of the commands whose event arrived."""
        for handle in list(self._kept):
            if handle < self._base or self._has[handle - self._base]:
                self._kept.pop(handle).free()

    def _handler(self, handler):
        """(number, program field) of `handler`: a Handler names its program; a name
        or a number is a handler of the run's first program."""
        if isinstance(handler, Handler):
            return int(handler), self.program_field(handler.program)
        if handler.__class__ is not int:
            handler = (
                self.program.handler(handler)
                if self.program is not None
                else int(handler)
            )
        return int(handler), 0

    def field_handler(self, number, program_field):
        """What submit takes for handler `number` of the program with field
        `program_field` (a command's header names both)."""
        if program_field == 0:
            return number
        if program_field >= len(self.programs):
            raise ValueError(f"program field {program_field}: no such program here")
        return self.programs[program_field].handler(number)

    def check_templates(self, templates):
        """Refuses a stream's or graph's command template whose program this run does
        not hold on its group (a group from registers: the dispatcher's to refuse)."""
        none = stream_module.TEMPLATE_NONE
        for template in templates:
            program_field, handler = template.rec[1] >> 24, template.rec[1] & 0xFF
            if program_field >= max(len(self.programs), 1):
                raise ValueError(
                    f"handler {handler}: program field {program_field}, and this run "
                    f"holds {len(self.programs)} programs"
                )
            first, count = template.rec[2] & 0xFFFF, template.rec[2] >> 16
            if template.first_register != none or template.count_register != none:
                continue
            if count == 0 or template.rec[1] >> 8 & self._flag_queue:
                self._check_holds(program_field, handler, 0, self.cores, "queued")
            elif first < self._cluster_base:
                self._check_holds(program_field, handler, first, count)

    def program_field(self, program):
        """The program field of `program`'s commands in this run (ValueError if the
        run does not hold it)."""
        for program_field, loaded in enumerate(self.programs):
            if loaded is program:
                return program_field
        raise ValueError(f"{program.name} is not a program of this run")

    def _check_holds(self, program_field, handler, first, count, queued=None):
        """Refuses a command whose group's cores do not hold its program (a queued
        one: every core, since the kernel picks them)."""
        if self._holds is None or handler == self._nop or not self.check_programs:
            return
        if not self._holds[program_field][first : first + count].all():
            name = self.programs[program_field].name
            where = "every core" if queued else f"group ({first}, {count})"
            raise ValueError(
                f"handler {handler} of {name}: {where} does not hold {name}"
            )

    def load(self, program, group):
        """Makes the cores of `group` (first, count) hold `program`, one of the dict
        given to session.load: its overlay and its buffers, set up again. A command,
        after every earlier one on those cores; their program's buffers are lost."""
        if self._switch is None:
            raise ProgramError(
                "run.load needs a run loaded with {program: group, ...}: its groups "
                "can switch program"
            )
        if self._switch["refused"]:
            raise ProgramError(f"run.load: {self._switch['refused']}")
        program_field = self.program_field(program)
        first, count = self._group(group)
        record = self._switch["records"][program_field]
        handle = self._queue_command(
            self.h["KM_HANDLER_LOAD"], (first, count), [record], False, 0, program_field
        )
        for held, cores in enumerate(self._holds or ()):
            cores[first : first + count] = held == program_field
        return handle

    def transfer(self, handler, group, args, timeout):
        """A PUSH or PULL (the session's transfers) on `group` (first, count, program
        field: the field its cores hold), waited for; returns its Event."""
        first, count, program_field = group
        handle = self._queue_command(
            handler, self._group((first, count)), args, False, 0, program_field
        )
        return self.wait(handle, timeout)

    def holds(self):
        """[field] -> the cores that hold each program (bool arrays), or None when every
        core holds the run's one program."""
        return self._holds

    def alive(self):
        """Whether the run takes commands: not stopped, nothing failed."""
        return self._failure is None and not self._stopped

    def submit_many(self, handler, firsts, counts, args=None, exclusive=False):
        """Queues n commands at once: `handler` a name, a number or n numbers, `firsts`
        and `counts` the groups (n each, or one for all), `args` an [n, m] matrix of
        32-bit words (m <= 10) or None. Returns the handles as a range."""
        started = time.perf_counter()
        self._check_alive()
        program_field = 0
        if isinstance(handler, str | Handler):
            handler, program_field = self._handler(handler)
        columns = [np.asarray(handler), np.asarray(firsts), np.asarray(counts)]
        if args is not None:
            args = np.asarray(args)
            if args.ndim != 2 or args.shape[1] > len(self._arg_at):
                raise ValueError("arguments are an [n, m] matrix with m <= 10")
            if args.dtype.kind not in "iu":
                raise ValueError("arguments are 32-bit unsigned words")
            columns.append(np.zeros(args.shape[0], np.int64))  # its row count
        if any(column.dtype.kind not in "iu" for column in columns[:3]):
            raise ValueError("handlers, firsts and counts are integers")
        try:
            shape = np.broadcast_shapes(*(column.shape for column in columns))
        except ValueError:
            shape = ()
        if len(shape) != 1:
            raise ValueError("handlers, firsts, counts and arguments: one row each")
        handlers, first, count = (
            np.broadcast_to(column, shape).astype(np.int64) for column in columns[:3]
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
        if self._holds is not None and self.check_programs:
            missing = np.concatenate([[0], np.cumsum(~self._holds[program_field])])
            bad = (missing[first + count] - missing[first] > 0) & (
                handlers != self._nop
            )
            if bad.any():
                i = int(np.argmax(bad))
                self._check_holds(
                    program_field, int(handlers[i]), int(first[i]), int(count[i])
                )
        words = np.zeros((shape[0], 16), np.uint32)
        words[:, self._hdr] = handlers & 0xFF | program_field << 24
        words[:, self._group_at] = first | count << 16
        if args is not None and args.size:
            if args.min() < 0 or args.max() > M32:
                raise ValueError("arguments are 32-bit unsigned words")
            words[:, self._arg_at[: args.shape[1]]] = args
        self._hand()
        start = self._next
        self._next += shape[0]
        if exclusive:
            words[:, self._hdr] |= self._flag_exclusive << 8
        self._send_block(words, first, count, np.zeros(shape[0], np.uint8))
        self.submit_s += time.perf_counter() - started
        return range(start, self._next)

    def _to_thread(self, handler, cluster, group, exclusive, queued):
        """(handler number, group) of a command to work cluster `cluster`'s thread; a
        name is one of that thread's handlers."""
        if group is not None or queued or exclusive:
            raise ValueError(
                "a command to a thread takes no group, and is not queued or exclusive"
            )
        if cluster.__class__ is not int or not 1 <= cluster <= self._work_clusters:
            raise ValueError(
                f"cluster {cluster!r}: work clusters are 1 to {self._work_clusters}"
            )
        owner = self.thread_program
        thread = None if owner is None else owner.threads.get(cluster)
        if owner is not None and thread is None:
            raise ValueError(f"{owner.name} has no thread on cluster {cluster}")
        if handler.__class__ is not int:
            handler = thread.handler(handler) if thread is not None else int(handler)
        return handler, (self._cluster_base + cluster, 1)

    def _group(self, group):
        if group is None:
            return (0, self.cores)
        first, count = int(group[0]), int(group[1])
        if count < 1 or first < 0 or first + count > self.cores:
            raise ValueError(
                f"group ({first}, {count}) is not inside {self.cores} cores"
            )
        return (first, count)

    def _queue_command(
        self, handler, group, args=(), exclusive=False, flags=0, program_field=0
    ):
        self._check_alive()
        first, count = group if group else (0, 0)
        flags |= self._flag_exclusive if exclusive else 0  # the kernel orders it
        words = [0] * 16
        words[self._hdr] = handler & 0xFF | flags << 8 | program_field << 24
        words[self._group_at] = first | count << 16
        for position, value in zip(self._arg_at, args):
            words[position] = int(value) & M32
        self._words.extend(words)
        self._firsts.append(-1 if flags & self._flag_queue else first)
        self._counts.append(count)
        self._fl.append(flags)
        handle = self._next
        self._next += 1
        return handle

    def _hand(self):
        """Gives the commands submitted one by one to the port, as one block."""
        if not self._counts:
            return
        if self._unfenced:  # one DRAM barrier for every block of these commands
            self.backend.fence()
            self._unfenced = False
        rows = len(self._counts)
        words = np.frombuffer(self._words, np.uint32).reshape(rows, 16)
        first = np.frombuffer(self._firsts, np.int32)
        count = np.frombuffer(self._counts, np.int32)
        flags = np.frombuffer(self._fl, np.uint8)
        self._send_block(words, first, count, flags)
        self._unsent()

    def _send_block(self, words, first, count, flags):
        """Notes handles handed .. handed + n - 1 (groups, flags, sequences) and queues
        their commands in the port. A queued command's first is -1 (the kernel's)."""
        rows = len(words)
        row = self._handed - self._base
        self._grow(row + rows)
        has_cores = count > 0
        self._first[row : row + rows] = first
        self._count[row : row + rows] = count
        self._flags[row : row + rows] = flags
        self._sequence[row : row + rows] = np.where(
            has_cores, self._next_sequence + np.cumsum(has_cores) - 1, 0
        )
        self._next_sequence += int(has_cores.sum())
        self.port.enqueue(np.ascontiguousarray(words))
        self._handed += rows

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
            elif event.kind == "timeout" and self._first[handle - self._base] < (
                self._cluster_base
            ):
                timeout = event  # a thread's late command does not end the run
        if timeout is not None:
            self._fail_timeout(timeout)

    def _unsequence(self, handle):
        """A rejected command took no sequence number: later commands move down one."""
        i = handle - self._base
        self._sequence[i] = 0
        later = self._sequence[i + 1 : self._handed - self._base]
        later[later > 0] -= 1
        self._next_sequence -= 1

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
        events = [self.event(handle) for handle in wanted.tolist()]
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
        self.forget([handle])
        if event.kind == "rejected":
            raise CommandRejected(handle, event.auxiliary, self._rejections)
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

    def _wait(self, wanted, timeout):
        """Waits for the events of `wanted` (sorted handles)."""
        started = time.perf_counter()
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
                for handle in sorted(self._rejected):
                    if np.isin(handle, wanted):
                        raise CommandRejected(
                            handle, self._rejected[handle], self._rejections
                        )
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
        """Sends everything queued and waits for every command's event (rejections do
        not raise here); returns how many events it waited for."""
        self._hand()
        oldest_row, end = self._oldest - self._base, self._handed - self._base
        open_ = (self._count[oldest_row:end] > 0) & ~self._has[oldest_row:end]
        wanted = np.flatnonzero(open_) + self._oldest
        if wanted.size:
            with contextlib.suppress(CommandRejected):
                self._wait(wanted, timeout)
        return int(wanted.size)

    def thread_result(self, timeout=10.0):
        """Waits (with a timeout) for the program's X280 thread, stops it and returns
        its result block (komorebi.thread.ThreadResult). Events go on as before."""
        if self.thread is None:
            raise KomorebiError("the program has no thread")
        result = self.thread.wait(timeout)
        if result.stuck:
            cluster = self.thread.thread.cluster
            raise ThreadFailed(
                stuck_message(cluster, result), cluster, {cluster: result}
            )
        return result

    def thread_results(self, timeout=10.0):
        """Waits (with a timeout) for every X280 thread, stops each and returns
        {cluster: ThreadResult}. One that fails (no return in time, a library error) has
        the others asked to stop at once, and ThreadFailed carries every result."""
        pending = dict(self._thread_runs())
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
            result = results[failed]
            how = (
                "did not return in time"
                if not result.ran
                else f"failed ({result.error})"
            )
            stuck = [cluster for cluster, outcome in results.items() if outcome.stuck]
            if stuck:
                cluster = failed if failed in stuck else stuck[0]
                what = stuck_message(cluster, results[cluster])
                if cluster != failed:
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
        wanted = self._flags[i] & self._flag_result and kind == "done"
        auxiliary = status >> 8
        if status & 0xFF in self._numbered:
            auxiliary &= self._auxiliary_low
        return Event(handle, kind, auxiliary, ticks, ticks if wanted else None)

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

    # Graphs -----------------------------------------------------------------------

    def run_hostfed(self, graph, bindings=None, timeout=10.0, engine=None):
        """Runs a graph or a stream host-fed: its executor on the host, its commands
        through this run's host ring. Returns a HostfedResult (status, error, registers,
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

    def upload(self, stream_or_graph, kind=None):
        """Writes a Stream's or a Graph's upload (or encoded bytes of `kind` "stream"
        or "graph") into the dispatcher's upload region, first fit beside the
        others; returns its Upload. ValueError, before any write, if it does not fit."""
        return self._engines().upload(stream_or_graph, kind)

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
        oldest_row, end = self._oldest - self._base, self._handed - self._base
        waiting = ~self._has[oldest_row:end]
        self._oldest += int(np.argmax(waiting)) if waiting.any() else end - oldest_row

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
        self._finish(_left(deadline))
        result = self._result
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
        if not result.get("result_ready", True):
            wrong.append("the kernel's result never became ready")
        if not result.get("cleaned", True):
            wrong.append(
                "the X280 runtime could not restore cluster 0 (reset the card)"
            )
        if result["state"] != self.h["KMD_STATE_DONE"]:
            wrong.append(f"state {result['state_name']}")
        if result["error"] != 0:
            wrong.append(f"error {result['error_name']} in {result['where_name']}")
        if result.get("failed"):
            wrong.append(result["failed"])
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

    def _fail_start(self, state, timeout=1.0):
        """The kernel did not report READY: finishes it, reads back, raises StartError
        with its result block."""
        # stopped after this, unless the runtime could not be closed: a later exit
        # path (Session.close, the exit hooks) then closes it
        self._finish(timeout, stop_grid=True)
        result = self._result
        raise self._remember(
            StartError(
                f"the dispatcher did not report READY (state {state}, core "
                f"result {result.get('result')}): {self._finish_problems(result)}",
                result=result,
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
        self._result = device_state.result_block(fin["words"], self.h)
        self._result.update(
            {key: value for key, value in fin.items() if key != "words"}
        )
        self._read_back(stop_grid)

    def _read_back(self, stop_grid=False):
        blocks, trace_rows = self.backend.read_back(stop_grid=stop_grid)
        for thread_run in self._thread_runs().values():
            thread_run.grid_stopped()
        self._blocks = blocks
        self._state = device_state.state_words(blocks, 0, self.h)
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
                    seqs[handle], groups[handle] = _sequence_number(count), group
            self._trace = device_state.Trace(
                trace_rows, seqs, groups, self.backend.mhz, self.h
            )

    def _remember(self, error):
        self._failure = error
        return error

    def _fail_timeout(self, event):
        self._finish(10.0)
        result = self._result
        group = self.group(event.handle)
        sequence = (
            result["sequence"]
            if result["error"] == self.h["KMD_ERROR_DEADLINE"]
            else None
        )
        if sequence is None:
            sequence = self.sequence(event.handle)
        known = group and group[0] is not None
        cores = (
            device_state.not_reported(
                self._state, group, event.auxiliary, sequence, self.h
            )
            if known
            else []
        )
        raise self._remember(
            CommandTimeout(
                event.handle, event.auxiliary, sequence, group, cores, result
            )
        )

    def _fail_kernel(self, why):
        self._finish(10.0)
        self.backend.log_state()
        result = self._result
        where = (
            f"exception on X280 core {result['req']} (mcause {result['slot']:#x}, mepc "
            f"{result['sequence']:#x}, mtval {result['seen']:#x})"
            if result["error"] == self.h["KMD_ERROR_TRAP"]
            else f"slot {result['slot']}, sequence {result['sequence']}, "
            f"command {result['req']}"
        )
        raise self._remember(
            KernelFailed(
                f"{why}: error {result['error_name']} in {result['where_name']}, "
                f"{where}",
                result=result,
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
            device_state.core_state(self._state, core, self.h)
            | {
                "ports": device_state.ports(self._blocks[core], self.h),
                "broadcasts": device_state.broadcasts(self._blocks[core], self.h),
                "dynamic": device_state.dynamic(self._blocks[core], self.h),
            }
            for core in range(self.cores)
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
