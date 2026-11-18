"""A loaded program under the dispatcher: submit commands, wait for them, stop,
read back state words and trace. The card is reached only through `backend`, so the
bookkeeping is the same against a fake transport.
"""

import contextlib
import time
from array import array

import numpy as np

# np.unique imports numpy.ma on its first call (about 8 ms): here, not in a first wait
import numpy.ma

from komorebi import abi, layout
from komorebi.chain import calls as chain_module
from komorebi.device import tensor as tensor_module
from komorebi.device.events import Event
from komorebi.dsl import first_use
from komorebi.dsl.chain import chain_calls, chain_stamps
from komorebi.dsl.front import front
from komorebi.dsl.front.front import ClassHandler, HandlerMethod
from komorebi.engines import control, graph_hostfed, stream_hostfed
from komorebi.engines import stream as stream_module
from komorebi.errors import (
    CommandRejected,
    CommandTimeout,
    KernelFailed,
    KomorebiError,
    MessagesGaveUp,
    ProgramError,
    StartError,
    ThreadFailed,
    WaitTimeout,
    describe_core,
)
from komorebi.merge import lifted, merged_holds
from komorebi.program import overlay
from komorebi.program.addon import code as added
from komorebi.program.addon import program as added_program
from komorebi.program.description import Handler, Program, unguarded
from komorebi.session import messages_gave, page_bounds
from komorebi.session import state as device_state

X280_TICKS_S = 50e6  # the X280's mtime, ticks per second
NO_LIST = 0xFF  # a core with no kept reset list (KMD_RESET_DIRECTORY)
LINK_EVENTS = ("gave", "hung", "lost")  # KMD_LINK_*: what a link event says
# KMD_RESET_*, from 1
RESET_STATUS = ("BAD_LIST", "NO_WINDOW", "NO_FIRMWARE", "NO_LIST", "HELD")

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


def _left(deadline):
    """Seconds left until the monotonic time `deadline` (0 once it passed)."""
    return max(deadline - time.monotonic(), 0.0)


def _sequence_number(number):
    """The kernel's sequence number of the n-th command: it counts in 32 bits and skips
    0 (it also skips a slot's last value, which only a wrap can repeat)."""
    return (number - 1) % M32 + 1


def _rectangle_here(rectangle, base):
    """A class's rectangle on the island whose first core is `base`: (its group in the
    card's cores, its rect); a split card's islands each have their own
    (dsl.program.island_rectangles)."""
    islands = rectangle.get("islands")
    if islands is not None and base in islands:
        return islands[base]
    return rectangle["group"], rectangle["rect"]


def _given_on_islands(method, args, group, at, stack, width):
    """km.UPTO(rect=)'s check on islands: the rectangle counts from the island of the
    group's first rank and runs on to the next islands, one below the other (`stack`,
    island_session), every cell a rank; the words, its y0 moved to the stack's rows
    (a split card's: the card's), as the cores count (km::rect_whole)."""
    x0, y0, columns, rows = (int(word) for word in args[at : at + 4])
    tops, heights = stack["tops"], stack["rows"]
    firsts = [first for _, first, _ in stack["islands"]]
    start = max(i for i, first in enumerate(firsts) if first <= group[0])
    top = tops[start] + y0
    named = f"{method.name}: rect=({x0}, {y0}, {columns}, {rows})"
    if x0 + columns > width or top + rows > tops[-1] + heights[-1]:
        raise ValueError(
            f"{named} from island {start}'s first row is off the islands' "
            f"{width} x {tops[-1] + heights[-1] - tops[start]} grid from there"
        )
    ends = [
        layout.stack_rank(stack, width, x0 + columns - 1, y)
        for y in range(top, top + rows)
    ]
    if None in ends:
        raise ValueError(
            f"{named} from island {start}'s first row takes a link core of an island "
            "(each island's link cores last): give one on the islands' ranks"
        )
    first = layout.stack_rank(stack, width, x0, top)
    want = (first, ends[-1] + 1 - first)
    if tuple(group) != want:
        raise ValueError(
            f"{named} is the group {want}; submit it to that group (not {tuple(group)})"
        )
    args = list(args)
    args[at + 1] = top  # the stack's row
    return args


def with_area(writes, address, area, h):
    """A reset list (whole 64-byte lines, ASSERT first) with the words `area` written
    to L1 `address` right after its ASSERT, while the core's CPUs are held."""
    entry = h["KMD_RESET_ENTRY"]
    if int.from_bytes(writes[:4], "little") != h["KMD_RESET_ASSERT"]:
        raise ValueError("a reset list that does not start with its ASSERT")
    data = np.asarray(area, "<u4").tobytes()
    data += bytes(-len(data) % 16)
    line = array("I", [h["KMD_RESET_WRITE"], address, 4 * len(area), 0]).tobytes()
    out = bytes(writes[:entry]) + line + data + bytes(writes[entry:])
    return out + bytes(-len(out) % 64)


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
        ranks=None,
        island_first=0,
        island_ranks=None,
        island_cores=None,
        island_span=None,
        thread_first=1,
    ):
        self.h = abi_headers or abi.headers()
        self._page_bounds = page_bounds.Checked()  # a class's ranges, at submit
        self.dram = None  # the session's DRAM tensors (extents.Registry): the stamps'
        self.backend = backend
        self.port = backend.port
        self.port.batch = batch
        self._keep = bool(getattr(backend, "trace_cap", 0))  # the trace needs groups
        self.grid = grid
        self.cores = grid[0] * grid[1]
        # groups are in flat ranks, of all islands (one island: rank = core)
        self.ranks = ranks or self.cores
        self.island_first = island_first  # rank island_first + c is core base + c
        # the island's thread numbers start here (flat over islands, KMX_THREAD)
        self.thread_first = thread_first
        # link core k (group first KM_GROUP_LINK_BASE + k) is core island_ranks + k
        self.island_ranks = self.cores if island_ranks is None else island_ranks
        self._link_base = self.h["KM_GROUP_LINK_BASE"]
        # (base, count): the island's cores of the card (a split card's); default all
        self.island_cores = island_cores or (0, self.cores)
        # cores of the island from its base, its link cores too (default all)
        self.island_span = island_span or self.cores
        self.program = program
        # the run's programs by program field; holds[field]: the cores holding it
        self.programs = list(programs or ([program] if program is not None else []))
        self._holds = holds
        self.thread_program = thread_program or program  # owns the X280 threads
        self._switch = switch  # {"records": {field: DRAM address}, "refused": why}
        self.check_programs = True  # False: the dispatcher alone refuses (reason 13)
        # an island's run: (group) -> the far cores of a group over islands, as their
        # dispatchers read them (IslandRun._far_cores); None on one card
        self.far_cores = None
        # an island's run: () -> its own cores a give-up already named left (on any
        # island's run: IslandRun._named_cores); None on one card
        self.named_cores = None
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
        self._flag_part = constants["KM_FLAG_PART"]
        # a command to the dispatcher itself (INSPECT, RESET): its flag, not its number
        self._flag_dispatcher = constants["KMD_FLAG_DISPATCHER"]
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
        self.half_added = {}  # an island's: IslandRun's adds marked unusable
        self._next_sequence = 1  # the kernel's sequence of the next command
        self._new_handles(1)
        self._kept = {}  # handle -> the DRAM tensor of its block, until its event
        self._unfenced = False  # a block written since the last fence
        self._rejected = {}
        self._given_up = {}  # handle -> (its timeout event, the cores at the deadline)
        self._given_up_seen = 0  # the dispatcher's count of give-ups, as last read
        self.drain_given_up = 0  # of the commands the last drain() waited for, given up
        self.stalls = []  # the watcher's KMD_EVENT_STALL events: stage, req, ticks
        self.link_events = []  # KMD_EVENT_LINK: a give-up on a link core (_links)
        # The dispatcher's kept reset list ({offset, bytes, seconds}), or why not
        self.recovery, self.recovery_refused = None, "not asked for"
        self._reserved = []  # (offset, bytes) of the upload region the run holds
        self.recoveries = []  # give-ups whose cores the dispatcher reset: handle, text
        self._reset_failed = (-1, 0)  # a give-up's failed reset: core, KMD_RESET_*
        self._timeout_cores = []  # a deadline's inspect, not given up: its cores
        self.inspect_ticks = None  # mtime ticks the last inspect() took on core 0
        self._failure = None
        self._stopped = False
        self._result = None
        self._state = None
        self._blocks = None
        self._trace = None
        self.own_commands = {}  # handle -> what: the runtime's own commands (a trace's)
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
        self._ports = {}  # a port command without its event yet: handle -> group
        self._unsent()

    def _unsent(self):
        # submit() appends here; new objects, since numpy may still view the old ones
        self._words, self._firsts, self._counts = array("I"), array("i"), array("i")
        self._fl = array("B")

    # Submitting --------------------------------------------------------------------

    def chain(self, calls, mode=0, flags=0, cores=None, prefetch=True, records_at=None):
        """Calls (chain.Call, or a class's handler calls) as one chain's records, their
        inputs proven or checked first (ChainRefused names call, tensor, core). `mode`
        a4, `flags` a1; a class's calls: `cores` the group of those without cores=,
        `prefetch` False: reads wait for marks; `records_at`: a held bank address."""
        if any(isinstance(call, front.HandlerCall) for call in calls):
            calls = self._class_calls(calls, cores, prefetch)
        in_dram = (getattr(calls, "direct", []), getattr(calls, "written", []))
        walk = self._walk(calls)
        report, word = chain_module.build(calls)
        calls = [  # a class's handler: the one built for its words' constants
            call._replace(
                handler=self._words_built_for(call.handler, [call.args], "a chain")
            )
            for call in calls
        ]
        numbers = self._chain_numbers(calls)
        for index, (_, field) in enumerate(numbers):
            if field != numbers[0][1]:
                raise chain_module.ChainRefused(
                    f"call {index} ({calls[index].name}): a handler of program field "
                    f"{field}, not of the chain's program (field {numbers[0][1]})"
                )
        handlers = [number for number, _ in numbers]
        first = min(call.group[0] for call in calls)
        end = max(call.group[0] + call.group[1] for call in calls)
        tensor = self.backend.block(
            bytes(64 * len(calls)),
            lambda address: (
                chain_module.records(calls, address, handlers).view(np.uint8).ravel()
            ),
            **({} if records_at is None else {"at": records_at}),
        )
        self._unfenced = True
        handler = self.field_handler(*numbers[0]) if walk is None else walk
        made = chain_module.Chain(
            handler,  # a number: submit makes no words of it
            (first, end - first),
            mode,
            flags,
            word,
            tensor,
            report,
            calls,
        )
        made.direct, made.written = in_dram
        return made

    def _chain_numbers(self, calls):
        """(number, program field) of each call's handler; a gather's (km.all_gather)
        the walk's of the other calls' program, refused unless it has the gather."""
        numbers = [
            None if call.gather is not None else self._handler(call.handler)
            for call in calls
        ]
        fields = [number[1] for number in numbers if number is not None]
        for index in range(len(calls)):
            if numbers[index] is not None:
                continue
            program = self.programs[fields[0]] if fields else None
            if program is None or not getattr(program, "chain_gather", False):
                raise chain_module.ChainRefused(
                    f"call {index} (km.all_gather): the chain's program states no "
                    "km.all_gather in a km.chain method, so its kernels have no "
                    "gather's case; write the chain as a km.chain method"
                )
            numbers[index] = (program.chain_handler, fields[0])
        return numbers

    def _class_calls(self, calls, cores, prefetch, group=None, starts=(0,)):
        """The framework's calls of a chain stated with a class's handler calls
        (komorebi.dsl.chain.chain_calls); `group` (flat ranks) in place of `cores`,
        `starts` each island's first rank (a gather's record)."""
        group = self._group(cores) if group is None else group
        return chain_calls.chain_calls(self, list(calls), group, prefetch, starts)

    def _walk(self, calls):
        """The chain command's handler when the first call is a class's handler of a
        program built with the chain's walk (its records each name their handler);
        None: the first call's handler, the program's own chain mode."""
        first = calls[0].handler if calls else None
        if not isinstance(first, HandlerMethod | ClassHandler):
            return None
        field = self._handler(first)[1]
        number = getattr(self.programs[field], "chain_handler", 0)
        return self.field_handler(number, field) if number else None

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
        part=False,
        **words,
    ):
        """Queues a command for `handler` on cores [first, first + count) (default all)
        or cluster `cluster`'s X280 thread; returns its handle. `result`: ask the result
        word; `queued`: the kernel picks the cores; `block`: bytes its handlers read;
        `part`: an island's part."""
        # a chain (run.chain): an exclusive command a piece of kChainRecordsMax
        # records, `words` its per-submit word; the pieces before the last waited for
        if isinstance(handler, chain_module.Chain):
            chain = handler
            for piece in range(chain.pieces):
                handle = self.submit(
                    chain.handler, chain.group, chain.args(words, piece), exclusive=True
                )
                if piece + 1 < chain.pieces:
                    self.wait(handle)
            tensor_module.sent_direct(chain.written, chain.direct)
            return handle
        if words:
            raise TypeError(f"submit: unknown arguments {sorted(words)}")
        added.refuse_half_added(self.half_added, [handler])  # an island's own run
        started = time.perf_counter()
        handler, group, args, flags, program_field, ports, kept = self._submit_words(
            handler, group, args, exclusive, result, queued, cluster, block, part
        )
        handle = self._queue_command(
            handler, group, args, exclusive, flags, program_field
        )
        if ports:
            self._ports[handle] = group
        if kept is not None:
            self._kept[handle] = kept
        self.submit_s += time.perf_counter() - started
        return handle

    def _submit_words(
        self,
        handler,
        group,
        args,
        exclusive,
        result,
        queued,
        cluster,
        block,
        part,
    ):
        """A command's (handler, group, words, flags, program field, whether it is a
        port's, its kept block) as submit checks them, refused before anything is
        queued (a prebuild's mock run checks its commands so)."""
        program_field, method, given = 0, None, args
        if isinstance(handler, HandlerMethod | ClassHandler):
            handler = self._built_for(handler, args)
            method = handler if isinstance(handler, HandlerMethod) else handler.method
            result = result or method.returns
            args = handler.args(*self._regions_named(method, args))
        if cluster is not None:
            handler, group = self._to_thread(handler, cluster, group, exclusive, queued)
        else:
            handler, program_field = self._handler(handler)
        if not (handler == self._nop or 1 <= handler <= self._user_max):
            raise ValueError(
                f"handler {handler}: not NOP and not in 1..KM_HANDLER_USER_MAX"
            )
        ports = False
        if cluster is None and queued:
            if exclusive:
                raise ValueError("a queued command cannot be exclusive")
            group = self._group((0, group))
            self._check_holds(program_field, handler, 0, self.cores, "queued")
        elif cluster is None:
            group = self._group(group)
            self._check_holds(program_field, handler, *group)
            ports = self._check_ports(program_field, handler, group, exclusive)
            if method is not None:
                self._check_rectangle(method, program_field, group)
                args = self._check_given_rectangle(method, args, group, program_field)
                self._check_reduce_groups(method, program_field, group)
                self._page_bounds.check(
                    method, given, args, group, self.programs[program_field]
                )
        if cluster is None:
            self._check_stamps_out(program_field, handler, args, group)
        if len(args) > len(self._arg_at):
            raise ValueError(f"{len(args)} arguments; a command has 10")
        if any(not 0 <= int(word) <= M32 for word in args):
            raise ValueError("arguments are 32-bit unsigned words")
        kept = None
        if block is not None:
            args, kept = self._block_words(block, args, cluster)
        flags = (
            (self._flag_result if result else 0)
            | (self._flag_queue if queued else 0)
            | (self._flag_part if part else 0)
        )
        return handler, group, args, flags, program_field, ports, kept

    def thread_words(self, handler, args=()):
        """(program field, handler number, [command words]) of a call as submit would
        send it, for an X280 thread to build the same command (kmx_command) of a class
        made after its image was built, an added one too; nothing is sent."""
        if isinstance(handler, HandlerMethod | ClassHandler):
            handler = self._built_for(handler, args)
            method = handler if isinstance(handler, HandlerMethod) else handler.method
            args = handler.args(*self._regions_named(method, args))
        number, program_field = self._handler(handler)
        return program_field, number, [int(word) for word in args]

    def _regions_named(self, method, args):
        """A class handler's arguments with each region parameter given by its name
        made (bank, bank address): the region of this run's thread that has it."""
        out = list(args)
        for index, ((parameter, kind), value) in enumerate(
            zip(method.parameters(), args)
        ):
            if kind != "region" or not isinstance(value, str):
                continue
            found = [
                thread.regions.address(value)
                for thread in self.threads.values()
                if thread.regions is not None and value in thread.regions.keys
            ]
            if len(found) != 1:
                raise ValueError(
                    f"{method.name}: {parameter} names region {value!r}, which "
                    f"{'no' if not found else 'more than one'} thread of this run "
                    "has; give (bank, address)"
                )
            out[index] = found[0]
        return out

    def _check_given_rectangle(self, method, args, group, program_field=0):
        """Refuses a command whose rectangle given at each submit (km.UPTO(rect=))
        is off the grid or whose group is not the rectangle's; the words, its y0
        moved to the card's rows. On islands (the program's stack) it counts from the
        island of the group's first rank and may run on (_given_on_islands)."""
        if front.rect_given(method.function) is None or method.words is None:
            return args
        at = method.words.index(("rect", "rect"))
        x0, y0, columns, rows = (int(word) for word in args[at : at + 4])
        width = self.grid[0]
        state = getattr(self.programs[program_field], "rectangle", None) or {}
        if state.get("stack") is not None:
            return _given_on_islands(method, args, group, at, state["stack"], width)
        height = self.island_span // width  # the island's rows: a split card's own
        if x0 + columns > width or y0 + rows > height:
            raise ValueError(
                f"{method.name}: rect=({x0}, {y0}, {columns}, {rows}) is off the "
                f"{width} x {height} grid"
            )
        first = y0 * width + x0
        want = (first + self.island_first,
                (y0 + rows - 1) * width + x0 + columns - first)  # fmt: skip
        if tuple(group) != want:
            raise ValueError(
                f"{method.name}: rect=({x0}, {y0}, {columns}, {rows}) is the group "
                f"{want}; submit it to that group (not {tuple(group)})"
            )
        args = list(args)
        args[at + 1] = y0 + self.island_cores[0] // width  # the card's row
        return args

    def _check_rectangle(self, method, program_field, group):
        """Refuses a broadcast's command of a class whose group is not the group of
        the rectangle the program was loaded with (fixed at load): on islands a rect=
        that fits one island is each island's own; one taller than island 0, or none
        (one island's in flat ranks), one over the islands (dsl.program.spans)."""
        program = self.programs[program_field]
        line = getattr(program, "broadcast_lines", {}).get(method.name)
        rectangle = getattr(program, "rectangle", None)
        if line is None or rectangle is None or rectangle["group"] is None:
            return
        (low, count), rect = _rectangle_here(rectangle, self.island_cores[0])
        want = (low + self.island_first - self.island_cores[0], count)
        one = ""
        if rectangle.get("span"):
            want, one = (
                tuple(rectangle["group"]),
                (
                    ", one rectangle over the islands from island 0's first row (it is "
                    "taller than island 0) in flat ranks"
                ),
            )
        if rectangle.get("span") == "flat":
            one = ", one island's rectangle in the islands' flat ranks (the default)"
        if tuple(group) != want:
            raise ValueError(
                f"{type(program).__name__}.{method.name}, {line}: the broadcast's "
                f"rectangle {rect or 'the whole grid'} is the group "
                f"{want}{one}, fixed at load; submit it to that group "
                f"(not {tuple(group)}), or make the program with rect="
            )

    def _port_numbers(self, program_field):
        """The handler numbers of the program's port_handlers."""
        program = self.programs[program_field] if self.programs else None
        return {
            program.handlers.get(name, -1) if isinstance(name, str) else int(name)
            for name in getattr(program, "port_handlers", ())
        }

    def _check_ports(self, program_field, handler, group, exclusive, handle=None):
        """Whether `handler` is one of its program's port_handlers; such a command on
        cores a port command without its event still holds is refused unless it is
        exclusive (a group's port commands do not overlap: ports.md)."""
        if handler not in self._port_numbers(program_field):
            return False
        handle = self._next if handle is None else int(handle)
        first, count = group
        for held, (low, cores) in list(self._ports.items()):
            row = held - self._base
            if held < self._base or (held < self._handed and self._has[row]):
                del self._ports[held]  # its event came: the cores are free
            elif not exclusive and low < first + count and first < low + cores:
                raise ValueError(
                    f"command {handle} (handler {handler}, cores {first} to "
                    f"{first + count - 1}) uses ports on cores that command {held} "
                    f"(cores {low} to {low + cores - 1}) still holds; submit it "
                    f"exclusive=True, or wait for command {held} first"
                )
        return True

    def _check_reduce_groups(self, method, program_field, group):
        """Refuses a command of a class's handler that reduces in groups of S ranks
        whose group's cores are not a multiple of S."""
        program = self.programs[program_field]
        found = getattr(program, "reduce_groups", {}).get(method.name)
        if found is not None and group[1] % found[0]:
            raise ValueError(
                f"{type(program).__name__}.{method.name}, {found[1]}: groups of "
                f"{found[0]} ranks, and the command's {group[1]} cores are not a "
                f"multiple of {found[0]}"
            )

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
            handler = lifted(handler, self.programs)  # a merged source's own
            return int(handler), self.program_field(handler.program)
        if isinstance(handler, HandlerMethod):
            fields = [
                field
                for field, program in enumerate(self.programs)
                if isinstance(program, handler.owner)
                or merged_holds(program, handler.owner)
            ]
            if not fields:
                raise ValueError(f"{handler.name}: no program of its class in the run")
            if len(fields) > 1:
                raise ValueError(self._class_form_refused(handler, fields))
            merged = getattr(self.programs[fields[0]], "merged_number", None)
            if merged is not None:  # a merged class's: its number there
                return merged(handler), fields[0]
            return int(handler), fields[0]
        if handler.__class__ is not int:
            handler = (
                self.program.handler(handler)
                if self.program is not None
                else int(handler)
            )
        return int(handler), 0

    def _built_for(self, handler, args):
        """A class's handler with constant parameters: the handler its program built
        for the values `args` gives at their places (left out: the defaults); a set
        not built is refused, naming the sets that were. Others: `handler`."""
        method = handler if isinstance(handler, HandlerMethod) else handler.method
        if not front.constant_parameters(method.function):
            return handler
        if isinstance(handler, ClassHandler):
            program = handler.program
        else:
            program = self.programs[self._handler(handler)[1]]
        number, built = first_use.constant_handler(
            program, method, method.constants_given(args), run=self
        )
        return ClassHandler(number, program, built)

    def _words_built_for(self, handler, rows, where):
        """A class's handler given command words (submit_many's rows, a chain call's
        words): the handler built for the constant values they carry (plain words:
        the defaults), as _built_for. Others: `handler`."""
        if not isinstance(handler, HandlerMethod | ClassHandler):
            return handler
        method = handler if isinstance(handler, HandlerMethod) else handler.method
        if not front.constant_parameters(method.function):
            return handler
        program = None
        if isinstance(handler, HandlerMethod):
            program = self.programs[self._handler(handler)[1]]
        return first_use.words_handler(handler, rows, program, where)

    def _class_form_refused(self, handler, fields):
        """Why `Class.handler` names no program when the run holds several of its
        class (at `fields`): each named by its constants."""
        named = []
        for field in fields:
            program = self.programs[field]
            values = getattr(program, "constant_values", None) or {}
            given = ", ".join(f"{k}={v}" for k, v in values.items())
            named.append(
                f"{type(program).__name__}({given}) at field {field}"
                f"{self._program_told(program, field)}"
            )
        owner = handler.owner.__name__ if handler.owner else "its class"
        return (
            f"{owner}.{handler.name}: the run holds {len(fields)} programs of the "
            f"class ({'; '.join(named)}), and the class's handler does not say which; "
            f"name it through the program (program.{handler.name})"
        )

    def _program_told(self, program, field):
        """What tells a program of the run from another of its class: added live or
        not, and the ranks holding it now (when the run tracks them)."""
        told = ["added live"] if getattr(program, "added_live", False) else []
        holds = getattr(self, "_holds", None)
        if holds is not None and field < len(holds):
            ranks = np.flatnonzero(holds[field])
            spans = np.split(ranks, np.flatnonzero(np.diff(ranks) != 1) + 1)
            told.append(
                "held by ranks "
                + ", ".join(
                    f"{s[0]}-{s[-1]}" if len(s) > 1 else f"{s[0]}" for s in spans
                )
                if len(ranks)
                else "held by no core yet"
            )
        return f" ({', '.join(told)})" if told else ""

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
        low, high = self._part(first, count)
        if not self._holds[program_field][low:high].all():
            name = self.programs[program_field].name
            where = "every core" if queued else f"group ({first}, {count})"
            raise ValueError(
                f"handler {handler} of {name}: {where} does not hold {name}"
            )

    def load_field(self, program):
        """The program field a load of `program` gives its cores; ProgramError with the
        reason when this run cannot load it (before anything is sent)."""
        if self._switch is None:
            raise ProgramError(
                "run.load needs a run loaded with {program: group, ...}: its groups "
                "can switch program"
            )
        if self._switch["refused"]:
            raise ProgramError(f"run.load: {self._switch['refused']}")
        program_field = self.program_field(program)
        unheld = self._switch.get("unheld", {})
        if program_field in unheld:
            raise ProgramError(
                f"run.load: {unheld[program_field]} has no group on this island, so "
                "its parts were not built here (a part is built against the kernels "
                "of a group that holds it, its buffers' formats); give it a group on "
                "every island it is to be loaded on"
            )
        return program_field

    def load(self, program, group, part=False):
        """Makes the cores of `group` (first, count) hold `program`, one of the dict
        given to session.load: its overlay and its buffers, set up again. A command,
        after every earlier one on those cores; their program's buffers are lost
        (`part`: an island's part of a flat group, as submit's)."""
        program_field, first, count, record = self._load_words(program, group)
        flags = self._flag_part if part else 0
        handle = self._queue_command(
            self.h["KM_HANDLER_LOAD"],
            (first, count),
            [record],
            False,
            flags,
            program_field,
        )
        self._held(program_field, first, count)
        return handle

    def _check_stamps_out(self, program_field, handler, args, group):
        """Refuses a chain stamps' read-out (the chain handler, a7 0) whose tensor at
        a0 is not this session's or has fewer pages of STAMPS_PAGE bytes than the
        group's flat ranks need: each core writes the page of its flat rank."""
        program = self.programs[program_field] if self.programs else self.program
        if not getattr(program, "chain_stamps", False) or len(args) < 8:
            return
        if handler != int(program.chain_handler) or int(args[7]) != 0:
            return
        needed = group[0] + group[1]
        tensor = None if self.dram is None else self.dram.at(int(args[0]))
        if tensor is None:
            if self.dram is None:
                return  # a run made without its session (a test's): the card's guard
            raise ValueError(
                f"chain stamps' read-out: a0 {int(args[0]):#x} is no DRAM tensor of "
                f"this session; give a tensor of chain_stamps.tensor_for({needed})"
            )
        page = chain_stamps.STAMPS_PAGE
        if tensor.page_bytes != page or tensor.pages < needed:
            raise ValueError(
                f"chain stamps' read-out to the group ({group[0]}, {group[1]}): its "
                f"tensor has {tensor.pages} pages of {tensor.page_bytes} B, and the "
                f"group's flat ranks need {needed} pages of {page} B (each core writes "
                f"the page of its flat rank): chain_stamps.tensor_for({needed})"
            )

    def _load_words(self, program, group):
        """A load's (program field, first, count, switch record) as run.load checks
        them, refused before anything is queued."""
        program_field = self.load_field(program)
        first, count = self._group(group)
        self._check_load_rectangle(program, first, count)
        return program_field, first, count, self._switch["records"][program_field]

    def _held(self, program_field, first, count):
        """The cores of (first, count) on this card now hold `program_field`."""
        low, high = self._part(first, count)
        for held, cores in enumerate(self._holds or ()):
            cores[low:high] = held == program_field

    def _check_load_rectangle(self, program, first, count):
        """Refuses a load whose group's part on this card does not hold the class's
        rectangle (its per-core args are its rectangle's): the rectangle's cores, this
        card's, taken as flat ranks (an island's: its first rank on), and never one of
        an island's link cores. One card: inside the group."""
        rectangle = getattr(program, "rectangle", None) or {}
        if rectangle.get("group") is None:
            return
        if rectangle.get("span"):  # one over the islands, its cells ranks (spans)
            low, cores = rectangle["group"]
            if low < first or low + cores > first + count:
                raise ProgramError(
                    f"run.load: {program.name}'s rectangle {rectangle['rect']} is "
                    f"the flat ranks ({low}, {cores}) over the islands, not inside the "
                    f"group ({first}, {count})"
                )
            return
        base = self.island_cores[0]
        (low, cores), rect = _rectangle_here(rectangle, base)
        flat = low + self.island_first - base
        part_low = max(first, self.island_first)
        part_high = min(first + count, self.island_first + self.island_ranks)
        links = low + cores > base + self.island_ranks
        if links or flat < part_low or flat + cores > part_high:
            island = self.island_ranks != self.ranks or self.island_first
            where = (
                f"ranks ({flat}, {cores}) of this island"
                + (", its link cores too" if links else "")
                if island
                else f"cores ({low}, {cores})"
            )
            span = max(part_high - part_low, 0)
            mine = f"; this island's part of it ({part_low}, {span})" if island else ""
            raise ProgramError(
                f"run.load: {program.name}'s rectangle {rect} is the "
                f"{where}, not inside the group ({first}, {count}){mine}"
            )

    def addon(
        self,
        name,
        parts=None,
        handler=False,
        number=None,
        defines=None,
        timeout=10.0,
        message=False,
        compile_args=None,
    ):
        """Adds overlay `name` (`parts` as Program.overlay's, built against the loaded
        kernels, `compile_args` in place of theirs) to the live run; returns its number.
        A class `name` in the runtime's form becomes one more program of a run loaded
        as a dict (komorebi.program.addon.program); returns its program field."""
        if isinstance(name, Program):
            return added_program.add(self, name)
        if parts is None:
            raise TypeError(f"run.addon({name!r}): parts, as Program.overlay's")
        return added.addon(
            self, name, parts, handler, number, defines, timeout, message, compile_args
        )

    @property
    def added(self):
        """What run.addon placed: {name: komorebi.program.addon.code.Added}."""
        room = getattr(self.backend, "room", None)
        return dict(room.added) if room is not None else {}

    def transfer(self, handler, group, args, timeout):
        """A PUSH or PULL (the session's transfers) on `group` (first, count, program
        field: the field its cores hold), waited for; returns its Event."""
        first, count, program_field = group
        handle = self._queue_command(
            handler, self._group((first, count)), args, False, 0, program_field
        )
        return self.wait(handle, timeout)

    def _area_resets(self, lists, core_list):
        """Adds the overlay area as the load wrote it (load_overlays) to each reset
        list, so a reset core's slots, parks and meets start again from there; the
        reason it cannot when the cores of one list hold different areas."""
        rows = getattr(self.backend, "control_rows", None)
        start = self.backend.control_offset + self.h["KM_CONTROL_BYTES"] // 4
        area = getattr(self.backend, "area_bytes", None)  # with records past 4 slots
        words = (area or overlay.constants()["kOverlayAreaBytes"]) // 4
        if rows is None or rows.shape[1] < start + words:
            return None  # no overlay area after the control blocks
        numbers = np.frombuffer(bytes(core_list), np.uint8)
        address = self.backend.control_addr + self.h["KM_CONTROL_BYTES"]
        for number, writes in enumerate(lists):
            area = rows[np.flatnonzero(numbers == number), start : start + words]
            if (area != area[0]).any():
                return f"the overlay areas of reset list {number}'s cores differ"
            lists[number] = with_area(writes, address, area[0], self.h)
        return None

    def keep_resets(self, exclude=(), timeout=10.0):
        """Hands the dispatcher a reset list a group of cores with the same kernels and
        each core's list number (load(recover=True)): a give-up then resets and loads
        again each core that did not leave; `exclude` (link cores) get none. Sets
        `recovery`, or `recovery_refused` with the reason; True when kept."""
        started = time.perf_counter()
        if self.recovery is not None:  # again (run.addon): the new lists replace them
            self._reserved.remove((self.recovery["offset"], self.recovery["bytes"]))
            self.recovery = None
        launch = getattr(self.backend, "launch", None)
        if launch is None:
            self.recovery_refused = "the session's backend has no launch to reload"
            return False
        groups = list(launch.reset_groups())[: self.cores]
        excluded = set(exclude)
        lists, owner = [], {}
        # A reset ends once the reader's start-up cleared the ring and wrote its magic.
        magic = self.h["kStateMagic"]
        ready = {
            "ready": self.backend.control_addr
            + self.h["KM_CONTROL_STATE"]
            + 4 * self.h["kStMagic"],
            "ready_mask": 0xFF000000,
            "ready_value": magic & 0xFF000000,
        }
        try:  # one list a group, built on its first core
            for core, group in enumerate(groups):
                if group >= 0 and core not in excluded and group not in owner:
                    owner[group] = len(lists)
                    x, y = core % self.grid[0], core // self.grid[0]
                    lists.append(launch.reset_list(x, y, every=True, **ready))
        except (ValueError, RuntimeError) as error:  # semaphores
            self.recovery_refused = str(error)
            return False
        core_list = bytearray([NO_LIST] * self.h["KM_CORES_MAX"])
        for core, group in enumerate(groups):
            if group in owner and core not in excluded:
                core_list[core] = owner[group]
        refused = self._area_resets(lists, core_list)
        if refused:
            self.recovery_refused = refused
            return False
        head = self.h["KMD_RESET_DIRECTORY_ENTRIES"]
        at = -(-(head + 16 * len(lists)) // 64) * 64
        entries, body = [], b""
        for writes in lists:
            entries.append((at + len(body), len(writes)))
            body += writes  # whole 64-byte lines
        directory = array("I", [self.h["KMD_RESET_DIRECTORY"], len(lists), 0, 0])
        directory = directory.tobytes() + bytes(core_list)
        for offset, size in entries:
            directory += array("I", [offset, size, 0, 0]).tobytes()
        blob = directory + bytes(at - len(directory)) + body
        ready = self.h["KMD_HOST_READY"] + control.offsets(self.h)["ready_bytes"]
        if not lists or len(blob) > self.port.host_word(ready) or self._reserved:
            self.recovery_refused = (
                f"{len(blob)} bytes do not fit the upload region"
                if lists
                else "no core with kernels to reload"
            )
            return False
        self.port.upload(0, blob)
        handle = self._queue_command(
            self.h["KMD_HANDLER_RESET"],
            (0, 0),
            [0, len(blob)],
            flags=self._flag_dispatcher,
        )
        self._counts[-1] = 0  # the kernel's own command: no sequence number
        event = self.wait(handle, timeout)
        if event.kind != "pong" or event.auxiliary != self.h["KMD_RESET_OK"]:
            self.recovery_refused = f"the dispatcher refused the lists: {event}"
            return False
        self._reserved.append((0, len(blob)))
        self.recovery = {
            "offset": 0,
            "bytes": len(blob),
            "lists": entries,
            "cores": [
                core for core, number in enumerate(core_list) if number != NO_LIST
            ],
            "seconds": time.perf_counter() - started,
        }
        self._core_list = core_list
        self.recovery_refused = None
        return True

    def reset_core(self, core, timeout=10.0, channels=(), drops=()):
        """Resets core `core` (this card's core number) and loads its kernels again
        while the other cores run: by the dispatcher from its kept list (`recovery`),
        else by the host: its connections on `channels` closed, `drops` emptied first.
        Returns {core, by, seconds, x280_seconds}; KomorebiError if it did not return.
        """
        if not 0 <= core < self.cores:
            raise ValueError(f"core {core}: the grid has {self.cores}")
        if not timeout >= 0.001:  # tt-metal reads a wait under 1 ms as its own limit
            raise ValueError(f"reset_core: timeout {timeout} s: at least 0.001 s")
        started = time.perf_counter()
        number = NO_LIST if self.recovery is None else self._core_list[core]
        if number == NO_LIST or channels or drops:
            x, y = core % self.grid[0], core // self.grid[0]
            parts = self.backend.launch.reset_core(
                x,
                y,
                min(timeout, 2.0),
                [int(channel) for channel in channels],
                [int(word) for word in drops],
            )
            if parts[1] == -2:
                raise KomorebiError(
                    f"core {core}: a router did not close its connection"
                )
            if parts[1] < 0:
                raise KomorebiError(f"core {core}: its firmware did not start again")
            return {
                "core": core,
                "by": "host",
                "seconds": time.perf_counter() - started,
                "x280_seconds": None,
                "parts": list(parts),  # reset (and close), firmware, load
            }
        offset, size = self.recovery["lists"][number]
        handle = self._queue_command(
            self.h["KMD_HANDLER_RESET"],
            (core, 1),
            [offset, size],
            flags=self._flag_dispatcher,
        )
        self._counts[-1] = 0  # the kernel's own command: no sequence number
        event = self.wait(handle, timeout)
        if event.kind != "pong" or event.auxiliary != self.h["KMD_RESET_OK"]:
            raise KomorebiError(f"core {core}: the dispatcher's reset failed: {event}")
        return {
            "core": core,
            "by": "dispatcher",
            "seconds": time.perf_counter() - started,
            "x280_seconds": event.ticks / X280_TICKS_S,
        }

    def given_up(self, handle):
        """The CommandTimeout of a given-up command whose event is held (event() after
        drain()), or None; its text names the cores reset and loaded again. Every such
        give-up is also in `recoveries`, kept after wait() raised it."""
        return self._given_up_error(handle) if handle in self._given_up else None

    def inspect(self, group=None, timeout=10.0):
        """While the run goes on: the dispatcher reads each rank of `group` (flat, of
        this island; default every core): core_state's dict, `live` (each kernel's wait
        and command), `trace` (the latest commands' entries); `read` False for a core
        it could not reach. inspect_ticks: X280 ticks the read took on core 0."""
        off = unguarded(self.programs)
        if off:
            raise KomorebiError(
                f"run.inspect: {', '.join(off)} built with guard=False: its kernels "
                "keep no live words, so there is nothing to read; build it with "
                "guard=True (the default)"
            )
        first, count = (0, self.cores)
        if group is not None:  # flat ranks, of this island: the dispatcher reads cores
            low, high = self._part(*self._group(group))
            first, count = int(low), int(high - low)
            if count != group[1]:
                raise ValueError(f"group {tuple(group)}: not this island's ranks")
        handle = self._queue_command(
            self.h["KMD_HANDLER_INSPECT"], (first, count), flags=self._flag_dispatcher
        )
        self._counts[-1] = 0  # the kernel's own command: no sequence number
        self.wait(handle, timeout)
        block = device_state.inspect(self.backend.inspect(count)["block"], self.h)
        if not (
            block["complete"]
            and block["kind"] == "host"
            and block["request"] == handle & M32
        ):
            raise KomorebiError(
                f"the inspect block does not answer command {handle}: another inspect"
                " (a deadline's) was read meanwhile; inspect again"
            )
        self.inspect_ticks = block["ticks"]
        return block["cores"]

    def holds(self):
        """[field] -> the cores that hold each program (bool arrays), or None when every
        core holds the run's one program."""
        return self._holds

    def alive(self):
        """Whether the run takes commands: not stopped, nothing failed."""
        return self._failure is None and not self._stopped

    def submit_many(
        self, handler, firsts, counts, args=None, exclusive=False, part=False
    ):
        """Queues n commands at once: `handler` a name, a number, n numbers or a
        class's handler, `firsts` and `counts` the groups (n each, or one for all),
        `args` an [n, m] matrix of 32-bit words (m <= 10) or None; `part` as for
        submit. Returns the handles as a range."""
        started = time.perf_counter()
        self._check_alive()
        named = handler if isinstance(handler, list | tuple) else [handler]
        added.refuse_half_added(
            self.half_added, list(np.ravel(np.asarray(named, object)))
        )
        handlers, first, count, args, shape, program_field = self._many_words(
            handler, firsts, counts, args, exclusive
        )
        words = np.zeros((shape[0], 16), np.uint32)
        words[:, self._hdr] = handlers & 0xFF | program_field << 24
        words[:, self._group_at] = first | count << 16
        if args is not None and args.size:
            words[:, self._arg_at[: args.shape[1]]] = args
        self._hand()
        start = self._next
        self._next += shape[0]
        if exclusive:
            words[:, self._hdr] |= self._flag_exclusive << 8
        if part:
            words[:, self._hdr] |= self._flag_part << 8
        self._send_block(words, first, count, np.zeros(shape[0], np.uint8))
        self.submit_s += time.perf_counter() - started
        return range(start, self._next)

    def _many_words(self, handler, firsts, counts, args, exclusive):
        """submit_many's (handlers, firsts, counts, words, shape, program field) as it
        checks them, its port rows taken (self._ports) before anything is queued."""
        program_field = 0
        rows = () if args is None else args  # a class's words: their constants' handler
        handler = self._words_built_for(handler, rows, "submit_many")
        if isinstance(handler, str | Handler | HandlerMethod):
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
        bad = (count < 1) | (first < 0) | (first + count > self.ranks)
        if bad.any():
            i = int(np.argmax(bad))
            raise ValueError(
                f"group ({int(first[i])}, {int(count[i])}) is not inside {self.ranks} "
                "ranks"
            )
        if self._holds is not None and self.check_programs:
            missing = np.concatenate([[0], np.cumsum(~self._holds[program_field])])
            low, high = self._part(first, count)
            bad = (missing[high] - missing[low] > 0) & (handlers != self._nop)
            if bad.any():
                i = int(np.argmax(bad))
                self._check_holds(
                    program_field, int(handlers[i]), int(first[i]), int(count[i])
                )
        numbers = self._port_numbers(program_field)
        rows = np.flatnonzero(np.isin(handlers, list(numbers))) if numbers else []
        try:  # each port row checked against those before it too
            for row in rows:
                group = (int(first[row]), int(count[row]))
                self._check_ports(
                    program_field,
                    int(handlers[row]),
                    group,
                    exclusive,
                    self._next + row,
                )
                self._ports[self._next + int(row)] = group
        except ValueError:
            for row in rows:
                self._ports.pop(self._next + int(row), None)
            raise
        if args is not None and args.size and (args.min() < 0 or args.max() > M32):
            raise ValueError("arguments are 32-bit unsigned words")
        return handlers, first, count, args, shape, program_field

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
        return handler, (self._cluster_base + self.thread_first - 1 + cluster, 1)

    def _part(self, first, count):
        """The cores [low, high) of flat group (first, count) on this island (numbers
        or arrays; low == high: none); a group of link cores, its link cores."""
        link = (first >= self._link_base) & (first < self._cluster_base)
        first = np.where(link, first - self._link_base + self.island_ranks, first)
        island_first = np.where(link, 0, self.island_first)
        base, cores = self.island_cores
        end = np.where(link, base + self.island_span, base + cores)
        low = np.clip(first - island_first + base, base, end)
        high = np.clip(first + count - island_first + base, low, end)
        return low, high

    def _group(self, group):
        if group is None:
            return (0, self.cores)
        first, count = int(group[0]), int(group[1])
        if count < 1 or first < 0 or first + count > self.ranks:
            raise ValueError(
                f"group ({first}, {count}) is not inside {self.ranks} ranks"
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
        taken = has_cores * np.ones(rows, np.int64)  # a sequence number a command
        self._sequence[row : row + rows] = np.where(
            has_cores, self._next_sequence + np.cumsum(taken) - taken, 0
        )
        self._next_sequence += int(taken.sum())
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
        count = self._local(group)[1] if group and group[0] is not None else 0
        read = self.backend.inspect(count)
        if read["given_up"] == self._given_up_seen:
            block = device_state.inspect(read["block"], self.h)
            failed = block["reset_failed"] if block["kind"] == "timeout" else 0
            self._reset_failed = (failed & 0xFFFF) - 1, failed >> 16
            self._timeout_cores = block["cores"] if block["kind"] == "timeout" else []
            return False
        self._given_up_seen += 1
        block = device_state.inspect(read["block"], self.h)
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
        if far is None or not group or group[0] is None:
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
        self.wait_kept(wanted, timeout)
        events = [self.event(handle) for handle in wanted.tolist()]
        self.forget(wanted)
        return events[0] if single else events

    def wait_kept(self, handles, timeout):
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
            self.drain_given_up = sum(int(h) in self._given_up for h in wanted)
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

    def dispatcher_given_up(self):
        """The dispatcher's count of the host's commands it gave up, read now from its
        memory through the runtime (no command)."""
        return int(self.backend.inspect(0)["given_up"])

    @property
    def blocks_held(self):
        """Command blocks the run still holds: each is freed at its command's event."""
        return len(self._kept)

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
        """Stops the grid, then the kernel through the runtime, and reads back what the
        cores left; for failure paths. result() then holds the kernel's result block
        with `abandoned`, the commands left without their DONE event. Its L1 tensors
        are given back as a stop gives them (each island's alike), unless the runtime
        could not be closed."""
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
        try:
            self._finish(10.0)
        except KomorebiError as error:
            if seen:
                error.args = (f"{error.args[0]}; {seen}", *error.args[1:])
            raise
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
                self._state, self._local(group), event.auxiliary, sequence, self.h
            )
            if known
            else []
        )
        gave = messages_gave.named(result, self._blocks, self.h, self._state)
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
        raise self._remember(error)

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
            if status == self.h["KMD_RESET_HELD"]:
                why = (
                    f"core {core} was not reset: other commands in flight are in "
                    "its ring, which a reset would clear"
                )
            parts.append(why)
        return "; ".join(parts)

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
                f"{where}"
                + (f"; {self._close_failed()}" if self._close_failed() else ""),
                result=result,
            )
        )

    # After the stop ---------------------------------------------------------------

    def result(self):
        """The dispatcher's result block (decoded); valid after the stop."""
        self._require_stopped("result")
        return self._result

    def state(self):
        """Every core's state words, port slots, broadcast areas, dynamic-run words
        and overlay epoch, as dicts; valid after the stop."""
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
            for core in range(self.cores)
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
