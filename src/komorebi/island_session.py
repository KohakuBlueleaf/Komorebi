"""A session over several cards, each card's chip an island: one fabric mesh, one X280
runtime over it, a dispatcher and the persistent grid on every island. Ranks are flat
and contiguous by island; a command's group is split by island on the host, a tensor
is placed by island. `Session(cards=[...])` returns this; one card is one island."""

import concurrent.futures
import copy
import os
import time

import numpy as np

from komorebi import (
    abi,
    added,
    added_program,
    dram,
    formats,
    islands,
    kernel_buffer,
    link_area,
    native,
    prebuild,
    remembered,
)
from komorebi import chain as chain_module
from komorebi import channel as channel_module
from komorebi import program as program_module
from komorebi import programs as programs_module
from komorebi import session as session_module
from komorebi import state as device_state
from komorebi.dsl import front
from komorebi.errors import (
    CardBusy,
    CommandTimeout,
    KernelSizeError,
    KomorebiError,
    L1BudgetError,
    WaitTimeout,
)
from komorebi.program import Program
from komorebi.trees import TreeCalls

M32 = 0xFFFFFFFF
SLICE_S = 0.0  # an island's turn in wait_together: one native round (IslandRun)
THREADS = 3  # thread numbers an island: one a work cluster (KMX_THREAD)
TENSORS_MAX = islands.TENSORS_MAX  # the tensor table's entries by default
# a word below a load's KM_TENSORS_MAX is an entry: every DRAM bank address is above
# the allocator's base (over 1 MiB on Blackhole), so the bound may go up to this
TENSORS_MOST = 1 << 16
LINK_PACKET_BYTES = 4352  # the least packet a session takes
ISLANDS_MAX = 16  # km_abi.h's KM_ISLANDS_MAX: a load of more islands defines its own
# A link core's L1 for deliveries (km_deliver.hpp), shared by the source islands
DELIVER_ROOM = 64 << 10
# a split card's shapes and how the log names them
SPLIT_SHAPES = {
    "mesh": "each joined to each",
    "line": "in a line",
    "ring": "in a ring",
    "grid": "a 2 x 2 grid with wrap",
}
# Two cards' kernel buffer, in steps of 16 KiB; kernel_bytes= overrides it.
KERNEL_BYTES = 160 * 1024
# The transfer chunk a session over two cards asks for: a page CPU passes pages through
# it, one L1 slot a read in flight (15 slots of 4 KiB); with no room: the default.
LINK_CHUNK = 64 * 1024
CROSS_PAGE = 64  # bytes of a link CPU's row of what crossed, as COUNTS writes it
# The packet payload on the links, told to the routers at the open. One link from L1
# slots: 31.3 GB/s at 4352 B (the default), 40.2 at 8192, 33.2 at 15232 (the largest).
PACKET_BYTES = 8192


def _rows(indices):
    """Ascending row indices as a slice when they are one run (a view of the rows, no
    copy), else as they are."""
    if indices[-1] - indices[0] == len(indices) - 1:
        return slice(int(indices[0]), int(indices[-1]) + 1)
    return indices


def _named(error, island):
    """`error` with the island it came from: its message starts "island i: " and
    `error.island` is i (once: an error raised again keeps its first island)."""
    if getattr(error, "island", None) is None:
        error.island = island
        if error.args:
            error.args = (f"island {island}: {error.args[0]}", *error.args[1:])
    return error


def _left(deadline):
    return max(deadline - time.monotonic(), 0.0)


def _one_command(handler, firsts, counts, args):
    """(handler, group, words) when submit_many's arguments are one command (one
    handler, one first and count, at most one row of integer words), else None: those
    submit takes as they are (any other form, a wrong one too, goes submit_many's way).
    """
    if not isinstance(handler, str | int):  # a Handler is an int
        number = np.asarray(handler)
        if number.ndim or number.dtype.kind not in "iu":
            return None
        handler = int(number)
    first, count = np.asarray(firsts), np.asarray(counts)
    if first.size != 1 or count.size != 1 or first.ndim > 1 or count.ndim > 1:
        return None
    if first.dtype.kind not in "iu" or count.dtype.kind not in "iu":
        return None
    words = ()
    if args is not None:
        args = np.asarray(args)
        if args.ndim != 2 or args.shape[0] != 1 or args.dtype.kind not in "iu":
            return None
        words = args[0].tolist()
    elif first.ndim + count.ndim == 0:
        return None  # no row at all (submit_many refuses it)
    return handler, (int(first.reshape(-1)[0]), int(count.reshape(-1)[0])), words


class Crossings:
    """What crossed the links since the run's start: `rows[(link, source, target)]`,
    what island `source` sent to `target` on its link `link` (FIELDS, each a count).
    `after - before` is what crossed between two reads; `total` sums links."""

    FIELDS = ("messages", "lines", "asks", "pages", "parts", "puts", "bytes")
    # Put-ring cycles of the link CPU: turns sent (some waiting for the router), between
    # turns by the next first slot, and the turns that sent (a count, mod 2**32).
    IDLE = ("ring_late", "ring_filling", "ring_turns", "ring_empty")
    CYCLES = ("ring_sending", "ring_router") + IDLE

    def __init__(self, rows):
        self.rows = dict(rows)

    def __getitem__(self, key):
        return self.rows[key]

    def __sub__(self, other):
        zero = dict.fromkeys(self.FIELDS + self.CYCLES, 0)
        return Crossings(
            {
                key: {
                    **{f: row[f] - other.rows.get(key, zero)[f] for f in self.FIELDS},
                    **{
                        f: (row[f] - other.rows.get(key, zero)[f]) % 2**32
                        for f in self.CYCLES
                    },
                }
                for key, row in self.rows.items()
            }
        )

    def total(self, source=None, target=None):
        """Every field summed over the links (from `source`, to `target`: None any)."""
        out = dict.fromkeys(self.FIELDS, 0)
        for (_, here, there), row in self.rows.items():
            if source in (None, here) and target in (None, there):
                for field in self.FIELDS:
                    out[field] += row[field]
        return out

    def __repr__(self):
        lines = [
            f"link {k} island {a} -> {b}: {row}" for (k, a, b), row in self.rows.items()
        ]
        return "\n".join(lines) or "no link"


class LinkPlan:
    """Island `island`'s link toward island `far` for commands across: its link core 0
    (`core`, its index in the island's cores, after the ranks), whose link 0 CPU holds
    link 0's connection and sends the lines, and that core's link area (`area`)."""

    def __init__(self, graph, island, far, core, area):
        self.graph, self.island, self.far = graph, island, far
        self.core, self.area = core, area
        # the relay words of the island area (km_relay.hpp): relay areas here and
        # far, the far link core's tile x | y << 16 (relay_tiles: each link core's)
        self.relay = [0, 0, 0]
        self.relay_tiles = {}
        self.forward = 0  # the forward area's address (more than two islands)
        self.own = 0  # the own words' address (KM_OWN_*), on every core
        # the link area and relay on the link cores alone ((x, y) a core, its words
        # with the area's address) when they run their own kernels; else None
        self.apart = None

    def clear(self, session):
        """Zeroes the link area and relay on each link core (the page CPUs' word set),
        through the device API: just before the launch, after every other write
        (a transfer kernel's buffer starts at the same L1 base)."""
        if self.apart is None:
            return
        cores, words = self.apart
        module = session._module or native.load_module(session.tree)
        for x, y in cores:
            module.write_l1(session.device, x, y, self.area, words)

    def line(self, dispatch, far=None, dispatches=None):
        """The dispatcher's link line and link area (link_area.line): every island's
        dispatcher of `dispatches`, else each taken as `dispatch` (two cards)."""
        ones = dispatches or [dispatch] * len(self.graph.islands)
        tiles = [(one.l2_x, one.l2_y) for one in ones]
        return link_area.line(
            self.graph, self.island, self.core, self.area, tiles, dispatch.lim_base
        )


class IslandTensor:
    """A DRAM tensor of an island session, one logical tensor: island i holds its part
    (`placement`; a copy: every page) in its own buffer `parts[i]` (None: no page).
    `address` is its reference word, which kernels hand to Komorebi's page calls: one
    island, the buffer's bank address; several, its entry in the tensor table."""

    def __init__(self, parts, placement, shape, dtype, layout, word, release=None):
        self.parts = tuple(parts)
        self._release = release
        self.placement = placement
        held = [part for part in self.parts if part is not None]
        first = held[0]
        self.shape, self.dtype, self.layout = tuple(shape), dtype, layout
        self.pages = placement.pages
        self.page_bytes = first.page_bytes
        self.aligned_page_bytes = first.aligned_page_bytes
        self.address = word
        self.in_dram = True
        self.session = None  # the IslandSession that made it: read() and write()

    def write(self, data):
        """Writes `data` (this shape) over the whole tensor, as a one-card tensor's
        write: each island's part through its own host channel."""
        raw, shape = formats.to_bytes(data, self.dtype, self.layout)
        if shape != self.shape:
            raise ValueError(f"data of shape {shape}: the tensor is {self.shape}")
        self.session.write(self, raw.reshape(self.pages, self.page_bytes))

    def read(self):
        """The whole tensor as a numpy array (bfloat16 as float32), as a one-card
        tensor's read: each island's pages gathered."""
        (rows,) = self.session.read(self)
        return formats.from_bytes(rows.reshape(-1), self.shape, self.dtype, self.layout)

    @property
    def addresses(self):
        """Each island's bank address of its part; 0 where it holds none."""
        return [0 if part is None else part.address for part in self.parts]

    def entry(self):
        """Its entry of the tensor table (islands.tensor_entry)."""
        return islands.tensor_entry(
            self.placement, self.aligned_page_bytes, self.addresses
        )

    def accessor_args(self):
        """The tensor accessor arguments, the same on every island."""
        args = [part.accessor_args() for part in self.parts if part is not None]
        if any(other != args[0] for other in args[1:]):
            raise KomorebiError("the islands' accessor arguments differ")
        return args[0]

    def tensor_args(self):
        """What km::Tensor reads (compile arguments): the accessor's; with the
        tensor's reference word (`address`), what a kernel's page calls take."""
        return self.accessor_args()

    def holders(self, first, pages):
        """[(island, first, pages, index)] of the pages first .. first + pages - 1:
        runs each held by one island at consecutive indices of its part (a copy:
        island 0's, index the page)."""
        return self.placement.runs(first, pages)

    def free(self):
        """Frees every island's part (once each) and its entry of the tensor table."""
        if self._release is not None:
            self._release(self)
        for part in {id(part): part for part in self.parts if part}.values():
            part.free()


class BankRange:
    """`bytes` of DRAM bank `bank` of island `island` from bank address `base` (an X280
    thread's region, say), as km::BankRange names it: `address` is its word, its entry
    of the tensor table (one island: the bank address itself)."""

    def __init__(self, island, bank, base, size, word, count=1, release=None):
        self.island, self.bank, self.base, self.bytes = island, bank, base, size
        self.address = word
        self._count = count  # the session's islands
        self._release = release

    def entry(self):
        """Its entry of the tensor table (islands.bank_entry)."""
        return islands.bank_entry(
            self._count, self.island, self.bank, self.base, self.bytes
        )

    def free(self):
        """Its entry of the tensor table is free again; the bank's bytes stay."""
        if self._release is not None:
            self._release(self)


def _check_reduce_islands(graph, handler, group):
    """Refuses a command of a class's handler that reduces in groups of S consecutive
    ranks when one such group of the command spans two islands (a reduce's cores
    reach each other on their own card only)."""
    method = getattr(handler, "method", None)
    program = getattr(handler, "program", None)
    found = getattr(program, "reduce_groups", {}).get(getattr(method, "name", None))
    if found is None:
        return
    size, line = found
    first, count = group
    for island in range(1, len(graph.islands)):
        edge = graph.first(island)
        if first < edge < first + count and (edge - first) % size:
            low = first + (edge - first) // size * size
            raise ValueError(
                f"{type(program).__name__}.{method.name}, {line}: its reduce group "
                f"of {size} ranks at flat ranks {low}..{low + size - 1} spans islands "
                f"{island - 1} and {island}; start the command so that island "
                f"{island}'s first rank {edge} begins a group"
            )


class IslandTrace:
    """An island run's trace: a ticket's rows are its parts' rows in island order (flat
    rank order). Each island's clock is its own: phases and leads are an island's."""

    def __init__(self, traces, parts):
        self.traces, self._parts = list(traces), parts

    @property
    def capacity(self):
        return min(trace.capacity for trace in self.traces)

    def island(self, island):
        """Island `island`'s own Trace (its handles, its clock)."""
        return self.traces[island]

    def command(self, ticket):
        """[count, 4] words (seen, done, sent, extra) of the ticket's ranks, each part's
        in its island's clock; None when a part was not traced."""
        rows = []
        for island, trace in enumerate(self.traces):
            handle = int(self._parts[island][ticket])
            if handle:
                rows.append(trace.command(handle))
        if not rows or any(part is None for part in rows):
            return None
        return np.concatenate(rows)


class IslandEvent(tuple):
    """A finished command of an island run: the tuple of its parts' events in island
    order, which also reads as one card's event (docs/reference/islands.md#runs)."""

    @property
    def parts(self):
        return tuple(self)

    def _worst(self):
        """The first part that is not done, else the slowest."""
        undone = [event for event in self if event.kind != "done"]
        return undone[0] if undone else max(self, key=lambda event: event.ticks)

    kind = property(lambda self: self._worst().kind)  # "done" only if every part is
    ticks = property(lambda self: max(event.ticks for event in self))
    microseconds = property(lambda self: max(event.microseconds for event in self))
    reason = property(lambda self: self._worst().auxiliary)  # rejected: its reason
    handle = property(lambda self: self[0].handle)  # the group's first rank's part
    auxiliary = property(lambda self: self[0].auxiliary)
    result = property(lambda self: self[0].result)


def _handler_label(handler):
    """A handler as a line names it: program.method, the method, or its number."""
    method = getattr(handler, "method", handler)
    if hasattr(method, "name"):
        program = getattr(handler, "program", None)
        return f"{program.name}.{method.name}" if program else method.name
    program = getattr(handler, "program", None)
    return f"{program.name} handler {int(handler)}" if program else f"handler {handler}"


class IslandRun(TreeCalls):
    """The runs of one program on every island, as one run over flat ranks. A ticket
    is a command of a group of flat ranks; its parts went to their islands' dispatchers
    and it completes when every part has. Errors name the island (`error.island`)."""

    def __init__(self, graph, runs, counts=None):
        self.graph = graph
        self.runs = tuple(runs)
        self._counts = counts  # () -> Crossings of the links, read now (None: no link)
        self.session = None  # the IslandSession with links: a link core's reset
        self._links_reset = {}  # (island, k) -> its run's link events at its reset
        self.cores = graph.ranks
        self.program = self.runs[0].program
        self._next = 1  # the next ticket
        # per island, the handle of each ticket's part on that island (0: none)
        self._parts = [np.zeros(1024, np.int64) for _ in self.runs]
        # a (handler, group)'s first command: what it moved across the links, said as
        # a line ("log"), raised after it ran ("error"), or not read ("off")
        self.crossing_line = "log"
        self._crossed_seen, self._measured = set(), {}
        for island, run in enumerate(self.runs):  # a time-out names every island's part
            run.far_cores = lambda group, island=island: self._far_cores(island, group)

    def island(self, island):
        """Island `island`'s own Run (local ranks)."""
        return self.runs[island]

    def _grow(self, tickets):
        while self._next + tickets > len(self._parts[0]):
            self._parts = [
                np.concatenate([parts, np.zeros_like(parts)]) for parts in self._parts
            ]

    def chain(self, calls, mode=0, flags=0, cores=None, prefetch=True):
        """Run.chain over flat ranks: refused when a mark or a credit would go from
        one island to another; else each island's records written in its own DRAM,
        each island walking them on its own cores. A class's calls are described
        with island 0's program, `cores` their group in flat ranks."""
        if any(isinstance(call, front.HandlerCall) for call in calls):
            group = (0, self.cores) if cores is None else tuple(cores)
            calls = self.runs[0]._class_calls(calls, None, prefetch, group)
        chain_module.build(calls, lambda rank: self.graph.resolve(rank)[0])
        chains = []
        try:
            for run in self.runs:
                chains.append(run.chain(calls, mode, flags))  # noqa: PERF401 (freed)
        except KomorebiError as error:
            for made in chains:
                made.free()
            raise self._error(error, len(chains)) from None
        return chain_module.IslandChain(chains, chains[0].group)

    def _submit_chain(self, chain, words):
        """An island chain as one ticket: a part an island it touches, each with its
        own island's records (the command's a7)."""
        self._grow(1)
        ticket = self._next
        for island, _first, _count in self.graph.split(chain.group):
            one = chain.chains[island]
            try:
                handle = self.runs[island].submit(
                    one.handler,
                    one.group,
                    one.args(words),
                    exclusive=True,
                    part=True,
                    span=one.span,
                )
            except KomorebiError as error:
                raise self._error(error, island) from None
            self._parts[island][ticket] = handle
        self._next += 1
        return ticket

    def submit(
        self, handler, group=None, args=(), exclusive=False, result=False, **words
    ):
        """Queues a command for `handler` on flat ranks [first, first + count)
        (default every rank): a part on each island it touches. Returns its ticket.
        A chain (run.chain): one ticket, `words` its per-submit word."""
        if isinstance(handler, chain_module.IslandChain):
            return self._submit_chain(handler, words)
        if words:
            raise TypeError(f"submit: unknown arguments {sorted(words)}")
        group = (0, self.cores) if group is None else tuple(group)
        parts = self.graph.split(group)
        _check_reduce_islands(self.graph, handler, group)
        before = self._first_use(handler, group)  # read before the command is queued
        self._grow(1)
        ticket = self._next
        for island, _first, _count in parts:
            try:  # the flat group word unchanged, marked a part: the dispatcher cuts
                # its part and does not cross the line (the other parts went as well)
                handle = self.runs[island].submit(
                    handler, group, args, exclusive=exclusive, result=result, part=True
                )
            except KomorebiError as error:
                raise self._error(error, island) from None
            self._parts[island][ticket] = handle
        self._next += 1
        if before is not None:  # earlier commands still out: counted with it
            label = _handler_label(handler)
            self._measured[ticket] = (label, group, before, self._out_before(ticket))
        return ticket

    def _first_use(self, handler, group):
        """The links' counts before a (handler, group)'s first command (a COUNTS
        command an island), or None: seen before, no link, crossing_line "off", or
        a run whose messages cross."""
        if self.crossing_line not in ("log", "error", "off"):
            raise ValueError(
                f"crossing_line={self.crossing_line!r}: 'log', 'error' or 'off'"
            )
        key = (_handler_label(handler), group)
        if self.crossing_line == "off" or self._counts is None:
            return None
        if key in self._crossed_seen:
            return None
        self._crossed_seen.add(key)
        return self.crossings()

    def _out_before(self, ticket):
        """The tickets before `ticket` whose events the host has not yet filed."""
        out = set()
        for island, run in enumerate(self.runs):
            handles = self._parts[island][1:ticket]
            rows = handles - run._base
            inside = (handles != 0) & (rows >= 0)  # below the base: filed and gone
            filed = np.zeros_like(inside)
            held = inside & (rows < run._has.size)
            filed[held] = run._has[rows[held]]
            late = inside & ~filed
            out.update((np.flatnonzero(late) + 1).tolist())
        return len(out)

    def _crossed(self, tickets, events=None):
        """After the events of first commands among `tickets` were seen done: the
        counts read again, a line each when pages, asks or puts crossed; under
        "error" raised here, the events on the error's `events`."""
        mine = [int(t) for t in tickets if int(t) in self._measured]
        if not mine:
            return
        after, raised = self.crossings(), []
        for ticket in mine:
            label, group, before, out = self._measured.pop(ticket)
            moved = (after - before).total()
            if not (moved["pages"] or moved["asks"] or moved["puts"]):
                continue
            others = out + self._next - ticket - 1  # before it, and queued after it
            many = "s" if others > 1 else ""
            flight = f" (with {others} other command{many} in flight)" if others else ""
            line = (
                f"{label} on ranks {group[0]}..{group[0] + group[1] - 1} moved "
                f"{moved['pages']} pages across the link (asks {moved['asks']}, puts "
                f"{moved['puts']}){flight}; if that is not meant, see its tensors' "
                "group="
            )
            if self.crossing_line == "error":
                raised.append(f"{line} (ticket {ticket} has run)")
            else:
                getattr(self.session, "log", print)(line)
        if raised:
            error = KomorebiError(
                f"{'; '.join(raised)} (crossing_line='error': its events on .events)"
            )
            error.events = events
            raise error

    def load(self, program, group):
        """Run.load over flat ranks: the cores of `group` hold `program`, one of the
        dict session.load took; a part on each island it touches. Its ticket."""
        group = tuple(group)
        parts = self.graph.split(group)
        for island, _first, _count in parts:  # every part refused before one is sent
            try:
                self.runs[island].load_field(program)
            except KomorebiError as error:
                raise self._error(error, island) from None
        self._grow(1)
        ticket = self._next
        for island, _first, _count in parts:
            try:
                handle = self.runs[island].load(program, group, part=True)
            except KomorebiError as error:
                raise self._error(error, island) from None
            self._parts[island][ticket] = handle
        self._next += 1
        return ticket

    def addon(self, program):
        """Run.addon(program) over islands: a class one more program of every island's
        run, all or none (added_program.add_islands), the same field and overlay numbers
        on each; its tensors made before the load or while the run lives (their table
        rows are written live). Returns its program field; run.load then loads it."""
        if not isinstance(program, Program):
            raise TypeError(
                'IslandRun.addon takes a class\'s program (form="runtime"); code added '
                "as overlay parts is one island's run.addon"
            )
        session = self.session
        if session is None:
            raise added.AddError(
                f"{program.name}: a card split into islands (one launch, its programs "
                "in the kernel form)"
            )
        session._hand_rectangles(program)
        try:
            return added_program.add_islands(
                list(self.runs), program, session._island_defines()
            )
        except KomorebiError as error:
            raise self._error(error, getattr(error, "added_island", 0)) from None

    @property
    def check_programs(self):
        """Whether the host refuses a command to cores not holding its program (every
        island's run); False: the dispatchers alone refuse."""
        return all(run.check_programs for run in self.runs)

    @check_programs.setter
    def check_programs(self, value):
        for run in self.runs:
            run.check_programs = value

    def forget(self, tickets):
        """Run.forget of every part of `tickets` on its island."""
        tickets = [tickets] if isinstance(tickets, int) else list(tickets)
        for island, run in enumerate(self.runs):
            handles = [int(self._parts[island][t]) for t in tickets]
            run.forget([handle for handle in handles if handle])

    def submit_many(self, handler, firsts, counts, args=None, exclusive=False):
        """n commands at once, as Run.submit_many takes them, the groups in flat
        ranks; each island gets its parts in one submit_many. The tickets, a range."""
        rows = () if args is None else args  # a class's words: their constants' handler
        handler = self.runs[0]._words_built_for(handler, rows, "submit_many")
        one = _one_command(handler, firsts, counts, args)
        if one is not None:  # one command: submit's Python numbers, not numpy's
            handler, group, words = one
            ticket = self.submit(handler, group, words, exclusive=exclusive)
            return range(ticket, ticket + 1)
        scalar = isinstance(handler, str) or np.ndim(handler) == 0
        handlers, first, count = np.broadcast_arrays(
            np.asarray(0 if scalar else handler),
            np.asarray(firsts, np.int64),
            np.asarray(counts, np.int64),
        )
        if args is not None:
            args = np.asarray(args)
            shape = np.broadcast_shapes(first.shape, (args.shape[0],))
            handlers, first, count = (
                np.broadcast_to(column, shape) for column in (handlers, first, count)
            )
        if first.ndim != 1:
            raise ValueError("handlers, firsts, counts and arguments: one row each")
        bad = (count < 1) | (first < 0) | (first + count > self.cores)
        if bad.any():
            i = int(np.argmax(bad))
            raise ValueError(
                f"group ({int(first[i])}, {int(count[i])}) of {self.cores} ranks"
            )
        rows = len(first)
        self._grow(rows)
        tickets = np.arange(self._next, self._next + rows)
        for island, run in enumerate(self.runs):
            low = np.maximum(first, self.graph.first(island))
            high = np.minimum(
                first + count, self.graph.first(island) + self.graph.count(island)
            )
            mine = high > low
            if not mine.any():
                continue
            named = handler if scalar else handlers[mine]
            try:  # the flat group words unchanged, marked parts (as in submit)
                handles = run.submit_many(
                    named,
                    first[mine],
                    count[mine],
                    None if args is None else args[mine],
                    exclusive,
                    part=True,
                )
            except KomorebiError as error:
                raise self._error(error, island) from None
            self._parts[island][tickets[mine]] = np.asarray(handles, np.int64)
        self._next += rows
        return range(int(tickets[0]), int(tickets[0]) + rows) if rows else range(0)

    def send(self):
        """Stores every island's queued commands on its card, so that the islands run
        their parts at once (a run over a link needs every part started): all handed
        first, then stored by one native call (every island's lines, one fence), then
        the events read meanwhile filed."""
        for island, run in enumerate(self.runs):
            try:
                run._hand()
            except KomorebiError as error:
                raise self._error(error, island) from None
        ports = [run.port for run in self.runs]
        together = getattr(type(ports[0]), "store_together", None)
        if together is not None and all(type(port) is type(ports[0]) for port in ports):
            # one native call: every island's lines, then one fence
            read = together(ports, [run._store_upto() for run in self.runs])
        else:
            read = []
            for island, run in enumerate(self.runs):
                try:
                    read.append(run._store_handed())
                except KomorebiError as error:
                    raise self._error(error, island) from None
        for island, (run, rows) in enumerate(zip(self.runs, read)):
            if len(rows):
                try:
                    run._take(rows)
                except KomorebiError as error:
                    raise self._error(error, island) from None

    def wait(self, tickets, timeout=10.0):
        """Sends what is queued and waits, within one `timeout` for all, for every
        part of the tickets. Returns per ticket the tuple of its parts' events in
        island order (one tuple, or a list for a list of tickets); the first holds the
        group's first rank, so its result word is the command's, as on one island."""
        single = isinstance(tickets, int | np.integer)
        wanted = np.asarray([tickets] if single else list(tickets), np.int64)
        if not wanted.size:
            return []
        if wanted.min() < 1 or wanted.max() >= self._next:
            raise ValueError("a ticket that was never submitted")
        deadline = time.monotonic() + timeout
        self.send()
        self._store(wanted, deadline)
        events = {}
        for island, run in enumerate(self.runs):
            handles = self._parts[island][wanted]
            mine = handles != 0
            if not mine.any():
                continue
            try:
                got = run.wait(handles[mine].tolist(), _left(deadline))
            except WaitTimeout as error:
                error.pending = self._tickets(island, error.pending)
                raise self._error(error, island) from None
            except KomorebiError as error:
                raise self._error(error, island) from None
            for ticket, event in zip(wanted[mine].tolist(), got):
                events.setdefault(ticket, []).append(event)
        out = [IslandEvent(events[ticket]) for ticket in wanted.tolist()]
        self._crossed(wanted, out[0] if single else out)
        return out[0] if single else out

    def _store(self, wanted, deadline):
        """Until the tickets' parts arrived or the deadline: one native round an island
        in turn (store what its ring has room for, file its events), so that every
        island is fed while the host waits; Run.wait then returns or raises."""
        pending = {}
        for island, run in enumerate(self.runs):  # what the stores filed is not waited
            handles = self._parts[island][wanted]
            handles = np.unique(handles[handles != 0])
            pending[island] = handles[~run._arrived(handles)]
        while any(handles.size for handles in pending.values()):
            if time.monotonic() > deadline:
                return
            for island, handles in pending.items():
                if not handles.size:
                    continue
                run = self.runs[island]
                try:
                    run._check_alive()
                    run._step(handles, int(handles[-1]), 0.0)
                except KomorebiError as error:
                    raise self._error(error, island) from None
                pending[island] = handles[~run._arrived(handles)]

    def inspect(self, ranks=None, timeout=10.0):
        """While the run goes on: what each rank of `ranks` (first, count; default every
        rank) is doing, as Run.inspect reads it, asked of each island's dispatcher in
        turn within one `timeout`; each core's dict also holds its `rank` and `island`.
        """
        group = (0, self.cores) if ranks is None else tuple(ranks)
        deadline = time.monotonic() + timeout
        out = []
        for island, first, count in self.graph.split(group):
            cores = self.graph.rank_cores[island][first : first + count]
            if cores[-1] - cores[0] + 1 != count:
                raise ValueError(
                    f"ranks {group}: not one run of island {island}'s cores"
                )
            try:
                read = self.runs[island].inspect(
                    (self.graph.first(island) + first, count), _left(deadline)
                )
            except KomorebiError as error:
                raise self._error(error, island) from None
            out.extend(self._ranked(island, read))
        return out

    def reset_core(self, rank, timeout=10.0):
        """Resets the core of flat rank `rank` and loads its kernels again while the
        other cores run, as Run.reset_core does on its island (by its dispatcher from
        its kept lists, else by the host): its result with `rank` and `island`.
        KomorebiError names the rank if the core did not come back."""
        island, local = self.graph.resolve(rank)
        core = self.graph.rank_cores[island][local]
        try:
            done = self.runs[island].reset_core(core, timeout)
        except KomorebiError as error:
            if error.args:
                where = f"rank {rank} (island {island}, core {core})"
                error.args = (f"{where}: {error.args[0]}", *error.args[1:])
            error.island = island
            raise
        return {**done, "rank": rank, "island": island}

    def reset_link(self, island, k, timeout=10.0):
        """Resets link core k of island `island` by the host and opens its connections
        again (the session's reset_link): {island, core, by, seconds, ...}."""
        if self.session is None:
            raise ValueError("a run without links has no link core to reset")
        done = self.session.reset_link(island, k, timeout)
        self._links_reset[island, k] = len(self.runs[island].link_events)
        return done

    @property
    def links_hung(self):
        """(island, link core) of each link core a give-up found hung (its dispatcher's
        KMD_EVENT_LINK) and not reset since: run.reset_link(island, k) recovers it."""
        out = []
        for island, run in enumerate(self.runs):
            for at, event in enumerate(run.link_events):
                key = (island, event["link"])
                if (
                    event["what"] in ("hung", "lost")
                    and at >= self._links_reset.get(key, 0)
                    and key not in out
                ):
                    out.append(key)
        return out

    def _far_cores(self, island, group):
        """At a time-out of island `island`'s run: every other island's part of flat
        `group` as its dispatcher's inspect of the deadline read it (a complete timeout
        of that part's cores), its cores ranked and marked `far`; [] where none is."""
        out = []
        for other, _first, _count in self.graph.split(tuple(group)):
            run = self.runs[other]
            local = run._local(group)
            if other == island or local[1] == 0 or not hasattr(run.backend, "inspect"):
                continue
            block = device_state.inspect(run.backend.inspect(local[1])["block"], run.h)
            if (
                block["complete"]
                and block["kind"] == "timeout"
                and (block["first"], block["count"]) == local
            ):
                out += [
                    {**core, "far": True}
                    for core in self._ranked(other, block["cores"])
                ]
        return out

    def _ranked(self, island, cores):
        """Core dicts of island `island` (its own core numbers in `core`) with their
        flat `rank` and `island`; a link core's rank is None, its `link` its index. A
        core of another island's part (`far`) is kept as it is."""
        mine, links = self.graph.rank_cores[island], self.graph.link_cores[island]
        first = self.graph.first(island)
        out = []
        for core in cores:
            if core.get("far"):
                out.append(core)
                continue
            named = {**core, "rank": None, "island": island}
            if core["core"] in mine:
                named["rank"] = first + mine.index(core["core"])
            elif core["core"] in links:
                named["link"] = links.index(core["core"])
            out.append(named)
        return out

    def _error(self, error, island):
        """`error` from island `island`'s run as this run's: named by its island, and a
        CommandTimeout by its ticket and the flat ranks of its cores."""
        if isinstance(error, CommandTimeout) and getattr(error, "island", None) is None:
            cores = self._ranked(island, error.cores)
            ranks = {
                core["core"]: core["rank"] for core in cores if not core.get("far")
            }
            tickets = self._tickets(island, [error.handle])
            CommandTimeout.__init__(
                error,
                tickets[0] if tickets else error.handle,
                error.slot,
                error.sequence,
                error.group,
                cores,
                error.result,
                error.given_up,
                [ranks.get(core, core) for core in error.recovered],
                ranks=True,
                links=error.links,
            )
        return _named(error, island)

    @property
    def recovery(self):
        """Each island's kept reset lists (Run.recovery), None where refused."""
        return [run.recovery for run in self.runs]

    @property
    def recoveries(self):
        """Every island's give-ups whose cores its dispatcher reset, island order:
        {island, ticket, handle, cores (its own core numbers), ranks, text}, the text
        that of the give-up, naming the flat ranks."""
        out = []
        for island, run in enumerate(self.runs):
            for entry in run.recoveries:
                tickets = self._tickets(island, [entry["handle"]])
                given_up = run.given_up(entry["handle"])
                text = f"island {island}: {entry['text']}"
                if given_up is not None:
                    text = str(self._error(given_up, island))
                ranked = self._ranked(island, [{"core": c} for c in entry["cores"]])
                out.append(
                    {"island": island, "ticket": next(iter(tickets), None)}
                    | entry
                    | {"ranks": [core["rank"] for core in ranked], "text": text}
                )
        return out

    def _tickets(self, island, handles):
        """The tickets whose part on `island` has one of these handles."""
        parts = self._parts[island][: self._next]
        return [int(t) for t in np.flatnonzero(np.isin(parts, handles)) if t]

    def drain(self, timeout=10.0):
        """Sends everything queued and waits, within one `timeout`, for every
        island's events; returns how many events it waited for, every island's
        together. `drain_given_up`: the parts of them given up, every island's."""
        deadline = time.monotonic() + timeout
        self.send()
        waited = 0
        for island, run in enumerate(self.runs):
            try:
                waited += run.drain(_left(deadline))
            except KomorebiError as error:
                raise self._error(error, island) from None
        self._crossed(list(self._measured))  # every command done: its events by event()
        return waited

    @property
    def drain_given_up(self):
        return sum(run.drain_given_up for run in self.runs)

    def given_up(self, ticket):
        """The CommandTimeout of a ticket's first given-up part, naming its island and
        the wait each rank sat in (event() after drain()), or None."""
        for island, run in enumerate(self.runs):
            part = int(self._parts[island][ticket])
            error = run.given_up(part) if part else None
            if error is not None:
                return self._error(error, island)
        return None

    def event(self, ticket):
        """The parts' events of a ticket (None for a part not arrived), island order."""
        return tuple(
            self.runs[island].event(int(self._parts[island][ticket]))
            for island in range(len(self.runs))
            if self._parts[island][ticket]
        )

    @property
    def stopped(self):
        return all(run.stopped for run in self.runs)

    def stop(self, timeout=10.0):
        """Stops every island's run within one `timeout`, each even if another
        failed; the first failure is raised, naming its island. The results, a list."""
        deadline = time.monotonic() + timeout
        failure = None
        for island, run in enumerate(self.runs):
            try:
                run.stop(_left(deadline))
            except BaseException as error:  # noqa: BLE001 (raised after the others)
                if failure is None:
                    failure = self._error(error, island)
        if failure is not None:
            raise failure
        return self.result()

    def abort(self):
        for run in self.runs:
            run.abort()

    def result(self):
        """Every island's result block, in island order; valid after the stop."""
        return [run.result() for run in self.runs]

    def trace(self):
        """The per-command trace over islands (IslandTrace), or None if the run was not
        traced; valid after the stop."""
        traces = [run.trace() for run in self.runs]
        return None if None in traces else IslandTrace(traces, self._parts)

    def state(self):
        """Every rank's state words, as Run.state gives them, in flat rank order, each
        with its `rank` and `island`."""
        states = [run.state() for run in self.runs]
        return [
            {
                **states[island][core],
                "rank": self.graph.first(island) + i,
                "island": island,
            }
            for island, cores in enumerate(self.graph.rank_cores)
            for i, core in enumerate(cores)
        ]

    def crossings(self):
        """What crossed each link since the run's start, by link and direction, read
        now from the link cores (a Crossings; none on one island). The difference of
        two reads, `after - before`, is what crossed between them."""
        return Crossings({}) if self._counts is None else self._counts()

    def all_gather(self, tensor, first=0, pages=None, timeout=10.0):
        """All-gather of pages first .. first + pages - 1 (default to the end) of the
        copy `tensor`, island c's share (layout.partition of the pages, c of the
        islands) written on island c: on return every island's copy holds every
        share. One island: returns at once. km::all_gather on every rank (GATHER)."""
        first, pages = self._collective(tensor, first, pages)
        if len(self.runs) < 2 or pages == 0:
            return None
        h = abi.headers()
        words = [0] * 7
        words[h["KM_COLL_TENSOR"]] = tensor.address
        words[h["KM_COLL_FIRST"]] = first
        words[h["KM_COLL_PAGES"]] = pages
        return self._served(h["KM_HANDLER_GATHER"], [words] * len(self.runs), timeout)

    def all_reduce(self, tensor, scratch, first=0, pages=None, timeout=10.0, tree=None):
        """All-reduce of pages first .. first + pages - 1 of the copy `tensor` (each
        island's copy its partial): every copy then holds the sum, in its format.
        `scratch` and `tree` as reduce_to's (a tree: every page up it, then down). One
        island: returns at once; else REDUCE on every rank (km::all_reduce)."""
        first, pages = self._collective(tensor, first, pages)
        kinds = {"int32": 0, "uint32": 0, "float32": 1, "bfloat16": 2}
        if tensor.dtype not in kinds:
            raise KomorebiError(
                f"all_reduce adds int32, float32, bfloat16: not {tensor.dtype}"
            )
        if len(self.runs) < 2 or pages == 0:
            return None
        self._collective(scratch, 0, self._ports() * pages)
        if scratch.aligned_page_bytes != tensor.aligned_page_bytes:
            raise KomorebiError("all_reduce: the scratch's pages differ in size")
        transfers = [run.backend.transfer for run in self.runs]
        if any(t is None or t.chunk < tensor.aligned_page_bytes for t in transfers):
            raise KomorebiError("all_reduce: the run has no transfer buffer of a page")
        h = abi.headers()
        words = [0] * 10
        words[h["KM_COLL_TENSOR"]] = tensor.address
        words[h["KM_COLL_FIRST"]] = first
        words[h["KM_COLL_PAGES"]] = pages
        words[h["KM_COLL_SCRATCH"]] = scratch.address
        words[h["KM_COLL_SUM"]] = kinds[tensor.dtype]
        self._tree(words, tree)
        by_island = []
        for transfer in transfers:  # the folds' L1: each island's run's own
            words[h["KM_COLL_BUFFER"]] = transfer.address
            words[h["KM_COLL_ROOM"]] = 2 * transfer.chunk
            by_island.append(list(words))
        return self._served(h["KM_HANDLER_REDUCE"], by_island, timeout)

    def _collective(self, tensor, first, pages):
        """(first, pages) of a collective on `tensor`, checked: a copy, the pages in
        it, the islands joined (the link services)."""
        pages = tensor.pages - first if pages is None else pages
        if not getattr(tensor.placement, "copy", False):
            raise KomorebiError("a collective's tensor is a copy (copy=True)")
        if first < 0 or pages < 0 or first + pages > tensor.pages:
            raise KomorebiError(
                f"pages {first} + {pages}: the tensor has {tensor.pages}"
            )
        if len(self.runs) > 1 and self._counts is None:
            raise KomorebiError("a collective crosses links: the session has none")
        return first, pages

    def _served(self, handler, words, timeout):
        """A command of the runtime's service overlay on every rank, a part on each
        island (as submit) with that island's `words`, waited for; its events, island
        order."""
        group = (0, self.cores)
        self._grow(1)
        ticket = self._next
        for island, _first, _count in self.graph.split(group):
            run = self.runs[island]
            try:
                handle = run._queue_command(
                    handler, run._group(group), words[island], False, run._flag_part
                )
            except KomorebiError as error:
                raise self._error(error, island) from None
            self._parts[island][ticket] = handle
        self._next += 1
        return self.wait(ticket, timeout)


def check_link_queues(places):
    """Refuses islands whose link cores' put queues (offset from their buffers' base,
    chunk bytes) differ: a sender writes its lines' arrival at its own queue's place."""
    seen = sorted({place for place in places if place is not None})
    if len(seen) > 1:
        raise KomorebiError(
            "the islands' link cores put queues at different places (offset, chunk "
            f"bytes): {seen}: a put line across would never be confirmed"
        )


class IslandSession:
    """Cards as islands of one 1 x n fabric mesh, the X280 runtime once over it, a
    Session an island (one card: its own; `islands` 2 to 4: it split into rows). Two
    cards use every link found (`links`: at most that many), a link core a pair. A
    context manager. `host_reaches`: see the islands reference (a test)."""

    LINKS_MAX = 4  # a byte each in a PAGES command's link word
    # How an ask for another island's pages reaches the holder: "core" (the read call
    # puts an ask line into link core 0), "host" (the host queues the holder's send).
    ASKS_BY = "core"

    def __init__(
        self,
        cards,
        links=None,
        islands=None,
        pll_mhz=None,
        tree=None,
        log=None,
        grid_timeout=10.0,
        kernel_bytes=None,
        channel=channel_module.REGION_DEFAULT,
        slow_code=False,
        host_reaches=None,
        card_dram=False,
        line=False,
        packet=None,
        shape=None,
        expect=None,
        tensors=None,
        tensor_cache=None,
    ):
        self.cards = tuple(int(card) for card in cards)
        if not self.cards or len(set(self.cards)) != len(self.cards):
            raise ValueError(f"cards {cards}: one or more different cards")
        if links and not 1 <= links <= self.LINKS_MAX:
            raise ValueError(
                f"links={links}: None (every link found), or 1 to {self.LINKS_MAX} "
                "between two islands"
            )
        self.split = islands or 1  # the islands one card is split into
        tensors = TENSORS_MAX if tensors is None else tensors
        if not 1 <= tensors <= TENSORS_MOST:
            raise ValueError(
                f"tensors={tensors}: 1 to {TENSORS_MOST}, the tensor table's entries "
                "(a word below it names an entry, any other a bank address); each "
                "entry takes L1 on every core, which the load's budget bounds"
            )
        # the table cached a RISC (K rows) or flat in L1; None: the smaller of the two
        if tensor_cache not in (None, False) and (
            isinstance(tensor_cache, bool)
            or not 4 <= tensor_cache <= 1024
            or tensor_cache % 4
        ):
            raise ValueError(
                f"tensor_cache={tensor_cache!r}: None (the smaller in L1), False (the "
                "table flat in L1) or K rows a RISC, a multiple of 4 from 4 to 1024"
            )
        self.tensor_cache = tensor_cache
        # a split card: tensors held by island (pages asked across, as over cards);
        # card_dram: one tensor in the card's DRAM, every page local to every island
        self.card_dram = bool(card_dram) and self.split > 1
        # a split card as a line: only neighbouring islands joined, the others
        # reached through those between (forwarded)
        self.line = bool(line)
        if self.line and self.split < 2:
            raise ValueError("line=True: a card split into islands")
        # its shape: "mesh" (each joined to each), "line", "ring" or "grid" (2 x 2)
        self.shape = shape or ("line" if self.line else "mesh")
        if self.shape not in SPLIT_SHAPES or (shape and self.split < 2):
            raise ValueError(
                f"shape={shape!r}: a card split into islands, one of {SPLIT_SHAPES}"
            )
        if self.line and self.shape != "line":
            raise ValueError(f"line=True and shape={shape!r}")
        self.line = self.shape == "line"
        if self.split == 1 and len(self.cards) > 2:
            raise ValueError(
                f"cards {self.cards}: a session wires one card or two; "
                f"{len(self.cards)} cards are not wired yet"
            )
        if host_reaches is not None and (
            len(self.cards) != 2 or not set(host_reaches) < {0, 1} or not host_reaches
        ):
            raise ValueError(
                f"host_reaches={host_reaches}: one island of a session over two cards"
            )
        self.host_reaches = None if host_reaches is None else tuple(host_reaches)
        self.packet = PACKET_BYTES if packet is None else int(packet)
        largest = abi.headers()["KM_PACKET_LARGEST"]
        if not LINK_PACKET_BYTES <= self.packet <= largest or self.packet % 64 != 0:
            raise ValueError(
                f"packet={packet}: {LINK_PACKET_BYTES} to {largest} bytes (a message "
                "packet's to the routers' largest), whole 64-byte lines"
            )
        if self.split == 1 and links and len(self.cards) != 2:
            raise ValueError(f"links={links}: between two cards, or a split card")
        if self.split > 1 and len(self.cards) != 1:
            raise ValueError(f"islands={islands}: one card")
        # two cards: every link found, at most `links` (set at the open), link 0
        # always (commands ride it); a split card: `links` wires a pair (default 1)
        self._links_asked = links or None
        self.links = (
            1 if len(self.cards) == 2 else (links or 1) if self.split > 1 else 0
        )
        self.log = log or (lambda message: None)
        self._options = {
            "pll_mhz": pll_mhz,
            "tree": tree,
            "grid_timeout": grid_timeout,
            "channel": channel,
            "slow_code": slow_code,
        }
        self._memory = None  # the remembered kernel buffer (komorebi.remembered)
        if kernel_bytes is None and (len(self.cards) == 2 or self.split > 1):
            kernel_bytes = self._remembered(expect)  # the page links' code
        self.kernel_bytes = kernel_bytes
        self._worker_l1 = session_module.worker_l1_size(kernel_bytes)
        self.tree = tree
        self.mesh = None
        self.chips = None
        self.sessions = ()
        self.graph = None
        self.run = None
        self.link_cores = ()  # per island: the local cores (core table) of its links
        self.channels = ()  # per island: the Ethernet channel of each link it sends on
        self._entries = [None] * tensors  # the tensor table, by word (`tensors`)
        self._tables = []  # per island: its cores' tensor table (L1), from a load
        self._drams = []  # per island: the table in DRAM, when its cores cache rows
        self._table_size = 0  # its entries

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, kind, error, traceback):
        try:
            if self.run is not None and not self.run.stopped:
                if kind is None:
                    self.run.stop()
                else:
                    self.run.abort()
        finally:
            self.close()

    def open(self):
        """Opens the mesh (the fabric set first), the X280 runtime over it, then a
        Session on each island's device; the graph of the islands' ranks. One card:
        its Session opens the card alone, the graph is one island."""
        if len(self.cards) == 1:
            session = session_module.Session(
                card=self.cards[0],
                log=self.log,
                kernel_bytes=self.kernel_bytes,
                **self._options,
            )
            session.open()
            session.packet = self.packet
            self.chips, self.sessions = [self.cards[0]], (session,) * self.split
            columns, rows = session.grid
            self.graph = islands.one(columns * rows, columns)
            if self.split > 1:  # one device, a dispatcher an island, wires between
                self.graph = _split_graph(
                    columns, rows, self.split, self.links, self.shape
                )
                check_wired(self.graph, self.split)
                self.link_cores = self.graph.link_cores
                self._forward_packet()
                self.channels = ((0,) * self.links,) * self.split  # no router: a wire
                session_module._LIVE.add(self)
                shape = SPLIT_SHAPES[self.shape]
                self.log(f"card {self.cards[0]} as {self.split} islands, {shape}")
            return
        listed = ",".join(str(card) for card in self.cards)
        visible = os.environ.get("TT_VISIBLE_DEVICES")
        if visible is not None and visible != listed:
            raise ValueError(f"cards {listed} are not the visible ones ({visible})")
        os.environ["TT_VISIBLE_DEVICES"] = listed
        try:
            self._open_mesh(listed)
        except BaseException:
            if visible is None:  # set by this open: unset again when it fails
                os.environ.pop("TT_VISIBLE_DEVICES", None)
            raise

    def _open_mesh(self, listed):
        """open() over two or more cards, TT_VISIBLE_DEVICES set: the mesh, the X280
        runtime and a Session on each island's device."""
        os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
        if session_module.exit_timeout(os.environ, self.log) == 0:
            self.log(
                f"{session_module.OP_TIMEOUT}=0: an exit after a grid that did not "
                "return hangs"
            )
        for card in self.cards:
            holder = session_module.chip_holder(card)
            if holder is not None:
                raise CardBusy(
                    f"card {card} is held by process {holder} "
                    f"({session_module._command(holder)}): an open would wait until "
                    "that process exits",
                    pid=holder,
                )
        session_module.refuse_reopen("mesh", self.cards)
        module = native.load_module(self.tree)
        self.log(f"opening cards {listed} as one 1 x {len(self.cards)} fabric mesh")
        self.log(f"routers told: packets of {self.packet} bytes")
        self.mesh = module.Islands(len(self.cards), self._worker_l1, self.packet)
        session_module._hook_exit()
        session_module._LIVE.add(self)
        try:
            self.chips = list(self.mesh.chips())
            if sorted(self.chips) != sorted(self.cards):
                raise KomorebiError(
                    f"the mesh's chips {self.chips} are not the cards {listed}"
                )
            if not prebuild.active():  # a mock has no X280
                self.mesh.open_x280()
            self.log(f"X280 runtime open over the mesh (islands: chips {self.chips})")
            self.sessions = tuple(
                session_module.Session(
                    card=chip,
                    device=self.mesh.device(island),
                    log=self._island_log(island),
                    **self._options,
                )
                for island, chip in enumerate(self.chips)
            )
            for island, session in enumerate(self.sessions):
                session.transfer_chunk = LINK_CHUNK
                session.l1_held = lambda island=island: self._held_l1(island)
                session.packet = self.packet
            if len(self.chips) == 2:
                found = int(self.mesh.link(0, 1, 0)[0])
                most = min(found, self.LINKS_MAX)
                self.links = min(self._links_asked or most, most)
                self.log(f"links: {self.links} of {found} found")
                for session in self.sessions:  # a relay ask's link: its rank's modulo
                    session.relay_links = self.links
            parts, taken, link_cores = [], (self.links + 1) // 2, []
            for session in self.sessions:
                columns, rows = session.grid
                cores = columns * rows
                parts.append(islands.Island(tuple(range(cores)), columns, THREADS))
                link_cores.append(tuple(range(cores - taken, cores)))
            links = [
                islands.Link(i, i + 1, max(self.links, 1))
                for i in range(len(parts) - 1)
            ]
            self.graph = islands.Graph(parts, links, link_cores=link_cores)
            self.link_cores = self.graph.link_cores
            self._forward_packet()
            if self.links:
                self._open_links(module)
        except BaseException:
            self.close()
            raise

    def _forward_packet(self):
        """More than two islands: the kernels' packets no larger than a forward slot's
        payload (KM_FORWARD_PAYLOAD; the routers may carry larger ones)."""
        most = abi.headers()["KM_FORWARD_PAYLOAD"]
        if len(self.graph.islands) > 2 and self.packet > most:
            self.log(f"packets of {most} bytes: a forward slot's, over the islands")
            self.packet = most
            for session in self.sessions:
                session.packet = most

    def _open_links(self, module):
        """The Ethernet channel of each link toward the other island, and the check
        that a page's DRAM bank sits at the same NoC place on every island."""
        channels = []
        for island in range(2):
            found = [self.mesh.link(island, 1 - island, k) for k in range(self.links)]
            if found[0][0] < self.links:
                raise KomorebiError(
                    f"island {island}: {found[0][0]} links found, {self.links} asked"
                )
            channels.append(tuple(int(link[1]) for link in found))
        self.channels = tuple(channels)
        banks = []
        for session in self.sessions:
            count = module.dram_banks(session.device)
            banks.append([tuple(session.device.dram_core(b)) for b in range(count)])
        if any(other != banks[0] for other in banks[1:]):
            raise KomorebiError(
                "the islands' DRAM banks sit at different NoC places: a page sent "
                "into the other island's DRAM would land in another bank"
            )
        self.log(f"page links: channels {self.channels}, link cores {self.link_cores}")

    def _island_log(self, island):
        return lambda message: self.log(f"island {island}: {message}")

    @property
    def ranks(self):
        return self.graph.ranks

    @property
    def grid(self):
        """Each island's grid (columns, rows), the same on every island: a class's
        rectangle is one island's (docs/reference/islands.md#classes-on-islands)."""
        grids = {tuple(session.grid) for session in self.sessions}
        if len(grids) != 1:
            raise KomorebiError(f"the islands' grids differ: {sorted(grids)}")
        return grids.pop()

    def island(self, island):
        """Island `island`'s own Session (local ranks, its own tensors)."""
        return self.sessions[island]

    def tensor(
        self, data, dtype, layout="row", split=None, group=None, like=None, copy=False
    ):
        """A DRAM tensor holding `data` as Session.tensor takes it, placed by the hint
        (empty() names them); each island's part written through its own host
        channel."""
        raw, shape = formats.to_bytes(data, dtype, layout)
        tensor = self.empty(shape, dtype, layout, split, group, like, copy)
        try:
            self.write(tensor, raw.reshape(tensor.pages, tensor.page_bytes))
        except BaseException:
            tensor.free()
            raise
        return tensor

    def empty(
        self, shape, dtype, layout="row", split=None, group=None, like=None, copy=False
    ):
        """A tensor placed by the hint, not written. `split` d over `group` (first
        rank, ranks; default every rank): the group's ranks partition dimension d as
        km::partition does and each rank's part is held by its island; `like`: as
        tensor `like` is placed; `copy`: on every island. No hint: split 0."""
        if like is not None:
            if split is not None or group is not None or copy:
                raise ValueError("`like` takes the other tensor's hint: no other hint")
            hint = like.placement
            split, group, copy = hint.split, hint.group, hint.copy
        placement = islands.place(self.graph, shape, layout, split, group, copy)
        if self.card_dram:  # one tensor in the card's DRAM, every page local
            part = self.sessions[0].empty(shape, dtype, layout)
            tensor = IslandTensor(
                [part] * self.split, placement, shape, dtype, layout, part.address
            )
            tensor.session = self
            return tensor
        word = None
        if len(self.sessions) > 1:
            free = [i for i, held in enumerate(self._entries) if held is None]
            if not free:
                raise KomorebiError(
                    f"{len(self._entries)} tensors at once: the tensor table is full "
                    "(free one first, or open with a larger tensors=)"
                )
            word = free[0]
        parts = []
        try:
            for island, session in enumerate(self.sessions):
                parts.append(
                    self._part(session, island, placement, shape, dtype, layout)
                )
        except BaseException:
            for part in parts:
                if part is not None:
                    part.free()
            raise
        tensor = IslandTensor(
            parts, placement, shape, dtype, layout, word, self._release_entry
        )
        tensor.session = self
        if word is None:
            tensor.address = parts[0].address  # one island: its bank address
        else:
            if any(
                0 < part.address < len(self._entries) for part in tensor.parts if part
            ):
                raise KomorebiError("a bank address below the tensor table's size")
            self._entries[word] = tensor
            try:
                self._table_row(word)
            except BaseException:
                tensor.free()
                raise
        return tensor

    def bank_range(self, island, bank, address, size):
        """A range of one DRAM bank as kernels on any island name it (km::BankRange):
        `size` bytes of bank `bank` of island `island` from bank address `address`.
        Takes an entry of the tensor table as a tensor does; free() frees it."""
        if not 0 <= island < len(self.sessions):
            raise ValueError(f"island {island} of {len(self.sessions)}")
        if not 0 <= bank < islands.DRAM_BANKS:
            raise ValueError(
                f"bank {bank}: the banks are 0 to {islands.DRAM_BANKS - 1}"
            )
        if (
            address % 64
            or address < len(self._entries)
            or size <= 0
            or address + size > 1 << 32
        ):
            raise ValueError(
                f"{size} bytes at bank address {address:#x}: whole 64-byte lines from "
                f"{len(self._entries)} on, inside the bank's 4 GiB"
            )
        if len(self.sessions) == 1 or self.card_dram:  # every page here: no entry
            return BankRange(island, bank, address, size, address)
        free = [i for i, held in enumerate(self._entries) if held is None]
        if not free:
            raise KomorebiError(
                "the tensor table has no free entry for a bank range (free a tensor "
                "first, or make it before the load)"
            )
        held = BankRange(
            island,
            bank,
            address,
            size,
            free[0],
            len(self.sessions),
            self._release_entry,
        )
        self._entries[held.address] = held
        try:
            self._table_row(held.address)
        except BaseException:
            held.free()
            raise
        return held

    def _part(self, session, island, placement, shape, dtype, layout):
        """Island `island`'s buffer of its part (None: it holds no page): the whole
        shape when it holds every page, else its pages with the same page bytes. A
        part that does not fit raises with the most pages that would."""
        pages = placement.part_pages(island)
        if pages == 0:
            return None
        whole = pages == placement.pages
        if whole:
            mine = shape
        elif layout == "tile":
            mine = (pages * formats.TILE, formats.TILE)
        else:
            mine = (pages, shape[-1])
        try:
            return session.empty(mine, dtype, layout)
        except KomorebiError:
            raise
        except RuntimeError as error:  # the allocator's: the DRAM is full
            fits = self._fits(session, mine, pages, dtype, layout)
            lines = str(error).splitlines() or [""]  # its sentence, not its backtrace
            said = next((line for line in lines if "Out of Memory" in line), lines[0])
            raise KomorebiError(
                f"island {island}: its part of {pages} pages of shape {tuple(shape)} "
                f"does not fit its DRAM ({said}); at most {fits} pages of this kind "
                "fit there now: split it over more islands' ranks (`group`) or free "
                "tensors"
            ) from None

    @staticmethod
    def _fits(session, shape, pages, dtype, layout):
        """The most pages of this kind an island's DRAM takes now (found by trying)."""
        low, high = 0, pages
        while low < high:
            middle = (low + high + 1) // 2
            trial = (
                (middle * formats.TILE, formats.TILE)
                if layout == "tile"
                else (middle, shape[-1])
            )
            try:
                session.empty(trial, dtype, layout).free()
                low = middle
            except RuntimeError:
                high = middle - 1
        return low

    def _table_row(self, word):
        """Entry `word` into every island's table: by the run's cores while a run
        lives (_table_row_live), else by the host (a load writes it otherwise)."""
        if self._tables and self.run is not None and not self.run.stopped:
            self._table_row_live(word)
        else:
            self._write_table()

    def _table_row_live(self, word, timeout=10.0):
        """Entry `word` written by every island's ranks (ENTRY with KM_ENTRY_TABLE): the
        row in a page of each island's DRAM at its L1 place's offset mod 64, one
        command an island, all handed before any is waited for."""
        h = abi.headers()
        size = islands.tensor_entry_bytes(len(self.sessions))
        row = np.frombuffer(self._entries[word].entry(), np.uint8)
        scratches, handles = [], []
        try:
            for island, run in enumerate(self.run.runs):
                if self._drams:  # the DRAM row written; the ranks forget their copies
                    page = np.zeros((1, self._cache_row()), np.uint8)
                    page[0, :size] = row
                    self.sessions[island].write(self._drams[island], page, first=word)
                    address = 0
                else:
                    offset = (self._tables[island].address + word * size) % 64
                    page = np.zeros(128, np.uint8)
                    page[offset : offset + size] = row
                    scratch = self.sessions[island].tensor(
                        page.view(np.uint32)[None], "uint32"
                    )
                    scratches.append(scratch)
                    address = scratch.address + offset
                part = run._flag_part if run.island_ranks != run.cores else 0
                words = (h["KM_ENTRY_TABLE"], word, address)
                group = (run.island_first, run.island_ranks)
                handles.append(
                    run._queue_command(h["KM_HANDLER_ENTRY"], group, words, False, part)
                )
            for run in self.run.runs:
                run._hand()
            for island, (run, handle) in enumerate(zip(self.run.runs, handles)):
                event = run.wait(handle, timeout)
                if event.kind != "done":
                    raise KomorebiError(
                        f"island {island}: the table row of tensor word {word} ended "
                        f"{event.kind}: {event}"
                    )
        finally:
            for scratch in scratches:
                scratch.free()

    def _release_entry(self, tensor):
        """A freed tensor's entry of the tensor table is free again."""
        if (
            tensor.address < len(self._entries)
            and self._entries[tensor.address] is tensor
        ):
            self._entries[tensor.address] = None

    def _table_entries(self, extra=0):
        """The entries of a load's tensor table: every word (`tensors`), so a tensor
        made while the run lives has its row; `extra` is kept for callers."""
        del extra
        return len(self._entries)

    def _table_rows(self):
        """The tensor table every core of an island holds: an entry a word, in word
        order ([entries x entry bytes / 4] uint32, the table's entries)."""
        size = islands.tensor_entry_bytes(len(self.sessions))
        table = bytearray(self._table_size * size)
        for word, tensor in enumerate(self._entries[: self._table_size]):
            if tensor is not None:
                table[word * size : (word + 1) * size] = tensor.entry()
        return np.frombuffer(bytes(table), np.uint32)

    def _cache_k(self):
        """The rows a RISC caches: tensor_cache's K, else KM_TABLE_CACHE_ENTRIES."""
        given = self.tensor_cache
        return given if given else abi.headers()["KM_TABLE_CACHE_ENTRIES"]

    def _cached(self):
        """Whether the table lives in each island's DRAM, its cores caching rows: as
        tensor_cache says, else where that takes less L1 than the table flat."""
        if self.tensor_cache is not None:
            return bool(self.tensor_cache)
        flat = len(self._entries) * islands.tensor_entry_bytes(len(self.sessions))
        return self._cache_bytes() < flat

    def _cache_bytes(self):
        """A core's L1 for a cached table (km_table_cache.hpp): the head, then the
        reader's and the writer's caches (tags, last uses, clock, misses; the rows)."""
        k, row = self._cache_k(), self._cache_row()
        head = -(-((2 * k + 2) * 4) // 64) * 64
        return abi.headers()["KM_TABLE_HEAD_BYTES"] + 2 * (head + k * row)

    def table_misses(self):
        """A cached table's misses since the load, per island: [cores, 2] (each core's
        reader, writer), read from the cores' L1 after run.stop(); None if flat."""
        if not self._drams:
            return None
        if self.run is not None and not self.run.stopped:
            raise KomorebiError(
                "table_misses: after run.stop() (a live run's L1 is read through its "
                "own transfer handler, which a hand program lacks)"
            )
        k = self._cache_k()
        head = -(-((2 * k + 2) * 4) // 64) * 64
        part = (head + k * self._cache_row()) // 4
        first = abi.headers()["KM_TABLE_HEAD_BYTES"] // 4 + 2 * k + 1
        out = []
        for session, table in zip(self.sessions, self._tables):
            (raw,) = session.read(table)
            words = np.ascontiguousarray(raw).view("<u4")
            out.append(words[:, [first, first + part]].astype(np.int64))
        return out

    def _cache_row(self):
        return -(-islands.tensor_entry_bytes(len(self.sessions)) // 64) * 64

    def _dram_rows(self):
        """The cached table's rows in DRAM: an entry a page of whole 64 B lines."""
        size = islands.tensor_entry_bytes(len(self.sessions))
        rows = np.zeros((self._table_size, self._cache_row()), np.uint8)
        rows[:, :size] = self._table_rows().view(np.uint8).reshape(-1, size)
        return rows

    def _l1_row(self, island):
        """What every core of `island` holds at the table's place: the table, or a
        cached table's head (word 0: its DRAM address) and empty caches."""
        if not self._drams:
            return self._table_rows()
        row = np.zeros(self._cache_bytes() // 4, np.uint32)
        row[0] = self._drams[island].address
        return row

    def _write_table(self):
        """Writes the tensor table into every core of every island, when a run holds
        it (a load writes it otherwise); a cached one into DRAM, its caches emptied."""
        if not self._tables:
            return
        pairs = {id(table): i for i, table in enumerate(self._tables)}
        for island in pairs.values():  # a split card: one table
            session, table = self.sessions[island], self._tables[island]
            if self._drams:
                session.write(self._drams[island], self._dram_rows())
            row = self._l1_row(island)
            session.write(table, np.tile(row, (table.pages, 1)).view(np.uint8))

    def _tables_for_load(self):
        """Each island's tensor table in the L1 of all its cores, made at the first
        load and written: its address for Session.load (several islands; else 0)."""
        if len(self.sessions) == 1:
            return [0]
        if abi.headers()["KM_TENSORS_MAX"] != islands.TENSORS_MAX:
            raise KomorebiError("islands.TENSORS_MAX differs from km_abi.h's")
        if self._tables and self._table_size < self._table_entries():
            for table in {id(t): t for t in [*self._tables, *self._drams]}.values():
                table.free()  # the load's tensors need more entries: made again
            self._tables, self._drams = [], []
        if not self._tables:
            self._table_size = self._table_entries()
            own = self.sessions[: 1 if self.split > 1 else None]
            if self._cached():
                rows = self._dram_rows().view(np.uint32)
                self._drams = [session.tensor(rows, "uint32") for session in own]
                self._drams *= len(self.sessions) // len(self._drams)
            for island, session in enumerate(own):
                columns, rows = session.grid
                self._tables.append(
                    session.tensor(
                        np.tile(self._l1_row(island), (columns * rows, 1)),
                        "uint32",
                        shard_grid=(columns, rows),
                    )
                )
            self._tables *= len(self.sessions) // len(self._tables)  # split: one
        else:
            self._write_table()
        return [table.address for table in self._tables]

    def read(self, *tensors, first=0, pages=None):
        """The raw pages of island tensors as Session.read gives them: each island's
        pages through its own host channel (with a run loaded, the islands at once),
        gathered into one array."""
        if (first or pages is not None) and len(tensors) != 1:
            raise ValueError("a page range is read from one tensor")
        if self.card_dram:  # one tensor in the card's DRAM
            parts = [tensor.parts[0] for tensor in tensors]
            return self.sessions[0].read(*parts, first=first, pages=pages)
        out = []
        for tensor in tensors:
            start, count = dram.page_range(first, pages, tensor.pages)
            rows = np.empty((count, tensor.page_bytes), np.uint8)

            def gather(island, at, low, high, tensor=tensor, rows=rows, start=start):
                part = tensor.parts[island]
                mine = _rows(at[low:high] - start)
                if isinstance(mine, slice):  # straight into the rows: no second copy
                    self.sessions[island].read(
                        part, first=low, pages=high - low, out=rows[mine]
                    )
                    return
                (got,) = self.sessions[island].read(part, first=low, pages=high - low)
                rows[mine] = got

            spans = self._spans(tensor, start, count, reading=True)
            direct, far = self._reached(spans)
            self._each_island(gather, direct)
            for island, at, low, high in far:
                rows[_rows(at[low:high] - start)] = self._read_across(
                    tensor.parts[island], island, low, high - low
                )
            out.append(rows)
        return out

    def _reached(self, spans):
        """The spans whose island the host reaches through its own channel, and the
        others (host_reaches)."""
        if self.host_reaches is None:
            return spans, []
        direct = [span for span in spans if span[0] in self.host_reaches]
        return direct, [span for span in spans if span[0] not in self.host_reaches]

    def _staging(self, part, island):
        """A buffer like `part` on the island the host reaches, for a part of island
        `island` crossing the page path; KomorebiError without a run with links."""
        if self.run is None or self.run.stopped or not self.links:
            raise KomorebiError(
                f"host_reaches={list(self.host_reaches)}: island {island}'s part "
                "crosses by the page path, which runs in a loaded run with links: "
                "write or read it after the load"
            )
        via = self.host_reaches[0]
        return via, self.sessions[via].empty(part.shape, part.dtype, part.layout)

    def _read_across(self, part, island, first, pages, timeout=30.0):
        """Pages first .. of island `island`'s part, sent by its link cores into a
        buffer of the island the host reaches, then read through that one's channel."""
        via, staging = self._staging(part, island)
        try:
            sent = self._send(
                island, via, part.address, part.aligned_page_bytes, first, pages,
                staging.address,
            )  # fmt: skip
            landed = self.wait_pages(via, self._across(island, via), pages)
            self.wait_together([sent, (via, landed)], timeout)
            (got,) = self.sessions[via].read(staging, first=first, pages=pages)
        finally:
            staging.free()
        return got

    def _write_across(self, part, island, first, rows, timeout=30.0):
        """`rows` into pages first .. of island `island`'s part: written through the
        reached island's channel into a buffer there, sent by its link cores."""
        via, staging = self._staging(part, island)
        try:
            self.sessions[via].write(staging, rows, first)
            sent = self._send(
                via, island, staging.address, part.aligned_page_bytes, first, len(rows),
                part.address,
            )  # fmt: skip
            landed = self.wait_pages(island, self._across(via, island), len(rows))
            self.wait_together([sent, (island, landed)], timeout)
        finally:
            staging.free()

    def _spans(self, tensor, start, count, reading=False):
        """[(island, its part's pages in order, first index, end index)] of the part
        indices whose pages lie in start .. start + count - 1 (a copy read: island 0
        alone)."""
        spans = []
        for island, part in enumerate(tensor.parts):
            if part is None or (reading and tensor.placement.copy and island):
                continue
            at = tensor.placement.part(island)
            low, high = np.searchsorted(at, [start, start + count])
            if high > low:
                spans.append((island, at, int(low), int(high)))
        return spans

    def _each_island(self, work, spans):
        """work(island, at, low, high) for every span: with a run loaded on every
        island, one thread an island at once; else in turn."""
        live = self.run is not None and not self.run.stopped
        if not live or len(spans) < 2 or self.split > 1:  # a split card: one channel
            for span in spans:
                work(*span)
            return
        with concurrent.futures.ThreadPoolExecutor(len(spans)) as pool:
            for done in [pool.submit(work, *span) for span in spans]:
                done.result()

    def read_bf16(self, *tensors):
        """bfloat16 tile tensors of rank 2 as float32 arrays, read as read() does."""
        out = []
        for tensor, rows in zip(tensors, self.read(*tensors)):
            if (
                tensor.layout != "tile"
                or tensor.dtype != "bfloat16"
                or len(tensor.shape) != 2
            ):
                raise ValueError("read_bf16 takes bfloat16 tile tensors of rank 2")
            out.append(dram.untilize_bf16(rows, *tensor.shape))
        return out

    def read_tiles_bf16(self, tensor, first, pages):
        """Tiles first .. first + pages - 1 of a bfloat16 tile tensor as float32
        [pages, 32, 32], as Session.read_tiles_bf16, read as read() does."""
        if tensor.layout != "tile" or tensor.dtype != "bfloat16":
            raise ValueError("read_tiles_bf16 takes a bfloat16 tile tensor")
        (raw,) = self.read(tensor, first=first, pages=pages)
        tile = dram.TILE
        return dram.untilize_bf16(raw, len(raw) * tile, tile).reshape(-1, tile, tile)

    def write(self, tensor, pages, first=0):
        """Writes `pages` ([n, page_bytes] uint8) into pages first .. first + n - 1:
        each island's part through its own host channel (with a run loaded, the
        islands at once; a copy: every island)."""
        pages = np.ascontiguousarray(pages, np.uint8)
        if pages.ndim != 2 or pages.shape[1] != tensor.page_bytes:
            raise ValueError(f"pages are [n, {tensor.page_bytes}] uint8 rows")
        if self.card_dram:  # one tensor in the card's DRAM
            self.sessions[0].write(tensor.parts[0], pages, first)
            return
        start, count = dram.page_range(first, len(pages), tensor.pages)

        def scatter(island, at, low, high):  # a run of rows: a view, not a copy
            mine = pages[_rows(at[low:high] - start)]
            self.sessions[island].write(tensor.parts[island], mine, low)

        direct, far = self._reached(self._spans(tensor, start, count))
        self._each_island(scatter, direct)
        for island, at, low, high in far:
            rows = pages[_rows(at[low:high] - start)]
            self._write_across(tensor.parts[island], island, low, rows)

    def load(self, program, **options):
        """Loads `program` on every island in turn, with Session.load's options; an
        IslandRun over flat ranks. As on one card: a Program, or programs in the
        runtime's form (a list, or {program: (first, count)} in flat ranks); or a list
        of kernel-form Programs, one an island. Each brings its own X280 threads."""
        if self.run is not None and not self.run.stopped:
            raise RuntimeError("a run is loaded: stop it first")
        if self.split > 1:  # one launch on the card, a dispatcher an island
            return self._load_split(program, options)
        self._hand_rectangles(program)
        programs = self._island_loads(program)
        if self.ASKS_BY not in ("core", "host"):
            raise ValueError(f"ASKS_BY={self.ASKS_BY!r}: 'core' or 'host'")
        if self.links and all(isinstance(one, Program) for one in programs):
            programs = self._linked(programs)
        owned = {}  # every Program the islands load, once: the island defines on each
        for one in programs:
            for each in (
                one.programs if isinstance(one, programs_module.Plan) else [one]
            ):
                owned[id(each)] = each
                for other in getattr(each, "added_ahead_programs", ()):
                    owned[id(other)] = other  # its parts built at the load, as added
        owned = list(owned.values())
        defines = [one.defines for one in owned]
        # an X280 thread's copies name tensor words too (kmx.h): the same bound
        threads = [t for one in owned for t in getattr(one, "threads", {}).values()]
        thread_defines = [t.defines for t in threads]
        runs = []
        try:
            for one, own in zip(owned, defines):
                one.defines = {**own, **self._island_defines()}
            for thread, own in zip(threads, thread_defines):
                thread.defines = {**own, **self._table_bound()}
            plans = self._links(programs)
            tables = self._tables_for_load()
            for island, (session, one) in enumerate(zip(self.sessions, programs)):
                try:
                    # one island: the card's run as it is without islands
                    island_of = (self.graph, island) if len(programs) > 1 else None
                    runs.append(
                        session.load(
                            one,
                            island=island_of,
                            link=plans[island],
                            tensors=tables[island],
                            **options,
                        )
                    )
                except KomorebiError as error:
                    raise _named(error, island) from None
            check_link_queues(
                [getattr(run.backend, "link_queue_at", None) for run in runs]
            )
            if prebuild.active():  # every island built: a mock run for the launch
                self.run = prebuild.Run(parts=runs)
                self._remember(program)
                return self.run
            self._check_links(runs, plans)
        except (KernelSizeError, L1BudgetError) as error:
            for run in runs:
                run.abort()
            self._refused(error)
            raise
        except BaseException:
            for run in runs:
                run.abort()
            raise
        finally:
            for one, own in zip(owned, defines):
                one.defines = own
            for thread, own in zip(threads, thread_defines):
                thread.defines = own
        self.run = IslandRun(
            self.graph,
            runs,
            counts=self._crossings if self.links else None,
        )
        self.run.session = self
        self._remember(program)
        if self.links:
            try:
                self._hold_links()
            except BaseException:
                self.run.abort()
                raise
        return self.run

    def _hand_rectangles(self, spec):
        """Each class with a rectangle the islands load (`spec`, the classes it names
        ahead) gets island 0's NoCs and the islands' rank count before the plan checks
        it: its default rectangle is an island's rank cores (dsl.program.rank_rectangle;
        islands of different counts refuse it by name)."""
        if len(self.sessions) < 2 or self.split > 1:
            return
        if isinstance(spec, programs_module.Plan):
            listed = list(spec.programs)
        elif isinstance(spec, dict | list | tuple):
            listed = list(spec)
        else:
            listed = [spec]
        listed += [o for p in listed for o in getattr(p, "added_ahead_programs", ())]
        needing = [
            p
            for p in listed
            if isinstance(p, Program)
            and getattr(p, "rectangle", None)
            and p.rectangle["args"] is None
        ]
        if not needing:
            return
        counts = {len(cores) for cores in self.graph.rank_cores}
        ranks = next(iter(counts)) if len(counts) == 1 else None
        nocs = self.sessions[0].nocs()
        for program in needing:
            state = program.rectangle
            if state["rect"] is None and ranks is None:
                raise KomorebiError(
                    f"{program.name}: its default rectangle is an island's rank cores, "
                    f"and the islands have {sorted(counts)}: make it with rect="
                )
            if state["nocs"] is None:
                state["nocs"] = nocs
            state["ranks"] = ranks

    def _island_defines(self):
        """The defines a load gives every program the islands load (and the classes
        named ahead, built as added): a tensor's word may name its table entry (several
        islands); the link cores' CPUs serve PAGES (km_pages.hpp)."""
        out = {"KM_TENSOR_TABLE": 1} if len(self.sessions) > 1 else {}
        out |= self._table_bound() | self._islands_bound()
        if self.links:
            asks = {"KM_ASKS_BY_HOST": 1} if self.ASKS_BY == "host" else {}
            out |= {"KM_PAGE_LINKS": 1, **asks, **self._forward}
        return out

    def _islands_bound(self):
        """KM_ISLANDS_MAX for more than ISLANDS_MAX islands: the per-island arrays'
        length on the cores."""
        count = len(self.graph.islands) if self.graph is not None else 1
        if abi.headers()["KM_ISLANDS_MAX"] != ISLANDS_MAX:
            raise KomorebiError("ISLANDS_MAX differs from km_abi.h's")
        return {"KM_ISLANDS_MAX": count} if count > ISLANDS_MAX else {}

    def _table_bound(self):
        """KM_TENSORS_MAX for a table of more than TENSORS_MAX entries; KM_TABLE_CACHE
        when the table is cached (_cached); a flat one of 32 or fewer: neither."""
        entries = len(self._entries)
        out = {"KM_TENSORS_MAX": entries} if entries > TENSORS_MAX else {}
        if self._cached() and len(self.sessions) != 1:  # one island: no table
            out["KM_TABLE_CACHE"] = self._cache_k()
        return out

    def _island_loads(self, spec):
        """What each island's session loads for `spec`: a kernel-form Program on every
        island, a list of them one an island, or (the runtime's form) each island's
        part of the plan over flat ranks: its groups cut to its ranks, the same
        program fields."""
        count = len(self.sessions)
        if isinstance(spec, Program) and spec.form() != "program":
            return [spec] * count
        kernel_form = isinstance(spec, list | tuple) and all(
            isinstance(one, Program) and one.form() != "program" for one in spec
        )
        if kernel_form and spec:
            if len(spec) != count:
                raise ValueError(
                    f"a list of kernel-form Programs is one an island: {count} here, "
                    f"{len(spec)} given (several programs a run are in the runtime's "
                    "form, km_program)"
                )
            return list(spec)
        whole = programs_module.plan(spec, self.graph.ranks)  # checks it as one card
        parts = []
        for island in range(count):
            first, ranks = self.graph.first(island), self.graph.count(island)
            groups = []
            for group in whole.groups:
                low = max(group.first, first)
                high = min(group.first + group.count, first + ranks)
                if high > low:
                    groups.append(
                        programs_module.Group(low - first, high - low, group.programs)
                    )
            parts.append(
                programs_module.Plan(
                    whole.programs, tuple(groups), ranks, whole.owner, whole.switch
                )
            )
        return parts

    @property
    def _forward(self):
        """KM_FORWARD over more than two islands: the link services forward (built only
        then, the kernels smaller without); KM_FAR_PUTS when a put crosses two links."""
        if len(self.graph.islands) <= 2:
            return {}
        diameter = self.graph.diameter()
        return {
            "KM_FORWARD": 1,
            # the rooms: a writer core and CPU, a hop level (km_abi.h KM_FORWARD_*)
            "KM_FORWARD_CORES": max(len(cores) for cores in self.link_cores),
            "KM_FORWARD_CPUS": max(min(link.count, 2) for link in self.graph.links),
            "KM_FORWARD_LEVELS": max(diameter - 1, 1),
            **({"KM_FAR_PUTS": 1} if diameter > 1 else {}),
        }

    def _load_split(self, program, options):
        """load() on a card split into islands: one launch, the links wires (NoC
        writes to the far island's tiles), the services as on two cards."""
        if not isinstance(program, Program) or program.form() == "program":
            raise ValueError(
                "a card split into islands loads one Program of the kernel form: its "
                "one launch builds no runtime's-form programs (km_program, a list or "
                "a dict) yet"
            )
        own = program.defines
        try:
            plans = None
            if self.links:
                (program,) = self._linked([program])
                asks = {"KM_ASKS_BY_HOST": 1} if self.ASKS_BY == "host" else {}
                # tensors held by island: a tensor's word may name its table entry
                table = {} if self.card_dram else {"KM_TENSOR_TABLE": 1}
                table |= {} if self.card_dram else self._table_bound()
                table |= self._islands_bound()
                program.defines = {
                    **program.defines,
                    "KM_PAGE_LINKS": 1,
                    "KM_WIRE": 1,
                    **self._forward,
                    **asks,
                    **table,
                }
                plans = self._links()
            tables = [0] if self.card_dram else self._tables_for_load()
            runs = self.sessions[0].load_islands(
                program, self.graph, links=plans, tensors=tables[0], **options
            )
        finally:
            program.defines = own
        if prebuild.active():  # every island built: a mock run for the launch
            self.run = prebuild.Run(parts=runs)
            return self.run
        self.run = IslandRun(
            self.graph,
            runs,
            counts=self._crossings if self.links else None,
        )
        if self.links:
            try:
                self._hold_links()
            except BaseException:
                self.run.abort()
                raise
        return self.run

    # Pages across a link

    def _hold_links(self, timeout=10.0):
        """Each link CPU of both islands opens its connection, held until its kernel
        ends, and says hello to the far link core's same CPU (an OPEN a link core,
        waited together): no send goes out before the far side runs."""
        tables = [list(run.backend.port.core_table()) for run in self.run.runs]
        if any(table != tables[0] for table in tables):
            raise KomorebiError(
                "the islands' core tables differ: a link core that pushes a rank's "
                "pages finds the far core in its own table"
            )
        waits = [
            (here, self._open_link(here, k))
            for here in range(len(self.graph.islands))
            for k in range(len(self.link_cores[here]))
        ]
        self.wait_together(waits, timeout)
        self.log(f"page links held, hello both ways: link cores {self.link_cores}")

    def _open_link(self, here, k):
        """Queues the OPEN of link core k of island `here` (its CPUs' connections,
        then a hello to the same CPUs of the far link core across, self.pair).
        Returns its handle."""
        h = abi.headers()
        there, first, far = self.pair(here, k)
        buffer, chunk = self.staging(here)
        links = list(self.channels[here]) + [h["KM_PAGES_NO_LINK"]] * self.LINKS_MAX
        words = [0] * 10
        words[h["KM_PAGES_FIRST"]] = first
        words[h["KM_PAGES_TO_CORE"]] = chunk // 64 << 16
        words[h["KM_PAGES_COUNT"]] = self.port_address(there, h["KM_PAGES_HELLO_PORT"])
        words[h["KM_PAGES_COUNTER"]] = (
            self.noc_of(there, far) | 1 << 16 | h["KM_PAGES_OPEN"] << 24
        )
        words[h["KM_PAGES_LINKS"]] = int.from_bytes(bytes(links[:4]), "little")
        words[h["KM_PAGES_BUFFER"]] = buffer
        return self.run.island(here)._queue_command(
            h["KM_HANDLER_PAGES"], self.graph.link_group(here, k, 1), words
        )

    def reset_link(self, island, k, timeout=10.0):
        """Resets link core k of `island` by the host and opens its connections again:
        each closed at its router first, the core reloaded, then its OPEN with the far
        link core's hello taken as said (the far side runs). {island, core, by, seconds,
        reset_seconds, open_seconds}; KomorebiError if a step did not finish."""
        h = abi.headers()
        if (
            self.run is None
            or not self.links
            or not 0 <= k < len(self.link_cores[island])
        ):
            raise ValueError(f"island {island}: no link core {k} in a run with links")
        run = self.run.island(island)
        core = self.link_cores[island][k]
        held = [
            2 * k + cpu for cpu in range(2) if 2 * k + cpu < len(self.channels[island])
        ]
        drops = []
        areas = getattr(self, "_areas", None)
        if k == 0 and areas:  # lines given up with their commands: never sent
            area = areas[island][0]
            for name in ("ASK", "PUT"):
                drops += [
                    area + h[f"KM_{name}_TAIL"],
                    area + h[f"KM_{name}_TAKEN"],
                    area + h[f"KM_{name}_LINES"],
                    h["KM_ASK_SLOTS"],
                ]
        started = time.perf_counter()
        try:
            reset = run.reset_core(
                core,
                timeout,
                channels=[self.channels[island][link] for link in held],
                drops=drops,
            )
            group = self.graph.link_group(island, k, 1)
            run.wait(run._queue_command(h["KM_HANDLER_NOP"], group), timeout)
            opened = time.perf_counter()
            backend = run.backend
            module, columns = backend.session._module, backend.port.grid_x
            for link in held:  # the far CPU said hello at the load: taken as heard
                port = self.port_address(
                    island, h["KM_PAGES_HELLO_PORT"] + link - 2 * k
                )
                module.write_l1(
                    backend.device, core % columns, core // columns, port, [2]
                )
            run.wait(self._open_link(island, k), timeout)
        except KomorebiError as error:
            raise _named(error, island) from None
        done = time.perf_counter()
        self.log(
            f"island {island}: link core {k} (core {core}) reset and opened again: "
            f"reset {(opened - started) * 1e3:.1f} ms, "
            f"open {(done - opened) * 1e3:.3f} ms"
        )
        return {
            "island": island,
            "core": core,
            "by": "host",
            "seconds": done - started,
            "reset_seconds": opened - started,
            "open_seconds": done - opened,
            "parts": reset["parts"],
        }

    def _crossings(self, timeout=10.0):
        """Crossings read now: a COUNTS command to every link core of each island (each
        CPU writes its row into the island's counts tensor), waited for together."""
        h = abi.headers()
        waits = []
        for island in range(len(self.graph.islands)):
            words = [0] * 10
            words[h["KM_PAGES_TO"]] = self._counts[island].address
            words[h["KM_PAGES_PAGE"]] = CROSS_PAGE
            words[h["KM_PAGES_COUNTER"]] = h["KM_PAGES_COUNTS"] << 24
            handle = self.run.island(island)._queue_command(
                h["KM_HANDLER_PAGES"], self.graph.link_group(island), words
            )
            waits.append((island, handle))
        self.wait_together(waits, timeout)
        fields = [h[f"KM_CROSS_{name.upper()}"] for name in Crossings.FIELDS[:-1]]
        rows = {}
        for island in range(len(self.graph.islands)):
            (raw,) = self.sessions[island].read(self._counts[island])
            words = raw.view("<u4").astype(np.int64)
            for at in range(2 * len(self.link_cores[island])):
                there, first, _ = self.pair(island, at // 2)
                link = at - 2 * first  # its link toward `there`
                if link >= self.links:
                    continue
                row = words[at]
                low = h["KM_CROSS_BYTES"]
                ring = h["KM_CROSS_ROW"] // 4  # the ring's words after the row
                rows[(link, island, there)] = {
                    **dict(zip(Crossings.FIELDS, (int(row[f]) for f in fields))),
                    "bytes": int(row[low]) | int(row[low + 1]) << 32,
                    "ring_sending": int(row[ring + h["KM_PUT_RING_SENDING"] // 4]),
                    "ring_router": int(row[ring + h["KM_PUT_RING_ROUTER"] // 4]),
                    **{
                        f: int(row[ring + 4 + h[f"KM_PUT_{f.upper()}"] // 4])
                        for f in Crossings.IDLE
                    },
                }
        return Crossings(rows)

    def _table_of(self, session):
        """The island table in the L1 of every core of `session`'s card, each core its
        own island's (graph.table): KM_ISLAND_TABLE, sized by the island count."""
        columns, rows = session.grid
        count = len(self.graph.islands)
        # the entries (2 words an island), far tiles (2), counts (1), children (1)
        words = np.zeros((columns * rows, -(-6 * count // 16) * 16), np.uint32)
        mine = [self.sessions.index(session)] if self.split == 1 else range(count)
        for island in mine:
            cores = list(self.graph.islands[island].cores)
            if self.split == 1:
                cores = range(columns * rows)
            words[cores, : 2 * count] = np.frombuffer(
                self.graph.table(island), np.uint32
            )
            words[cores, 2 * count : 4 * count] = np.concatenate(
                self._far_tiles(island)
            )
            words[cores, 5 * count : 6 * count] = self.graph.children(island)
        return session.tensor(words, "uint32", shard_grid=(columns, rows))

    def _first_link(self, here, there):
        """Island `here`'s first link core toward its neighbour `there` (an index into
        its link_cores)."""
        return self.graph.next_link(here, there)

    def _far_tiles(self, island):
        """(enter, leave): for each island x, the tile (x | y << 16) of x's link core
        where `island`'s path to x enters it, and where x's path to `island` leaves it
        (km::Islands::far_tile, KM_FAR_PUTS); 0 for `island` itself."""
        count = len(self.graph.islands)
        enter, leave = np.zeros(count, np.uint32), np.zeros(count, np.uint32)
        if not self.link_cores or not all(self.link_cores):
            return enter, leave

        def tile(x, toward):
            place = self.noc_of(x, self.link_cores[x][self._first_link(x, toward)])
            return (place & 0xFF) | (place >> 8) << 16

        for x in range(count):
            if x == island or self.graph.hops(island, x) == 0:
                continue
            last = island  # the island before x on island's path to it
            while self.graph.next_island(last, x) != x:
                last = self.graph.next_island(last, x)
            enter[x] = tile(x, last)
            leave[x] = tile(x, self.graph.next_island(x, island))
        return enter, leave

    def _port_words(self, island):
        """Island `island`'s port table (KM_ISLAND_PORTS): port p (neighbour p), link
        l: its link core, CPU, far island, the port's links, the far link core's tile
        and the far island's port toward `island`."""
        h = abi.headers()
        per = h["KM_ISLAND_PORT_LINKS"]
        words = np.zeros((h["KM_ISLAND_PORTS_MAX"] * per, 2), np.uint32)
        words[:, 0] = h["KM_ISLAND_NO_CORE"]

        first = self._first_link
        for port, there in enumerate(self.graph.neighbours(island)):
            count = self.graph.between(island, there).count
            if count > per:
                raise ValueError(
                    f"islands {island} and {there}: {count} links, a port's "
                    f"KM_ISLAND_PORT_LINKS {per}"
                )
            for link in range(count):
                core = self.link_cores[island][first(island, there) + link // 2]
                peer = self.link_cores[there][first(there, island) + link // 2]
                place = self.noc_of(there, peer)
                back = self.graph.neighbours(there).index(island)  # its port here
                words[port * per + link] = (
                    core | link % 2 << 8 | there << 16 | count << 24,
                    (place & 0xFF) | (place >> 8) << 16 | back << 24,
                )
        return words.reshape(-1)

    def pair(self, island, k):
        """Link core `k` of `island`: (the island its links go to, the first link core
        toward it, the far island's link core across from this one). A neighbour's
        links take ceil(links / 2) link cores in neighbour order, two CPUs a core."""
        per = (self.links + 1) // 2
        neighbours = self.graph.neighbours(island)
        there = neighbours[k // per]
        far = self.graph.neighbours(there).index(island) * per + k % per
        return there, k // per * per, self.link_cores[there][far]

    def noc_of(self, island, core):
        """x | y << 8: the NoC place of local core `core` of `island` (core table)."""
        columns = self.sessions[island].grid[0]
        x, y = self.sessions[island].device.worker_core(core % columns, core // columns)
        return int(x) | int(y) << 8

    def port_address(self, island, port):
        """The L1 address of port `port`'s count, the same on every core of `island`
        (KM_PORT_PAGES and the one after it: the page calls' own)."""
        h = abi.headers()
        if not 0 <= port < h["KM_PORT_SLOTS"]:
            raise ValueError(f"port {port} of {h['KM_PORT_SLOTS']}")
        control = self.run.island(island).backend.control_addr
        return control + h["KM_CONTROL_PORT"] + port * h["KM_PORT_STRIDE"]

    def staging(self, island):
        """The L1 every core of `island` moves transfers through: (address, bytes of a
        chunk); two chunks. Pages asked from another island land there."""
        transfer = self.run.island(island).backend.transfer
        if transfer is None:
            raise KomorebiError(f"island {island}: the run has no transfer buffer")
        return transfer.address, transfer.chunk

    def send_pages(
        self,
        source,
        first,
        pages,
        to,
        core=None,
        counted=None,
        port=None,
        there=None,
        shape=0,
    ):
        """Sends pages first .. of `source` by its holder's link cores into island
        `there` (default: the neighbour): the same pages of tensor `to`, or (`core`) L1
        from address `to`; each batch adds to port `port` of `counted` (default `core`,
        else the far link core). `shape`: the loop shape. Returns (island, handle)."""
        if not self.links:
            raise KomorebiError(
                "no page links: pages cross in a session over two cards"
            )
        holders = source.holders(first, pages)
        if len(holders) != 1:
            raise ValueError(f"pages {first} + {pages}: held by islands {holders}")
        here, _, _, index = holders[0]
        if there is None:
            if len(self.graph.neighbours(here)) != 1:
                raise ValueError(f"island {here} has more neighbours: name `there`")
            there = self.graph.neighbours(here)[0]
        k, cores = self._toward(here, there)
        if core is None and (
            not isinstance(to, IslandTensor)
            or to.aligned_page_bytes != source.aligned_page_bytes
        ):
            raise ValueError("a tensor target has the source's aligned page bytes")
        if core is None and to.placement.runs(first, pages, there) != [
            (there, first, pages, index)
        ]:
            raise ValueError(
                f"pages {first} + {pages}: island {there} does not hold them at the "
                f"indices island {here} does ({index} on)"
            )
        return self._send(
            here,
            there,
            source.addresses[here],
            source.aligned_page_bytes,
            index,
            pages,
            to.addresses[there] if core is None else to,
            core,
            counted,
            port,
            shape,
        )

    def _send(
        self,
        here,
        there,
        start,
        page,
        first,
        pages,
        to,
        core=None,
        counted=None,
        port=None,
        shape=0,
    ):
        """Queues the send by `here`'s link cores toward `there` of pages first .. of
        the DRAM buffer at `start` (aligned `page` bytes): into the same pages of the
        buffer at `to`, or (`core`) L1 `to` of that core on; send_pages' counts."""
        h = abi.headers()
        k, cores = self._toward(here, there)
        counted = counted if counted is not None else core
        counted = counted if counted is not None else self.pair(here, k)[2]
        port = h["KM_PORTS"] - 1 if port is None else port
        buffer, chunk = self.staging(here)
        links = list(self.channels[here]) + [h["KM_PAGES_NO_LINK"]] * self.LINKS_MAX
        words = [0] * 10
        words[h["KM_PAGES_FROM"]] = start
        words[h["KM_PAGES_PAGE"]] = page
        words[h["KM_PAGES_FIRST"]] = first
        words[h["KM_PAGES_PAGES"]] = pages
        words[h["KM_PAGES_TO"]] = to
        target = h["KM_PAGES_DRAM"] if core is None else self.noc_of(there, core)
        words[h["KM_PAGES_TO_CORE"]] = target | chunk // 64 << 16
        words[h["KM_PAGES_COUNT"]] = self.port_address(there, port)
        words[h["KM_PAGES_COUNTER"]] = (
            self.noc_of(there, counted) | 1 << 16 | h["KM_PAGES_SEND"] << 24
        )
        words[h["KM_PAGES_LINKS"]] = int.from_bytes(bytes(links[:4]), "little")
        shape |= h["KM_PAGES_SHAPE_LINK0"]  # link 0 carries pages too
        words[h["KM_PAGES_BUFFER"]] = buffer | shape << h["KM_PAGES_SHAPE_SHIFT"]
        handle = self.run.island(here)._queue_command(
            h["KM_HANDLER_PAGES"], self.graph.link_group(here, k, cores), words
        )
        return here, handle

    def _across(self, here, there):
        """The link core of `there` that counts the pages `here` sends it."""
        return self.pair(here, self._toward(here, there)[0])[2]

    def _toward(self, here, there):
        """(first, count) of `here`'s link cores whose links go to island `there`."""
        if there not in self.graph.neighbours(here):
            raise ValueError(f"islands {here} and {there} are not neighbours")
        if len(self.graph.neighbours(here)) == 1:
            return 0, len(self.link_cores[here])
        per = (self.links + 1) // 2
        return self.graph.neighbours(here).index(there) * per, per

    def ask(self, tensor, first, pages, island, core, to, port):
        """Pages first .. of `tensor`, held by another island, into L1 address `to` of
        local core `core` of `island`, counted at its port `port`: what that core waits
        on (km::port_wait). Returns (holder, handle) of the holder's send."""
        if self.ASKS_BY != "host":
            raise KomorebiError(
                f"asks by {self.ASKS_BY!r}: the host asks only with ASKS_BY = 'host'"
            )
        holder = tensor.holders(first, pages)[0][0]
        if holder == island:
            raise ValueError(f"pages {first} + {pages}: island {island} holds them")
        return self.send_pages(tensor, first, pages, to, core, core, port, island)

    def ask_range(self, tensor, first, pages, group, to, port):
        """One ask for every rank of `group` (first rank, ranks; one island): rank first
        + i gets pages first + i * pages .. of `tensor` (held by another island) into L1
        address `to`, counted at port `port`. One command to the holder's link cores,
        which split the ranks among their CPUs. Returns (holder, handle)."""
        h = abi.headers()
        if self.ASKS_BY != "host":
            raise KomorebiError(
                f"asks by {self.ASKS_BY!r}: the host asks only with ASKS_BY = 'host'"
            )
        rank, ranks = group
        island, local = self.graph.resolve(rank)
        if ranks < 1 or self.graph.resolve(rank + ranks - 1)[0] != island:
            raise ValueError(f"group {group}: not one island's ranks")
        holders = tensor.holders(first, pages * ranks)
        if len(holders) != 1 or holders[0][0] == island:
            raise ValueError(
                f"pages {first} + {pages * ranks}: held by islands {holders}, not by "
                f"one other than island {island}"
            )
        holder, _, _, index = holders[0]
        if not 0 <= port < h["KM_PORT_SLOTS"] or ranks > 0xFFFF:
            raise ValueError(f"port {port}, ranks {ranks}")
        buffer, chunk = self.staging(holder)
        links = list(self.channels[holder]) + [h["KM_PAGES_NO_LINK"]] * self.LINKS_MAX
        words = [0] * 10
        words[h["KM_PAGES_FROM"]] = tensor.addresses[holder]
        words[h["KM_PAGES_PAGE"]] = tensor.aligned_page_bytes
        words[h["KM_PAGES_FIRST"]] = index
        words[h["KM_PAGES_PAGES"]] = pages
        words[h["KM_PAGES_TO"]] = to
        core = self.graph.rank_cores[island][local]
        words[h["KM_PAGES_TO_CORE"]] = core | chunk // 64 << 16
        words[h["KM_PAGES_COUNT"]] = self.port_address(island, port)
        words[h["KM_PAGES_COUNTER"]] = ranks | 1 << 16 | h["KM_PAGES_RANGE"] << 24
        words[h["KM_PAGES_LINKS"]] = int.from_bytes(bytes(links[:4]), "little")
        words[h["KM_PAGES_BUFFER"]] = buffer
        handle = self.run.island(holder)._queue_command(
            h["KM_HANDLER_PAGES"],
            self.graph.link_group(holder, *self._toward(holder, island)),
            words,
        )
        return holder, handle

    def wait_pages(self, island, core, pages, port=None):
        """Queues on local core `core` of `island` a wait (bounded) until port `port`'s
        count has reached `pages`, which it then takes off. Returns the handle."""
        h = abi.headers()
        port = h["KM_PORTS"] - 1 if port is None else port
        words = [0] * 10
        words[h["KM_PAGES_PAGES"]] = pages
        words[h["KM_PAGES_COUNT"]] = self.port_address(island, port)
        words[h["KM_PAGES_COUNTER"]] = h["KM_PAGES_WAIT"] << 24
        return self.run.island(island)._queue_command(
            h["KM_HANDLER_PAGES"], self.core_group(island, core), words
        )

    def core_group(self, island, core):
        """The group of local core `core` alone in a command to `island`: a link core
        by its name (Graph.link_group), a rank's core by its flat rank."""
        if core in self.link_cores[island]:
            k = self.link_cores[island].index(core)
            return self.graph.link_group(island, k, 1)
        if core not in self.graph.rank_cores[island]:
            raise ValueError(f"island {island} has no core {core}")
        return self.graph.first(island) + self.graph.rank_cores[island].index(core), 1

    def wait_together(self, waits, timeout=10.0):
        """Waits, within one `timeout`, for [(island, handle)] of the islands' own runs,
        one native round an island in turn: a host's commands reach an island's
        ring only while the host waits on it, and pages crossing need both islands'.
        Returns the events in the order given; errors name their island."""
        mine = {}
        for island, handle in waits:
            mine.setdefault(island, []).append(handle)
        pending = {i: np.unique(np.asarray(h, np.int64)) for i, h in mine.items()}
        deadline = time.monotonic() + timeout
        for island in pending:
            self.run.island(island)._hand()
        while any(handles.size for handles in pending.values()):
            if time.monotonic() > deadline:
                island = next(i for i, h in pending.items() if h.size)
                raise _named(
                    WaitTimeout(
                        f"{pending[island].size} handles not done in {timeout} s",
                        pending[island].tolist(),
                    ),
                    island,
                )
            for island, handles in pending.items():
                if handles.size:
                    run = self.run.island(island)
                    try:
                        run._check_alive()
                        run._step(handles, int(handles[-1]), SLICE_S)
                    except KomorebiError as error:
                        raise _named(error, island) from None
                    pending[island] = handles[~run._arrived(handles)]
        events = {}
        for island, handles in mine.items():
            try:
                got = self.run.island(island).wait(handles, _left(deadline))
            except KomorebiError as error:
                raise _named(error, island) from None
            events.update({(island, h): e for h, e in zip(handles, got)})
        return [events[(island, handle)] for island, handle in waits]

    def cross(self, source, target, first=0, pages=None, timeout=10.0, shape=0):
        """Writes pages first .. of `source` into the same pages of `target` where
        another island holds them (a copy target: every other island's copy), sent
        over the links, waited for on the far island; `shape` as send_pages. Returns
        pages, bytes and seconds, the submit to the last count."""
        start, count = dram.page_range(first, pages, source.pages)
        sends = []
        for here, low, length, _ in source.holders(start, count):
            for there in range(len(self.sessions)):
                if there != here:
                    sends += [
                        (holder, part, part_length)
                        for holder, part, part_length, _ in target.placement.runs(
                            low, length, there
                        )
                        if holder == there
                    ]
        started = time.perf_counter()
        waits, far = [], []
        for there, low, length in sends:
            here = source.holders(low, length)[0][0]
            counted = self.pair(here, self._toward(here, there)[0])[2]
            far.append((there, self.wait_pages(there, counted, length)))
            waits.append(
                self.send_pages(source, low, length, target, there=there, shape=shape)
            )
        # the senders first: a failure names the side that failed
        self.wait_together(waits + far, timeout)
        seconds = time.perf_counter() - started
        moved = sum(length for _, _, length in sends)
        return {
            "pages": moved,
            "bytes": moved * source.aligned_page_bytes,
            "seconds": seconds,
        }

    def _linked(self, programs):
        """The dynamic programs whose messages to another island's ranks are
        deliveries by the link service (km_deliver.hpp); X280 ranks refused."""
        dynamic = [one.messages for one in programs if one.messages is not None]
        if any(messages.x280_ranks for messages in dynamic):
            raise ValueError("X280 ranks over a link are not done yet")
        return [self._delivered(one) for one in programs]

    def _delivered(self, program):
        """`program` (dynamic) with its messages to other islands delivered by the link
        service: KM_DELIVER and its places' layout, DELIVER_ROOM shared by the source
        islands (a slot each at least: a bin's 32 messages and its body)."""
        messages = program.messages
        if messages is None:
            return program
        islands_count = len(self.graph.islands)
        constants, _ = program_module.messages_header()
        body_at = constants["kMessagesPacket"] * messages.message_bytes
        slot = body_at + -(-messages.body_bytes // 64) * 64
        slots = max(1, DELIVER_ROOM // (islands_count * slot))
        delivered = copy.copy(program)
        delivered.defines = {
            **program.defines,
            "KM_DELIVER": 1,
            "KM_DELIVER_ISLANDS": islands_count,
            "KM_DELIVER_SLOTS": slots,
            "KM_DELIVER_SLOT_BYTES": slot,
            "KM_DELIVER_BODY_AT": body_at,
        }
        size = session_module.deliver_bytes(delivered.defines)
        self.log(
            f"deliveries: {slots} slot(s) a source island of {slot:,} B, "
            f"{size:,} B of each link core's L1 over {islands_count} islands"
        )
        return delivered

    def _held_l1(self, island):
        """What island `island`'s session keeps in the L1 of every core, at the top
        (the load's L1 budget counts it): [(row name, bytes)] of the tensors it holds
        now: its own words, its link area and relay when the link cores do not hold
        them alone, its tensor table."""
        held = dict(getattr(self, "_held", [{}] * (island + 1))[island])
        form = (
            f"tensor table, cached ({self._cache_k()} rows a RISC)"
            if self._drams
            else f"tensor table, flat ({self._table_size} entries)"
        )
        held[form] = self._tables[island : island + 1]
        rows = []
        for name, tensors in held.items():
            size = sum(t.aligned_page_bytes for t in tensors if not t.freed)
            if size:
                rows.append((f"session: {name}", size))
        return rows

    def held_bytes(self, apart=True, extra=0):
        """The L1 bytes a load keeps on every core (the rows of l1_held it will have):
        the own words; the link area and relay unless `apart` (the link cores run
        their own kernels and hold them alone, _apart); the tensor table with `extra`
        tensors more than now."""
        if len(self.sessions) == 1:
            return 0
        h = abi.headers()
        size = self._table_entries(extra) * islands.tensor_entry_bytes(
            len(self.sessions)
        )
        size = self._cache_bytes() if self._cached() else size
        if self.links:
            size += h["KM_OWN_ALL_BYTES"]
            if not apart:
                size += h["KM_LINK_AREA_BYTES"] + -(-h["KM_RELAY_BYTES"] // 64) * 64
        return size

    def _apart(self, programs):
        """Whether the link cores of a load of `programs` (programs or plans) run their
        own kernels (session._link_kernels): the link area and relay then live on them
        alone, at their circular buffers' base."""
        held = [
            each
            for one in programs
            for each in (
                one.programs if isinstance(one, programs_module.Plan) else [one]
            )
        ]
        return self.split == 1 and all(
            isinstance(one, Program) and one.messages is None for one in held
        )

    def _links(self, programs=None):
        """Each island's LinkPlan (two islands, or a split card), else None each: link
        core 0, own words on every core, a link area and a relay at one address on every
        island, each link core's far one for the relay, a forward area beyond two
        islands. Own-kernel link cores (_apart): the area and relay are theirs."""
        if len(self.sessions) != 2 and self.split == 1:
            return [None] * len(self.sessions)
        h = abi.headers()
        bytes_ = h["KM_LINK_AREA_BYTES"]
        relay_bytes = -(-h["KM_RELAY_BYTES"] // 64) * 64
        apart = programs is not None and self._apart(programs)
        plans = []
        self._link_areas = []  # the tensors held while the run lives
        self._held = []  # per island: {row: tensors} of what every core keeps
        self._areas = []  # per island: (link area, relay) addresses
        self._counts = []
        relays = []
        cpus = self.links  # the page CPUs: every link
        landings = self._landings(h)
        for island, session in enumerate(self.sessions):
            far = self.graph.neighbours(island)[0]
            columns, rows = session.grid
            if self.split > 1 and island:  # one card: island 0's areas
                (area, relay), own = self._areas[0], self._held[0]["own words"][0]
                self._held.append({})
            else:
                own = session.tensor(  # zeros: no line, answer, done word or batch
                    np.zeros((columns * rows, h["KM_OWN_ALL_BYTES"] // 4)),
                    "uint32",
                    shard_grid=(columns, rows),
                )
                held = {"own words": [own], "island table": [self._table_of(session)]}
                if apart:  # the link cores' buffers' base, the same on every island
                    area = session_module.L1_BYTES - session.l1()[1]
                    relay = area + bytes_
                else:
                    words = np.zeros((columns * rows, bytes_ // 4), np.uint32)
                    words[:, h["KM_PUT_CPUS"] // 4] = cpus
                    words[:, h["KM_PUT_LANDING"] // 4] = landings[far].address
                    words[:, h["KM_PUT_LANDED"] // 4] = landings[island].address
                    tensors = [
                        session.tensor(words, "uint32", shard_grid=(columns, rows)),
                        session.tensor(  # zeros: no ask, no count, no done word
                            np.zeros((columns * rows, relay_bytes // 4)),
                            "uint32",
                            shard_grid=(columns, rows),
                        ),
                    ]
                    area, relay = (tensor.address for tensor in tensors)
                    held |= {"link area": tensors[:1], "relay": tensors[1:]}
                self._held.append(held)
                self._link_areas += [t for each in held.values() for t in each]
                self._link_areas.append(landings[island])
            if area != (self._areas or [(area, 0)])[0][0]:
                raise KomorebiError(
                    f"island {island}: the link area is at {area:#x}, "
                    f"island 0's at {self._areas[0][0]:#x}"
                )
            if plans and own.address != plans[0].own:  # a far confirmation's place
                raise KomorebiError(
                    f"island {island}: the own words are at {own.address:#x}, "
                    f"island 0's at {plans[0].own:#x}"
                )
            self._areas.append((area, relay))
            relays.append(relay)
            # what crossed: a page a link CPU, as COUNTS writes it (_crossings)
            self._counts.append(
                session.empty(
                    (2 * len(self.link_cores[island]), CROSS_PAGE // 4), "uint32"
                )
            )
            core = self.link_cores[island][0]
            plans.append(LinkPlan(self.graph, island, far, core, area))
            plans[-1].own = own.address
            plans[-1].table = self._held[-1 if self.split == 1 else 0]["island table"]
            plans[-1].table = plans[-1].table[0].address
            plans[-1].ports = self._port_words(island)
            if apart:
                words = np.zeros((bytes_ + relay_bytes) // 4, np.uint32)
                words[h["KM_PUT_CPUS"] // 4] = cpus
                words[h["KM_PUT_LANDING"] // 4] = landings[far].address
                words[h["KM_PUT_LANDED"] // 4] = landings[island].address
                plans[-1].apart = (
                    [(c % columns, c // columns) for c in self.link_cores[island]],
                    words.tolist(),
                )
            for k, link_core in enumerate(self.link_cores[island]):
                there, _, peer = self.pair(island, k)
                place = self.noc_of(there, peer)
                plans[-1].relay_tiles[link_core] = (place & 0xFF) | (place >> 8) << 16
            plans[-1].relay = [relays[island], 0, plans[-1].relay_tiles[core]]
            self.log(
                f"island {island}: commands across on link core {core}, channel "
                f"{self.channels[island][0]}, area {area:#x}"
                + (" (link cores only)" if apart else "")
                + f", own words {own.address:#x}"
            )
        for plan in plans:
            plan.relay[1] = relays[plan.far]
        return plans

    def _landings(self, h):
        """Each island's landing of km::put_bytes' batches, in DRAM: a page an island
        and put line owner (a split card: island 0's for every island)."""
        size = h["KM_PUT_BATCH_BYTES"]
        landings = []
        for island, session in enumerate(self.sessions):
            if self.split > 1 and island:
                landings.append(landings[0])
                continue
            landings.append(
                session.empty(
                    (len(self.sessions) * h["KM_QUEUE_LINES"], size // 4), "uint32"
                )
            )
            if landings[-1].aligned_page_bytes != size:
                pages = landings[-1].aligned_page_bytes
                for landing in landings:
                    landing.free()
                raise KomorebiError(f"the put batches' landing: {pages} bytes a page")
        return landings

    def _check_links(self, runs, plans):
        """Every island's dispatcher has the tile and LIM base its link line took for
        the far one's (its own); the control blocks are at one address (what crosses
        through the relay lands at its sender's addresses)."""
        if plans[0] is None or self.split > 1:  # one card: a dispatcher a cluster
            return
        ports = [run.backend.port for run in runs]
        seen = {
            (port.l2_x, port.l2_y, port.lim_base, port.upload_bytes) for port in ports
        }
        if len(seen) != 1:
            raise KomorebiError(
                "the islands' dispatchers differ in tile, LIM base or upload region: "
                f"{sorted(seen)}"
            )
        controls = sorted({run.backend.control.address for run in runs})
        if len(controls) != 1:
            raise KomorebiError(f"the islands' control blocks differ: {controls}")
        direct = {
            getattr(run.backend, "direct", None) and run.backend.direct.address
            for run in runs
        }
        if len(direct) != 1:  # a direct line across lands at its own address there
            raise KomorebiError(f"the islands' direct areas differ: {direct}")

    def _remembered(self, expect):
        """The kernel buffer this session opens at: the size remembered for the script
        (or `expect`, a made Program) with headroom, else KERNEL_BYTES."""
        shape = f"islands {len(self.cards)} cards x {self.split}"
        self._memory = remembered.Memory(
            self.log, expect, shape, default=f"{KERNEL_BYTES:,} B", room=True
        )
        size = self._memory.opened or KERNEL_BYTES
        return min(size, session_module.KERNEL_BUFFER_MAX)

    def _remember(self, program):
        """The kernel buffer the islands' load needed (its largest), kept for the
        script's next run."""
        total = max((one._last_total for one in self.sessions), default=0)
        if self._memory is not None and total:
            self._memory.loaded(program, total)

    def _refused(self, error):
        """A load refused, its kernels over this session's buffer or its L1 table
        failed below a larger buffer than they need: their size kept, so that the
        script's next run opens at it; the log says so."""
        total = getattr(getattr(error, "sizes", None), "total", 0) or max(
            (one._last_total for one in self.sessions), default=0
        )
        if self._memory is None or not total:
            return
        need = kernel_buffer.need(total)
        smaller = isinstance(error, L1BudgetError) and need < self.kernel_bytes
        if not (smaller or isinstance(error, KernelSizeError)):
            return
        self._memory.wanted(total, smaller)
        self.log(
            f"kernel buffer: the kernels take {total:,} B, this session's buffer is "
            f"{self.kernel_bytes:,} B; remembered, the script's next run opens at "
            f"{need:,} B"
        )

    def close(self):
        """Stops every island's run through the X280 runtime, closes the islands'
        sessions, then the mesh (not while a grid did not return). Every island's
        session is tried; the first error is raised after, its island counted stuck."""
        if self._memory is not None:
            self._memory.closed()
        failed, first = [], None
        try:
            if self.split > 1 and self.run is not None and not self.run.stopped:
                self.run.abort()
            for island, session in enumerate(self.sessions):
                try:
                    session.close()
                except BaseException as error:  # noqa: BLE001 (raised after the rest)
                    failed.append(island)
                    first = error if first is None else first
        finally:
            stuck = [
                i
                for i, session in enumerate(self.sessions)
                if session.stuck or i in failed
            ]
            if self.mesh is not None and not stuck:
                self.log("closing the mesh")
                self.mesh.close()
                self.mesh = None
                session_module._LIVE.discard(self)
            elif stuck and self.mesh is not None:
                self.log(f"mesh not closed: the grid of islands {stuck} did not return")
            if self.mesh is None and not stuck:
                session_module._LIVE.discard(self)
        if first is not None:
            raise first

    def _release(self, why, close_device):
        """The exit hooks' clean-up: every island's session first, then the mesh."""
        if self.split > 1 and self.run is not None and not self.run.stopped:
            try:
                self.run.abort()
            except Exception as error:  # noqa: BLE001 (the process ends: logged)
                self.log(f"{why}: abort raised {type(error).__name__}: {error}")
        for session in self.sessions:
            session._release(why, False)
        if close_device:
            try:
                self.close()
            except Exception as error:  # noqa: BLE001 (the process ends: logged)
                self.log(f"{why}: close raised {type(error).__name__}: {error}")


def _split_pairs(parts, links, shape):
    """The wired pairs of a split card's `shape` (SPLIT_SHAPES) of `parts` islands:
    Link values, `links` wires a pair (the 2 x 2 grid: its wrap doubles each pair)."""
    if shape == "grid":
        if parts != 4:
            raise ValueError(f"shape='grid': 4 islands (2 x 2), not {parts}")
        order = (0, 1, 3, 2)  # the 2 x 2 row by row: 0 1 / 2 3, around it
        return [
            islands.Link(*sorted((order[i], order[(i + 1) % 4])), 2 * links)
            for i in range(4)
        ]
    ring = shape == "ring" and parts > 2
    return [
        islands.Link(a, b, links)
        for a in range(parts)
        for b in range(a + 1, parts)
        if shape == "mesh" or b == a + 1 or (ring and (a, b) == (0, parts - 1))
    ]


def check_wired(graph, split):
    """Refuses by name a graph the session cannot wire (Graph itself refuses more
    than islands.ISLANDS_MAX): more than PORTS_MAX neighbours, and beyond two islands
    more than two wires a pair (a link core a neighbour, a link a CPU)."""
    count = len(graph.islands)
    for island in range(count):
        near = graph.neighbours(island)
        if len(near) > islands.PORTS_MAX:
            raise ValueError(
                f"island {island}: {len(near)} neighbours, a card's ports "
                f"{islands.PORTS_MAX}"
            )
    wires = {link.count for link in graph.links}
    if count > 2 and max(wires) > 2:
        raise ValueError(
            f"{count} islands with {max(wires)} wires a pair: beyond two islands a "
            "link core a neighbour carries two"
        )


def _split_graph(columns, rows, parts, links, shape="mesh"):
    """One card of `columns` x `rows` cores as `parts` islands of whole rows, the pairs
    of `shape` joined by `links` wires (_split_pairs); an island's link cores its last
    cores (two CPUs a core, a link each, a neighbour after another), its ranks the
    cores before them."""
    pairs = _split_pairs(parts, links, shape)
    parts_, link_cores = [], []
    for part in range(parts):
        start, length = (
            columns * size for size in islands.partition(rows, part, parts)
        )
        cores = tuple(range(start, start + length))
        taken = sum((pair.count + 1) // 2 for pair in pairs if part in (pair.a, pair.b))
        parts_.append(islands.Island(cores, columns))
        link_cores.append(cores[length - taken :])
    return islands.Graph(parts_, pairs, link_cores=link_cores)
