"""A session over several cards, each card's chip an island: one fabric mesh, one X280
runtime over it, a dispatcher and the persistent grid on every island. Ranks are flat
and contiguous by island; a command's group is split by island on the host, a tensor
is placed by island. `Session(cards=[...])` with more than one card returns this."""

import copy
import os
import secrets
import time

import numpy as np

from komorebi import channel as channel_module
from komorebi import dram, formats, islands, native
from komorebi import session as session_module
from komorebi.errors import CardBusy, KomorebiError, WaitTimeout
from komorebi.program import Program

M32 = 0xFFFFFFFF
# The link service (km_link.hpp): LinkArgs words, the region's layout, its sizes.
LINK_ARGS = 12
LINK_TEARDOWN, LINK_CURSOR = 0x60, 0x70
LINK_OUTBOX_AT, LINK_PREFIX, LINK_PACKET_BYTES = 0x190, 16, 4352
LINK_TAIL = 0x20 + 2 * 16 * 256  # the reports, two passes of 256 status lines
LINK_OUTBOX, LINK_INBOX = 128, 4  # entries; packets


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


class IslandTensor:
    """A DRAM tensor of an island session: one tensor of the whole shape on every
    island at one address, page p in the DRAM of the island `placement` gives it (a
    copy: every island). Kernels take its `address` and `accessor_args()` unchanged."""

    def __init__(self, parts, placement):
        self.parts = tuple(parts)
        self.placement = placement
        first = self.parts[0]
        for island, part in enumerate(self.parts):
            if part.address != first.address:
                raise KomorebiError(
                    f"island {island}: the tensor is at {part.address:#x}, island 0 "
                    f"at {first.address:#x}; make an island session's tensors before "
                    "anything else is allocated on one island alone"
                )
        self.shape, self.dtype, self.layout = first.shape, first.dtype, first.layout
        self.pages, self.page_bytes = first.pages, first.page_bytes
        self.aligned_page_bytes = first.aligned_page_bytes
        self.address = first.address
        self.in_dram = True

    def accessor_args(self):
        """The tensor accessor arguments, the same on every island."""
        args = [part.accessor_args() for part in self.parts]
        if any(other != args[0] for other in args[1:]):
            raise KomorebiError("the islands' accessor arguments differ")
        return args[0]

    def holders(self, first, pages):
        """[(island, first, pages)] of the pages first .. first + pages - 1 by the
        island that holds each (a copy: island 0)."""
        if self.placement.copy:
            return [(0, first, pages)]
        out = []
        for island, (start, length) in enumerate(self.placement.parts):
            low, high = max(first, start), min(first + pages, start + length)
            if high > low:
                out.append((island, low, high - low))
        return out

    def free(self):
        for part in self.parts:
            part.free()


class IslandRun:
    """The runs of one program on every island, as one run over flat ranks. A ticket
    is a command of a group of flat ranks; its parts went to their islands' dispatchers
    and it completes when every part has. Errors name the island (`error.island`)."""

    def __init__(self, graph, runs, linked=False):
        self.graph = graph
        self.runs = tuple(runs)
        self.linked = linked  # a link joins the islands: a dynamic RUN takes them all
        self.cores = graph.ranks
        self.program = self.runs[0].program
        self._next = 1  # the next ticket
        # per island, the handle of each ticket's part on that island (0: none)
        self._parts = [np.zeros(1024, np.int64) for _ in self.runs]

    def island(self, island):
        """Island `island`'s own Run (local ranks)."""
        return self.runs[island]

    def _grow(self, tickets):
        while self._next + tickets > len(self._parts[0]):
            self._parts = [
                np.concatenate([parts, np.zeros_like(parts)]) for parts in self._parts
            ]

    def submit(self, handler, group=None, args=(), exclusive=False, result=False):
        """Queues a command for `handler` on flat ranks [first, first + count)
        (default every rank): a part on each island it touches. Returns its ticket."""
        group = (0, self.cores) if group is None else tuple(group)
        parts = self.graph.split(group)
        self._linked_group(handler, group)
        self._grow(1)
        ticket = self._next
        for island, _first, _count in parts:
            try:  # the flat group word unchanged: the dispatcher cuts its part
                handle = self.runs[island].submit(
                    handler, group, args, exclusive=exclusive, result=result
                )
            except KomorebiError as error:
                raise _named(error, island) from None
            self._parts[island][ticket] = handle
        self._next += 1
        return ticket

    def submit_many(self, handler, firsts, counts, args=None, exclusive=False):
        """n commands at once, as Run.submit_many takes them, the groups in flat
        ranks; each island gets its parts in one submit_many. The tickets, a range."""
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
        for row in range(rows if self.linked else 0):
            named = handler if scalar else handlers[row]
            self._linked_group(named, (int(first[row]), int(count[row])))
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
            try:  # the flat group words unchanged: the dispatcher cuts its part
                handles = run.submit_many(
                    named,
                    first[mine],
                    count[mine],
                    None if args is None else args[mine],
                    exclusive,
                )
            except KomorebiError as error:
                raise _named(error, island) from None
            self._parts[island][tickets[mine]] = np.asarray(handles, np.int64)
        self._next += rows
        return range(int(tickets[0]), int(tickets[0]) + rows) if rows else range(0)

    def _linked_group(self, handler, group):
        """Over a link, a dynamic RUN's group is every rank: each island's part ends
        at its link core."""
        if not self.linked or self.program.messages is None:
            return
        number, _ = self.runs[0]._handler(handler)
        if number in self.program.messages.handlers and group != (0, self.cores):
            raise ValueError(
                f"group {group}: a dynamic run over a link takes every rank "
                f"(0, {self.cores})"
            )

    def send(self):
        """Stores every island's queued commands on its card, so that the islands run
        their parts at once (a run over a link needs every part started) while the
        host waits on one."""
        for island, run in enumerate(self.runs):
            try:
                run.store()
            except KomorebiError as error:
                raise _named(error, island) from None

    def wait(self, tickets, timeout=10.0):
        """Sends what is queued and waits, within one `timeout` for all, for every
        part of the tickets. Returns per ticket the tuple of its parts' events in
        island order (one tuple, or a list for a list of tickets)."""
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
                raise _named(error, island) from None
            except KomorebiError as error:
                raise _named(error, island) from None
            for ticket, event in zip(wanted[mine].tolist(), got):
                events.setdefault(ticket, []).append(event)
        out = [tuple(events[ticket]) for ticket in wanted.tolist()]
        return out[0] if single else out

    def _store(self, wanted, deadline):
        """Until the tickets' parts arrived or the deadline: one native round an island
        in turn (store what its ring has room for, file its events), so that every
        island is fed while the host waits; Run.wait then returns or raises."""
        pending = {}
        for island in range(len(self.runs)):
            handles = self._parts[island][wanted]
            pending[island] = np.unique(handles[handles != 0])
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
                    raise _named(error, island) from None
                pending[island] = handles[~run._arrived(handles)]

    def inspect(self, island, cores=None, timeout=10.0):
        """Run.inspect of island `island`'s cores [first, first + count) of `cores`
        (its own core numbers; default all), asked of that island's dispatcher."""
        try:
            return self.runs[island].inspect(cores, timeout)
        except KomorebiError as error:
            raise _named(error, island) from None

    def _tickets(self, island, handles):
        """The tickets whose part on `island` has one of these handles."""
        parts = self._parts[island][: self._next]
        return [int(t) for t in np.flatnonzero(np.isin(parts, handles)) if t]

    def drain(self, timeout=10.0):
        """Sends everything queued and waits, within one `timeout`, for every
        island's events; returns how many events it waited for, every island's
        together."""
        deadline = time.monotonic() + timeout
        self.send()
        waited = 0
        for island, run in enumerate(self.runs):
            try:
                waited += run.drain(_left(deadline))
            except KomorebiError as error:
                raise _named(error, island) from None
        return waited

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
                    failure = _named(error, island)
        if failure is not None:
            raise failure
        return self.result()

    def abort(self):
        for run in self.runs:
            run.abort()

    def result(self):
        """Every island's result block, in island order; valid after the stop."""
        return [run.result() for run in self.runs]

    def state(self):
        """Every rank's state words, as Run.state gives them, in flat rank order."""
        states = [run.state() for run in self.runs]
        return [
            states[island][core]
            for island, cores in enumerate(self.graph.rank_cores)
            for core in cores
        ]


class IslandSession:
    """Cards `cards` as islands of one 1 x n fabric mesh in this process: the X280
    runtime once over the mesh, then per island a Session given its device. `links` 0:
    nothing crosses; 1 (two cards): a dynamic run's messages cross through each
    island's link core (its last core, no message rank). Use as a context manager."""

    def __init__(
        self,
        cards,
        links=0,
        pll_mhz=None,
        tree=None,
        log=None,
        grid_timeout=10.0,
        kernel_bytes=None,
        channel=channel_module.REGION_DEFAULT,
    ):
        self.cards = tuple(int(card) for card in cards)
        if len(self.cards) < 2 or len(set(self.cards)) != len(self.cards):
            raise ValueError(f"cards {cards}: two or more different cards")
        if links not in (0, 1) or (links and len(self.cards) != 2):
            raise ValueError(f"links={links}: 0, or 1 between two cards")
        self.links = links
        self.log = log or (lambda message: None)
        self._options = {
            "pll_mhz": pll_mhz,
            "tree": tree,
            "grid_timeout": grid_timeout,
            "channel": channel,
        }
        self.kernel_bytes = kernel_bytes
        self._worker_l1 = session_module.worker_l1_size(kernel_bytes)
        self.tree = tree
        self.mesh = None
        self.chips = None
        self.sessions = ()
        self.graph = None
        self.run = None
        self._regions = []  # the link's L1 region on every core, an island each

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
        Session on each island's device; the graph of the islands' ranks."""
        listed = ",".join(str(card) for card in self.cards)
        visible = os.environ.get("TT_VISIBLE_DEVICES")
        if visible is not None and visible != listed:
            raise ValueError(f"cards {listed} are not the visible ones ({visible})")
        os.environ["TT_VISIBLE_DEVICES"] = listed
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
        module = native.load_module(self.tree)
        self.log(f"opening cards {listed} as one 1 x {len(self.cards)} fabric mesh")
        self.mesh = module.Islands(len(self.cards), self._worker_l1)
        session_module._hook_exit()
        session_module._LIVE.add(self)
        try:
            self.chips = list(self.mesh.chips())
            if sorted(self.chips) != sorted(self.cards):
                raise KomorebiError(
                    f"the mesh's chips {self.chips} are not the cards {listed}"
                )
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
            parts = []
            for session in self.sessions:
                columns, rows = session.grid
                parts.append(islands.Island(tuple(range(columns * rows)), columns))
            links = [islands.Link(i, i + 1) for i in range(len(parts) - 1)]
            # a link core stays a core of the group (no message rank): ranks are cores
            self.graph = islands.Graph(parts, links, reserve=False)
        except BaseException:
            self.close()
            raise

    def _island_log(self, island):
        return lambda message: self.log(f"island {island}: {message}")

    @property
    def ranks(self):
        return self.graph.ranks

    def island(self, island):
        """Island `island`'s own Session (local ranks, its own tensors)."""
        return self.sessions[island]

    def tensor(self, data, dtype, layout="row", like=None, copy=False):
        """A DRAM tensor holding `data` as Session.tensor takes it, placed by
        islands.place: split by island along the leading dimension as its ranks
        partition it (`like`: the group (first, count) it follows), or a `copy`."""
        raw, shape = formats.to_bytes(data, dtype, layout)
        tensor = self.empty(shape, dtype, layout, like, copy)
        rows = raw.reshape(tensor.pages, tensor.page_bytes)
        for island, part in enumerate(tensor.parts):
            if tensor.placement.copy:
                mine = rows
            else:
                start, length = tensor.placement.parts[island]
                mine = np.zeros_like(rows)
                mine[start : start + length] = rows[start : start + length]
            self.sessions[island].write(part, mine)
        return tensor

    def empty(self, shape, dtype, layout="row", like=None, copy=False):
        """A tensor as tensor() places it, not written."""
        parts = [session.empty(shape, dtype, layout) for session in self.sessions]
        first = parts[0]
        placement = islands.place(
            self.graph,
            first.pages,
            islands.leading_units(shape, layout),
            like,
            copy,
        )
        try:
            return IslandTensor(parts, placement)
        except KomorebiError:
            for part in parts:
                part.free()
            raise

    def read(self, *tensors, first=0, pages=None):
        """The raw pages of island tensors as Session.read gives them, each page read
        from the island that holds it, through that island's host channel."""
        if (first or pages is not None) and len(tensors) != 1:
            raise ValueError("a page range is read from one tensor")
        out = []
        for tensor in tensors:
            start, count = dram.page_range(first, pages, tensor.pages)
            rows = np.empty((count, tensor.page_bytes), np.uint8)
            for island, low, length in tensor.holders(start, count):
                (got,) = self.sessions[island].read(
                    tensor.parts[island], first=low, pages=length
                )
                rows[low - start : low - start + length] = got
            out.append(rows)
        return out

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
        """Writes `pages` ([n, page_bytes] uint8) into pages first .. first + n - 1,
        each into the island that holds it (a copy: into every island)."""
        pages = np.ascontiguousarray(pages, np.uint8)
        if pages.ndim != 2 or pages.shape[1] != tensor.page_bytes:
            raise ValueError(f"pages are [n, {tensor.page_bytes}] uint8 rows")
        start, count = dram.page_range(first, len(pages), tensor.pages)
        if tensor.placement.copy:
            targets = [(island, start, count) for island in range(len(self.sessions))]
        else:
            targets = tensor.holders(start, count)
        for island, low, length in targets:
            self.sessions[island].write(
                tensor.parts[island], pages[low - start : low - start + length], low
            )

    def load(self, program, **options):
        """Loads `program` on every island in turn (its grid, then its dispatcher), or
        a list of Programs, one an island in island order, with Session.load's
        options; returns an IslandRun over flat ranks."""
        if self.run is not None and not self.run.stopped:
            raise RuntimeError("a run is loaded: stop it first")
        programs = list(program) if isinstance(program, list | tuple) else [program]
        if len(programs) == 1:
            programs *= len(self.sessions)
        if len(programs) != len(self.sessions) or not all(
            isinstance(one, Program) for one in programs
        ):
            raise ValueError(
                f"an island session loads one Program or a list of "
                f"{len(self.sessions)}, one an island"
            )
        for one in programs:
            if one.threads:
                raise ValueError(
                    f"{one.name}: X280 threads over islands are not done yet"
                )
        if self.links:
            programs = self._linked(programs)
        runs = []
        try:
            for island, (session, one) in enumerate(zip(self.sessions, programs)):
                try:
                    island_of = (self.graph, island)
                    runs.append(session.load(one, island=island_of, **options))
                except KomorebiError as error:
                    raise _named(error, island) from None
        except BaseException:
            for run in runs:
                run.abort()
            raise
        self.run = IslandRun(self.graph, runs, linked=bool(self._regions))
        return self.run

    def link_core(self, island):
        """Island `island`'s link core, its own core number: its last core (the one
        place the host names it)."""
        return self.graph.count(island) - 1

    def _linked(self, programs):
        """The dynamic programs with the link's arguments after their own reader and
        writer arguments, KM_LINKS defined, a region on every core of each island
        and one token for both sides' hello."""
        for region in self._regions:
            region.free()
        self._regions = []
        dynamic = [one.messages for one in programs if one.messages is not None]
        if not dynamic:
            return programs
        if any(messages.x280_ranks for messages in dynamic):
            raise ValueError("X280 ranks over a link are not done yet")
        entry = LINK_PREFIX + max(messages.message_bytes for messages in dynamic)
        size = (
            LINK_OUTBOX_AT + LINK_OUTBOX * entry + (LINK_INBOX + 1) * LINK_PACKET_BYTES
        ) + LINK_TAIL
        for session in self.sessions:
            columns, rows = session.grid
            self._regions.append(
                session.empty(
                    (columns * rows, (size + 63) // 64 * 16),
                    "uint32",
                    shard_grid=session.grid,
                )
            )
        token = secrets.randbits(31) | 1
        out = []
        for island, one in enumerate(programs):
            if one.messages is None:
                out.append(one)
                continue
            words = self._link_words(island, one.messages.message_bytes, token)
            out.append(_with_link(one, words))
        return out

    def _link_words(self, island, message_bytes, token):
        """LinkArgs of island `island`'s cores toward the other island."""
        far = 1 - island
        far_core = self.link_core(far)
        columns = self.sessions[far].grid[0]
        x, y = self.sessions[far].device.worker_core(
            far_core % columns, far_core // columns
        )
        _count, channel, _x, _y = self.mesh.link(island, far, 0)
        region = self._regions[island].address
        words = [
            self.link_core(island),
            region,
            LINK_OUTBOX,
            LINK_INBOX,
            LINK_PACKET_BYTES // (LINK_PREFIX + message_bytes),
            island,
            x | y << 16,
            self._regions[far].address,
            channel,
            region + LINK_TEARDOWN,
            region + LINK_CURSOR,
            token,
        ]
        assert len(words) == LINK_ARGS
        return words

    def close(self):
        """Stops every island's run through the X280 runtime, closes the islands'
        sessions, then the mesh (not while a grid did not return)."""
        self._regions = []
        try:
            for session in self.sessions:
                session.close()
        finally:
            stuck = [i for i, session in enumerate(self.sessions) if session.stuck]
            if self.mesh is not None and not stuck:
                self.log("closing the mesh")
                self.mesh.close()
                self.mesh = None
                session_module._LIVE.discard(self)
            elif stuck:
                self.log(f"mesh not closed: the grid of islands {stuck} did not return")

    def _release(self, why, close_device):
        """The exit hooks' clean-up: every island's session first, then the mesh."""
        for session in self.sessions:
            session._release(why, False)
        if close_device:
            try:
                self.close()
            except Exception as error:  # noqa: BLE001 (the process ends: logged)
                self.log(f"{why}: close raised {type(error).__name__}: {error}")


def _with_link(program, words):
    """A copy of `program` whose reader and writer take `words` after their own
    arguments (as many on both), built with KM_LINKS."""
    own = program.core_args(0)
    after = len(own[0])
    if len(own[1]) != after:
        raise ValueError(
            f"{program.name}: over a link the reader and the writer take as many "
            f"arguments ({len(own[0])}, {len(own[1])})"
        )

    def args(core):
        reader, writer, compute = program.core_args(core)
        if len(reader) != after or len(writer) != after:
            raise ValueError(
                f"{program.name}: core {core}'s reader and writer arguments are not "
                f"{after} each"
            )
        return [*reader, *words], [*writer, *words], compute

    linked = copy.copy(program)
    linked.args = args
    linked.defines = {**program.defines, "KM_LINKS": 1, "KM_LINK_ARGS_AFTER": after}
    return linked
