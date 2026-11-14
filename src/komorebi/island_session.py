"""A session over several cards, each card's chip an island: one fabric mesh, one X280
runtime over it, a dispatcher and the persistent grid on every island. Ranks are flat
and contiguous by island; a command's group is split by island on the host, a tensor
is placed by island. `Session(cards=[...])` returns this; one card is one island."""

import copy
import os
import secrets
import struct
import time

import numpy as np

from komorebi import abi, dram, formats, islands, native
from komorebi import channel as channel_module
from komorebi import session as session_module
from komorebi.errors import CardBusy, KomorebiError, WaitTimeout
from komorebi.program import Program

M32 = 0xFFFFFFFF
SLICE_S = 0.0  # an island's turn in wait_together: one native round (IslandRun)
THREADS = 3  # thread numbers an island: one a work cluster (KMX_THREAD)
# The message service on link core 0 (km_link.hpp): LinkArgs words, the region's
# sizes (words, outbox, inbox, sending packet, reports and two passes of lines).
LINK_ARGS = 9
LINK_OUTBOX_AT, LINK_PREFIX, LINK_PACKET_BYTES = 0x190, 16, 4352
LINK_TAIL = 0x20 + 2 * 16 * 256
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


class LinkPlan:
    """Island `island`'s link toward island `far` for commands across: its link core 0
    (`core`, its index in the island's cores, after the ranks), whose link 0 CPU holds
    link 0's connection and sends the lines, and that core's link area (`area`)."""

    def __init__(self, graph, island, far, core, area):
        self.graph, self.island, self.far = graph, island, far
        self.core, self.area = core, area
        # the relay words of the island area (km_relay.hpp): this island's relay
        # area, the far island's, the far link core 0's NoC tile x | y << 16
        self.relay = [0, 0, 0]

    def line(self, dispatch):
        """The dispatcher's link line and island table (KMD_LIM_LINK). The far island's
        dispatcher tile and link block are taken as this one's (checked after load)."""
        headers = abi.headers()
        words = [0] * (headers["KMD_LINK_COUNT"] // 4)
        words[headers["KMD_L_MARK"]] = headers["KMD_LINK_MARK"]
        words[headers["KMD_L_ISLAND"]] = self.island
        words[headers["KMD_L_ISLANDS"]] = len(self.graph.islands)
        words[headers["KMD_L_CORE"]] = self.core
        words[headers["KMD_L_AREA"]] = self.area
        words[headers["KMD_L_FAR"]] = self.far
        words[headers["KMD_L_FAR_TILE"]] = dispatch.l2_x | dispatch.l2_y << 16
        words[headers["KMD_L_FAR_BLOCK"]] = dispatch.lim_base + headers["KMD_LIM_LINK"]
        table = headers["KMD_LINK_TABLE"] // 4
        for island in range(len(self.graph.islands)):
            threads = self.graph.thread_numbers(island)
            words[table + 2 * island] = (
                self.graph.first(island) | self.graph.count(island) << 16
            )
            words[table + 2 * island + 1] = threads.start | len(threads) << 16
        return struct.pack(f"<{len(words)}I", *words)


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

    def placement_args(self):
        """What km::Placement reads (compile arguments, after the accessor's): the
        islands, copy, then (first page, pages) of each island."""
        parts = [word for part in self.placement.parts for word in part]
        return [len(self.placement.parts), int(self.placement.copy), *parts]

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
        for part in {id(part): part for part in self.parts}.values():  # once each
            part.free()


class IslandRun:
    """The runs of one program on every island, as one run over flat ranks. A ticket
    is a command of a group of flat ranks; its parts went to their islands' dispatchers
    and it completes when every part has. Errors name the island (`error.island`)."""

    def __init__(self, graph, runs, linked=False):
        self.graph = graph
        self.runs = tuple(runs)
        # a dynamic program's messages cross: a dynamic RUN takes every rank, and link
        # core 0 of each island serves it (a PAGES command of mode MESSAGES)
        self.linked = linked
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
        if self._dynamic(handler):
            self._serve_links(group)
        self._grow(1)
        ticket = self._next
        for island, _first, _count in parts:
            try:  # the flat group word unchanged, marked a part: the dispatcher cuts
                # its part and does not cross the line (the other parts went as well)
                handle = self.runs[island].submit(
                    handler, group, args, exclusive=exclusive, result=result, part=True
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
            if self._dynamic(handler if scalar else handlers[row]):
                raise ValueError(
                    "a dynamic RUN over a link is submitted alone (submit)"
                )
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
                raise _named(error, island) from None
            self._parts[island][tickets[mine]] = np.asarray(handles, np.int64)
        self._next += rows
        return range(int(tickets[0]), int(tickets[0]) + rows) if rows else range(0)

    def _dynamic(self, handler):
        """Whether `handler` is a dynamic handler of a program whose messages cross."""
        if not self.linked:
            return False
        number, _ = self.runs[0]._handler(handler)
        return number in self.program.messages.handlers

    def _serve_links(self, group):
        """Before a dynamic RUN over a link: its group is every rank, and link core 0
        of each island is given the RUN's group word to serve its messages across."""
        if group != (0, self.cores):
            raise ValueError(
                f"group {group}: a dynamic RUN over a link takes every rank "
                f"(0, {self.cores})"
            )
        h = abi.headers()
        words = [0] * 10
        words[h["KM_PAGES_FROM"]] = group[0] | group[1] << 16
        words[h["KM_PAGES_COUNTER"]] = h["KM_PAGES_MESSAGES"] << 24 | 1 << 16
        for island, run in enumerate(self.runs):
            try:
                run._queue_command(
                    h["KM_HANDLER_PAGES"], self.graph.link_group(island, 0, 1), words
                )
            except KomorebiError as error:
                raise _named(error, island) from None

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
    """Cards as islands of one 1 x n fabric mesh, the X280 runtime once over it, a
    Session an island (one card: its own; `islands` 2 to 4: it split into islands of
    whole rows). `links` 1 to 4 (two cards): links pages cross on, a link core (the last
    cores) a pair; 0: every core a rank. A context manager."""

    LINKS_MAX = 4  # a byte each in a PAGES command's link word
    # How a core's ask for pages another island holds reaches the holder: "core" (its
    # own read call, km::read_pages, puts an ask line into its link core 0, sent on
    # link 0); "host" (the host queues the holder's send: ask, ask_range; the read call
    # only waits for the pages).
    ASKS_BY = "core"

    def __init__(
        self,
        cards,
        links=0,
        islands=None,
        pll_mhz=None,
        tree=None,
        log=None,
        grid_timeout=10.0,
        kernel_bytes=None,
        channel=channel_module.REGION_DEFAULT,
        slow_code=False,
    ):
        self.cards = tuple(int(card) for card in cards)
        if not self.cards or len(set(self.cards)) != len(self.cards):
            raise ValueError(f"cards {cards}: one or more different cards")
        if not 0 <= links <= self.LINKS_MAX or (links and len(self.cards) != 2):
            raise ValueError(
                f"links={links}: 0, or 1 to {self.LINKS_MAX} between two islands"
            )
        # two islands always have link 0 (commands across ride it): links 0 is 1
        self.links = links or (1 if len(self.cards) == 2 else 0)
        self.split = islands or 1  # the islands one card is split into
        if self.split > 1 and (len(self.cards) != 1 or links):
            raise ValueError(f"islands={islands}: one card, without links")
        self.log = log or (lambda message: None)
        self._options = {
            "pll_mhz": pll_mhz,
            "tree": tree,
            "grid_timeout": grid_timeout,
            "channel": channel,
            "slow_code": slow_code,
        }
        self.kernel_bytes = kernel_bytes
        self._worker_l1 = session_module.worker_l1_size(kernel_bytes)
        self.tree = tree
        self.mesh = None
        self.chips = None
        self.sessions = ()
        self.graph = None
        self.run = None
        self.link_cores = ()  # per island: the local cores (core table) of its links
        self._regions = []  # the message service's L1 region on every core, by island
        self.channels = ()  # per island: the Ethernet channel of each link it sends on

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
            self.chips, self.sessions = [self.cards[0]], (session,) * self.split
            columns, rows = session.grid
            self.graph = islands.one(columns * rows, columns)
            if self.split > 1:  # one device, a dispatcher an island
                self.graph = islands.split_card(
                    columns * rows, columns, self.split, rows=True, reserve=False
                )
                session_module._LIVE.add(self)
                self.log(f"card {self.cards[0]} as {self.split} islands")
            return
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
            if self.links:
                self._open_links(module)
        except BaseException:
            self.close()
            raise

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

    def island(self, island):
        """Island `island`'s own Session (local ranks, its own tensors)."""
        return self.sessions[island]

    def tensor(self, data, dtype, layout="row", like=None, copy=False, by="rank"):
        """A DRAM tensor holding `data` as Session.tensor takes it, placed by
        islands.place: split along the leading dimension as the ranks partition it
        (`like`: the group it follows), even parts by island (`by` "island"), or a copy.
        """
        raw, shape = formats.to_bytes(data, dtype, layout)
        tensor = self.empty(shape, dtype, layout, like, copy, by)
        rows = raw.reshape(tensor.pages, tensor.page_bytes)
        if self.split > 1:  # one tensor in the card's DRAM
            self.sessions[0].write(tensor.parts[0], rows)
            return tensor
        for island, part in enumerate(tensor.parts):
            if tensor.placement.copy:
                mine = rows
            else:
                start, length = tensor.placement.parts[island]
                mine = np.zeros_like(rows)
                mine[start : start + length] = rows[start : start + length]
            self.sessions[island].write(part, mine)
        return tensor

    def empty(self, shape, dtype, layout="row", like=None, copy=False, by="rank"):
        """A tensor as tensor() places it, not written."""
        if self.split > 1:
            parts = [self.sessions[0].empty(shape, dtype, layout)] * self.split
        else:
            parts = [session.empty(shape, dtype, layout) for session in self.sessions]
        first = parts[0]
        placement = islands.place(
            self.graph,
            first.pages,
            islands.leading_units(shape, layout),
            like,
            copy,
            by,
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
        options; returns an IslandRun over flat ranks. Each Program brings its own
        X280 threads."""
        if self.run is not None and not self.run.stopped:
            raise RuntimeError("a run is loaded: stop it first")
        if self.split > 1:  # one launch on the card, a dispatcher an island
            runs = self.sessions[0].load_islands(program, self.graph, **options)
            self.run = IslandRun(self.graph, runs)
            return self.run
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
        if self.ASKS_BY not in ("core", "host"):
            raise ValueError(f"ASKS_BY={self.ASKS_BY!r}: 'core' or 'host'")
        if self.links:
            programs = self._linked(programs)
        defines = [one.defines for one in programs]
        if self.links:  # the link cores' CPUs serve PAGES (km_pages.hpp), while built
            asks = {"KM_ASKS_BY_HOST": 1} if self.ASKS_BY == "host" else {}
            for one, own in zip(programs, defines):
                one.defines = {**own, "KM_PAGE_LINKS": 1, **asks}
        plans = self._links()
        runs = []
        try:
            for island, (session, one) in enumerate(zip(self.sessions, programs)):
                try:
                    # one island: the card's run as it is without islands
                    island_of = (self.graph, island) if len(programs) > 1 else None
                    runs.append(
                        session.load(
                            one, island=island_of, link=plans[island], **options
                        )
                    )
                except KomorebiError as error:
                    raise _named(error, island) from None
            self._check_links(runs, plans)
        except BaseException:
            for run in runs:
                run.abort()
            raise
        finally:
            for one, own in zip(programs, defines):
                one.defines = own
        self.run = IslandRun(self.graph, runs, linked=bool(self._regions))
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
        h = abi.headers()
        tables = [list(run.backend.port.core_table()) for run in self.run.runs]
        if tables[0] != tables[1]:
            raise KomorebiError(
                "the islands' core tables differ: a link core that pushes a rank's "
                "pages finds the far core in its own table"
            )
        waits = []
        for here in range(2):
            there = 1 - here
            buffer, chunk = self.staging(here)
            links = list(self.channels[here]) + [h["KM_PAGES_NO_LINK"]] * self.LINKS_MAX
            for k, far in enumerate(self.link_cores[there]):
                words = [0] * 10
                words[h["KM_PAGES_TO_CORE"]] = chunk // 64 << 16
                words[h["KM_PAGES_COUNT"]] = self.port_address(
                    there, h["KM_PAGES_HELLO_PORT"]
                )
                words[h["KM_PAGES_COUNTER"]] = (
                    self.noc_of(there, far) | 1 << 16 | h["KM_PAGES_OPEN"] << 24
                )
                words[h["KM_PAGES_LINKS"]] = int.from_bytes(bytes(links[:4]), "little")
                words[h["KM_PAGES_BUFFER"]] = buffer
                handle = self.run.island(here)._queue_command(
                    h["KM_HANDLER_PAGES"], self.graph.link_group(here, k, 1), words
                )
                waits.append((here, handle))
        self.wait_together(waits, timeout)
        self.log(f"page links held, hello both ways: link cores {self.link_cores}")

    def noc_of(self, island, core):
        """x | y << 8: the NoC place of local core `core` of `island` (core table)."""
        columns = self.sessions[island].grid[0]
        x, y = self.sessions[island].device.worker_core(core % columns, core // columns)
        return int(x) | int(y) << 8

    def port_address(self, island, port):
        """The L1 address of port `port`'s count, the same on every core of `island`."""
        h = abi.headers()
        if not 0 <= port < h["KM_PORTS"]:
            raise ValueError(f"port {port} of {h['KM_PORTS']}")
        control = self.run.island(island).backend.control_addr
        return control + h["KM_CONTROL_PORT"] + port * h["KM_PORT_STRIDE"]

    def staging(self, island):
        """The L1 every core of `island` moves transfers through: (address, bytes of a
        chunk); two chunks. Pages asked from another island land there."""
        transfer = self.run.island(island).backend.transfer
        if transfer is None:
            raise KomorebiError(f"island {island}: the run has no transfer buffer")
        return transfer.address, transfer.chunk

    def send_pages(self, source, first, pages, to, core=None, counted=None, port=None):
        """Queues a send of pages first .. of `source` by its holder's link cores into
        the other island: the same pages of tensor `to`, or (`core`: a local core) L1
        from address `to`. Each batch adds its pages to port `port` of core `counted`
        (default `core`, else the first link core). Returns (island, handle)."""
        h = abi.headers()
        if not self.links:
            raise KomorebiError("no page links: open the session with links=1 to 4")
        holders = source.holders(first, pages)
        if len(holders) != 1:
            raise ValueError(f"pages {first} + {pages}: held by islands {holders}")
        here = holders[0][0]
        there = 1 - here
        if core is None and (
            not isinstance(to, IslandTensor)
            or to.aligned_page_bytes != source.aligned_page_bytes
        ):
            raise ValueError("a tensor target has the source's aligned page bytes")
        counted = counted if counted is not None else core
        counted = counted if counted is not None else self.link_cores[there][0]
        port = h["KM_PORTS"] - 1 if port is None else port
        buffer, chunk = self.staging(here)
        links = list(self.channels[here]) + [h["KM_PAGES_NO_LINK"]] * self.LINKS_MAX
        words = [0] * 10
        words[h["KM_PAGES_FROM"]] = source.address
        words[h["KM_PAGES_PAGE"]] = source.aligned_page_bytes
        words[h["KM_PAGES_FIRST"]] = first
        words[h["KM_PAGES_PAGES"]] = pages
        words[h["KM_PAGES_TO"]] = to.address if core is None else to
        target = h["KM_PAGES_DRAM"] if core is None else self.noc_of(there, core)
        words[h["KM_PAGES_TO_CORE"]] = target | chunk // 64 << 16
        words[h["KM_PAGES_COUNT"]] = self.port_address(there, port)
        words[h["KM_PAGES_COUNTER"]] = (
            self.noc_of(there, counted) | 1 << 16 | h["KM_PAGES_SEND"] << 24
        )
        words[h["KM_PAGES_LINKS"]] = int.from_bytes(bytes(links[:4]), "little")
        words[h["KM_PAGES_BUFFER"]] = buffer
        handle = self.run.island(here)._queue_command(
            h["KM_HANDLER_PAGES"], self.graph.link_group(here), words
        )
        return here, handle

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
        return self.send_pages(tensor, first, pages, to, core, core, port)

    def ask_range(self, tensor, first, pages, group, to, port):
        """One ask for every rank of `group` (first rank, ranks; all on one island):
        rank first + i gets pages first + i * pages .. of `tensor` (held by another
        island) into L1 address `to` of its core, counted at its port `port`. One
        command to the holder's link cores, which split the ranks among their CPUs.
        Returns (holder, handle)."""
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
        holder = holders[0][0]
        if not 0 <= port < h["KM_PORTS"] or ranks > 0xFFFF:
            raise ValueError(f"port {port}, ranks {ranks}")
        buffer, chunk = self.staging(holder)
        links = list(self.channels[holder]) + [h["KM_PAGES_NO_LINK"]] * self.LINKS_MAX
        words = [0] * 10
        words[h["KM_PAGES_FROM"]] = tensor.address
        words[h["KM_PAGES_PAGE"]] = tensor.aligned_page_bytes
        words[h["KM_PAGES_FIRST"]] = first
        words[h["KM_PAGES_PAGES"]] = pages
        words[h["KM_PAGES_TO"]] = to
        core = self.graph.rank_cores[island][local]
        words[h["KM_PAGES_TO_CORE"]] = core | chunk // 64 << 16
        words[h["KM_PAGES_COUNT"]] = self.port_address(island, port)
        words[h["KM_PAGES_COUNTER"]] = ranks | 1 << 16 | h["KM_PAGES_RANGE"] << 24
        words[h["KM_PAGES_LINKS"]] = int.from_bytes(bytes(links[:4]), "little")
        words[h["KM_PAGES_BUFFER"]] = buffer
        handle = self.run.island(holder)._queue_command(
            h["KM_HANDLER_PAGES"], self.graph.link_group(holder), words
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

    def cross(self, source, target, first=0, pages=None, timeout=10.0):
        """Writes pages first .. of `source` into the same pages of `target` where
        the two are held by different islands: sent over the links, waited for on the
        far island. Returns pages, bytes and seconds, the submit to the last count."""
        start, count = dram.page_range(first, pages, source.pages)
        sends = []
        for here, low, length in source.holders(start, count):
            for there, part, part_length in target.holders(low, length):
                if there != here:
                    sends.append((there, part, part_length))
        started = time.perf_counter()
        waits, far = [], []
        for there, low, length in sends:
            far.append(
                (there, self.wait_pages(there, self.link_cores[there][0], length))
            )
            waits.append(self.send_pages(source, low, length, target))
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
        """The dynamic programs with the message service's arguments after their own
        reader and writer arguments and KM_LINKS defined: a region on every core of
        each island, link core 0's NoC place across, one token for both hellos."""
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
        """LinkArgs of island `island`'s cores: its link core 0, the far one's."""
        far = 1 - island
        place = self.noc_of(far, self.link_cores[far][0])
        words = [
            self.link_cores[island][0],
            self._regions[island].address,
            LINK_OUTBOX,
            LINK_INBOX,
            LINK_PACKET_BYTES // (LINK_PREFIX + message_bytes),
            island,
            (place & 0xFF) | (place >> 8) << 16,
            self._regions[far].address,
            token,
        ]
        assert len(words) == LINK_ARGS
        return words

    def _links(self):
        """Each island's LinkPlan for commands across (two islands), else None each:
        its link core 0, a link area on every core (zeroed: the asks' counts start at
        0), at one address on both islands (a link core writes the far one's)."""
        if len(self.sessions) != 2:
            return [None] * len(self.sessions)
        h = abi.headers()
        bytes_ = h["KM_LINK_AREA_BYTES"]
        plans = []
        self._link_areas = []
        relays = []
        for island, session in enumerate(self.sessions):
            far = 1 - island
            columns, rows = session.grid
            area = session.tensor(
                np.zeros((columns * rows, bytes_ // 4), np.uint32),
                "uint32",
                shard_grid=(columns, rows),
            )
            self._link_areas.append(area)
            if area.address != self._link_areas[0].address:
                raise KomorebiError(
                    f"island {island}: the link area is at {area.address:#x}, "
                    f"island 0's at {self._link_areas[0].address:#x}"
                )
            relay = session.tensor(  # zeros: no ask, no count, no done word
                np.zeros((columns * rows, -(-h["KM_RELAY_BYTES"] // 64) * 16)),
                "uint32",
                shard_grid=(columns, rows),
            )
            self._link_areas.append(relay)
            relays.append(relay.address)
            core = self.link_cores[island][0]
            plans.append(LinkPlan(self.graph, island, far, core, area.address))
            place = self.noc_of(far, self.link_cores[far][0])
            plans[-1].relay = [
                relays[island],
                0,
                (place & 0xFF) | (place >> 8) << 16,
            ]
            self.log(
                f"island {island}: commands across on link core {core}, channel "
                f"{self.channels[island][0]}, area {area.address:#x}"
            )
        for plan in plans:
            plan.relay[1] = relays[plan.far]
        return plans

    def _check_links(self, runs, plans):
        """Every island's dispatcher has the tile and LIM base its link line took for
        the far one's (its own)."""
        if plans[0] is None:
            return
        ports = [run.backend.port for run in runs]
        seen = {(port.l2_x, port.l2_y, port.lim_base) for port in ports}
        if len(seen) != 1:
            raise KomorebiError(
                f"the islands' dispatchers differ in tile or LIM base: {sorted(seen)}"
            )

    def close(self):
        """Stops every island's run through the X280 runtime, closes the islands'
        sessions, then the mesh (not while a grid did not return)."""
        try:
            if self.split > 1 and self.run is not None and not self.run.stopped:
                self.run.abort()
            for session in self.sessions:
                session.close()
        finally:
            stuck = [i for i, session in enumerate(self.sessions) if session.stuck]
            if self.mesh is not None and not stuck:
                self.log("closing the mesh")
                self.mesh.close()
                self.mesh = None
                session_module._LIVE.discard(self)
            elif stuck and self.mesh is not None:
                self.log(f"mesh not closed: the grid of islands {stuck} did not return")
            if self.mesh is None and not stuck:
                session_module._LIVE.discard(self)

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
