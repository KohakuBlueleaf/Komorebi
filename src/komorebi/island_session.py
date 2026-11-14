"""A session over several cards, each card's chip an island: one fabric mesh, one X280
runtime over it, a dispatcher and the persistent grid on every island. Ranks are flat
and contiguous by island; a command's group is split by island on the host, a tensor
is placed by island. `Session(cards=[...])` returns this; one card is one island."""

import concurrent.futures
import copy
import os
import secrets
import struct
import time

import numpy as np

from komorebi import abi, dram, formats, islands, native
from komorebi import channel as channel_module
from komorebi import session as session_module
from komorebi.errors import CardBusy, CommandTimeout, KomorebiError, WaitTimeout
from komorebi.program import Program

M32 = 0xFFFFFFFF
SLICE_S = 0.0  # an island's turn in wait_together: one native round (IslandRun)
THREADS = 3  # thread numbers an island: one a work cluster (KMX_THREAD)
TENSORS_MAX = islands.TENSORS_MAX  # the tensor table's entries
# The message service on link core 0 (km_link.hpp): LinkArgs words, the region's
# sizes (words, outbox, inbox, sending packet, reports and two passes of lines).
LINK_ARGS = 9
LINK_OUTBOX_AT, LINK_PREFIX, LINK_PACKET_BYTES = 0x190, 16, 4352
LINK_TAIL = 0x20 + 2 * 16 * 256
LINK_OUTBOX, LINK_INBOX = 128, 4  # entries; packets
# The kernel buffer of a session over two cards: the largest build with page links
# (90,112 B) and at least 16 KiB more, in steps of 16 KiB. kernel_bytes= overrides it.
KERNEL_BYTES = 112 * 1024
# The transfer buffer's chunk a session over two cards asks for: a page CPU passes
# pages through its chunk, one L1 slot a read in flight (15 slots of 4 KiB and their
# headers); a run with no room for it keeps the default.
LINK_CHUNK = 64 * 1024
CROSS_PAGE = 64  # bytes of a link CPU's row of what crossed, as COUNTS writes it


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


class Crossings:
    """What crossed the links since the run's start: `rows[(link, source, target)]`,
    what island `source` sent to `target` on its link `link` (FIELDS, each a count).
    `after - before` is what crossed between two reads; `total` sums links."""

    FIELDS = ("messages", "lines", "asks", "pages", "parts", "puts", "bytes")

    def __init__(self, rows):
        self.rows = dict(rows)

    def __getitem__(self, key):
        return self.rows[key]

    def __sub__(self, other):
        zero = dict.fromkeys(self.FIELDS, 0)
        return Crossings(
            {
                key: {f: row[f] - other.rows.get(key, zero)[f] for f in self.FIELDS}
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


class IslandRun:
    """The runs of one program on every island, as one run over flat ranks. A ticket
    is a command of a group of flat ranks; its parts went to their islands' dispatchers
    and it completes when every part has. Errors name the island (`error.island`)."""

    def __init__(self, graph, runs, linked=False, counts=None):
        self.graph = graph
        self.runs = tuple(runs)
        self._counts = counts  # () -> Crossings of the links, read now (None: no link)
        # a dynamic program's messages cross: a dynamic RUN takes every rank, and link
        # core 0 of each island serves it (a PAGES command of mode MESSAGES)
        self.linked = linked
        self.session = None  # the IslandSession with links: a link core's reset
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
                raise self._error(error, island) from None
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
                raise self._error(error, island) from None
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
                raise self._error(error, island) from None

    def send(self):
        """Stores every island's queued commands on its card, so that the islands run
        their parts at once (a run over a link needs every part started) while the
        host waits on one."""
        for island, run in enumerate(self.runs):
            try:
                run.store()
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
                read = self.runs[island].inspect((cores[0], count), _left(deadline))
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
        return self.session.reset_link(island, k, timeout)

    def _ranked(self, island, cores):
        """Core dicts of island `island` (its own core numbers in `core`) with their
        flat `rank` and `island`; a link core's rank is None, its `link` its index."""
        mine, links = self.graph.rank_cores[island], self.graph.link_cores[island]
        first = self.graph.first(island)
        out = []
        for core in cores:
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
            ranks = {core["core"]: core["rank"] for core in cores}
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
        together."""
        deadline = time.monotonic() + timeout
        self.send()
        waited = 0
        for island, run in enumerate(self.runs):
            try:
                waited += run.drain(_left(deadline))
            except KomorebiError as error:
                raise self._error(error, island) from None
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


class IslandSession:
    """Cards as islands of one 1 x n fabric mesh, the X280 runtime once over it, a
    Session an island (one card: its own; `islands` 2 to 4: it split into islands of
    whole rows). Two cards use every link found between them (`links`: at most that
    many), a link core (the last cores) a pair. A context manager."""

    LINKS_MAX = 4  # a byte each in a PAGES command's link word
    # How a core's ask for pages another island holds reaches the holder: "core" (its
    # own read call, km::read_pages, puts an ask line into its link core 0, sent on
    # link 0); "host" (the host queues the holder's send: ask, ask_range; the read call
    # only waits for the pages).
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
    ):
        self.cards = tuple(int(card) for card in cards)
        if not self.cards or len(set(self.cards)) != len(self.cards):
            raise ValueError(f"cards {cards}: one or more different cards")
        if links and (not 1 <= links <= self.LINKS_MAX or len(self.cards) != 2):
            raise ValueError(
                f"links={links}: None (every link found), or 1 to {self.LINKS_MAX} "
                "between two cards"
            )
        # two cards: every link found, at most `links` (set at the open); link 0 is
        # always there (commands across ride it)
        self._links_asked = links or None
        self.links = 1 if len(self.cards) == 2 else 0
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
        if kernel_bytes is None and len(self.cards) == 2:
            kernel_bytes = KERNEL_BYTES  # the link services' code in every kernel
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
        self._entries = [None] * TENSORS_MAX  # the tensor table, by word
        self._tables = []  # per island: its cores' tensor table (L1), from a load

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
            for session in self.sessions:
                session.transfer_chunk = LINK_CHUNK
            if len(self.chips) == 2:
                found = int(self.mesh.link(0, 1, 0)[0])
                most = min(found, self.LINKS_MAX)
                self.links = min(self._links_asked or most, most)
                self.log(f"links: {self.links} of {found} found")
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

    def tensor(
        self, data, dtype, layout="row", split=None, group=None, like=None, copy=False
    ):
        """A DRAM tensor holding `data` as Session.tensor takes it, placed by the hint
        (empty() names them); each island's part written through its own host
        channel."""
        raw, shape = formats.to_bytes(data, dtype, layout)
        tensor = self.empty(shape, dtype, layout, split, group, like, copy)
        self.write(tensor, raw.reshape(tensor.pages, tensor.page_bytes))
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
        if self.split > 1:  # one tensor in the card's DRAM, every page local
            part = self.sessions[0].empty(shape, dtype, layout)
            return IslandTensor(
                [part] * self.split, placement, shape, dtype, layout, part.address
            )
        word = None
        if len(self.sessions) > 1:
            free = [i for i, held in enumerate(self._entries) if held is None]
            if not free:
                raise KomorebiError(
                    f"{islands.TENSORS_MAX} tensors at once: the tensor table is full "
                    "(free one first)"
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
        if word is None:
            tensor.address = parts[0].address  # one island: its bank address
        else:
            if any(
                0 < part.address < islands.TENSORS_MAX for part in tensor.parts if part
            ):
                raise KomorebiError("a bank address below the tensor table's size")
            self._entries[word] = tensor
            self._write_table()
        return tensor

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
            raise KomorebiError(
                f"island {island}: its part of {pages} pages of shape {tuple(shape)} "
                f"does not fit its DRAM ({error}); at most {fits} pages of this kind "
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

    def _release_entry(self, tensor):
        """A freed tensor's entry of the tensor table is free again."""
        if (
            tensor.address < len(self._entries)
            and self._entries[tensor.address] is tensor
        ):
            self._entries[tensor.address] = None

    def _table_rows(self):
        """The tensor table every core of an island holds: an entry a word, in word
        order ([TENSORS_MAX x entry bytes / 4] uint32)."""
        size = islands.tensor_entry_bytes(len(self.sessions))
        table = bytearray(islands.TENSORS_MAX * size)
        for word, tensor in enumerate(self._entries):
            if tensor is not None:
                table[word * size : (word + 1) * size] = tensor.entry()
        return np.frombuffer(bytes(table), np.uint32)

    def _write_table(self):
        """Writes the tensor table into every core of every island, when a run holds
        it (a load writes it otherwise)."""
        if not self._tables:
            return
        row = self._table_rows()
        for session, table in zip(self.sessions, self._tables):
            rows = np.tile(row, (table.pages, 1)).view(np.uint8)
            session.write(table, rows)

    def _tables_for_load(self):
        """Each island's tensor table in the L1 of all its cores, made at the first
        load and written: its address for Session.load (several islands; else 0)."""
        if len(self.sessions) == 1:
            return [0]
        if abi.headers()["KM_TENSORS_MAX"] != islands.TENSORS_MAX:
            raise KomorebiError("islands.TENSORS_MAX differs from km_abi.h's")
        if not self._tables:
            row = self._table_rows()
            for session in self.sessions:
                columns, rows = session.grid
                self._tables.append(
                    session.tensor(
                        np.tile(row, (columns * rows, 1)),
                        "uint32",
                        shard_grid=(columns, rows),
                    )
                )
        else:
            self._write_table()
        return [table.address for table in self._tables]

    def read(self, *tensors, first=0, pages=None):
        """The raw pages of island tensors as Session.read gives them: each island's
        pages through its own host channel (with a run loaded, the islands at once),
        gathered into one array."""
        if (first or pages is not None) and len(tensors) != 1:
            raise ValueError("a page range is read from one tensor")
        if self.split > 1:  # one tensor in the card's DRAM
            parts = [tensor.parts[0] for tensor in tensors]
            return self.sessions[0].read(*parts, first=first, pages=pages)
        out = []
        for tensor in tensors:
            start, count = dram.page_range(first, pages, tensor.pages)
            rows = np.empty((count, tensor.page_bytes), np.uint8)

            def gather(island, at, low, high, tensor=tensor, rows=rows, start=start):
                (got,) = self.sessions[island].read(
                    tensor.parts[island], first=low, pages=high - low
                )
                rows[at[low:high] - start] = got

            self._each_island(gather, self._spans(tensor, start, count, reading=True))
            out.append(rows)
        return out

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
        if not live or len(spans) < 2:
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
        if self.split > 1:  # one tensor in the card's DRAM
            self.sessions[0].write(tensor.parts[0], pages, first)
            return
        start, count = dram.page_range(first, len(pages), tensor.pages)

        def scatter(island, at, low, high):
            self.sessions[island].write(
                tensor.parts[island], pages[at[low:high] - start], low
            )

        self._each_island(scatter, self._spans(tensor, start, count))

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
        tables = self._tables_for_load()
        runs = []
        try:
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
            self._check_links(runs, plans)
        except BaseException:
            for run in runs:
                run.abort()
            raise
        finally:
            for one, own in zip(programs, defines):
                one.defines = own
        self.run = IslandRun(
            self.graph,
            runs,
            linked=bool(self._regions),
            counts=self._crossings if self.links else None,
        )
        self.run.session = self
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
        if tables[0] != tables[1]:
            raise KomorebiError(
                "the islands' core tables differ: a link core that pushes a rank's "
                "pages finds the far core in its own table"
            )
        waits = [
            (here, self._open_link(here, k))
            for here in range(2)
            for k in range(len(self.link_cores[1 - here]))
        ]
        self.wait_together(waits, timeout)
        self.log(f"page links held, hello both ways: link cores {self.link_cores}")

    def _open_link(self, here, k):
        """Queues the OPEN of link core k of island `here` (its CPUs' connections,
        then a hello to the far link core k's same CPUs). Returns its handle."""
        h = abi.headers()
        there = 1 - here
        buffer, chunk = self.staging(here)
        links = list(self.channels[here]) + [h["KM_PAGES_NO_LINK"]] * self.LINKS_MAX
        words = [0] * 10
        words[h["KM_PAGES_TO_CORE"]] = chunk // 64 << 16
        words[h["KM_PAGES_COUNT"]] = self.port_address(there, h["KM_PAGES_HELLO_PORT"])
        words[h["KM_PAGES_COUNTER"]] = (
            self.noc_of(there, self.link_cores[there][k])
            | 1 << 16
            | h["KM_PAGES_OPEN"] << 24
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
        started = time.perf_counter()
        try:
            reset = run.reset_core(
                core, timeout, channels=[self.channels[island][link] for link in held]
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
        for island in range(2):
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
        for island in range(2):
            (raw,) = self.sessions[island].read(self._counts[island])
            words = raw.view("<u4").astype(np.int64)
            for link in range(self.links):
                row = words[link]
                low = h["KM_CROSS_BYTES"]
                rows[(link, island, 1 - island)] = {
                    **dict(zip(Crossings.FIELDS, (int(row[f]) for f in fields))),
                    "bytes": int(row[low]) | int(row[low + 1]) << 32,
                }
        return Crossings(rows)

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
            raise KomorebiError(
                "no page links: pages cross in a session over two cards"
            )
        holders = source.holders(first, pages)
        if len(holders) != 1:
            raise ValueError(f"pages {first} + {pages}: held by islands {holders}")
        here, _, _, index = holders[0]
        there = 1 - here
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
        counted = counted if counted is not None else core
        counted = counted if counted is not None else self.link_cores[there][0]
        port = h["KM_PORTS"] - 1 if port is None else port
        buffer, chunk = self.staging(here)
        links = list(self.channels[here]) + [h["KM_PAGES_NO_LINK"]] * self.LINKS_MAX
        words = [0] * 10
        words[h["KM_PAGES_FROM"]] = source.addresses[here]
        words[h["KM_PAGES_PAGE"]] = source.aligned_page_bytes
        words[h["KM_PAGES_FIRST"]] = index
        words[h["KM_PAGES_PAGES"]] = pages
        words[h["KM_PAGES_TO"]] = to.addresses[there] if core is None else to
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
        holder, _, _, index = holders[0]
        if not 0 <= port < h["KM_PORTS"] or ranks > 0xFFFF:
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
        another island holds them (a copy target: every other island's copy), sent
        over the links, waited for on the far island. Returns pages, bytes and
        seconds, the submit to the last count."""
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
        self._counts = []
        relays = []
        for island, session in enumerate(self.sessions):
            far = 1 - island
            columns, rows = session.grid
            words = np.zeros((columns * rows, bytes_ // 4), np.uint32)
            # the page CPUs a put line waits for (links 1 on, or link 0 alone)
            words[:, h["KM_PUT_CPUS"] // 4] = max(self.links - 1, 1)
            area = session.tensor(words, "uint32", shard_grid=(columns, rows))
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
            # what crossed: a page a link CPU, as COUNTS writes it (_crossings)
            self._counts.append(
                session.empty(
                    (2 * len(self.link_cores[island]), CROSS_PAGE // 4), "uint32"
                )
            )
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
        the far one's (its own); the control blocks are at one address (what crosses
        through the relay lands at its sender's addresses)."""
        if plans[0] is None:
            return
        ports = [run.backend.port for run in runs]
        seen = {(port.l2_x, port.l2_y, port.lim_base) for port in ports}
        if len(seen) != 1:
            raise KomorebiError(
                f"the islands' dispatchers differ in tile or LIM base: {sorted(seen)}"
            )
        controls = sorted({run.backend.control.address for run in runs})
        if len(controls) != 1:
            raise KomorebiError(f"the islands' control blocks differ: {controls}")

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
