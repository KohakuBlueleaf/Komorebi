"""The island session's mesh and links: the cards' sessions, the fabric."""

import os
from collections.abc import Callable
from typing import Any

from komorebi import abi
from komorebi.chain import records as chain_records
from komorebi.device import mock, native
from komorebi.errors import (
    CardBusy,
    KomorebiError,
)
from komorebi.islands import graph as islands
from komorebi.session import process
from komorebi.session import session as session_module

# The transfer chunk a session over two cards asks for: a page CPU passes pages through
# it, one L1 slot a read in flight (15 slots of 4 KiB); with no room: the default.
LINK_CHUNK = 64 * 1024


THREADS = 3  # thread numbers an island: one a work cluster (KMX_THREAD)


def cards_graph(grids, pairs):
    """Cards as islands joined as `pairs` (islands.links_of): each island's last cores
    its link cores, ceil(links / 2) a neighbour."""
    joined = islands.links_of(pairs, len(grids))
    parts, link_cores = [], []
    for island, (columns, rows) in enumerate(grids):
        cores = columns * rows
        taken = sum(
            (link.count + 1) // 2 for link in joined if island in (link.a, link.b)
        )
        parts.append(islands.Island(tuple(range(cores)), columns, THREADS))
        link_cores.append(tuple(range(cores - taken, cores)))
    return islands.Graph(parts, joined, link_cores=link_cores)


def check_wired(graph, split):
    """Refuses by name a graph the session cannot wire (Graph itself refuses one not
    connected): more than PORTS_MAX neighbours an island, more than
    MeshMixin.LINKS_MAX links a pair (a byte each in a PAGES command's word)."""
    for island in range(len(graph.islands)):
        near = graph.neighbours(island)
        if len(near) > islands.PORTS_MAX:
            raise ValueError(
                f"island {island}: {len(near)} neighbours, a card's ports "
                f"{islands.PORTS_MAX}"
            )
    for link in graph.links:
        if link.count > MeshMixin.LINKS_MAX:
            raise ValueError(
                f"islands {link.a} and {link.b}: {link.count} links, at most "
                f"{MeshMixin.LINKS_MAX} a pair (a byte each in a PAGES command)"
            )


def link_slots(graph, island, channel_of):
    """`island`'s links as two slots a link core (a CPU each) in link core order:
    `channel_of(neighbour, k)` for link k toward each neighbour, KM_PAGES_NO_LINK in
    a core's second slot when its neighbour's links are odd."""
    none = abi.headers()["KM_PAGES_NO_LINK"]
    slots = []
    for there in graph.neighbours(island):
        count = graph.between(island, there).count
        slots += [channel_of(there, k) for k in range(count)] + [none] * (count % 2)
    return tuple(slots)


def _split_graph(columns, rows, parts, pairs):
    """One card of `columns` x `rows` cores as `parts` islands of whole rows joined as
    `pairs` (islands.links_of: a wire a pair given); an island's link cores its last
    cores (two CPUs a core, a link each, a neighbour after another), its ranks the
    cores before them."""
    pairs = islands.links_of(pairs, parts)
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


class MeshMixin:
    """The island session's mesh and links: the cards' sessions, the fabric."""

    # its state the constructor sets (komorebi.islands.session)
    split: int
    links: int
    card_dram: bool
    kernel_bytes: int | None
    _links_asked: int | None
    _options: dict
    _worker_l1: int
    packet: int
    pairs: list | None
    tree: Any
    cards: tuple
    # the state of the session's other parts this one reads or sets, owned there
    _memory: Any  # IslandSession
    log: Callable  # IslandSession

    LINKS_MAX = 4  # a byte each in a PAGES command's link word

    def _init_mesh(self):
        """The topology before the open: no mesh, sessions or links yet."""
        self.mesh = None
        self.chips = None
        self.sessions = ()
        self.graph = None
        self.run = None
        self.link_cores = ()  # per island: the local cores (core table) of its links
        self._regions = []  # the message service's L1 region on every core, by island
        self.channels = ()  # per island: the Ethernet channel of each link it sends on
        self.records = None  # chain records at one address on every island (cards)

    def _fabric_pairs(self):
        """The pairs of the open mesh as the fabric joins them, a pair as many times
        as its links (at most LINKS_MAX, and `links`); graph= takes a part of them."""
        fabric = {}
        for a in range(len(self.chips)):
            for b in range(a + 1, len(self.chips)):
                if self.mesh.adjacent(a, b):
                    found = int(self.mesh.link(a, b, 0)[0])
                    fabric[a, b] = min(
                        found, self.LINKS_MAX, self._links_asked or found
                    )
        self.log(f"the fabric's links: {fabric}")
        if self.pairs is None:
            return [pair for pair, count in fabric.items() for _ in range(count)]
        for link in islands.links_of(self.pairs, len(self.chips)):
            if link.count > fabric.get((link.a, link.b), 0):
                raise ValueError(
                    f"graph=: {link.count} links between cards {link.a} and {link.b}, "
                    f"the fabric has {fabric.get((link.a, link.b), 0)}"
                )
        return self.pairs

    def _forward_packet(self):
        """More than two islands: the kernels' packets no larger than a forward slot's
        payload (KM_FORWARD_PAYLOAD; the routers may carry larger ones)."""
        most = abi.headers()["KM_FORWARD_PAYLOAD"]
        if len(self.graph.islands) > 2 and self.packet > most:
            self.log(f"packets of {most} bytes: a forward slot's, over the islands")
            self.packet = most
            for session in self.sessions:
                session.packet = most

    def _island_log(self, island):
        return lambda message: self.log(f"island {island}: {message}")

    def _open_links(self, module):
        """The Ethernet channel of each link toward the other island, and the check
        that a page's DRAM bank sits at the same NoC place on every island."""
        channels = []
        for island in range(len(self.sessions)):
            for there in self.graph.neighbours(island):
                asked = self.graph.between(island, there).count
                found = int(self.mesh.link(island, there, 0)[0])
                if found < asked:
                    raise KomorebiError(
                        f"island {island}: {found} links found to island {there}, "
                        f"{asked} asked"
                    )
            channels.append(  # in link core order: each neighbour's links
                link_slots(
                    self.graph,
                    island,
                    lambda there, k, island=island: int(
                        self.mesh.link(island, there, k)[1]
                    ),
                )
            )
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

    def _open_mesh(self, listed):
        """open() over two or more cards, TT_VISIBLE_DEVICES set: the mesh, the X280
        runtime and a Session on each island's device."""
        os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
        if process.exit_timeout(os.environ, self.log) == 0:
            self.log(
                f"{process.OP_TIMEOUT}=0: an exit after a grid that did not "
                "return hangs"
            )
        for card in self.cards:
            holder = process.chip_holder(card)
            if holder is not None:
                raise CardBusy(
                    f"card {card} is held by process {holder} "
                    f"({process._command(holder)}): an open would wait until "
                    "that process exits",
                    pid=holder,
                )
        process.refuse_reopen("mesh", self.cards)
        module = native.load_module(self.tree)
        self.log(f"opening cards {listed} as one 1 x {len(self.cards)} fabric mesh")
        self.log(f"routers told: packets of {self.packet} bytes")
        self.mesh = module.Islands(len(self.cards), self._worker_l1, self.packet)
        process._hook_exit()
        process._LIVE.add(self)
        try:
            self.chips = list(self.mesh.chips())
            if sorted(self.chips) != sorted(self.cards):
                raise KomorebiError(
                    f"the mesh's chips {self.chips} are not the cards {listed}"
                )
            if not mock.active():  # a mock has no X280
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
            self.records = chain_records.Region(self.sessions)  # allocators alike now
            pairs = []
            if len(self.chips) > 1:
                pairs = self._fabric_pairs()
            self.graph = cards_graph([session.grid for session in self.sessions], pairs)
            check_wired(self.graph, 1)
            if pairs:  # the page CPUs a pair: the fewest links of any pair
                self.links = min(link.count for link in self.graph.links)
            self.link_cores = self.graph.link_cores
            self._forward_packet()
            if self.links:
                self._open_links(module)
        except BaseException:
            self.close()
            raise

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

    def _split_pairs(self):
        """A split card's pairs: graph=, else every pair `links` times (default 1)."""
        if self.pairs is not None:
            return self.pairs
        every = [(a, b) for a in range(self.split) for b in range(a + 1, self.split)]
        return every * (self._links_asked or 1)

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
                process._LIVE.discard(self)
            elif stuck and self.mesh is not None:
                self.log(f"mesh not closed: the grid of islands {stuck} did not return")
            if self.mesh is None and not stuck:
                process._LIVE.discard(self)
        if first is not None:
            raise first

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
            if self.split > 1:  # the L1 its first load makes, counted before it
                session.l1_held = self._split_held
            self.chips, self.sessions = [self.cards[0]], (session,) * self.split
            columns, rows = session.grid
            self.graph = islands.one(columns * rows, columns)
            if self.split > 1:  # one device, a dispatcher an island, wires between
                pairs = self._split_pairs()
                self.graph = _split_graph(columns, rows, self.split, pairs)
                check_wired(self.graph, self.split)
                self.link_cores = self.graph.link_cores
                self._forward_packet()
                self.channels = tuple(  # no router: a wire
                    link_slots(self.graph, island, lambda there, k: 0)
                    for island in range(self.split)
                )
                process._LIVE.add(self)
                self.log(
                    f"card {self.cards[0]} as {self.split} islands, "
                    f"pairs {self.graph.pairs()}"
                )
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

    @property
    def ranks(self):
        return self.graph.ranks
