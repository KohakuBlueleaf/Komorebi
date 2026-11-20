"""The island session's mesh and links: the cards' sessions, the fabric."""

import os
from collections.abc import Callable
from typing import Any

from komorebi import abi
from komorebi.chain import records as chain_records
from komorebi.device import chips, mock, native
from komorebi.errors import (
    CardBusy,
    ChipsHidden,
    KomorebiError,
)
from komorebi.islands import graph as islands
from komorebi.session import dispatch, process
from komorebi.session import session as session_module

# The transfer chunk a session over two cards asks for: a page CPU passes pages through
# it, one L1 slot a read in flight (15 slots of 4 KiB); with no room: the default.
LINK_CHUNK = 64 * 1024


THREADS = 3  # thread numbers an island: one a work cluster (KMX_THREAD)


def graph_line(chips, graph):
    """The log's line of a session's islands: each island's chip, in island order, the
    pairs the graph joins (a pair once) and its diameter."""
    joined = [(link.a, link.b) for link in graph.links]
    return (
        f"the islands' chips, in island order: {list(chips)}; {len(joined)} pairs "
        f"{joined}, diameter {graph.diameter()}"
    )


# tt-metal's one-hop fabric forms an open tries (in try_order): the first whose pairs
# are all the machine's own (Islands.physical) is kept, else the one with the most
FORMS = (("FABRIC_1D_RING", True), ("FABRIC_1D", False))


def has_cycle(pairs):
    """Whether the pairs (a, b) hold a cycle: a pair whose chips another path joins."""
    root = {}

    def find(chip):
        while root.setdefault(chip, chip) != chip:
            chip = root[chip]
        return chip

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            return True
        root[ra] = rb
    return False


def try_order(machine):
    """FORMS in the order an open tries them: the ring form first when the machine's
    own pairs between the visible chips (`machine`, (a, b, links) each) hold a cycle,
    else the plain form first (the order costs time only, never the form kept)."""
    ring_first = has_cycle((a, b) for a, b, _ in machine)
    return sorted(FORMS, key=lambda form: form[1] != ring_first)


def visible_ids(chips, visible):
    """The open mesh's `chips` as the driver's card ids: tt-metal numbers the
    TT_VISIBLE_DEVICES cards (`visible`, a list or None) from 0 on Wormhole."""
    if visible and sorted(chips) == list(range(len(visible))) != sorted(visible):
        return [visible[chip] for chip in chips]
    return list(chips)


# tt-metal's refusal when the chips it sees do not make the mesh an open asks for
MESH_REFUSAL = "Requested mesh shape"


def hidden_chips(visible, machine):
    """The machine's cards (`machine`) that TT_VISIBLE_DEVICES' cards (`visible`, a
    list or None) leave out, sorted: none when it is unset."""
    return sorted(set(machine) - set(visible)) if visible is not None else []


def form_pairs(mesh, count):
    """The pairs of mesh columns a < b the open fabric joins."""
    return {
        (a, b) for a in range(count) for b in range(a + 1, count) if mesh.adjacent(a, b)
    }


def unusable(physical, joined):
    """The machine's pairs (`physical`, {(a, b): links} by card id) the fabric's
    `joined` (a set of card id pairs) does not join, sorted."""
    return sorted(pair for pair in physical if pair not in joined)


def cards_graph(grids, pairs, per=2, dispatcher_cores=0):
    """Cards as islands joined as `pairs` (islands.links_of): each island's last cores
    its link cores, ceil(links / per) a neighbour (`per` links a core); a Tensix
    dispatcher's core, the grid's last (`dispatcher_cores` 1), no island's."""
    joined = islands.links_of(pairs, len(grids))
    parts, link_cores = [], []
    for island, (columns, rows) in enumerate(grids):
        cores = columns * rows - dispatcher_cores
        taken = sum(
            -(-link.count // per) for link in joined if island in (link.a, link.b)
        )
        parts.append(islands.Island(tuple(range(cores)), columns, THREADS))
        link_cores.append(tuple(range(cores - taken, cores)))
    return islands.Graph(parts, joined, link_cores=link_cores, per=per)


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
    """`island`'s links as graph.per slots a link core (a CPU each) in link core order:
    `channel_of(neighbour, k)` for link k toward each neighbour, KM_PAGES_NO_LINK in
    a core's last slots when its neighbour's links do not fill them."""
    none = abi.headers()["KM_PAGES_NO_LINK"]
    slots = []
    for there in graph.neighbours(island):
        count = graph.between(island, there).count
        slots += [channel_of(there, k) for k in range(count)]
        slots += [none] * (-count % graph.per)
    return tuple(slots)


def second_links(pairs, count):
    """A thin second sender's pairs (chips second_sender "thin"): a pair's k links as
    ceil(k / 2) links of the graph (a link core each), the rest the second links of
    its first link cores. (The graph's pairs, {(a, b): second links}.)"""
    kept, seconds = [], {}
    for link in islands.links_of(pairs, count):
        cores = -(-link.count // 2)
        kept += [(link.a, link.b)] * cores
        seconds[link.a, link.b] = link.count - cores
    return kept, seconds


def second_slots(graph, island, seconds, channel_of):
    """`island`'s second links, a slot a link core in link core order: link
    `count + j` toward its neighbour for its j-th link core toward it (`count` the
    pair's links in the graph), KM_PAGES_NO_LINK past the pair's second links."""
    none = abi.headers()["KM_PAGES_NO_LINK"]
    slots = []
    for there in graph.neighbours(island):
        count = graph.between(island, there).count
        given = seconds.get((min(island, there), max(island, there)), 0)
        slots += [
            channel_of(there, count + j) if j < given else none for j in range(count)
        ]
    return tuple(slots)


def _split_graph(columns, rows, parts, pairs, per=2, dispatcher_cores=0):
    """One card of `columns` x `rows` cores as `parts` islands of whole rows joined as
    `pairs` (islands.links_of: a wire a pair given); an island's link cores its last
    cores (`per` CPUs a core, a link each, a neighbour after another), its ranks the
    cores before them; a Tensix dispatcher's core, its rows' last (1), no island's."""
    pairs = islands.links_of(pairs, parts)
    parts_, link_cores = [], []
    for part in range(parts):
        start, length = (
            columns * size for size in islands.partition(rows, part, parts)
        )
        length -= dispatcher_cores
        cores = tuple(range(start, start + length))
        taken = sum(-(-pair.count // per) for pair in pairs if part in (pair.a, pair.b))
        parts_.append(islands.Island(cores, columns))
        link_cores.append(cores[length - taken :])
    return islands.Graph(parts_, pairs, link_cores=link_cores, per=per)


class MeshMixin:
    """The island session's mesh and links: the cards' sessions, the fabric."""

    # its state the constructor sets (komorebi.islands.session)
    split: int
    links: int
    card_dram: bool
    kernel_bytes: int | None
    _kernel_followed: bool
    _links_asked: int | None
    _options: dict
    arch: str

    @property
    def per(self):
        """The links a link core holds (chips links_a_core): a CPU each."""
        return chips.chip(self.arch).links_a_core

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
        # per island: each link core's second link's channel (a thin second sender)
        self.seconds = ()
        self._seconds = {}  # (a, b): the pair's second links
        self._columns = ()  # per island: its chip's column in the open mesh (cards)
        self._ring = None  # the fabric form kept at the open (FORMS), for a reopen
        self._visible = None  # TT_VISIBLE_DEVICES' cards at the open, else None
        self.records = None  # chain records at one address on every island (cards)

    def _fabric_pairs(self):
        """The pairs of the open mesh as the fabric joins them, a pair as many times
        as its links (at most LINKS_MAX, and `links` or the chip's default_links);
        graph= takes a part of them."""
        asked = self._links_asked or chips.chip(self.arch).default_links
        fabric, at = {}, self._columns
        for a in range(len(self.chips)):
            for b in range(a + 1, len(self.chips)):
                if self.mesh.adjacent(at[a], at[b]):
                    found = int(self.mesh.link(at[a], at[b], 0)[0])
                    fabric[a, b] = min(found, self.LINKS_MAX, asked or found)
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
        channels, at = [], self._columns
        for island in range(len(self.sessions)):
            for there in self.graph.neighbours(island):
                asked = self.graph.between(island, there).count + self._seconds.get(
                    (min(island, there), max(island, there)), 0
                )
                found = int(self.mesh.link(at[island], at[there], 0)[0])
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
                        self.mesh.link(at[island], at[there], k)[1]
                    ),
                )
            )
        self.channels = tuple(channels)
        if any(self._seconds.values()):
            self.seconds = tuple(
                second_slots(
                    self.graph,
                    island,
                    self._seconds,
                    lambda there, k, island=island: int(
                        self.mesh.link(at[island], at[there], k)[1]
                    ),
                )
                for island in range(len(self.sessions))
            )
        banks = []
        for session in self.sessions:
            count = session.dram_banks
            banks.append([tuple(session.device.dram_core(b)) for b in range(count)])
        if any(other != banks[0] for other in banks[1:]):
            raise KomorebiError(
                "the islands' DRAM banks sit at different NoC places: a page sent "
                "into the other island's DRAM would land in another bank"
            )
        self.log(f"page links: channels {self.channels}, link cores {self.link_cores}")
        if self.seconds:
            self.log(f"second links (the link cores' writers): channels {self.seconds}")

    def _open_form(self, module, ring, visible):
        """The mesh of every chip tt-metal sees under one of FORMS (`ring`), its chips
        as card ids, the card pairs it joins and the machine's own ({pair: links})."""
        with mock.cluster_visible(self.cards):  # a prebuild: the mock, not the cards
            try:
                mesh = module.Islands(0, self._worker_l1, self.packet, ring)
            except RuntimeError as error:
                hidden = hidden_chips(visible, process.card_entries())
                if not hidden or MESH_REFUSAL not in str(error):
                    raise
                raise ChipsHidden(hidden, visible) from error
        try:
            raw = list(mesh.chips())
            ids = visible_ids(raw, visible)
            card = dict(zip(raw, ids))
            physical = {
                (min(card[a], card[b]), max(card[a], card[b])): int(links)
                for a, b, links in mesh.physical()
            }
            joined = {
                (min(ids[a], ids[b]), max(ids[a], ids[b]))
                for a, b in form_pairs(mesh, len(ids))
            }
        except BaseException:
            mesh.close()
            raise
        return mesh, ids, joined, physical

    def _open_fabric(self, module, visible):
        """Opens every chip tt-metal sees as one mesh under the first of FORMS
        (try_order) joining every machine pair (Islands.physical), else the one joining
        most (opened again); the form kept for a reopen (_ring). Each asked card's mesh
        column; the log names the pairs the form leaves out."""
        tried = []  # (pairs joined, index in forms)
        forms = try_order(module.machine_links())
        forms = [form for form in forms if self._ring in (None, form[1])]
        for index, (name, ring) in enumerate(forms):
            mesh, ids, joined, physical = self._open_form(module, ring, visible)
            left = unusable(physical, joined)
            self.log(
                f"{name}: {len(joined)} of the machine's {len(physical)} pairs between "
                f"chips {sorted(ids)}"
            )
            tried.append((len(joined), index))
            if not left or index == len(forms) - 1:
                break
            mesh.close()
        best = max(tried, key=lambda one: one[0])[1]  # the first on a tie
        if best != tried[-1][1]:  # the open one is not the best: the best again
            mesh.close()
            name, ring = forms[best]
            mesh, ids, joined, physical = self._open_form(module, ring, visible)
            left = unusable(physical, joined)
        self.mesh, self._ring = mesh, ring
        if left:
            self.log(f"{name} kept: the machine's pairs it leaves out: {left}")
        else:
            self.log(f"{name} kept: every pair of the machine between the open chips")
        missing = [card for card in self.cards if card not in ids]
        if missing:
            raise KomorebiError(
                f"cards {missing} are not among the chips the mesh opened {sorted(ids)}"
            )
        self._columns = tuple(ids.index(card) for card in self.cards)
        return ids

    def _open_mesh(self, listed, visible):
        """open() over two or more cards: every chip tt-metal sees opened as one
        fabric mesh (_open_fabric), the X280 runtime, a Session on each asked card's
        device (island i: card i); the chips not asked hold only tt-metal's routers."""
        os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
        if process.exit_timeout(os.environ, self.log) == 0:
            self.log(
                f"{process.OP_TIMEOUT}=0: an exit after a grid that did not "
                "return hangs"
            )
        if visible is not None:
            held = visible
        elif mock.active() or os.environ.get(mock.MOCK):
            held = list(self.cards)  # a stand-in: no card is held
        else:
            held = process.card_entries()
        for card in held:
            holder = process.chip_holder(card)
            if holder is not None:
                raise CardBusy(
                    f"card {card} is held by process {holder} "
                    f"({process._command(holder)}): an open would wait until "
                    "that process exits",
                    pid=holder,
                )
        # driver files: CardNotReady before tt-metal, every card asked among them
        process.card_ready(sorted(set(held) | set(self.cards)))
        process.refuse_reopen("mesh", self.cards)
        mock.choose_cluster(self.cards)  # set before the native module loads
        module = native.load_module(self.tree)
        # each island's dispatcher: the X280 where the build and the chip have it
        has_x280 = mock.active() or bool(getattr(module, "X280", False))
        has_x280 = has_x280 and chips.chip(self.arch).x280
        kind = dispatch.choose(self._dispatcher_asked, has_x280)
        self._dispatcher_kind = kind
        if kind == dispatch.X280 and not mock.active():
            native.require_dispatcher(module, f"islands on cards {listed}")
        self.log(
            f"opening every chip tt-metal sees as one fabric mesh (cards {listed})"
        )
        self.log(f"routers told: packets of {self.packet} bytes")
        process._hook_exit(self.log)
        process._LIVE.add(self)
        self._visible = visible  # for a reopen (_open_fabric)
        try:
            self._open_fabric(module, visible)
            self.chips = list(self.cards)
            if kind == dispatch.X280:
                if not mock.active():  # a mock has no X280
                    self.mesh.open_x280()
                self.log(
                    f"X280 runtime open over the mesh (islands: chips {self.chips})"
                )
            self.sessions = tuple(
                session_module.Session(
                    card=chip,
                    device=self.mesh.device(self._columns[island]),
                    log=self._island_log(island),
                    dispatcher=kind,
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
            if chips.chip(self.arch).second_sender == "thin":  # two links a link core
                pairs, self._seconds = second_links(pairs, len(self.chips))
            self.graph = cards_graph(
                [session.grid for session in self.sessions],
                pairs,
                self.per,
                dispatch.DISPATCHER_CORES[kind],
            )
            check_wired(self.graph, 1)
            self.log(graph_line(self.chips, self.graph))
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
        session is tried; the first error is raised after. An island whose close raised
        is named with its error; the mesh is closed unless a grid did not return."""
        failed, first = [], None
        try:
            if self._memory is not None:
                self._memory.closed()
        except Exception as error:  # noqa: BLE001 (raised after the rest)
            first = error
        try:
            if self.split > 1 and self.run is not None and not self.run.stopped:
                self.run.abort()
            for island, session in enumerate(self.sessions):
                try:
                    session.close()
                except BaseException as error:  # noqa: BLE001 (raised after the rest)
                    failed.append((island, error))
                    first = error if first is None else first
        finally:
            for island, error in failed:
                self.log(
                    f"island {island}: close raised {type(error).__name__}: {error}"
                )
            stuck = [i for i, session in enumerate(self.sessions) if session.stuck]
            if self.mesh is not None and not stuck:
                # each island's device closed first: one held on (by a tensor or the
                # session) keeps it open, and tt-metal refuses the next mesh
                for session in self.sessions:
                    if session.device is not None:
                        session.device.close()
                        session.device = None
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
                dispatcher=self._dispatcher_asked,
                **self._options,
            )
            session.kernel_followed = self._kernel_followed
            session.open()
            session.packet = self.packet
            if self.split > 1:  # the L1 its first load makes, counted before it
                session.l1_held = self._split_held
            self.chips, self.sessions = [self.cards[0]], (session,) * self.split
            self._dispatcher_kind = session.dispatcher
            columns, rows = session.grid
            self.graph = islands.one(session.ranks, columns)  # a Tensix core's: none
            if self.split > 1:  # one device, a dispatcher an island, wires between
                pairs = self._split_pairs()
                self.graph = _split_graph(
                    columns,
                    rows,
                    self.split,
                    pairs,
                    self.per,
                    dispatch.DISPATCHER_CORES[session.dispatcher],
                )
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
        if visible is not None:
            visible = [int(one) for one in visible.split(",") if one.strip()]
            if not set(self.cards) <= set(visible):
                listed_visible = ",".join(str(card) for card in visible)
                raise ValueError(
                    f"cards {listed} are not all visible ({listed_visible})"
                )
        self._open_mesh(listed, visible)

    @property
    def ranks(self):
        return self.graph.ranks
