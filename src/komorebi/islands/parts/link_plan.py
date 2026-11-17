"""The island session's link service plan: link cores' areas, ports and words."""

import copy
import dataclasses
import secrets
from collections.abc import Callable
from typing import Any

import numpy as np

from komorebi import abi
from komorebi.errors import (
    KomorebiError,
)
from komorebi.islands import records as island_records
from komorebi.program import description as program_module
from komorebi.program import several as programs_module
from komorebi.program.description import Program
from komorebi.session import sizes

# The message service on link core 0 (km_link.hpp): LinkArgs words, the region's
# sizes (words, outbox, inbox, sending packet, reports and two passes of lines).
LINK_ARGS = 14
LINK_OUTBOX_AT, LINK_PREFIX, LINK_PACKET_BYTES = 0x190, 16, 4352
ISLANDS_MAX = 16  # km_abi.h's KM_ISLANDS_MAX: a load of more islands defines its own


def link_tail(count):
    """The message region's tail (km_link.hpp): lines, a report an island, 3 lines."""
    return 0x20 + 2 * 16 * 256 + 16 * max(count, ISLANDS_MAX) + 3 * 16


LINK_OUTBOX, LINK_INBOX = 128, 4  # entries; packets
LINK_NONE = 0xFFFFFFFF  # no link core (km_link.hpp kLinkNone)
# A link core's L1 for deliveries (km_deliver.hpp), shared by the source islands
DELIVER_ROOM = 64 << 10
# Less than this: a dynamic program's messages on link 0's service alone.
KERNEL_BYTES_SERVICES = 160 * 1024
CROSS_PAGE = 64  # bytes of a link CPU's row of what crossed, as COUNTS writes it
X280_FLAT = 0xFFF0  # km_abi.h KM_X280_FLAT(0): X280 rank r's flat rank over islands


def _with_link(program, words, services=1):
    """A copy of `program` whose reader and writer take `words` (or a core's
    `words(core)`) after their own arguments (as many on both), built with KM_LINKS
    (and KM_MESSAGE_SERVICES for more than one message service a side)."""
    words_of = words if callable(words) else (lambda core: words)
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
        mine = words_of(core)
        return [*reader, *mine], [*writer, *mine], compute

    linked = copy.copy(program)
    linked.args = args
    linked.defines = {**program.defines, "KM_LINKS": 1, "KM_LINK_ARGS_AFTER": after}
    if services > 1:
        linked.defines["KM_MESSAGE_SERVICES"] = services
    return linked


class LinkPlanMixin:
    """The island session's link service plan: link cores' areas, ports and words."""

    # the state of the session's other parts this one reads or sets, owned there
    log: Callable  # IslandSession
    message_links: bool  # IslandSession
    _regions: list  # MeshMixin
    channels: tuple  # MeshMixin
    graph: Any  # MeshMixin
    kernel_bytes: int | None  # MeshMixin
    link_cores: tuple  # MeshMixin
    links: int  # MeshMixin
    sessions: tuple  # MeshMixin
    split: int  # MeshMixin

    def _apart(self, programs):
        """Whether the link cores of a load of `programs` (programs or plans) run their
        own kernels (session._link_kernels): the link area then lives on them
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

    def _check_links(self, runs, plans):
        """Every island's dispatcher has the tile and LIM base its link line took for
        the far one's (its own); the control blocks are at one address (what crosses
        across lands at its sender's addresses)."""
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

    def _delivered(self, program, island=0):
        """`program` (dynamic) with its messages to other islands delivered by the link
        service: KM_DELIVER and its places' layout, DELIVER_ROOM shared by the source
        islands (a slot each at least: a bin's 32 messages and its body). X280 ranks
        are island 0's: another island's cores reach them, without their thread."""
        messages = program.messages
        if messages is None:
            return program
        islands_count = len(self.graph.islands)
        constants, _ = program_module.messages_header()
        body_at = constants["kMessagesPacket"] * messages.message_bytes
        slot = body_at + -(-messages.body_bytes // 64) * 64
        slots = max(1, DELIVER_ROOM // (islands_count * slot))
        delivered = copy.copy(program)
        if island != 0 and messages.x280_ranks:
            delivered.messages = dataclasses.replace(messages, x280_ranks=0)
            delivered.x280_elsewhere = messages.x280_ranks  # compute's X280 ranks
            delivered.threads = {
                cluster: thread
                for cluster, thread in program.threads.items()
                if cluster != messages.x280_cluster
            }
        delivered.defines = {
            **program.defines,
            "KM_DELIVER": 1,
            "KM_DELIVER_ISLANDS": islands_count,
            "KM_DELIVER_SLOTS": slots,
            "KM_DELIVER_SLOT_BYTES": slot,
            "KM_DELIVER_BODY_AT": body_at,
        }
        size = sizes.deliver_bytes(delivered.defines)
        self.log(
            f"deliveries: {slots} slot(s) a source island of {slot:,} B, "
            f"{size:,} B of each link core's L1 over {islands_count} islands"
        )
        return delivered

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

    def _first_link(self, here, there):
        """Island `here`'s first link core toward its neighbour `there` (an index into
        its link_cores)."""
        return self.graph.next_link(here, there)

    def _landings(self, h):
        """Each island's landing of km::put_bytes' batches, in DRAM: a page a holder,
        source island and put line owner, then a holder and far place (a split card:
        island 0's for every island). Three or more cards: at one address on each."""
        size = h["KM_PUT_BATCH_BYTES"]
        n = len(self.sessions)
        pages = n * n * h["KM_QUEUE_LINES"] + (
            n * n * h["KM_FAR_LINES"] if n > 2 else 0
        )
        landings = []
        for island, session in enumerate(self.sessions):
            if self.split > 1 and island:
                landings.append(landings[0])
                continue
            try:
                landings.append(session.empty((pages, size // 4), "uint32"))
            except RuntimeError as caught:
                for landing in landings[:1] if self.split > 1 else landings:
                    landing.free()
                raise KomorebiError(
                    f"the put batches' landing: {pages} pages of {size} B "
                    f"({pages * size / 2**20:,.1f} MiB) for {n} islands do not fit "
                    f"island {island}'s DRAM ({caught})"
                ) from caught
            wrong = None
            if landings[-1].aligned_page_bytes != size:
                wrong = f"{landings[-1].aligned_page_bytes} bytes a page"
            elif len(self.sessions) > 2 and landings[-1].address != landings[0].address:
                wrong = f"island {island}'s at {landings[-1].address:#x}, not 0's"
            if wrong:
                for landing in landings:
                    landing.free()
                raise KomorebiError(f"the put batches' landing: {wrong}")
        return landings

    def _link_words(self, island, message_bytes, token):
        """LinkArgs of island `island`'s cores: its link core 0, the far one's; over
        more than two islands link cores 1 and 2's far ones, and island 0's link core
        0 (the island reports' end); between two cards with links over two, link core
        1's far one; the services and the second region's stride."""

        def at(there, core):  # x | y << 16
            noc = self.noc_of(there, core)
            return (noc & 0xFF) | (noc >> 8) << 16

        far, _, core = self.pair(island, 0)
        place = at(far, core)
        fars = [LINK_NONE, LINK_NONE]
        root = LINK_NONE
        if len(self.graph.islands) > 2:
            for k in range(1, len(self.link_cores[island])):
                there, _, core = self.pair(island, k)
                fars[k - 1] = at(there, core)
            root = at(0, self.link_cores[0][0])
        elif self.message_services() > 2:
            there, _, core = self.pair(island, 1)
            fars[0] = at(there, core)
        words = [
            self.link_cores[island][0],
            self._regions[island].address,
            LINK_OUTBOX,
            LINK_INBOX,
            LINK_PACKET_BYTES // (LINK_PREFIX + message_bytes),
            island,
            place,
            self._regions[far].address,
            token,
            *fars,
            root,
            self.message_services(),
            getattr(self, "_stride", 0),
        ]
        assert len(words) == LINK_ARGS
        return words

    def _linked(self, programs):
        """The dynamic programs with the message service's arguments after their own
        reader and writer arguments and KM_LINKS defined: a region on every core of
        each island, link core 0's NoC place across, one token for both hellos."""
        for region in {id(region): region for region in self._regions}.values():
            region.free()
        self._regions = []
        dynamic = [one.messages for one in programs if one.messages is not None]
        if not dynamic:
            return programs
        if (
            self.message_links
            and self.message_services() == 1
            and self.links > 1
            and self.split == 1
        ):
            self.log(
                f"messages on link 0 alone: a service a link takes kernel_bytes >= "
                f"{KERNEL_BYTES_SERVICES} (this session's: {self.kernel_bytes})"
            )
        held = {messages.hold_links for messages in dynamic}
        if any(messages.x280_ranks for messages in dynamic):
            if True in held:
                raise ValueError(
                    "X280 ranks with hold_links=True are not done yet: deliveries "
                    "(hold_links=False, the default) take them"
                )
            if self.graph.ranks >= X280_FLAT:
                raise ValueError(
                    f"X280 ranks over islands: at most {X280_FLAT - 1} ranks, "
                    f"{self.graph.ranks} here"
                )
        if held == {True} and (self.split > 1 or len(self.graph.islands) > 2):
            raise ValueError(
                "hold_links=True (the message service, every rank) runs between two "
                "cards only: this session has "
                f"{'a split card' if self.split > 1 else 'more than two islands'}"
            )
        if len(held) > 1:
            raise ValueError("the dynamic programs of a load differ in hold_links")
        if held == {False}:  # deliveries by the link service (km_deliver.hpp)
            return [self._delivered(one, island) for island, one in enumerate(programs)]
        entry = LINK_PREFIX + max(messages.message_bytes for messages in dynamic)
        for messages in dynamic:  # a message and its body cross in one packet
            one = LINK_PREFIX + messages.message_bytes
            most = (LINK_PACKET_BYTES // one - 1) * one // 64 * 64
            if messages.body_bytes > most:
                raise ValueError(
                    f"body_bytes {messages.body_bytes} over a link: at most {most} "
                    f"bytes, a message of {messages.message_bytes} bytes and its body "
                    f"cross in one packet of {LINK_PACKET_BYTES} bytes"
                )
        size = (
            LINK_OUTBOX_AT + LINK_OUTBOX * entry + (LINK_INBOX + 1) * LINK_PACKET_BYTES
        ) + link_tail(len(self.graph.islands))
        self._stride = (size + 63) // 64 * 64 if self.message_services() > 1 else 0
        # a split card: one region, every island's at one address; a second CPU's
        # services: a second region after the first
        for session in self.sessions[: 1 if self.split > 1 else None]:
            columns, rows = session.grid
            self._regions.append(
                session.empty(
                    (
                        columns * rows,
                        (size + 63) // 64 * 16 * (2 if self._stride else 1),
                    ),
                    "uint32",
                    shard_grid=session.grid,
                )
            )
        token = secrets.randbits(31) | 1
        if self.split > 1:
            self._regions *= self.split
            (one,) = programs
            if one.messages is None:
                return programs
            words = [
                self._link_words(island, one.messages.message_bytes, token)
                for island in range(self.split)
            ]
            island_of = {
                core: island
                for island, part in enumerate(self.graph.islands)
                for core in part.cores
            }
            return [_with_link(one, lambda core: words[island_of[core]])]
        out = []
        for island, one in enumerate(programs):
            if one.messages is None:
                out.append(one)
                continue
            words = self._link_words(island, one.messages.message_bytes, token)
            out.append(_with_link(one, words, self.message_services()))
        return out

    def _links(self, programs=None):
        """Each island's LinkPlan (two islands, or a split card), else None each: link
        core 0, own words on every core, a link area at one address on every island,
        each link core's far tile, a forward area beyond two islands. Own-kernel link
        cores (_apart): the area is theirs."""
        if len(self.sessions) < 2 and self.split == 1:
            return [None] * len(self.sessions)
        h = abi.headers()
        bytes_ = h["KM_LINK_AREA_BYTES"]
        apart = programs is not None and self._apart(programs)
        plans = []
        self._link_areas = []  # the tensors held while the run lives
        self._held = []  # per island: {row: tensors} of what every core keeps
        self._areas = []  # per island: the link area's address
        self._counts = []
        self._counts_puts = []  # the same with each put block (crossings(puts=True))
        # the page CPUs (links 1 on when a pair has more): every link when no message
        # service holds link 0, else links 1 on (link 0 alone)
        cpus = self.links if not self._regions else max(self.links - 1, 1)
        landings = self._landings(h)
        for island, session in enumerate(self.sessions):
            far = self.graph.neighbours(island)[0]
            columns, rows = session.grid
            if self.split > 1 and island:  # one card: island 0's areas
                area, own = self._areas[0], self._held[0]["own words"][0]
                self._held.append({})
            else:
                own = session.tensor(  # zeros: no line, answer, done word or batch
                    np.zeros((columns * rows, h["KM_OWN_ALL_BYTES"] // 4)),
                    "uint32",
                    shard_grid=(columns, rows),
                )
                held = {"own words": [own], "island table": [self._table_of(session)]}
                if apart:  # the link cores' buffers' base, the same on every island
                    area = sizes.L1_BYTES - session.l1()[1]
                else:
                    words = np.zeros((columns * rows, bytes_ // 4), np.uint32)
                    words[:, h["KM_PUT_CPUS"] // 4] = cpus
                    words[:, h["KM_PUT_LANDING"] // 4] = landings[far].address
                    words[:, h["KM_PUT_LANDED"] // 4] = landings[island].address
                    tensor = session.tensor(words, "uint32", shard_grid=(columns, rows))
                    area = tensor.address
                    held |= {"link area": [tensor]}
                self._held.append(held)
                self._link_areas += [t for each in held.values() for t in each]
                self._link_areas.append(landings[island])
            if area != (self._areas or [area])[0]:
                raise KomorebiError(
                    f"island {island}: the link area is at {area:#x}, "
                    f"island 0's at {self._areas[0]:#x}"
                )
            if plans and own.address != plans[0].own:  # a far confirmation's place
                raise KomorebiError(
                    f"island {island}: the own words are at {own.address:#x}, "
                    f"island 0's at {plans[0].own:#x}"
                )
            self._areas.append(area)
            # what crossed: a page a link CPU, as COUNTS writes it (_crossings)
            for counts, page in (
                (self._counts, CROSS_PAGE),
                (self._counts_puts, h["KM_COUNTS_PUT_PAGE"]),
            ):
                counts.append(
                    session.empty(
                        (2 * len(self.link_cores[island]), page // 4), "uint32"
                    )
                )
            core = self.link_cores[island][0]
            plans.append(island_records.LinkPlan(self.graph, island, far, core, area))
            plans[-1].own = own.address
            plans[-1].table = self._held[-1 if self.split == 1 else 0]["island table"]
            plans[-1].table = plans[-1].table[0].address
            plans[-1].ports = self._port_words(island)
            if apart:
                words = np.zeros(bytes_ // 4, np.uint32)
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
                plans[-1].far_tiles[link_core] = (place & 0xFF) | (place >> 8) << 16
            plans[-1].relay = [0, 0, plans[-1].far_tiles[core]]
            self.log(
                f"island {island}: commands across on link core {core}, channel "
                f"{self.channels[island][0]}, area {area:#x}"
                + (" (link cores only)" if apart else "")
                + f", own words {own.address:#x}"
            )
        return plans

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

    def message_services(self):
        """The message services an island runs: between two cards with
        message_links and a kernel buffer of KERNEL_BYTES_SERVICES or more, one a link
        (a rank's messages take link rank % links); else 1 (link core 0's)."""
        if self.split > 1 or len(self.sessions) != 2 or not self.message_links:
            return 1
        if (self.kernel_bytes or 0) < KERNEL_BYTES_SERVICES:
            return 1
        return max(1, self.links)

    def noc_of(self, island, core):
        """x | y << 8: the NoC place of local core `core` of `island` (core table)."""
        columns = self.sessions[island].grid[0]
        x, y = self.sessions[island].device.worker_core(core % columns, core // columns)
        return int(x) | int(y) << 8
