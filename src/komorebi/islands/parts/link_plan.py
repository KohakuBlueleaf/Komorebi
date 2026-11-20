"""The island session's link service plan: link cores' areas, ports and words."""

import copy
import dataclasses
import secrets
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from komorebi import abi
from komorebi.device import chips, mock
from komorebi.errors import (
    KomorebiError,
    _CoresNotFree,
)
from komorebi.islands import records as island_records
from komorebi.program import description as program_module
from komorebi.program import several as programs_module
from komorebi.program.checks import far_collectives
from komorebi.program.description import Program
from komorebi.session import sizes
from komorebi.session import state as device_state

# The message service on link core 0 (km_link.hpp): LinkArgs words, the region's
# sizes (words, outbox, inbox, sending packet, reports and two passes of lines).
LINK_ARGS = 14
LINK_OUTBOX_AT, LINK_PREFIX, LINK_PACKET_BYTES = 0x190, 16, 4352
ISLANDS_MAX = 16  # km_abi.h's KM_ISLANDS_MAX: a load of more islands defines its own
ROUND_ISLANDS_MAX = 64  # a dispatcher's round answers, a bit an island (waves.h)
# The X280 dispatcher's wave (waves.h): a packet to every other island in one turn, into
# link core 0's queue of 16 (KMD_LINK_OUT); the Tensix dispatcher's goes over turns
X280_WAVE_ISLANDS_MAX = 17


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


def _address(landing):
    """A landing's DRAM address as the link words take it: 0 when there is none."""
    return 0 if landing is None else landing.address


def _held(programs):
    """The Programs of a load of `programs` (programs, or plans of the runtime's
    form: their programs)."""
    return [
        each
        for one in programs
        for each in (one.programs if isinstance(one, programs_module.Plan) else [one])
    ]


def _buffer_bytes(one, free):
    """The L1 the buffers of `one` take on a core: a Program's buffers; a plan's (the
    runtime's form) largest group's joint buffers, and on a plan whose groups switch
    its largest program's too (the room every group keeps). `free`: the bytes a core
    leaves for buffers (joint_buffers' refusal)."""
    if not isinstance(one, programs_module.Plan):
        return sum(b.page_bytes * b.pages for b in one.buffers)
    held = [[one.programs[index] for index in group.programs] for group in one.groups]
    if one.switch:
        held += [[program] for program in one.programs]
    return max(
        (
            sum(b.page_bytes * b.pages for b in programs_module.joint_buffers(h, free))
            for h in held
        ),
        default=0,
    )


# The collectives' landing (km_ask.hpp land_run): its slots a port, and the most bytes
# a slot holds (whole pages)
LANDING_SLOTS = 2
LANDING_CHUNK = 32768


def quiet_faults(read, h, n, rooms, far):
    """What one READ of the clear (_quiet_read's) says is still on its way, as lines
    naming the island and link core, or the rank; [] when every count of lines sent
    equals its count taken, answered, confirmed or credited back and none is pending
    (hazards.md, the links before the raise); `n`, `far`, `rooms` as _quiet_page's."""
    tail = h["KM_PUT_TAIL"]
    put, ask = h["KM_QUIET_PUT"] // 4, h["KM_QUIET_ASK"] // 4
    pending = h["KM_QUIET_QUEUE"] // 4 + h["KM_QUEUE_PENDING_COUNT"]
    ring = h["KM_QUIET_RING"] // 4
    places = h["KM_QUIET_FAR"] // 4
    forward = (h["KM_QUIET_FAR"] + far) // 4
    links, faults = read["links"], []
    for (island, core), (there, across, first, words) in sorted(links.items()):
        name = f"island {island}'s link core {core} (toward island {there})"
        cpus = int(words[put + (h["KM_PUT_CPUS"] - tail) // 4]) or 1
        taken = [
            int(words[put + (h["KM_PUT_TAKEN"] - tail) // 4 + j]) for j in range(cpus)
        ]
        served = [
            int(words[ask + (h["KM_ASK_SERVED"] - h["KM_ASK_TAIL"]) // 4 + j])
            for j in range(cpus)
        ]
        if core == first and any(t != int(words[put]) for t in taken):
            faults.append(
                f"{name}: put queue lines numbered {int(words[put])}, taken by its "
                f"page CPUs {taken}"
            )
        gets = int(words[ask + (h["KM_ASK_TAKEN"] - h["KM_ASK_TAIL"]) // 4])
        if any(s != gets for s in served):
            faults.append(f"{name}: gets sent across {gets}, served back {served}")
        far_words = links.get((there, across))
        sent = int(words[put + (h["KM_PUT_DONE"] - tail) // 4])
        if far_words is not None:
            took = int(far_words[-1][put + (h["KM_PUT_CONFIRMED"] - tail) // 4])
            if sent != took:
                faults.append(
                    f"{name}: put lines sent across {sent}, taken across {took}"
                )
        if int(words[pending]):
            faults.append(f"{name}: {int(words[pending])} lines taken, not yet done")
        for cpu in range(2):
            written = int(words[ring + 4 * cpu + h["KM_PUT_RING_TAIL"] // 4])
            gone = int(words[ring + 4 * cpu + h["KM_PUT_RING_SENT"] // 4])
            if written != gone:
                faults.append(
                    f"{name}: CPU {cpu}'s ring words written {written}, sent {gone}"
                )
        if far:
            sent = words[places : places + n].astype(np.int64)
            back = words[places + n : places + 2 * n].astype(np.int64)
            faults.extend(
                f"{name}: far put lines to island {int(holder)} sent "
                f"{int(sent[holder])}, done there {int(back[holder])}"
                for holder in np.flatnonzero(sent != back)
            )
            counted = words[places + 3 * n : places + 3 * n + h["KM_FAR_LINES"] * n]
            if np.any(counted):
                faults.append(
                    f"{name}: far places hold {int(np.count_nonzero(counted))} lines "
                    "with pages counted, not yet done"
                )
        for room in range(rooms):
            landed = int(words[forward + 4 * room])
            back = int(words[forward + 4 * rooms + 4 * room])
            taken_here = int(words[forward + 8 * rooms + room])
            sent = int(words[forward + 8 * rooms + rooms + room])
            if sent != back:
                faults.append(
                    f"{name}: forward room {room}'s packets sent on {sent}, taken "
                    f"there {back}"
                )
            if landed != taken_here:
                faults.append(
                    f"{name}: forward room {room}'s packets landed {landed}, passed on "
                    f"{taken_here}"
                )
    number, confirmed = (
        (h[at] - h["KM_OWN_PUT_SENT"]) // 4
        for at in ("KM_OWN_PUT_NUMBER", "KM_OWN_PUT_CONFIRMED")
    )
    for (island, core), words in sorted(read["ranks"].items()):
        for cpu in range(2):
            own = words[16 * cpu : 16 * cpu + 16]
            for parity in range(2):
                lines = int(own[number + parity])
                done = int(own[confirmed + parity]) & ~h["KM_OWN_REFUSED"]
                if lines != done:
                    faults.append(
                        f"island {island}'s core {core}: CPU {cpu}'s put lines "
                        f"(parity {parity}) numbered {lines:#x}, confirmed {done:#x}"
                    )
    return faults


def _same(before, now):
    """Two READs of the clear read the same words everywhere."""
    was = {("link", *k): v[-1] for k, v in before["links"].items()}
    was |= {("rank", *k): v for k, v in before["ranks"].items()}
    new = {("link", *k): v[-1] for k, v in now["links"].items()}
    new |= {("rank", *k): v for k, v in now["ranks"].items()}
    return was.keys() == new.keys() and all(
        np.array_equal(words, new[at]) for at, words in was.items()
    )


class LinkPlanMixin:
    """The island session's link service plan: link cores' areas, ports and words."""

    # the state of the session's other parts this one reads or sets, owned there
    arch: str  # IslandSession
    log: Callable  # IslandSession
    message_links: bool  # IslandSession
    _regions: list  # MeshMixin
    channels: tuple  # MeshMixin
    graph: Any  # MeshMixin
    kernel_bytes: int | None  # MeshMixin
    link_cores: tuple  # MeshMixin
    links: int  # MeshMixin
    per: int  # MeshMixin
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
                "the islands' dispatchers differ in tile, LIM base or upload region "
                f"({sorted(seen)}), and a put across the link lands at its sender's "
                "addresses: the session prepares every island alike, so this is a "
                "defect of the framework, not of the program; report these values"
            )
        controls = sorted({run.backend.control.address for run in runs})
        if len(controls) != 1:
            raise KomorebiError(
                f"the islands' control blocks differ ({controls}), and a put across "
                "the link lands at its sender's addresses: the session places them "
                "alike, so this is a defect of the framework, not of the program; "
                "report these values"
            )

    def _delivered(self, program, island=0):
        """`program` (dynamic) with its messages to other islands delivered by the link
        service: KM_DELIVER and its places' layout, DELIVER_ROOM shared by the source
        islands (a slot each at least: a bin's 32 messages and its body). X280 ranks
        are island 0's: another island's cores reach them (its thread has no ranks)."""
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
        delivered.defines = {
            **program.defines,
            "KM_DELIVER": 1,
            "KM_DELIVER_ISLANDS": islands_count,
            "KM_DELIVER_SLOTS": slots,
            "KM_DELIVER_SLOT_BYTES": slot,
            "KM_DELIVER_BODY_AT": body_at,
        }
        if messages.x280_ranks:  # their rings on every island's link cores
            delivered.defines["KM_X280_FAR"] = messages.x280_ranks
            if island == 0:
                delivered.x280_far = self._x280_far()
        size = sizes.deliver_bytes(delivered.defines)
        self.log(
            f"deliveries: {slots} slot(s) a source island of {slot:,} B, "
            f"{size:,} B of each link core's L1 over {islands_count} islands"
        )
        return delivered

    def _x280_far(self):
        """Island 0's X280 ranks' table of the other islands (kmx.h's far area): per
        island its first flat rank, its ranks and the tile (x | y << 16) of island 0's
        link core 0 toward it, where the ranks' rings are (0: island 0's own); None
        without link cores (no send reaches another island)."""
        if not self.link_cores or not all(self.link_cores):
            return None
        rows = []
        for x in range(len(self.graph.islands)):
            tile = 0
            if x != 0:
                place = self.noc_of(0, self.link_cores[0][self._first_link(0, x)])
                tile = (place & 0xFF) | (place >> 8) << 16
            rows.append((self.graph.first(x), self.graph.count(x), tile))
        return rows

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
        """Each island's landing of km::put_bytes' batches, in DRAM: what is put to it,
        a page a source island and put line owner, then a far place (a split card:
        island 0's, every holder's). Three or more cards: at one address on each. None
        on a chip whose links write no far DRAM: its batches are refused, none lands."""
        size = h["KM_PUT_BATCH_BYTES"]
        n = len(self.sessions)
        if not chips.chip(self.arch).link_far_dram:  # km_ask.hpp: KM_LINK_NO_FAR_DRAM
            return [None] * n
        holders = n if self.split > 1 else 1  # km_pages.hpp's landing_holders
        pages = holders * n * h["KM_QUEUE_LINES"] + (
            holders * n * h["KM_FAR_LINES"] if n > 2 else 0
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
        self._quiet_links = []  # the clear's rows of the words that count lines (QUIET)
        self._quiet_ranks = {}  # and each card's rows of its ranks' own put words
        # the page CPUs (links 1 on when a pair has more): every link when no message
        # service holds link 0, else links 1 on (link 0 alone)
        cpus = self.links if not self._regions else max(self.links - 1, 1)
        if cpus < self.links and self.per == 1:  # km_pages.hpp's page_lead
            raise ValueError(
                f"message services on link 0 beside {self.links - 1} page links: on "
                f"{self.arch}, one link a link core, link core 0 would hold no page "
                "link, and the lead page CPU, which confirms the put queue's lines, is "
                "link core 0's"
            )
        landings = self._landings(h)
        self._coll_gen = {}  # a card's collectives' generation pages (session: tensor)
        self._tree_landing = self._tree_landing_of(programs)
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
                    area = session.l1_bytes - session.l1()[1]
                else:
                    words = np.zeros((columns * rows, bytes_ // 4), np.uint32)
                    words[:, h["KM_PUT_CPUS"] // 4] = cpus
                    words[:, h["KM_PUT_LANDING"] // 4] = _address(landings[far])
                    words[:, h["KM_PUT_LANDED"] // 4] = _address(landings[island])
                    tensor = session.tensor(words, "uint32", shard_grid=(columns, rows))
                    area = tensor.address
                    held |= {"link area": [tensor]}
                self._held.append(held)
                self._link_areas += [t for each in held.values() for t in each]
                if landings[island] is not None:
                    self._link_areas.append(landings[island])
            if area != (self._areas or [area])[0]:
                raise KomorebiError(
                    f"island {island}: the link area is at {area:#x}, island 0's at "
                    f"{self._areas[0]:#x}: the islands' L1 differs (an L1 tensor made "
                    "on one island alone, with session.island(i)); make every L1 "
                    "tensor on the session (shard_grid=), which places it on every "
                    "island at one address"
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
                        (self.per * len(self.link_cores[island]), page // 4), "uint32"
                    )
                )
            # the clear's (quiet): zeros, the words a core without them never writes
            quiet = self.per * len(self.link_cores[island]), self._quiet_page()[0] // 4
            self._quiet_links.append(
                session.tensor(np.zeros(quiet, np.uint32), "uint32")
            )
            if self.split == 1 or not island:  # a card's ranks: a page a core
                columns, rows = session.grid
                self._quiet_ranks[island] = session.tensor(
                    np.zeros((columns * rows, h["KM_QUIET_RANK_PAGE"] // 4), np.uint32),
                    "uint32",
                )
            core = self.link_cores[island][0]
            plans.append(island_records.LinkPlan(self.graph, island, far, core, area))
            plans[-1].own = own.address
            plans[-1].table = self._held[-1 if self.split == 1 else 0]["island table"]
            plans[-1].table = plans[-1].table[0].address
            if self._tree_landing and plans[-1].table != plans[0].table:
                raise KomorebiError(  # a sender writes its neighbour's slots by its own
                    f"island {island}: the island table (the collectives' landing) is "
                    f"at {plans[-1].table:#x}, island 0's at {plans[0].table:#x}: the "
                    "islands' L1 differs (an L1 tensor made on one island alone); make "
                    "every L1 tensor on the session (shard_grid=)"
                )
            plans[-1].ports = self._port_words(island)
            if apart:
                words = np.zeros(bytes_ // 4, np.uint32)
                words[h["KM_PUT_CPUS"] // 4] = cpus
                words[h["KM_PUT_LANDING"] // 4] = _address(landings[far])
                words[h["KM_PUT_LANDED"] // 4] = _address(landings[island])
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
                core = self.link_cores[island][first(island, there) + link // self.per]
                peer = self.link_cores[there][first(there, island) + link // self.per]
                place = self.noc_of(there, peer)
                back = self.graph.neighbours(there).index(island)  # its port here
                words[port * per + link] = (
                    core | link % self.per << 8 | there << 16 | count << 24,
                    (place & 0xFF) | (place >> 8) << 16 | back << 24,
                )
        return words.reshape(-1)

    def _tree_landing_of(self, programs):
        """(slots a port, bytes a slot) of the collectives' landing (land_run) where
        links may not target a far DRAM and the load's movers call a collective, else
        None: whole pages up to LANDING_CHUNK a slot, from half the L1 the buffers leave
        at most; two slots of one page the floor, refused by name below."""
        if chips.chip(self.arch).link_far_dram or not programs:
            return None
        if not far_collectives.wanted(_held(programs)):
            return None
        session = self.sessions[0]
        page = max(session.largest_page, 16)
        free = session.l1()[1]
        left = free - max(  # its buffers (the load's budget, the rest)
            _buffer_bytes(one, free) for one in programs
        )
        slots, ports = LANDING_SLOTS, abi.headers()["KM_ISLAND_PORTS_MAX"]
        credits = 32 * ports * slots

        row = abi.headers()["KM_COLL_GEN_ALIGN"] + abi.headers()["KM_COLL_GEN_BYTES"]

        def core(chunk):  # room to align, the generation row, the credits, a staging
            return row + credits + (1 + ports * slots) * (32 + chunk)  # slot, the slots

        fit = (left // 2 - row - credits) // (1 + ports * slots) - 32
        chunk = min(fit, max(LANDING_CHUNK // page, 1) * page) // page * page
        if chunk < page:
            raise KomorebiError(
                f"the collectives' landing (km::all_reduce and the others on "
                f"{self.arch}: their pages land in the far leader's L1) needs "
                f"{core(page)} B of L1 on every core at least, {slots} slots of one "
                f"{page} B page a port, at most half of what the load leaves "
                f"({max(left, 0)} B): fewer or smaller buffers"
            )
        named = "the floor: " if chunk == page else ""
        self.log(
            f"collectives' landing: {slots} slots a port of {chunk} B ({named}"
            f"{chunk // page} pages of {page} B), {core(chunk)} B a core"
        )
        return slots, chunk

    def _table_of(self, session):
        """The island table on every core of `session`'s card (graph.table,
        KM_ISLAND_TABLE), then the meeting words (a rank each), the generation pages'
        address and page bytes (`_coll_gen`), the landing's slots and bytes (else 0), at
        KM_COLL_GEN_ALIGN bytes the row, the landing."""
        columns, rows = session.grid
        count = len(self.graph.islands)
        ranks = max(self.graph.count(island) for island in range(count))
        h = abi.headers()
        # entries (2 words an island), far tiles, counts, children, meeting words (their
        # number, a rank each), generation's 2 words, landing's 2, room for the row
        meet = 6 * count + 1 + ranks
        size = meet + 4 + h["KM_COLL_GEN_ALIGN"] // 4 + h["KM_COLL_GEN_BYTES"] // 4
        landing = getattr(self, "_tree_landing", None)
        if landing:  # only a load with a converted collective reads these words
            slots, chunk = landing
            ports = h["KM_ISLAND_PORTS_MAX"]
            size += 8 * ports * slots  # the credits, after the row
            size += (1 + ports * slots) * (8 + chunk // 4)
        # 64 B a page, its first 32 B the words: a live transfer moves whole 64 B
        gen = np.zeros((columns * rows, 16), np.uint32)
        gen[:, h["KM_COLL_GEN_NOW"]] = 1
        gen = session.tensor(gen, "uint32")
        self._coll_gen[self.sessions.index(session)] = gen
        words = np.zeros((columns * rows, -(-size // 16) * 16), np.uint32)
        words[:, 6 * count] = ranks
        words[:, meet : meet + 2] = [gen.address, gen.aligned_page_bytes]
        if landing:
            words[:, meet + 2 : meet + 4] = landing
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

    def _gen_words(self):
        """[(session, tensor, {core: island}, words)]: each card's collectives'
        generation pages now, [cores, 16] uint32, the first KM_COLL_GEN_BYTES / 4 the
        words."""
        out = []
        for index, tensor in getattr(self, "_coll_gen", {}).items():
            session = self.sessions[index]
            columns, rows = session.grid
            islands = dict.fromkeys(range(columns * rows), index)
            if self.split > 1:  # one card: each core its island's
                for island, each in enumerate(self.graph.islands):
                    islands |= dict.fromkeys(each.cores, island)
            words = session.read(tensor)[0].view(np.uint32)
            out.append((session, tensor, islands, words.reshape(columns * rows, -1)))
        return out

    def landing_drops(self):
        """The slots of an earlier generation (a collective given up before) the
        collectives' landing dropped since the load: {(island, core): count}, every
        core that dropped one."""
        drop, out = abi.headers()["KM_COLL_GEN_DROPS"], {}
        for _session, _tensor, islands, words in self._gen_words():
            drops = words[:, drop]
            for core in np.flatnonzero(drops):
                out[(islands[int(core)], int(core))] = int(drops[core])
        return out

    def coll_marked(self):
        """Every core marked under the generation now (a leader that gave up inside a
        collective since the load or the last clear), earliest stage first, then by
        island: [(island, core, gave, sequence, stage)], `gave` its give-up word as
        state.gave_up reads it (None: none recorded), `stage` its kCs* number."""
        h = abi.headers()
        now, mark = h["KM_COLL_GEN_NOW"], h["KM_COLL_GEN_MARK"]
        gave, sequence = h["KM_COLL_GEN_GAVE"], h["KM_COLL_GEN_SEQUENCE"]
        stage = h["KM_COLL_GEN_STAGE"]
        out = []
        for _session, _tensor, islands, words in self._gen_words():
            for core in np.flatnonzero(words[:, mark] == words[:, now]):
                row = words[core]
                out.append(
                    (
                        islands[int(core)],
                        int(core),
                        device_state.gave_up(row[gave], row[sequence], h),
                        int(row[sequence]),
                        int(row[stage]),
                    )
                )
        return sorted(out, key=lambda each: (each[4], each[0], each[1]))

    def raise_generation(self):
        """Every core's collectives' generation one more (the host's, after a
        collective is given up, with nothing running): a leader that gave up leads
        again, and each leader's first collective after it starts its counts at 0.
        Returns the new generation, None on a load without islands."""
        now = abi.headers()["KM_COLL_GEN_NOW"]
        generation = None
        for session, tensor, _islands, words in self._gen_words():
            words = words.copy()
            words[:, now] += 1
            generation = int(words[0, now])
            session.write(tensor, words.view(np.uint8))
        if generation is not None:
            drops = self.landing_drops()
            self.log(
                f"collectives: generation {generation}"
                + (
                    f", {sum(drops.values())} slots of before dropped so far"
                    if drops
                    else ""
                )
            )
        return generation

    def _quiet_page(self):
        """(bytes of a link CPU's QUIET row, the islands of its far words, the forward
        rooms, the far words' bytes), as the load's KM_FAR_PUTS and KM_FORWARD built
        the link cores (km_pages.hpp pages_quiet)."""
        h = abi.headers()
        defines = self._forward
        count = len(self.graph.islands)
        n = count if count > ISLANDS_MAX else h["KM_ISLANDS_MAX"]
        far = 0
        if defines.get("KM_FAR_PUTS"):
            far = -(-(12 + 4 * h["KM_FAR_LINES"]) * n // 64) * 64
        rooms = 0
        if defines.get("KM_FORWARD"):
            rooms = (
                defines["KM_FORWARD_CORES"]
                * defines["KM_FORWARD_CPUS"]
                * defines["KM_FORWARD_LEVELS"]
            )
        forward = -(-(40 * rooms + 4) // 64) * 64 if rooms else 0
        return h["KM_QUIET_FAR"] + far + forward, n, rooms, far

    def _quiet_read(self, timeout=10.0):
        """One READ of run.clear_collectives, every command done: a QUIET command to
        each link core and an ENTRY (KM_ENTRY_QUIET) to each rank, waited for. {"links":
        {(island, link core): (island faced, link core across, first toward it, words)},
        "ranks": {(island, core): words}}."""
        h = abi.headers()
        page = self._quiet_page()[0]
        waits = []
        for island in range(len(self.graph.islands)):
            words = [0] * 10
            words[h["KM_PAGES_TO"]] = self._quiet_links[island].address
            words[h["KM_PAGES_PAGE"]] = page
            words[h["KM_PAGES_COUNTER"]] = h["KM_PAGES_QUIET"] << 24
            part = self.run.island(island)
            handle = part._queue_command(
                h["KM_HANDLER_PAGES"],
                self.graph.link_group(island),
                words,
                budget=timeout,
            )
            part.own_commands[handle] = "the links' line counts for clear_collectives"
            waits.append((island, handle))
        self.wait_together(waits, timeout)
        to = [
            self._quiet_ranks[0 if self.split > 1 else island].address
            for island in range(len(self.graph.islands))
        ]
        read = [[h["KM_ENTRY_QUIET"], h["KM_QUIET_READ"], at] for at in to]
        self.run._served(h["KM_HANDLER_ENTRY"], read, timeout)
        links = {}
        for island in range(len(self.graph.islands)):
            (raw,) = self.sessions[island].read(self._quiet_links[island])
            rows = raw.view("<u4")
            per = self.graph.per
            for at in range(per * len(self.link_cores[island])):
                there, first, _ = self.pair(island, at // per)
                if at - per * first >= self.graph.between(island, there).count:
                    continue  # no link of this CPU: no row
                if (island, at // per) not in links:  # a core's words: its first row
                    across = self.graph.link_cores_toward(there, island)[0]
                    links[(island, at // per)] = (
                        there,
                        across + at // per - first,
                        first,
                        rows[at].copy(),
                    )
        ranks = {}
        for index, tensor in self._quiet_ranks.items():
            session = self.sessions[index]
            columns, rows = session.grid
            islands = dict.fromkeys(range(columns * rows), index)
            if self.split > 1:  # one card: each core its island's
                for island, each in enumerate(self.graph.islands):
                    islands |= dict.fromkeys(each.cores, island)
            words = session.read(tensor)[0].view("<u4").reshape(columns * rows, -1)
            for core in range(columns * rows):
                ranks[(islands[core], core)] = words[core].copy()
        return {"links": links, "ranks": ranks}

    def _quiet(self, timeout=10.0, named=None):
        """run.clear_collectives' proof that no line is anywhere: READ until two passes
        in a row read the same balanced words (quiet_faults) within `timeout`, then
        SETTLE on every rank; else KomorebiError by name, as while a core still settles
        (the run goes on). `named`: the clear's limit as its errors say it."""
        if mock.active():  # a mock run moves no line and serves no command
            return
        h = abi.headers()
        _page, n, rooms, far = self._quiet_page()
        until = time.monotonic() + timeout
        named = timeout if named is None else named
        before, faults = None, ["no READ finished"]
        while True:
            left = max(until - time.monotonic(), 0.1)
            try:
                now = self._quiet_read(left)
            except _CoresNotFree as error:  # a core still settles a given-up command
                raise KomorebiError(
                    f"clear_collectives: a core still waits for a line of the "
                    f"given-up command after {named:.3g} s, so no rank was settled and "
                    "the generation was not raised; the run goes on: call it again "
                    "once the line passes (a core waits up to the load's wait limit "
                    f"for its own lines): {error}"
                ) from None
            faults = quiet_faults(now, h, n, rooms, far)
            if not faults and before is not None and _same(before, now):
                break
            if time.monotonic() >= until:
                if not faults:
                    faults = ["two READs in a row never read the same words"]
                raise KomorebiError(
                    "clear_collectives: the links were not quiet within "
                    f"{named:.3g} s, so no rank was settled and the generation was "
                    "not raised: "
                    + "; ".join(faults[:8])
                    + (f"; and {len(faults) - 8} more" if len(faults) > 8 else "")
                    + "; a line of a given-up command is still on its way or lost: "
                    "this load cannot be made safe: load it again"
                )
            before = now
        words = [h["KM_ENTRY_QUIET"], h["KM_QUIET_SETTLE"]]
        self.run._served(
            h["KM_HANDLER_ENTRY"], [words] * len(self.graph.islands), timeout
        )

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
