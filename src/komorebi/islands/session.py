"""A session over several cards, each card's chip an island: one fabric mesh, one X280
runtime over it, a dispatcher and the persistent grid on every island. Ranks are flat
and contiguous by island; a command's group is split by island on the host, a tensor
is placed by island. `Session(cards=[...])` returns this; one card is one island."""

from komorebi import abi
from komorebi.device import channel as channel_module
from komorebi.device import chips
from komorebi.islands import graph as islands
from komorebi.islands.parts.calls import CallsMixin
from komorebi.islands.parts.entries import EntriesMixin
from komorebi.islands.parts.io import IoMixin
from komorebi.islands.parts.link_plan import LINK_PACKET_BYTES, LinkPlanMixin
from komorebi.islands.parts.load import LoadMixin
from komorebi.islands.parts.mesh import MeshMixin
from komorebi.islands.parts.table import TableMixin
from komorebi.islands.same import SessionCalls
from komorebi.session import process, sizes

TENSORS_MAX = islands.TENSORS_MAX  # the tensor table's entries at the open by default
TENSORS_MOST = islands.TENSORS_MOST  # the most it grows to
# The packet payload on the links, told to the routers at the open. One link from L1
# slots: 31.3 GB/s at 4352 B (the default), 40.2 at 8192, 33.2 at 15232 (the largest).
PACKET_BYTES = 8192


def largest_packet(arch):
    """The largest packet payload on the links between chips of `arch`: the routers'
    (chips fabric_payload_bytes; 7,616 B on Wormhole) within the runtime's, whole 64s.
    """
    most = min(
        abi.headers()["KM_PACKET_LARGEST"], chips.chip(arch).fabric_payload_bytes
    )
    return most // 64 * 64


def default_packet(arch):
    """The session's packet on chips of `arch` when none is given: PACKET_BYTES where
    the routers take it, else their largest."""
    return min(PACKET_BYTES, largest_packet(arch))


def _links_of(pairs, count=None):
    """islands.links_of (IslandSession's argument `islands` hides the module)."""
    return islands.links_of(pairs, count)


class IslandSession(
    MeshMixin,
    LinkPlanMixin,
    CallsMixin,
    LoadMixin,
    TableMixin,
    EntriesMixin,
    IoMixin,
    SessionCalls,
):
    """Cards as islands of one fabric mesh joined as the fabric reports, the X280
    runtime once over it, a Session an island (one card: its own; `islands` 2 to 4: it
    split into rows). `graph`: island pairs (a pair k times: k links). A context
    manager. `host_reaches`: see the islands reference (a test)."""

    # How an ask for another island's pages reaches the holder: "core" (the read call
    # puts an ask line into link core 0), "host" (the host queues the holder's send).
    ASKS_BY = "core"

    @property
    def dispatcher(self):
        """The dispatcher of its islands: "x280" (each island's X280 cluster; the Tensix
        dispatcher does not serve islands)."""
        return "x280"

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
        packet=None,
        message_links=False,
        graph=None,
        expect=None,
        tensors=None,
        tensor_cache=None,
    ):
        self.cards = tuple(int(card) for card in cards)
        # two cards: a dynamic program's messages over every link, a service each
        # (opt-in: one run of three of example 05 over four links stalled)
        self.message_links = bool(message_links)
        if not self.cards or len(set(self.cards)) != len(self.cards):
            raise ValueError(f"cards {cards}: one or more different cards")
        if links and not 1 <= links <= self.LINKS_MAX:
            raise ValueError(
                f"links={links}: None (the chip's default: every link found on "
                f"Blackhole, one on Wormhole), or 1 to {self.LINKS_MAX} between two "
                "islands"
            )
        if islands is not None and (
            isinstance(islands, bool)
            or not isinstance(islands, int)
            or not 0 <= islands <= sizes.SPLIT_ISLANDS_MAX
        ):
            raise ValueError(
                f"islands={islands!r}: a card split into 2 to "
                f"{sizes.SPLIT_ISLANDS_MAX} islands (None, 0 or 1: not split)"
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
        # the islands' pairs, a pair k times k links (None: a split card's every pair
        # `links` times; cards: the fabric's graph); islands beyond a pair forward
        self.pairs = None if graph is None else [tuple(pair) for pair in graph]
        if self.pairs is not None:
            if links:
                raise ValueError(
                    f"links={links} with graph=: a pair k times is k links"
                )
            if self.split < 2 and len(self.cards) < 2:
                raise ValueError("graph=: cards, or a card split into islands")
            _links_of(self.pairs, max(self.split, len(self.cards)))
        if host_reaches is not None and (
            len(self.cards) != 2 or not set(host_reaches) < {0, 1} or not host_reaches
        ):
            raise ValueError(
                f"host_reaches={host_reaches}: one island of a session over two cards"
            )
        self.host_reaches = None if host_reaches is None else tuple(host_reaches)
        self.arch = process.card_arch(self.cards[0])  # its cards' (one kind a session)
        self.packet = default_packet(self.arch) if packet is None else int(packet)
        largest = largest_packet(self.arch)
        if not LINK_PACKET_BYTES <= self.packet <= largest or self.packet % 64 != 0:
            raise ValueError(
                f"packet={packet}: {LINK_PACKET_BYTES} to {largest} bytes (a message "
                f"packet's to the routers' largest on {self.arch}), whole 64-byte lines"
            )
        if self.split == 1 and links and len(self.cards) < 2:
            raise ValueError(f"links={links}: between cards, or a split card")
        if self.split > 1 and len(self.cards) != 1:
            raise ValueError(f"islands={islands}: one card")
        # cards: the links found, at most `links` or default_links (at the open), link 0
        # always (commands ride it); a split card: `links` wires a pair (default 1)
        self._links_asked = links or None
        # the page CPUs a pair (the fewest links of any pair; set at the open on cards)
        self.links = 1 if len(self.cards) > 1 else 0
        if self.split > 1:
            self.links = min(link.count for link in _links_of(self._split_pairs()))
        self.log = log or (lambda message: None)
        self._options = {
            "pll_mhz": pll_mhz,
            "tree": tree,
            "grid_timeout": grid_timeout,
            "channel": channel,
            "slow_code": slow_code,
        }
        self._memory = None  # the remembered kernel buffer (komorebi.device.remembered)
        # a split card's card and two cards follow their kernels, as a card's own
        # session does: the remembered size they open at is not a pin (a given is)
        self._kernel_followed = kernel_bytes is None and (
            len(self.cards) == 2 or self.split > 1
        )
        if kernel_bytes is None and (len(self.cards) == 2 or self.split > 1):
            kernel_bytes = self._remembered(expect)  # the page links' code
        self.kernel_bytes = kernel_bytes
        self._worker_l1 = sizes.worker_l1_size(kernel_bytes, self.arch)
        self.tree = tree
        self._init_mesh()
        self._init_entries(tensors)
        self._init_table()

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

    # Pages across a link
