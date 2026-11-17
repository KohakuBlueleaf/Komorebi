"""The island session's records: what crossed the links (Crossings), a load's link plan
(LinkPlan), a tensor placed by island (IslandTensor), a bank range, a trace by island.
"""

import numpy as np

from komorebi import formats
from komorebi.device import native
from komorebi.device import tensor as tensor_module
from komorebi.errors import KomorebiError
from komorebi.islands import graph as islands
from komorebi.islands import link_area
from komorebi.islands.same import TENSOR_REFUSED, refused


class Crossings:
    """What crossed the links since the run's start: `rows[(link, source, target)]`,
    what island `source` sent to `target` on its link `link` (FIELDS, each a count).
    `after - before` is what crossed between two reads; `total` sums links."""

    FIELDS = ("messages", "lines", "asks", "pages", "puts", "bytes")
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
        # the island area's words after the table: two unused (0), the far link core
        # 0's tile x | y << 16 (far_tiles: each link core's far tile)
        self.relay = [0, 0, 0]
        self.far_tiles = {}
        self.forward = 0  # the forward area's address (more than two islands)
        self.own = 0  # the own words' address (KM_OWN_*), on every core
        # the link area on the link cores alone ((x, y) a core, its words
        # with the area's address) when they run their own kernels; else None
        self.apart = None

    def clear(self, session):
        """Zeroes the link area on each link core (the page CPUs' word set),
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
        self.page_tiles = getattr(first, "page_tiles", 1)
        self.address = word
        self.in_dram = True
        self.freed = False
        self._transposed = False
        self.written = False  # the host wrote its data (transposed_tiles is fixed)
        self.session = None  # the IslandSession that made it: read() and write()
        self.loaded = None  # its session's test: a run is loaded (read refused)

    def __repr__(self):
        return (
            f"IslandTensor({self.shape}, {self.dtype}, {self.layout}, "
            f"{sum(part is not None for part in self.parts)} islands' parts)"
        )

    def __getattr__(self, name):
        raise refused(self, TENSOR_REFUSED, "tensor", name)

    @property
    def transposed_tiles(self):
        """Whether each tile is held transposed (Tensor.transposed_tiles): every
        island's part alike; set before the data is written, refused after."""
        return self._transposed

    @transposed_tiles.setter
    def transposed_tiles(self, value):
        tensor_module.refuse_turned_after(self, value)
        self._transposed = bool(value)
        for part in self.parts:
            if part is not None:
                part.transposed_tiles = value

    @property
    def nbytes(self):
        """Its pages' bytes, as a one-card tensor's buffer holds them."""
        return self.pages * self.page_bytes

    def write(self, data):
        """Writes `data` (this shape) over the whole tensor, as a one-card tensor's
        write: each island's part through its own host channel."""
        raw, shape = formats.to_bytes(
            data, self.dtype, self.layout, self.transposed_tiles
        )
        if shape != self.shape:
            raise ValueError(f"data of shape {shape}: the tensor is {self.shape}")
        tensor_module.refuse_live_write(self)
        self.session.write(self, raw.reshape(self.pages, self.page_bytes))

    def read(self):
        """The whole tensor as a numpy array (bfloat16 as float32), as a one-card
        tensor's read: each island's pages gathered."""
        return formats.from_bytes(
            self.read_bytes(),
            self.shape,
            self.dtype,
            self.layout,
            self.transposed_tiles,
        )

    def read_bytes(self):
        """The whole tensor's bytes (uint8), read as read() reads them; refused while
        a run is loaded, as a one-card tensor's (session.read reads live)."""
        tensor_module.refuse_live(self)
        (rows,) = self.session.read(self)
        return rows.reshape(-1)

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
        self.freed = True


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
        in its island's clock; None when a part was not traced; a part past its
        island's trace slots refused (KomorebiError), naming the island."""
        rows = []
        for island, trace in enumerate(self.traces):
            handle = int(self._parts[island][ticket])
            if handle:
                try:
                    rows.append(trace.command(handle))
                except KomorebiError as error:
                    raise KomorebiError(
                        f"ticket {ticket}, island {island}'s part: {error}"
                    ) from None
        if not rows or any(part is None for part in rows):
            return None
        return np.concatenate(rows)
