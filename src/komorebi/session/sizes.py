"""A card's sizes: a Tensix core's L1, the runtime's kernel buffer in it, the link
cores' places and a card split into islands' rows."""

from komorebi import abi

# A Blackhole Tensix core's L1 and where the runtime's kernel buffer starts in it
# (MEM_L1_SIZE, MEM_MAP_END); the buffer ends where the allocator's L1 base begins.
L1_BYTES = 1536 * 1024
KERNEL_BUFFER_BASE = 41088
KERNEL_BUFFER_DEFAULT = 69 * 1024  # the runtime's own (70,656 B)
KERNEL_BUFFER_ALIGN = 64  # the allocator's base is aligned up to this
KERNEL_BUFFER_MAX = L1_BYTES - KERNEL_BUFFER_BASE - KERNEL_BUFFER_ALIGN
SPLIT_ISLANDS_MAX = 4  # a card split into islands: a dispatcher an X280 cluster
NOC_ALIGN = 16  # a NoC write's source and destination share their low 4 address bits


def worker_l1_size(kernel_bytes):
    """The runtime's worker_l1_size for a kernel buffer of `kernel_bytes` (None: the
    default, nothing passed); a size the runtime would not give exactly is rejected."""
    if kernel_bytes is None:
        return None
    if (
        not isinstance(kernel_bytes, int)
        or kernel_bytes % KERNEL_BUFFER_ALIGN
        or not KERNEL_BUFFER_ALIGN <= kernel_bytes <= KERNEL_BUFFER_MAX
    ):
        raise ValueError(
            f"kernel_bytes={kernel_bytes!r}: a multiple of {KERNEL_BUFFER_ALIGN} from "
            f"{KERNEL_BUFFER_ALIGN} to {KERNEL_BUFFER_MAX} (default "
            f"{KERNEL_BUFFER_DEFAULT})"
        )
    return L1_BYTES - KERNEL_BUFFER_BASE - kernel_bytes


def far_bytes(defines):
    """The link cores' places for puts between islands past the next (KM_FAR_PUTS),
    after the put queue (km_abi.h KM_FAR_BYTES); 0 without."""
    if not defines.get("KM_FAR_PUTS"):
        return 0
    h = abi.headers()
    n, lines = (
        int(defines.get("KM_ISLANDS_MAX", h["KM_ISLANDS_MAX"])),
        h["KM_FAR_LINES"],
    )
    return -(-(12 + 4 * lines) * n // 64) * 64 + 64 * lines * n


def deliver_bytes(defines):
    """The link cores' places for deliveries (KM_DELIVER), after the forward rooms
    (km_abi.h KM_DELIVER_BYTES); 0 without."""
    if not defines.get("KM_DELIVER"):
        return 0
    n, slots, size = (
        int(defines[name])
        for name in ("KM_DELIVER_ISLANDS", "KM_DELIVER_SLOTS", "KM_DELIVER_SLOT_BYTES")
    )
    return -(-(12 + 4 * slots) * n // 64) * 64 + 64 + 64 * slots * n + size * slots * n


def forward_bytes(defines):
    """A link core's forward rooms (KM_FORWARD), after its far places (km_abi.h
    KM_FORWARD_BYTES): a room a writer core and CPU and a hop level; 0 without."""
    if not defines.get("KM_FORWARD"):
        return 0
    h = abi.headers()
    cpus = int(defines.get("KM_FORWARD_CPUS", 1))
    rooms = (
        int(defines.get("KM_FORWARD_CORES", 3))
        * cpus
        * int(defines.get("KM_FORWARD_LEVELS", 1))
    )
    headers = -(-(32 * rooms + 4 * 2 * rooms + 4) // 64) * 64
    slots = rooms * h["KM_FORWARD_SLOTS"] * h["KM_FORWARD_SLOT_BYTES"]
    return headers + 2 * 0x80 * cpus + slots


def _split_rows(graph, grid):
    """[(first row, rows)] of each island of a card split into islands: its cores whole
    rows of the grid in order, its ranks then its link cores (the dispatcher's
    mapping), at most one island an X280 cluster."""
    columns, rows = grid
    out, at = [], 0
    if not 2 <= len(graph.islands) <= SPLIT_ISLANDS_MAX:
        raise ValueError(f"{len(graph.islands)} islands: 2 to {SPLIT_ISLANDS_MAX}")
    for island, ranks in enumerate(graph.rank_cores):
        cores = ranks + graph.link_cores[island]
        if (
            cores != tuple(range(at, at + len(cores)))
            or at % columns
            or len(cores) % columns
        ):
            raise ValueError(
                f"island {island}: its ranks, then its link cores, are not whole rows "
                f"of the {columns} x {rows} grid after the island before it"
            )
        out.append((at // columns, len(cores) // columns))
        at += len(cores)
    if at != columns * rows:
        raise ValueError(f"the islands hold {at} cores of {columns * rows}")
    return out
