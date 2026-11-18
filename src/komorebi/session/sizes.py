"""A card's sizes: a Tensix core's L1, the runtime's kernel buffer in it, the link
cores' places and a card split into islands' rows."""

from komorebi import abi
from komorebi.device import chips

# The kernel buffer ends where the allocator's L1 base begins; it starts at the chip's
# kernel_buffer_base (chips), in its l1_bytes.
KERNEL_BUFFER_DEFAULT = 69 * 1024  # the runtime's own (70,656 B on both chips)
KERNEL_BUFFER_ALIGN = 64  # the allocator's base is aligned up to this
SPLIT_ISLANDS_MAX = 4  # a card split into islands: a dispatcher an X280 cluster
NOC_ALIGN = 16  # a NoC write's source and destination share their low 4 address bits
EVENT_DEPTH_MAX = 128  # the dispatcher's event ring at most (kEventDepthMax)


def check_depths(depth, host_depth, event_depth):
    """The dispatcher's refusal of a load's rings (Dispatch.prepare's), before it is
    asked, so a prebuild's mock refuses what the card's load would."""
    h = abi.headers()

    def power(value):
        return hasattr(value, "__index__") and value > 0 and not value & (value - 1)

    if not (
        hasattr(depth, "__index__")
        and 1 <= depth <= h["KM_RING_SLOTS"]
        and power(host_depth)
        and host_depth <= h["KMD_HOST_DEPTH_MAX"]
        and power(event_depth)
        and event_depth <= EVENT_DEPTH_MAX
    ):
        raise ValueError(
            f"depth={depth!r}, host_depth={host_depth!r}, event_depth="
            f"{event_depth!r}: depth out of range, or host_depth or event_depth not a "
            f"power of two in range (depth 1 to {h['KM_RING_SLOTS']}, host_depth to "
            f"{h['KMD_HOST_DEPTH_MAX']}, event_depth to {EVENT_DEPTH_MAX})"
        )


def kernel_buffer_max(arch):
    """The largest kernel buffer on a chip of `arch`: its L1 less the runtime's part."""
    chip = chips.chip(arch)
    return chip.l1_bytes - chip.kernel_buffer_base - KERNEL_BUFFER_ALIGN


def worker_l1_size(kernel_bytes, arch):
    """The runtime's worker_l1_size for a kernel buffer of `kernel_bytes` (None: the
    default, nothing passed) on a chip of `arch`; a size it would not give is refused.
    """
    if kernel_bytes is None:
        return None
    most = kernel_buffer_max(arch)
    if (
        not isinstance(kernel_bytes, int)
        or kernel_bytes % KERNEL_BUFFER_ALIGN
        or not KERNEL_BUFFER_ALIGN <= kernel_bytes <= most
    ):
        raise ValueError(
            f"kernel_bytes={kernel_bytes!r}: a multiple of {KERNEL_BUFFER_ALIGN} from "
            f"{KERNEL_BUFFER_ALIGN} to {most} on {arch} (default "
            f"{KERNEL_BUFFER_DEFAULT})"
        )
    chip = chips.chip(arch)
    return chip.l1_bytes - chip.kernel_buffer_base - kernel_bytes


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
    (km_abi.h KM_DELIVER_BYTES), and the X280 ranks' rings (KM_X280_FAR); 0 without."""
    if not defines.get("KM_DELIVER"):
        return 0
    n, slots, size = (
        int(defines[name])
        for name in ("KM_DELIVER_ISLANDS", "KM_DELIVER_SLOTS", "KM_DELIVER_SLOT_BYTES")
    )
    rings = abi.headers()["KM_X280_RINGS_BYTES"] if defines.get("KM_X280_FAR") else 0
    return (
        -(-(12 + 4 * slots) * n // 64) * 64 + 64 + 64 * slots * n + size * slots * n
    ) + rings


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


def link_rows(load):
    """The L1 budget's row of the link cores' put queue (KM_LINK_CB, `load`'s
    link_bytes) when they also hold the program's buffers and slots (no link kernels
    of their own, not apart): it ends their buffers' region, above the slots."""
    if not load.link_bytes or load.link_kernels or load.apart_links:
        return []
    return [("link cores: their put queue (KM_LINK_CB)", load.link_bytes)]


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
