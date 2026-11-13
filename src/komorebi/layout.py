"""The grid, its panels and core groups, as pure functions; no device needed.

A grid is (grid_x, grid_y) cores, a core is (x, y). A panel is an inclusive range of
columns (x0, x1) with consecutive NoC0 x; a multicast rectangle never crosses one.
"""

from dataclasses import dataclass

from komorebi import abi


def panels(noc0_x):
    """Panels from `noc0_x`, the NoC0 x coordinate of each logical column in order.

    A column joins the previous panel when its NoC0 x is the previous one plus 1.
    """
    found = []
    for x, noc0 in enumerate(noc0_x):
        if found and noc0 == noc0_x[found[-1][1]] + 1:
            found[-1] = (found[-1][0], x)
        else:
            found.append((x, x))
    return found


# NoC0 x of a Blackhole chip's Tensix columns, in the order of its harvesting mask.
BLACKHOLE_TENSIX_X = (1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16)


def blackhole_noc0_x(harvest_mask):
    """NoC0 x of each logical column: the Tensix columns whose mask bit is clear."""
    return [x for i, x in enumerate(BLACKHOLE_TENSIX_X) if not harvest_mask >> i & 1]


def core_order(grid):
    """Every core of the grid in the fixed core order: row by row, x fastest."""
    gx, gy = grid
    return [(x, y) for y in range(gy) for x in range(gx)]


@dataclass(frozen=True)
class Nocs:
    """A grid's NoC coordinates by core (x, y): `noc_xy` the translated address (the
    dispatcher's core table), `noc0` the NoC0 coordinates; `words` the translated
    addresses as x | y << 16 in core order; `panels` from the NoC0 x of the columns."""

    grid: tuple
    noc_xy: dict
    noc0: dict
    words: tuple
    panels: list


def nocs(grid, table, noc0):
    """Nocs from the x | y << 16 words of every core in core order: `table` translated,
    `noc0` NoC0 (what Session.nocs() reads from the device)."""
    order = core_order(grid)
    if len(table) != len(order) or len(noc0) != len(order):
        raise ValueError(f"{len(table)} and {len(noc0)} words for {len(order)} cores")

    def by_core(words):
        return {c: (int(w) & 0xFFFF, int(w) >> 16) for c, w in zip(order, words)}

    at0 = by_core(noc0)
    xs = [at0[(x, 0)][0] for x in range(grid[0])]
    return Nocs(
        tuple(grid), by_core(table), at0, tuple(int(w) for w in table), panels(xs)
    )


def partition(items, index, count):
    """The part of the range `items` that member `index` of `count` members takes.

    Parts are contiguous and cover the range once; the first len(items) % count
    members take one item more.
    """
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"member {index} of {count}")
    if items.step != 1:
        raise ValueError("share needs a range with step 1")
    base, extra = divmod(len(items), count)
    start = items.start + index * base + min(index, extra)
    return range(start, start + base + (index < extra))


@dataclass(frozen=True)
class Counters:
    """A dynamic run's counter tiles: the last `k` cores of the group (core order
    `first`, `count` cores) hold the other members' tails and never submit an atomic."""

    workers: range  # the cores that send and receive items
    counters: range  # the counter tiles
    counter_of: dict  # member core: the counter tile that holds its tail
    serves: dict  # counter tile: the member cores whose tails it holds


def counter_tiles(first, count, k):
    """The counter tiles of a dynamic run of group (`first`, `count`) with `k` of them:
    member core c reserves on counter tile c mod k (as km::dyn_counter)."""
    most = abi.headers()["KM_COUNTERS_MAX"]
    if not 1 <= k <= most:
        raise ValueError(f"counters {k}: not in 1..{most}")
    if count <= k:
        raise ValueError(
            f"a group of {count} cores leaves no member beside {k} counters"
        )
    workers = range(first, first + count - k)
    counters = range(workers.stop, first + count)
    counter_of = {c: counters[(c - first) % k] for c in workers}
    serves = {t: [c for c in workers if counter_of[c] == t] for t in counters}
    return Counters(workers, counters, counter_of, serves)


def split_range(total, cores):
    """Contiguous (start, count) of `total` items per core, as a dict keyed by core."""
    if not cores:
        raise ValueError("split_range needs at least one core")
    items = range(total)
    return {
        core: (part.start, len(part))
        for i, core in enumerate(cores)
        for part in [partition(items, i, len(cores))]
    }


def queue_pages(core, queue_messages, page_messages=32):
    """The DRAM pages of core `core`'s queue of a dynamic handler (`page_messages` items
    a page, the queues of all cores in core order): a range of page indices."""
    if queue_messages % page_messages:
        raise ValueError(f"queue_messages {queue_messages} is not whole pages")
    per = queue_messages // page_messages
    return range(core * per, (core + 1) * per)


@dataclass(frozen=True)
class Group:
    """The cores `first` to `first + count - 1` of the grid's core order."""

    first: int
    count: int

    def __post_init__(self):
        if self.first < 0 or self.count < 1:
            raise ValueError(f"{self}: first must be >= 0 and count >= 1")

    @property
    def indices(self):
        """The core-order indices of the group, as a range."""
        return range(self.first, self.first + self.count)

    def cores(self, grid):
        """The group's cores, in core order."""
        order = core_order(grid)
        if self.first + self.count > len(order):
            raise ValueError(f"{self} does not fit a {grid[0]} by {grid[1]} grid")
        return order[self.first : self.first + self.count]

    def partition(self, items, index):
        """Member `index` of this group's part of the range `items`."""
        return partition(items, index, self.count)

    def rectangles(self, grid, grid_panels):
        """The group as inclusive rectangles (x0, y0, x1, y1), none across a panel.

        The rectangles are disjoint and cover exactly the group's cores.
        """
        self.cores(grid)
        gx = grid[0]
        if [x for p0, p1 in grid_panels for x in range(p0, p1 + 1)] != list(range(gx)):
            raise ValueError(f"panels {grid_panels} do not tile {gx} columns")
        ya, xa = divmod(self.first, gx)
        yb, xb = divmod(self.first + self.count - 1, gx)
        # Row bands (y0, y1, x0, x1): partial first row, full rows, partial last row.
        if ya == yb:
            bands = [(ya, ya, xa, xb)]
        else:
            bands, low, high = [], ya, yb
            if xa:
                bands.append((ya, ya, xa, gx - 1))
                low += 1
            if xb != gx - 1:
                high -= 1
            if low <= high:
                bands.append((low, high, 0, gx - 1))
            if xb != gx - 1:
                bands.append((yb, yb, 0, xb))
        return [
            (max(x0, p0), y0, min(x1, p1), y1)
            for y0, y1, x0, x1 in bands
            for p0, p1 in grid_panels
            if max(x0, p0) <= min(x1, p1)
        ]


BROADCAST_NONE = (
    0xFF  # rank of a group member outside every broadcast (km::kBroadcastNone)
)
BROADCAST_CORES_MAX = 16  # a sender and up to 15 receivers (KM_BROADCAST_RANKS)
BROADCAST_WINDOW_MAX = 8  # km::kBroadcastWindowMax


@dataclass(frozen=True)
class Broadcast:
    """The cores (x, y) of one broadcast in rank order; rank 0 is the sender."""

    cores: tuple

    @property
    def sender(self):
        return self.cores[0]


def broadcasts(grid, grid_panels, rect, along, noc):
    """The broadcasts of `rect` = (x0, y0, columns, rows) along "column" or "row".

    One broadcast per column of the rectangle, or per row and panel. The sender is where
    the NoC's direction starts: NoC0 top or left, NoC1 bottom or right."""
    x0, y0, nx, ny = rect
    if nx < 1 or ny < 1 or x0 < 0 or y0 < 0 or x0 + nx > grid[0] or y0 + ny > grid[1]:
        raise ValueError(f"rectangle {rect} does not fit a {grid[0]} by {grid[1]} grid")
    if noc not in (0, 1) or along not in ("column", "row"):
        raise ValueError(f"broadcasts along {along!r} on NoC {noc}")
    xs, ys = range(x0, x0 + nx), range(y0, y0 + ny)
    if along == "column":
        runs = [[(x, y) for y in ys] for x in xs]
    else:
        runs = [
            [(x, y) for x in xs if p0 <= x <= p1]
            for y in ys
            for p0, p1 in grid_panels
            if p0 <= xs[-1] and xs[0] <= p1
        ]
    out = [Broadcast(tuple(run if noc == 0 else run[::-1])) for run in runs]
    if any(len(broadcast.cores) > BROADCAST_CORES_MAX for broadcast in out):
        raise ValueError(
            f"a broadcast of {rect} has more than {BROADCAST_CORES_MAX} cores"
        )
    return out


def broadcast_rect(broadcast, noc0, noc_xy, noc):
    """Multicast corners (start, end) of the broadcast's receivers as x | y << 16 words.

    noc0[core]: NoC0 (x, y); noc_xy[core]: core table address. Receivers follow the
    sender one NoC0 step at a time in the NoC's direction. None for one core."""
    if len(broadcast.cores) < 2:
        return None
    step = 1 if noc == 0 else -1
    sx, sy = noc0[broadcast.sender]
    dx, dy = (step, 0) if broadcast.cores[1][1] == broadcast.sender[1] else (0, step)
    for rank, core in enumerate(broadcast.cores):
        if noc0[core] != (sx + rank * dx, sy + rank * dy):
            raise ValueError(f"{core} is not NoC0 step {rank} from the sender")
    first, last = noc_xy[broadcast.cores[1]], noc_xy[broadcast.cores[-1]]
    return first[0] | first[1] << 16, last[0] | last[1] << 16


def broadcast_args(grid, grid_panels, group, rect, along, noc, noc0, noc_xy, buffer):
    """Each group member's broadcast arguments by core index (km::BroadcastArgs fields).

    buffer = (broadcast id, circular buffer index, window B, pages P). The rectangle
    must lie in the group; members outside it get rank BROADCAST_NONE."""
    broadcast_id, cb, window, pages = buffer
    if not 1 <= window <= BROADCAST_WINDOW_MAX or pages < 1:
        raise ValueError(f"window {window} or pages {pages} out of range")
    gx = grid[0]
    workers = set(group.indices)
    common = {
        "broadcast": broadcast_id,
        "group": group.first | group.count << 16,
        "noc": noc,
        "cb": cb,
        "window": window,
        "pages": pages,
    }
    none = common | {"rank": BROADCAST_NONE, "receivers": 0, "sender": 0}
    out = {i: none | {"mc_start": 0, "mc_end": 0} for i in workers}
    for broadcast in broadcasts(grid, grid_panels, rect, along, noc):
        index = [y * gx + x for x, y in broadcast.cores]
        if not workers.issuperset(index):
            raise ValueError(f"rectangle {rect} is not inside {group}")
        corners = broadcast_rect(broadcast, noc0, noc_xy, noc) or (0, 0)
        for rank, i in enumerate(index):
            out[i] = common | {
                "rank": rank,
                "receivers": len(index) - 1,
                "sender": index[0],
                "mc_start": corners[0],
                "mc_end": corners[1],
            }
    return out
