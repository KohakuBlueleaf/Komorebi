"""The grid, its panels and teams of cores, as pure functions; no device needed.

A grid is (grid_x, grid_y) cores, a core is (x, y). A panel is an inclusive range of
columns (x0, x1) with consecutive NoC0 x; a multicast rectangle never crosses one.
"""

from dataclasses import dataclass


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
    dispatch kernel's core table), `noc0` the NoC0 coordinates; `words` the translated
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


def share(items, index, count):
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


def split_range(total, cores):
    """Contiguous (start, count) of `total` items per core, as a dict keyed by core."""
    if not cores:
        raise ValueError("split_range needs at least one core")
    items = range(total)
    return {
        core: (part.start, len(part))
        for i, core in enumerate(cores)
        for part in [share(items, i, len(cores))]
    }


@dataclass(frozen=True)
class Team:
    """The cores `first` to `first + count - 1` of the grid's core order."""

    first: int
    count: int

    def __post_init__(self):
        if self.first < 0 or self.count < 1:
            raise ValueError(f"{self}: first must be >= 0 and count >= 1")

    @property
    def members(self):
        """The core-order indices of the team, as a range."""
        return range(self.first, self.first + self.count)

    def cores(self, grid):
        """The team's cores, in core order."""
        order = core_order(grid)
        if self.first + self.count > len(order):
            raise ValueError(f"{self} does not fit a {grid[0]} by {grid[1]} grid")
        return order[self.first : self.first + self.count]

    def share(self, items, index):
        """Member `index` of this team's part of the range `items`."""
        return share(items, index, self.count)

    def rectangles(self, grid, grid_panels):
        """The team as inclusive rectangles (x0, y0, x1, y1), none across a panel.

        The rectangles are disjoint and cover exactly the team's cores.
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


LANE_NONE = 0xFF  # rank of a team member outside every lane (km::kLaneNone)
LANE_CORES_MAX = 16  # a sender and up to 15 receivers (KM_LANE_RANKS)
LANE_WINDOW_MAX = 8  # km::kLaneWindowMax


@dataclass(frozen=True)
class Lane:
    """The cores (x, y) of one lane in rank order; rank 0 is the sender."""

    cores: tuple

    @property
    def sender(self):
        return self.cores[0]


def lanes(grid, grid_panels, rect, along, noc):
    """The lanes of `rect` = (x0, y0, columns, rows) along "column" or "row".

    One lane per column of the rectangle, or per row and panel. The sender is where
    the NoC's direction starts: NoC0 top or left, NoC1 bottom or right."""
    x0, y0, nx, ny = rect
    if nx < 1 or ny < 1 or x0 < 0 or y0 < 0 or x0 + nx > grid[0] or y0 + ny > grid[1]:
        raise ValueError(f"rectangle {rect} does not fit a {grid[0]} by {grid[1]} grid")
    if noc not in (0, 1) or along not in ("column", "row"):
        raise ValueError(f"lanes along {along!r} on NoC {noc}")
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
    out = [Lane(tuple(run if noc == 0 else run[::-1])) for run in runs]
    if any(len(lane.cores) > LANE_CORES_MAX for lane in out):
        raise ValueError(f"a lane of {rect} has more than {LANE_CORES_MAX} cores")
    return out


def lane_rect(lane, noc0, noc_xy, noc):
    """Multicast corners (start, end) of the lane's receivers as x | y << 16 words.

    noc0[core]: NoC0 (x, y); noc_xy[core]: core table address. Receivers follow the
    sender one NoC0 step at a time in the NoC's direction. None for one core."""
    if len(lane.cores) < 2:
        return None
    step = 1 if noc == 0 else -1
    sx, sy = noc0[lane.sender]
    dx, dy = (step, 0) if lane.cores[1][1] == lane.sender[1] else (0, step)
    for rank, core in enumerate(lane.cores):
        if noc0[core] != (sx + rank * dx, sy + rank * dy):
            raise ValueError(f"{core} is not NoC0 step {rank} from the sender")
    first, last = noc_xy[lane.cores[1]], noc_xy[lane.cores[-1]]
    return first[0] | first[1] << 16, last[0] | last[1] << 16


def lane_args(grid, grid_panels, team, rect, along, noc, noc0, noc_xy, buffer):
    """Each team member's lane arguments by core index (km::LaneArgs fields).

    buffer = (lane id, circular buffer index, window B, pages P). The rectangle must
    lie in the team; members outside it get rank LANE_NONE."""
    lane_id, cb, window, pages = buffer
    if not 1 <= window <= LANE_WINDOW_MAX or pages < 1:
        raise ValueError(f"window {window} or pages {pages} out of range")
    gx = grid[0]
    members = set(team.members)
    common = {
        "lane": lane_id,
        "team": team.first | team.count << 16,
        "noc": noc,
        "cb": cb,
        "window": window,
        "pages": pages,
    }
    none = common | {"rank": LANE_NONE, "receivers": 0, "sender": 0}
    out = {i: none | {"mc_start": 0, "mc_end": 0} for i in members}
    for lane in lanes(grid, grid_panels, rect, along, noc):
        index = [y * gx + x for x, y in lane.cores]
        if not members.issuperset(index):
            raise ValueError(f"rectangle {rect} is not inside {team}")
        corners = lane_rect(lane, noc0, noc_xy, noc) or (0, 0)
        for rank, i in enumerate(index):
            out[i] = common | {
                "rank": rank,
                "receivers": len(index) - 1,
                "sender": index[0],
                "mc_start": corners[0],
                "mc_end": corners[1],
            }
    return out
