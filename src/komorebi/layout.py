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


def core_order(grid):
    """Every core of the grid in the fixed core order: row by row, x fastest."""
    gx, gy = grid
    return [(x, y) for y in range(gy) for x in range(gx)]


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
