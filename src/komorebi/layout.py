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
    columns, rows = grid
    return [(x, y) for y in range(rows) for x in range(columns)]


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
        return {
            core: (int(word) & 0xFFFF, int(word) >> 16)
            for core, word in zip(order, words)
        }

    at0 = by_core(noc0)
    column_xs = [at0[(x, 0)][0] for x in range(grid[0])]
    return Nocs(
        tuple(grid),
        by_core(table),
        at0,
        tuple(int(word) for word in table),
        panels(column_xs),
    )


def partition(items, index, count):
    """The part of the range `items` that rank `index` of `count` cores takes.

    Parts are contiguous and cover the range once; the first len(items) % count
    cores take one item more.
    """
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"rank {index} of {count}")
    if items.step != 1:
        raise ValueError("partition needs a range with step 1")
    base, extra = divmod(len(items), count)
    start = items.start + index * base + min(index, extra)
    return range(start, start + base + (index < extra))


@dataclass(frozen=True)
class Counters:
    """A dynamic run's counter tiles: the last `k` cores of the group (core order
    `first`, `count` cores) hold the other cores' tails and never submit an atomic."""

    workers: range  # the cores that send and receive messages
    counters: range  # the counter tiles
    counter_of: dict  # core: the counter tile that holds its tail
    serves: dict  # counter tile: the cores whose tails it holds


def counter_tiles(first, count, counter_count):
    """The counter tiles of a dynamic run of group (`first`, `count`) with
    `counter_count` of them: a core reserves on counter tile (core - first) mod
    `counter_count` (as km::messages_counter)."""
    most = abi.headers()["KM_COUNTERS_MAX"]
    if not 1 <= counter_count <= most:
        raise ValueError(f"counters {counter_count}: not in 1..{most}")
    if count <= counter_count:
        raise ValueError(
            f"a group of {count} cores leaves no core beside {counter_count} counters"
        )
    workers = range(first, first + count - counter_count)
    counters = range(workers.stop, first + count)
    counter_of = {core: counters[(core - first) % counter_count] for core in workers}
    serves = {
        tile: [core for core in workers if counter_of[core] == tile]
        for tile in counters
    }
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
    """The DRAM pages of core `core`'s queue of a dynamic handler (`page_messages`
    messages a page, the queues of all cores in core order): a range of page indices."""
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
        """Rank `index` of this group's part of the range `items`."""
        return partition(items, index, self.count)

    def rectangles(self, grid, grid_panels):
        """The group as inclusive rectangles (x0, y0, x1, y1), none across a panel.

        The rectangles are disjoint and cover exactly the group's cores.
        """
        self.cores(grid)
        columns = grid[0]
        if [
            x
            for panel_first, panel_last in grid_panels
            for x in range(panel_first, panel_last + 1)
        ] != list(range(columns)):
            raise ValueError(f"panels {grid_panels} do not tile {columns} columns")
        first_row, first_column = divmod(self.first, columns)
        last_row, last_column = divmod(self.first + self.count - 1, columns)
        # Row bands (y0, y1, x0, x1): partial first row, full rows, partial last row.
        if first_row == last_row:
            bands = [(first_row, first_row, first_column, last_column)]
        else:
            bands, low, high = [], first_row, last_row
            if first_column:
                bands.append((first_row, first_row, first_column, columns - 1))
                low += 1
            if last_column != columns - 1:
                high -= 1
            if low <= high:
                bands.append((low, high, 0, columns - 1))
            if last_column != columns - 1:
                bands.append((last_row, last_row, 0, last_column))
        return [
            (max(x0, panel_first), y0, min(x1, panel_last), y1)
            for y0, y1, x0, x1 in bands
            for panel_first, panel_last in grid_panels
            if max(x0, panel_first) <= min(x1, panel_last)
        ]


BROADCAST_NONE = (
    0xFF  # rank of a core of the group outside every broadcast (km::kBroadcastNone)
)
BROADCAST_CORES_MAX = 16  # a sender and up to 15 receivers (KM_BROADCAST_RANKS)
BROADCAST_WINDOW_MAX = 8  # km::kBroadcastWindowMax
BROADCAST_NO_NEXT = 0xFFFFFFFF  # no part on another island after this one


@dataclass(frozen=True)
class Broadcast:
    """The cores (x, y) of one broadcast in rank order; rank 0 is the sender."""

    cores: tuple

    @property
    def sender(self):
        return self.cores[0]


def broadcasts(grid, grid_panels, rect, along, noc, length=None):
    """The broadcasts of `rect` = (x0, y0, columns, rows) along "column" or "row".

    One per column of the rectangle, or per row and panel (with `length`, per band of
    so many cores of each), sent from the NoC's start: NoC0 top or left, NoC1 bottom."""
    x0, y0, columns, rows = rect
    if (
        columns < 1
        or rows < 1
        or x0 < 0
        or y0 < 0
        or x0 + columns > grid[0]
        or y0 + rows > grid[1]
    ):
        raise ValueError(f"rectangle {rect} does not fit a {grid[0]} by {grid[1]} grid")
    if noc not in (0, 1) or along not in ("column", "row"):
        raise ValueError(f"broadcasts along {along!r} on NoC {noc}")
    x_range, y_range = range(x0, x0 + columns), range(y0, y0 + rows)
    if along == "column":
        runs = [[(x, y) for y in y_range] for x in x_range]
    else:
        runs = [
            [(x, y) for x in x_range if panel_first <= x <= panel_last]
            for y in y_range
            for panel_first, panel_last in grid_panels
            if panel_first <= x_range[-1] and x_range[0] <= panel_last
        ]
    if length is not None:
        if length < 1:
            raise ValueError(f"a broadcast's band of {length} cores")
        runs = [run[i : i + length] for run in runs for i in range(0, len(run), length)]
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
    sender_x, sender_y = noc0[broadcast.sender]
    step_x, step_y = (
        (step, 0) if broadcast.cores[1][1] == broadcast.sender[1] else (0, step)
    )
    for rank, core in enumerate(broadcast.cores):
        if noc0[core] != (sender_x + rank * step_x, sender_y + rank * step_y):
            raise ValueError(f"{core} is not NoC0 step {rank} from the sender")
    first, last = noc_xy[broadcast.cores[1]], noc_xy[broadcast.cores[-1]]
    return first[0] | first[1] << 16, last[0] | last[1] << 16


def broadcast_args(
    grid, grid_panels, group, rect, along, noc, noc0, noc_xy, buffer, length=None
):
    """Each core's broadcast arguments by core index (km::BroadcastArgs fields).

    buffer = (broadcast id, circular buffer index, window B, pages P), B = P = 0 for
    words alone; `length`: bands. The rectangle is in the group; others get no rank."""
    broadcast_id, cb, window, pages = buffer
    if (window, pages) != (0, 0) and (
        not 1 <= window <= BROADCAST_WINDOW_MAX or pages < 1
    ):
        raise ValueError(f"window {window} or pages {pages} out of range")
    columns = grid[0]
    workers = set(group.indices)
    common = {
        "broadcast": broadcast_id,
        "group": group.first | group.count << 16,
        "noc": noc,
        "cb": cb,
        "window": window,
        "pages": pages,
    }
    common |= {"next": BROADCAST_NO_NEXT, "fed": 0, "from": 0}
    none = common | {"rank": BROADCAST_NONE, "receivers": 0, "sender": 0}
    out = {i: none | {"mc_start": 0, "mc_end": 0} for i in workers}
    for broadcast in broadcasts(grid, grid_panels, rect, along, noc, length):
        index = [y * columns + x for x, y in broadcast.cores]
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


def stack_rank(stack, columns, x, y):
    """The flat rank of cell (x, y) of islands stacked one below the other (a dict of
    "islands" (nocs, first rank, ranks), "tops" each one's first row in the stack and
    "rows" its rows, as island sessions give a class), or None: past every island's
    rows or one of an island's link cores (after its ranks)."""
    tops, rows, islands = stack["tops"], stack["rows"], stack["islands"]
    island = max(i for i, top in enumerate(tops) if top <= y)
    local = (y - tops[island]) * columns + x
    if y >= tops[island] + rows[island] or local >= islands[island][2]:
        return None
    return islands[island][1] + local


def stack_islands(stack, y, height):
    """The islands (their numbers) whose rows hold some of rows y .. y + height - 1 of
    islands stacked one below the other (`stack`, as stack_rank's)."""
    tops, rows = stack["tops"], stack["rows"]
    return [
        i
        for i, (top, n) in enumerate(zip(tops, rows))
        if top < y + height and y < top + n
    ]


def island_broadcast_args(
    islands, group, rect, along, noc, buffer, stack="below", bands=None, length=None
):
    """broadcast_args per island (nocs, first rank, ranks; core = local rank), the grids
    one `stack` "below" or "beside" the other, `rect` and `group` in that whole; a split
    card: `bands` its islands' (first row, rows), sender words card cores; `length` as
    broadcast_args's. A part an island, fed across by the previous part's first core."""
    broadcast_id, cb, window, pages = buffer
    if (window, pages) != (0, 0) and (
        not 1 <= window <= BROADCAST_WINDOW_MAX or pages < 1
    ):
        raise ValueError(f"window {window} or pages {pages} out of range")
    grid = islands[0][0].grid
    if any(nocs.grid != grid for nocs, _, _ in islands):
        raise ValueError("island broadcasts need one grid shape on every island")
    if stack not in ("below", "beside") or along not in ("column", "row"):
        raise ValueError(f"islands {stack!r}, broadcasts along {along!r}")
    columns, rows = grid
    below = stack == "below" or bands is not None
    whole = (columns, rows * len(islands)) if below else (columns * len(islands), rows)
    if bands is not None:
        whole = grid
    x0, y0, width, height = rect
    if width < 1 or height < 1 or x0 < 0 or y0 < 0:
        raise ValueError(f"rectangle {rect}")
    if x0 + width > whole[0] or y0 + height > whole[1]:
        raise ValueError(f"rectangle {rect} does not fit {whole[0]} by {whole[1]}")

    def place(x, y):  # (island, x, y) of a cell of the whole
        if bands is not None:  # the card's own cells
            (island,) = [i for i, (top, n) in enumerate(bands) if top <= y < top + n]
            return island, x, y
        island = y // rows if below else x // columns
        return island, x % columns, y % rows

    def panel(cell):
        island, x, _ = cell
        index = [first <= x <= last for first, last in islands[island][0].panels]
        return island, index.index(True)

    xs, ys = range(x0, x0 + width), range(y0, y0 + height)
    if along == "column":
        runs = [[place(x, y) for y in ys] for x in xs]
    else:
        runs = []
        for y in ys:
            run = []
            for x in xs:
                cell = place(x, y)
                if run and run[-1][0] == cell[0] and panel(run[-1]) != panel(cell):
                    runs.append(run)
                    run = []
                run.append(cell)
            runs.append(run)
    if length is not None:
        if length < 1:
            raise ValueError(f"a broadcast's band of {length} cores")
        runs = [run[i : i + length] for run in runs for i in range(0, len(run), length)]
    workers = set(group.indices)
    common = {
        "broadcast": broadcast_id,
        "group": group.first | group.count << 16,
        "noc": noc,
        "cb": cb,
        "window": window,
        "pages": pages,
    }
    none = common | {
        "rank": BROADCAST_NONE,
        "receivers": 0,
        "sender": 0,
        "mc_start": 0,
        "mc_end": 0,
        "next": BROADCAST_NO_NEXT,
        "fed": 0,
        "from": 0,
    }
    out = []
    for _, first, count in islands:
        mine = range(
            max(first, group.first), min(first + count, group.first + group.count)
        )
        out.append({rank - first: none for rank in mine})
    for run in runs:
        ordered = run if noc == 0 else run[::-1]
        parts = []
        for cell in ordered:
            if parts and parts[-1][0][0] == cell[0]:
                parts[-1].append(cell)
            else:
                parts.append([cell])
        if len({part[0][0] for part in parts}) != len(parts):
            raise ValueError(f"a broadcast of {rect} enters an island twice")
        heads = []
        for part in parts:
            island = part[0][0]
            nocs, first, count = islands[island]
            local = Broadcast(tuple((x, y) for _, x, y in part))
            top = bands[island][0] if bands is not None else 0
            index = [(y - top) * columns + x for x, y in local.cores]
            flat = [first + i for i in index]
            if max(index) >= count or not workers.issuperset(flat):
                raise ValueError(f"rectangle {rect} is not inside {group}")
            if len(index) > BROADCAST_CORES_MAX - (part is not parts[-1]):
                raise ValueError(f"a part of a broadcast of {rect} has too many cores")
            corners = broadcast_rect(local, nocs.noc0, nocs.noc_xy, noc) or (0, 0)
            heads.append((island, index, flat[0], corners, top * columns))
        for k, (island, index, _, corners, base) in enumerate(heads):
            feeder = heads[k - 1] if k else None
            after = heads[k + 1][2] if k + 1 < len(heads) else BROADCAST_NO_NEXT
            for rank, i in enumerate(index):
                out[island][i] = common | {
                    "rank": rank,
                    "receivers": len(index) - 1,
                    "sender": base + index[0],  # its core of the card
                    "mc_start": corners[0],
                    "mc_end": corners[1],
                    "next": after if rank == 0 else BROADCAST_NO_NEXT,
                    "fed": len(feeder[1]) if feeder and rank == 0 else 0,
                    "from": feeder[2] if feeder and rank == 0 else 0,
                }
    return out
