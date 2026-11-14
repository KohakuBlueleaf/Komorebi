"""Islands: sets of cores that reach each other only through links, each with an
entry in the island table (one card can be described as several). Pure host values: the
graph, ranks flat and contiguous by island, the table a core holds, placement and trees.
On one island every answer is today's."""

import struct
from bisect import bisect_right
from collections import deque
from typing import NamedTuple

from komorebi import channel

RANKS_MAX = 65024  # a group's first below 0xFE00, where link cores' names start
LINK_BASE = 0xFE00  # KM_GROUP_LINK_BASE: first = this + k names link cores k on
ISLANDS_MAX = 256  # an island's number is 8 bits
CORES_MAX = 128  # KM_CORES_MAX: an island's core table
THREAD_NUMBERS = 256  # 0xFF00 + number in 16 bits; numbers 1 to 255, 0 names none
NO_LINK = 0xFF  # an island's own entry: no link toward it

# An island table entry, little-endian (km_abi.h mirrors it): u16 first rank, u16
# count, u8 next link (the holder's own link number), u8 hops, u16 zero.
ENTRY = struct.Struct("<HHBBH")
ENTRY_BYTES = ENTRY.size


class Island(NamedTuple):
    """An island's cores: `cores` are core numbers of its card in core order (core c at
    column c % columns, row c // columns of the card's grid); `threads` how many X280
    threads it serves."""

    cores: tuple
    columns: int
    threads: int = 0


class Link(NamedTuple):
    """`count` links between islands `a` and `b`, each of `rate` bytes a second and
    packets of at most `packet` bytes (None: not measured)."""

    a: int
    b: int
    count: int = 1
    rate: float | None = None
    packet: int | None = None


class Placement(NamedTuple):
    """Where a tensor's pages live: `parts[island]` (first page, pages) for a split,
    or `copy` True and every island holds every page."""

    parts: tuple
    copy: bool = False


class Tree(NamedTuple):
    """A collect or broadcast over islands: `leaders[island]` folds the group's part
    there (the root, else the part's first rank); `parent[island]` is where it crosses
    to; `order` the crossing islands deepest first (collect in it, broadcast reversed).
    On one island: no parent, no crossing."""

    root: int
    leaders: dict
    parent: dict
    order: tuple


def partition(items, part, parts):
    """(start, length) of part `part` of `parts`: contiguous, the first items % parts
    parts one item more. The one rule of the host and km::partition on a core; on one
    island, partition(items, 0, 1) is all items."""
    if not 0 <= part < parts:
        raise ValueError(f"part {part} of {parts}")
    return channel.partition(items, part, parts)


def spread(cores, columns, count):
    """`count` of `cores` for link cores, spread over rows and columns: each is asked
    near its own point of the island's diagonal, preferring a row and a column no
    earlier one took (senders in one row share NoC paths). On one island: none."""
    cores = tuple(cores)
    if count == 0:
        return ()
    if count >= len(cores):
        raise ValueError(f"{count} link cores leave no rank of {len(cores)} cores")
    where = {core: divmod(core, columns) for core in cores}
    rows = sorted({row for row, _ in where.values()})
    cols = sorted({col for _, col in where.values()})
    chosen, rows_used, cols_used = [], set(), set()
    for link in range(count):
        row = rows[(2 * link + 1) * len(rows) // (2 * count)]
        col = cols[(2 * link + 1) * len(cols) // (2 * count)]

        def rank_of(core, row=row, col=col):
            at_row, at_col = where[core]
            near = abs(at_row - row) + abs(at_col - col)
            return at_row in rows_used, at_col in cols_used, near, core

        best = min((core for core in cores if core not in chosen), key=rank_of)
        chosen.append(best)
        rows_used.add(where[best][0])
        cols_used.add(where[best][1])
    return tuple(chosen)


class Graph:
    """Islands and their links (any connected graph). A used link takes a core on each
    island it joins (`reserve` False: none; `link_cores`: per island, these cores);
    ranks and thread numbers (from 1) are flat and contiguous by island, ranks = cores
    less link cores in core order. Routes by hops, ties to the lowest island. On one
    island: no link cores, rank = core."""

    def __init__(self, islands, links=(), reserve=True, link_cores=None):
        self.islands = tuple(
            Island(tuple(i.cores), i.columns, i.threads) for i in islands
        )
        self.links = tuple(Link(*link) for link in links)
        self._check()
        count = len(self.islands)
        # Each island's own links, numbered by neighbour then by link: (neighbour, k)
        self.ports = [[] for _ in range(count)]
        for link in sorted(
            self.links, key=lambda link: (min(link.a, link.b), max(link.a, link.b))
        ):
            for k in range(link.count):
                self.ports[link.a].append((link.b, k))
                self.ports[link.b].append((link.a, k))
        self.ports = tuple(tuple(sorted(ports)) for ports in self.ports)
        if link_cores is None:
            self.link_cores = tuple(
                spread(
                    island.cores, island.columns, len(self.ports[i]) if reserve else 0
                )
                for i, island in enumerate(self.islands)
            )
        else:
            self.link_cores = tuple(tuple(cores) for cores in link_cores)
            if len(self.link_cores) != len(self.islands):
                raise ValueError(f"link cores of {len(self.link_cores)} islands")
            for i, (island, taken) in enumerate(zip(self.islands, self.link_cores)):
                if len(set(taken)) != len(taken) or not set(taken) <= set(island.cores):
                    raise ValueError(f"island {i}: link cores {taken} not its cores")
        self.rank_cores = tuple(
            tuple(core for core in island.cores if core not in taken)
            for island, taken in zip(self.islands, self.link_cores)
        )
        self.firsts, total = [], 0
        for i, cores in enumerate(self.rank_cores):
            if not cores:
                raise ValueError(f"island {i}: its link cores leave no rank")
            self.firsts.append(total)
            total += len(cores)
        if total > RANKS_MAX:
            raise ValueError(f"{total} ranks: at most RANKS_MAX {RANKS_MAX}")
        self.ranks = total
        self.thread_firsts, threads = [], 1
        for island in self.islands:
            self.thread_firsts.append(threads)
            threads += island.threads
        if threads > THREAD_NUMBERS:
            raise ValueError(
                f"{threads - 1} threads: numbers 1 to {THREAD_NUMBERS - 1} "
                f"(THREAD_NUMBERS {THREAD_NUMBERS})"
            )
        self._routes()

    def _check(self):
        count = len(self.islands)
        if not 1 <= count <= ISLANDS_MAX:
            raise ValueError(f"{count} islands: 1 to ISLANDS_MAX {ISLANDS_MAX}")
        for i, island in enumerate(self.islands):
            cores = island.cores
            if not 1 <= len(cores) <= CORES_MAX:
                raise ValueError(
                    f"island {i}: {len(cores)} cores, 1 to CORES_MAX {CORES_MAX}"
                )
            if len(set(cores)) != len(cores) or min(cores) < 0 or island.columns < 1:
                raise ValueError(
                    f"island {i}: cores {cores} in {island.columns} columns"
                )
            if island.threads < 0:
                raise ValueError(f"island {i}: {island.threads} threads")
        pairs = set()
        for link in self.links:
            pair = (min(link.a, link.b), max(link.a, link.b))
            if link.a == link.b or not 0 <= pair[0] < pair[1] < count:
                raise ValueError(f"{link}: two different islands of {count}")
            if pair in pairs:
                raise ValueError(f"{link}: islands {pair} are joined twice")
            if link.count < 1:
                raise ValueError(f"{link}: at least one link")
            pairs.add(pair)

    def _routes(self):
        count = len(self.islands)
        near = [sorted({neighbour for neighbour, _ in ports}) for ports in self.ports]
        self._hops = []
        for target in range(count):
            hops = [None] * count
            hops[target], queue = 0, deque([target])
            while queue:
                at = queue.popleft()
                for neighbour in near[at]:
                    if hops[neighbour] is None:
                        hops[neighbour] = hops[at] + 1
                        queue.append(neighbour)
            if None in hops:
                raise ValueError(
                    f"island {hops.index(None)} has no path to island {target}"
                )
            self._hops.append(hops)  # self._hops[target][source]
        self._next = [
            [
                (
                    None
                    if source == target
                    else min(
                        n
                        for n in near[source]
                        if self._hops[target][n] == self._hops[target][source] - 1
                    )
                )
                for target in range(count)
            ]
            for source in range(count)
        ]
        self._neighbours = tuple(tuple(n) for n in near)

    # The graph's questions

    def hops(self, source, target):
        """Links crossed from island `source` to `target`; 0 on one island."""
        return self._hops[target][source]

    def next_island(self, source, target):
        """The neighbour of `source` on the way to `target`; None when they are one."""
        return self._next[source][target]

    def next_link(self, source, target):
        """`source`'s own link number of the first link toward `target` (the one that
        carries messages and commands, in order); NO_LINK when they are one island."""
        step = self._next[source][target]
        return NO_LINK if step is None else self.ports[source].index((step, 0))

    def neighbours(self, island):
        """The islands one link away; none on one island."""
        return self._neighbours[island]

    def between(self, a, b):
        """The Link joining neighbours `a` and `b` (its count, rate, largest packet)."""
        for link in self.links:
            if {link.a, link.b} == {a, b}:
                return link
        raise ValueError(f"islands {a} and {b} are not neighbours")

    # Ranks

    def first(self, island):
        """The island's first rank; 0 on one island."""
        return self.firsts[island]

    def count(self, island):
        """The island's ranks: its cores less its link cores; every core on one
        island."""
        return len(self.rank_cores[island])

    def link_group(self, island, k=0, count=None):
        """The group (first, count) that names link cores k to k + count - 1 of
        `island` (default: k on) in a command to that island: KM_GROUP_LINK_BASE + k,
        never cut by island; a link core's rank in it is its index less k."""
        cores = len(self.link_cores[island])
        count = cores - k if count is None else count
        if count < 1 or k < 0 or k + count > cores:
            raise ValueError(f"link cores {k} + {count} of island {island}'s {cores}")
        return LINK_BASE + k, count

    def resolve(self, rank):
        """(island, local rank) of `rank`; (0, rank) on one island."""
        if not 0 <= rank < self.ranks:
            raise ValueError(f"rank {rank} of {self.ranks}")
        island = bisect_right(self.firsts, rank) - 1
        return island, rank - self.firsts[island]

    def core_of(self, rank):
        """(island, core of its card) that runs `rank`; (0, rank) on one island whose
        cores are 0 on."""
        island, local = self.resolve(rank)
        return island, self.rank_cores[island][local]

    def split(self, group):
        """The part of `group` (first, count) of ranks on each island it touches, as
        [(island, local first, count)] in island order: what each island's dispatcher
        is given. On one island: [(0, first, count)]."""
        first, count = group
        if count < 1 or first < 0 or first + count > self.ranks:
            raise ValueError(f"group {group} of {self.ranks} ranks")
        end, parts = first + count, []
        for island in range(self.resolve(first)[0], self.resolve(end - 1)[0] + 1):
            low = max(first, self.firsts[island])
            high = min(end, self.firsts[island] + self.count(island))
            parts.append((island, low - self.firsts[island], high - low))
        return parts

    # Threads

    def thread_numbers(self, island):
        """The thread numbers the island serves; 1 to its threads on one island."""
        first = self.thread_firsts[island]
        return range(first, first + self.islands[island].threads)

    def thread(self, number):
        """(island, its own thread number from 1) of thread `number`; (0, number) on
        one island."""
        for island in range(len(self.islands)):
            if number in self.thread_numbers(island):
                return island, number - self.thread_firsts[island] + 1
        raise ValueError(f"no thread {number}")

    # The table a core holds

    def table(self, island):
        """The island table a core of `island` holds: ENTRY_BYTES an island (layout at
        ENTRY). On one island: one entry (0, ranks, NO_LINK, 0)."""
        return b"".join(
            ENTRY.pack(
                self.firsts[other],
                self.count(other),
                self.next_link(island, other),
                self.hops(island, other),
                0,
            )
            for other in range(len(self.islands))
        )


def read_table(data):
    """[(first rank, count, next link, hops)] of a table's bytes."""
    if len(data) % ENTRY_BYTES:
        raise ValueError(f"{len(data)} bytes: not whole entries of {ENTRY_BYTES}")
    return [ENTRY.unpack_from(data, at)[:4] for at in range(0, len(data), ENTRY_BYTES)]


# The usual graphs


def one(cores, columns, threads=0):
    """One island of cores 0 to cores - 1: today's run, no link cores."""
    return Graph([Island(tuple(range(cores)), columns, threads)])


def line(islands, links=1, rate=None, packet=None):
    """`islands` (Island values) in a line, `links` links each neighbouring pair."""
    return Graph(
        islands, [Link(i, i + 1, links, rate, packet) for i in range(len(islands) - 1)]
    )


def ring(islands, links=1, rate=None, packet=None):
    """`islands` in a ring (a line for fewer than three: a pair is joined once)."""
    count = len(islands)
    if count < 3:
        return line(islands, links, rate, packet)
    return Graph(
        islands, [Link(i, (i + 1) % count, links, rate, packet) for i in range(count)]
    )


def split_card(
    cores, columns, parts, shape="line", links=1, rate=None, packet=None, threads=0
):
    """One card of cores 0 to cores - 1 described as `parts` islands of contiguous
    cores (sizes by partition), joined as a "line" or a "ring"; `threads` on island 0.
    parts 1: the card as today."""
    islands = []
    for part in range(parts):
        start, length = partition(cores, part, parts)
        islands.append(
            Island(
                tuple(range(start, start + length)),
                columns,
                threads if part == 0 else 0,
            )
        )
    if shape == "line":
        return line(islands, links, rate, packet)
    if shape == "ring":
        return ring(islands, links, rate, packet)
    raise ValueError(f"shape {shape!r}: 'line' or 'ring'")


# Data


def leading_units(shape, layout="row"):
    """How many steps of the leading dimension a tensor's pages make (each step the
    same number of pages): rows of a "row" tensor's first dimension; tile rows of a
    2-D "tile" tensor, else its first dimension."""
    shape = tuple(shape)
    if layout == "row":
        return shape[0] if len(shape) >= 2 else 1
    if layout == "tile":
        return shape[0] // 32 if len(shape) == 2 else shape[0] if len(shape) > 2 else 1
    raise ValueError(f"layout {layout!r}: 'row' or 'tile'")


def place(graph, pages, units=None, like=None, copy=False):
    """Where a tensor's pages live: split along `units` leading steps (None: a page a
    step) as partition gives them to the ranks of `like` (first, count; its producer,
    consumer or a named group; default all ranks), so each rank's part is on its own
    island; `copy`: a copy on every island. On one island: every page on island 0."""
    islands = len(graph.islands)
    if copy:
        return Placement(tuple((0, pages) for _ in range(islands)), True)
    units = pages if units is None else units
    if units < 1 or pages % units:
        raise ValueError(f"{pages} pages are not {units} equal steps")
    step = pages // units
    first, count = like if like is not None else (0, graph.ranks)
    graph.split((first, count))  # the group must be the graph's
    parts = []
    for island in range(islands):
        low = min(max(graph.first(island), first), first + count) - first
        high = (
            min(max(graph.first(island) + graph.count(island), first), first + count)
            - first
        )
        start = channel.partition(units, low, count)[0] * step
        end = channel.partition(units, high, count)[0] * step
        parts.append((start, end - start))
    return Placement(tuple(parts))


def island_of(placement, page, island=0):
    """The island that holds `page`; for a copy, the asker's own `island`. On one
    island: 0."""
    if placement.copy:
        return island
    for holder, (start, length) in enumerate(placement.parts):
        if start <= page < start + length:
            return holder
    raise ValueError(f"page {page} is outside the tensor")


# Trees


def tree(graph, group, root=None):
    """The collect and broadcast of `group` (first, count): inside each island first,
    then one crossing an island toward the root's island. A part's crossing goes to
    the first island on its shortest path that holds part of the group (the links
    pass it through the others). `root`: a rank of the group, its first by default."""
    parts = graph.split(group)
    root = group[0] if root is None else root
    if not group[0] <= root < group[0] + group[1]:
        raise ValueError(f"root {root} is not in group {group}")
    home = graph.resolve(root)[0]
    held = {island for island, _, _ in parts}
    leaders = {island: graph.first(island) + local for island, local, _ in parts}
    leaders[home] = root
    parent = {}
    for island in held - {home}:
        step = graph.next_island(island, home)
        while step not in held:
            step = graph.next_island(step, home)
        parent[island] = step

    def depth(island):
        return 0 if island == home else 1 + depth(parent[island])

    order = tuple(sorted(parent, key=lambda island: (-depth(island), island)))
    return Tree(home, leaders, parent, order)
