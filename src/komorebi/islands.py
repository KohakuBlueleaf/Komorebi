"""Islands: sets of cores that reach each other only through links, each with an
entry in the island table (one card can be described as several). Pure host values: the
graph, ranks flat and contiguous by island, the table a core holds, placement and trees.
On one island every answer is today's."""

import math
import struct
from bisect import bisect_right
from collections import deque
from typing import NamedTuple

import numpy as np

from komorebi import channel

RANKS_MAX = 65024  # a group's first below 0xFE00, where link cores' names start
LINK_BASE = 0xFE00  # KM_GROUP_LINK_BASE: first = this + k names link cores k on
ISLANDS_MAX = 256  # an island's number is 8 bits
CORES_MAX = 128  # KM_CORES_MAX: an island's core table
THREAD_NUMBERS = 256  # 0xFF00 + number in 16 bits; numbers 1 to 255, 0 names none
NO_LINK = 0xFF  # an island's own entry: no link toward it

# An island table entry, little-endian (km_abi.h mirrors it): u16 first, u16 count, u8
# next link, u8 hops, u8 ports toward it (bit p: neighbour p on a shortest path), u8 0.
ENTRY = struct.Struct("<HHBBBB")
PORTS_MAX = 4  # a card's ports (north, south, east, west): neighbours an island
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
    """Where a tensor's pages live. The pages are `outer` blocks of `steps` steps of the
    split dimension, `inner` pages a step; island i holds steps `parts[i]` (first step,
    steps) of every block, in that order (its part). `copy`: every island holds every
    page. `split` and `group` are the hint it came from (None: a copy)."""

    parts: tuple
    copy: bool = False
    steps: int = 1
    inner: int = 1
    outer: int = 1
    split: int | None = 0
    group: tuple | None = None

    @property
    def pages(self):
        return self.outer * self.steps * self.inner

    def part_pages(self, island):
        """The pages island `island` holds: its part's length."""
        return (
            self.pages if self.copy else self.outer * self.parts[island][1] * self.inner
        )

    def where(self, page, island=0):
        """(island that holds `page`, its index in that island's part); a copy's
        page: the asker's own `island`, index `page`. One island: (0, page)."""
        if not 0 <= page < self.pages:
            raise ValueError(f"page {page} is outside the tensor's {self.pages}")
        if self.copy:
            return island, page
        block, rest = divmod(page, self.steps * self.inner)
        step, offset = divmod(rest, self.inner)
        for holder, (first, steps) in enumerate(self.parts):
            if first <= step < first + steps:
                return holder, (block * steps + step - first) * self.inner + offset
        raise ValueError(f"page {page}: step {step} is in no island's part")

    def runs(self, first, pages, island=0):
        """[(holder, first page, pages, index in the holder's part)] of pages first ..
        first + pages - 1: each run held by one island at consecutive indices. A step
        of the walk is a holder's whole stretch of a block (its steps there), not a
        page: the host's cost is its runs', not its pages'."""
        out = []
        end = first + pages
        if pages > 0:
            self.where(end - 1, island)  # the last page is in the tensor
        block_pages = self.steps * self.inner
        page = first
        while page < end:
            holder, index = self.where(page, island)
            if self.copy:
                stop = end
            else:
                start, steps = self.parts[holder]
                block = page // block_pages
                stop = min(end, block * block_pages + (start + steps) * self.inner)
            length = stop - page
            if out and out[-1][0] == holder and out[-1][3] + out[-1][2] == index:
                out[-1] = (holder, out[-1][1], out[-1][2] + length, out[-1][3])
            else:
                out.append((holder, page, length, index))
            page = stop
        return out

    def part(self, island):
        """The pages of island `island`'s part in its order (an int64 array): what
        its page k is in the whole tensor."""
        if self.copy:
            return np.arange(self.pages, dtype=np.int64)
        first, steps = self.parts[island]
        blocks = np.arange(self.outer, dtype=np.int64)[:, None, None]
        held = np.arange(first, first + steps, dtype=np.int64)[None, :, None]
        inner = np.arange(self.inner, dtype=np.int64)[None, None, :]
        return ((blocks * self.steps + held) * self.inner + inner).reshape(-1)


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
    island it joins (`reserve` False: none; `link_cores`: these cores); ranks and thread
    numbers (from 1) are flat and contiguous by island, ranks = cores less link cores.
    Routes by hops, ties to the lowest island. On one island: rank = core."""

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
        """The islands one link away (port p is neighbour p); none on one island."""
        return self._neighbours[island]

    def ports_toward(self, source, target):
        """The mask of `source`'s ports (bit p: neighbour p) on a shortest path to
        `target`: the links its pages may take; 0 when they are one island."""
        if source == target:
            return 0
        hops = self._hops[target]
        return sum(
            1 << port
            for port, neighbour in enumerate(self._neighbours[source])
            if hops[neighbour] == hops[source] - 1
        )

    def diameter(self):
        """The most links any route crosses; 0 on one island."""
        return max(max(hops) for hops in self._hops)

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
        """The island table a core of `island` holds: ENTRY_BYTES an island (ENTRY; the
        last byte: the ports whose neighbour's next island toward that entry's is
        `island`, its children in its tree). One island: one entry (0, ranks)."""
        return b"".join(
            ENTRY.pack(
                self.firsts[other],
                self.count(other),
                self.next_link(island, other),
                self.hops(island, other),
                self.ports_toward(island, other),
                sum(
                    1 << p
                    for p, n in enumerate(self._neighbours[island])
                    if self._next[n][other] == island
                ),
            )
            for other in range(len(self.islands))
        )


def read_table(data, ports=False):
    """[(first rank, count, next link, hops)] of a table's bytes (`ports`: and the
    ports' mask)."""
    if len(data) % ENTRY_BYTES:
        raise ValueError(f"{len(data)} bytes: not whole entries of {ENTRY_BYTES}")
    fields = 5 if ports else 4
    return [
        ENTRY.unpack_from(data, at)[:fields] for at in range(0, len(data), ENTRY_BYTES)
    ]


# The usual graphs


def one(cores, columns, threads=0):
    """One island of cores 0 to cores - 1: today's run, no link cores."""
    return Graph([Island(tuple(range(cores)), columns, threads)])


def line(islands, links=1, rate=None, packet=None, reserve=True):
    """`islands` (Island values) in a line, `links` links each neighbouring pair."""
    return Graph(
        islands,
        [Link(i, i + 1, links, rate, packet) for i in range(len(islands) - 1)],
        reserve,
    )


def ring(islands, links=1, rate=None, packet=None, reserve=True):
    """`islands` in a ring (a line for fewer than three: a pair is joined once)."""
    count = len(islands)
    if count < 3:
        return line(islands, links, rate, packet, reserve)
    return Graph(
        islands,
        [Link(i, (i + 1) % count, links, rate, packet) for i in range(count)],
        reserve,
    )


def grid(islands, rows, columns, wrap=True, links=1, rate=None, packet=None):
    """`islands` (rows x columns of them, row by row) joined to their north, south,
    east and west neighbours, around the edges too (`wrap`); a pair met twice (a side
    of 2 with wrap) is joined by 2 x `links` links."""
    if len(islands) != rows * columns or rows < 1 or columns < 1:
        raise ValueError(f"{len(islands)} islands are not {rows} x {columns}")
    counts = {}
    for row in range(rows):
        for column in range(columns):
            here = row * columns + column
            steps = [(row, column + 1), (row + 1, column)]
            for r, c in steps:
                if not wrap and (r >= rows or c >= columns):
                    continue
                there = (r % rows) * columns + c % columns
                if there != here:
                    pair = (min(here, there), max(here, there))
                    counts[pair] = counts.get(pair, 0) + links
    return Graph(
        islands, [Link(a, b, n, rate, packet) for (a, b), n in sorted(counts.items())]
    )


def split_card(
    cores,
    columns,
    parts,
    shape="line",
    links=1,
    rate=None,
    packet=None,
    threads=0,
    rows=False,
    reserve=True,
):
    """One card of cores 0 to cores - 1 described as `parts` islands of contiguous
    cores (sizes by partition; `rows`: of whole rows), joined as a "line" or a "ring";
    `threads` on island 0; `reserve` False: no link cores. parts 1: the card as today.
    """
    unit = columns if rows else 1
    if cores % unit:
        raise ValueError(f"{cores} cores are not whole rows of {columns}")
    islands = []
    for part in range(parts):
        start, length = (unit * size for size in partition(cores // unit, part, parts))
        islands.append(
            Island(
                tuple(range(start, start + length)),
                columns,
                threads if part == 0 else 0,
            )
        )
    if shape == "line":
        return line(islands, links, rate, packet, reserve)
    if shape == "ring":
        return ring(islands, links, rate, packet, reserve)
    raise ValueError(f"shape {shape!r}: 'line' or 'ring'")


# Data


def page_dims(shape, layout="row"):
    """The dimensions a tensor's pages make, leading first: a "row" tensor's every
    dimension but the last (a page is one row of it); a "tile" tensor's leading
    dimensions, then its tile rows and tile columns."""
    shape = tuple(int(size) for size in shape)
    if layout == "row":
        return shape[:-1]
    if layout == "tile":
        if len(shape) < 2:
            raise ValueError(f"shape {shape}: a tile layout needs two dimensions")
        return (*shape[:-2], -(-shape[-2] // 32), -(-shape[-1] // 32))
    raise ValueError(f"layout {layout!r}: 'row' or 'tile'")


def place(graph, shape, layout="row", split=None, group=None, copy=False):
    """Where a tensor's pages live, by the hint in the kernel's words: `split` d and
    `group` (first rank, ranks; default every rank): the ranks of the group partition
    dimension d (in its page steps) as km::partition does, and rank r's steps are held
    by r's island; `copy`: on every island. No hint: split 0 over every rank."""
    islands = len(graph.islands)
    dims = page_dims(shape, layout)
    if copy:
        if split is not None or group is not None:
            raise ValueError("a copy is not split: no `split` or `group` with it")
        pages = math.prod(dims)
        return Placement(((0, 1),) * islands, True, 1, pages, 1, None, None)
    split = 0 if split is None else split
    if not dims and split == 0:
        dims = (1,)  # one page: one step
    if not -len(dims) <= split < len(dims):
        raise ValueError(
            f"split={split}: a {layout} tensor of shape {tuple(shape)} has pages along "
            f"dimensions 0 to {len(dims) - 1} (the rest is inside a page)"
        )
    split %= len(dims)
    first, count = group if group is not None else (0, graph.ranks)
    graph.split((first, count))  # the group must be the graph's
    steps = dims[split]
    parts = []
    for island in range(islands):
        low = min(max(graph.first(island), first), first + count) - first
        high = (
            min(max(graph.first(island) + graph.count(island), first), first + count)
            - first
        )
        start = channel.partition(steps, low, count)[0] if low < count else steps
        end = channel.partition(steps, high, count)[0] if high < count else steps
        parts.append((start, end - start))
    return Placement(
        tuple(parts),
        False,
        steps,
        math.prod(dims[split + 1 :]),
        math.prod(dims[:split]),
        split,
        (first, count),
    )


def island_of(placement, page, island=0):
    """The island that holds `page`; for a copy, the asker's own `island`. On one
    island: 0."""
    return placement.where(page, island)[0]


# The tensor table at word KM_ISLAND_TENSORS of a core's island area (km_abi.h mirrors):
# a reference word below TENSORS_MAX is its entry's index, any other a bank address.
TENSORS_MAX = 32
TENSOR_HEADER = struct.Struct("<4I")  # page bytes, steps, inner, copy | pages << 2
TENSOR_PART = struct.Struct("<4I")  # part's bank address, first step, steps, bytes
TENSOR_BANK = 2  # header word 3 of a range of one DRAM bank (km::BankRange)
DRAM_BANKS = 8  # an interleaved part's pages over its island's banks


def tensor_entry_bytes(islands):
    """An entry's bytes in a table of `islands` islands."""
    return TENSOR_HEADER.size + islands * TENSOR_PART.size


def tensor_entry(placement, aligned_page_bytes, addresses):
    """A tensor's entry: its header, then each island's part (its bank address in
    `addresses`, 0 where the island holds none; its bytes in each bank)."""
    head = TENSOR_HEADER.pack(
        aligned_page_bytes,
        placement.steps,
        placement.inner,
        int(placement.copy) | placement.pages << 2,  # its pages: KM_GUARD's bound
    )
    return head + b"".join(
        TENSOR_PART.pack(
            address,
            first,
            steps,
            (
                bank_bytes(placement.part_pages(island), aligned_page_bytes)
                if address
                else 0
            ),
        )
        for island, (address, (first, steps)) in enumerate(
            zip(addresses, placement.parts)
        )
    )


def bank_bytes(pages, aligned_page_bytes):
    """The bytes an interleaved buffer of `pages` pages holds in each bank."""
    return -(-pages // DRAM_BANKS) * aligned_page_bytes


def bank_entry(islands, holder, bank, address, size):
    """The entry of `size` bytes of DRAM bank `bank` from bank address `address`, held
    by island `holder` of `islands`: no page bytes (the kernel gives them)."""
    head = TENSOR_HEADER.pack(holder, bank, size, TENSOR_BANK)
    return head + b"".join(
        TENSOR_PART.pack(address if island == holder else 0, 0, 0, 0)
        for island in range(islands)
    )


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
