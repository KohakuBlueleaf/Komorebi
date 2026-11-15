"""A linear layer Y = X x W written for islands (kernels/linear), one command a call on
every rank, its form by the tensors' placements: output dim sharded (--place out),
input dim sharded (--place in: each island a partial of all of Y), W gathered (rows).
Prints one JSON line; exit 1: a failed check."""

import argparse
import contextlib
import dataclasses
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi import abi, layout, power  # noqa: E402
from projects.dense import bench, card, run  # noqa: E402
from projects.dense import plan as pl  # noqa: E402
from reference import dense as ref  # noqa: E402

KERNELS = pl.HERE / "kernels" / "linear"
LINEAR_H = KERNELS / "linear.h"
L = abi.constants(LINEAR_H, known=abi.headers().constants)
G = pl.G
# Two cards, the load's L1 table rows past the buffers and the session's own
# (IslandSession.held_bytes): the host transfer buffer, control block and overlay, slot.
L1_TRANSFER = 32_848
L1_RUNTIME = 17_024 + 21_568
# on islands the reader's slot holds the larger service overlay (the collectives')
L1_RUNTIME_ISLANDS = 17_024 + 40_896
# The kernel buffer of two cards (one card's follows the kernels): 106 K, the
# addition or gather 118,000 B (mock builds); two-card decode 115,248 B (the cards).
KERNEL_BYTES = {
    "two": 108 * 1024,
    "add": 116 * 1024,
    "decode": 116 * 1024,
}
log = run.log
# (height, width) of every piece's subblock in place of the plan's (--subblock)
SUBBLOCK = None
# one island: the plan on the grid's first columns only (--columns)
COLUMNS = None
# the reader's and the writer's NoC, X's and W's broadcasts with them (--nocs)
NOCS = (0, 1)
# both broadcasts mark a block valid once it is sent, not acknowledged (--valid sent)
VALID_SENT = False
# the kernels' account of each CPU's phases in T (--account; LINEAR_ACCOUNT)
ACCOUNT = False
# the K-split plan's reduce inside a group (--reduce)
SPLIT_REDUCE = "tree"
# the K-split plan's chunks: their movers (--pattern; ABABAABA: 5 of 8 on mover A,
# Kohaku-TT's decode kernel's share) and the K chunks mover A's half holds (B's: 2)
SPLIT_PATTERN = "AB"
SPLIT_DEPTH = {"AB": 2, "AAAB": 3, "ABABAABA": 3}
# input dim sharded: the addition across in the kernels (kernel) or the all-reduce of Y
# after the product (reduce) (--add)
ADD = "kernel"
# output dim sharded into a copy of Y: the kernels' all-gather at the call's end
# (kernel) or the host's after it (host) (--gather)
GATHER = "kernel"
# the grid plan: W's read spread over the column broadcast's ranks (--spread), and X's
# over the row's (--xspread): "auto" where the plan's model (k_block_ns) says it wins
SPREAD = "auto"
X_SPREAD = "auto"
# the island grid's panels (first, last column; layout.panels): a row broadcast lies in
# one, so an X block is shared by the ranks of the row's part in a panel
PANELS = None
# The account's words (kernels/linear/linear.h kAcWFree ..), by name: cycles, but the
# tiles and products (counts) and the ends and start (wall clocks)
ACCOUNT_WORDS = (
    ("w_free", "kAcWFree"),
    ("w_read", "kAcWRead"),
    ("w_cast", "kAcWCast"),
    ("w_tiles", "kAcWTiles"),
    ("y_wait", "kAcYWait"),
    ("y_write", "kAcYWrite"),
    ("x_free", "kAcXFree"),
    ("x_read", "kAcXRead"),
    ("x_cast", "kAcXCast"),
    ("input_wait", "kAcInputWait"),
    ("math", "kAcMath"),
    ("pack", "kAcPack"),
    ("products", "kAcProducts"),
)
# the K-split plan's: the movers' chunks, the reduce, Y written (cycles from the last)
SPLIT_ACCOUNT_WORDS = (
    ("move_a", "kAcKMoveA"),
    ("move_b", "kAcKMoveB"),
    ("reduce", "kAcKReduce"),
    ("y_write", "kAcKWrite"),
    ("reader_all", "kAcKReaderReduce"),
)
# and its wall clocks: compute's own part packed, each round's part taken and added;
# the writer's part to send in hand
SPLIT_ACCOUNT_CLOCKS = (("own", L["kAcKOwn"]), ("held", L["kAcKHeld"])) + tuple(
    (f"{name}{step}", L[word] + step)
    for name, word in (("taken", "kAcKTaken"), ("added", "kAcKAdded"))
    for step in range(L["kKRounds"])
)
ACCOUNT_CLOCKS = (
    ("reader_end", "kAcReaderEnd"),
    ("compute_start", "kAcComputeStart"),
    ("compute_end", "kAcComputeEnd"),
)
# Placements by hint (X, W, Y): "rows" (W gathered: X, Y split along M); "out" (W, Y
# split along N, X a copy); "in" (X, W along K, Y partials).
PLACES = {
    "rows": ({"split": 0}, {"copy": True}, {"split": 0}),
    "out": ({"copy": True}, {"split": 1}, {"split": 1}),
    "in": ({"split": 1}, {"split": 0}, {"copy": True}),
}


def even_group(graph):
    """The hint's group that gives each island half of a split dimension: the last
    rank of island 0 and the first of island 1 (None on one island)."""
    islands = len(graph.islands)
    if islands == 1:
        return None
    if islands != 2:
        raise SystemExit(f"the halves' group is for two islands, not {islands}")
    return (graph.first(1) - 1, 2)


def spans(placement, outer_split):
    """Each island's (first, count) of the split dimension `outer_split` of a [rows,
    columns] tile tensor, as the kernels' k_part() and columns() find it."""
    out = []
    for first, steps in placement.parts:
        if placement.copy or placement.pages == steps * placement.inner * (
            placement.outer
        ):
            out.append((0, placement.steps))
        elif steps and placement.split == outer_split:
            out.append((first, steps))
        else:
            out.append((0, 0))
    return out


def k_spans(placement, k_tiles):
    """Each island's K tiles of W ([K, N] tiles), as the kernels' k_part() finds them:
    a copy holds all of K, a split along N all of K of its columns."""
    if placement.copy:
        return [(0, k_tiles)] * len(placement.parts)
    if placement.split == 1:
        return [(0, k_tiles if steps else 0) for _, steps in spans(placement, 1)]
    return spans(placement, 0)


def w_column_spans(placement, n_tiles):
    """Each island's columns of W, as the kernels' w_columns() finds them: its own of a
    W split along N, else all."""
    if not placement.copy and placement.split == 1:
        return spans(placement, 1)
    return [(0, n_tiles)] * len(placement.parts)


def made_spans(placement, m_tiles):
    """Each island's rows of the product (first, count), as the kernels' made_rows()
    finds them from X ([M, K] tiles): its rows when X is split along M, else all."""
    if placement.split == 0 and not placement.copy:
        return spans(placement, 0)
    return [
        (0, m_tiles if placement.part_pages(island) else 0)
        for island in range(len(placement.parts))
    ]


def kept_spans(placement, m_tiles, n_tiles):
    """Each island's ((first row, rows), (first column, columns)) of Y as the
    kernels' kept_rows() and columns() find them."""
    if placement.copy or placement.split is None:
        return [((0, m_tiles), (0, n_tiles))] * len(placement.parts)
    if placement.split == 0:
        return [
            ((first, steps), (0, n_tiles if steps else 0))
            for first, steps in spans(placement, 0)
        ]
    return [
        ((0, m_tiles if steps else 0), (first, steps))
        for first, steps in spans(placement, 1)
    ]


def extra_buffers(plan):
    """The linear kernels' buffers past the grid plan's: a partial's subblocks added
    (LINEAR_ADD), the token and its slot, T's row."""
    subblock = plan.sh * plan.sw
    return (
        [
            (L["kCbSent"], "bfloat16", L["kBf16Bytes"], 2 * subblock),
            (L["kCbToken"], "uint32", L["kSignalBytes"], 1),
            (L["kCbArrive"], "uint32", L["kSignalBytes"], 1),
            (L["kCbStamp"], "uint32", L["kStampBytes"], 1),
        ]
        + (
            # a receiver's K rows of a W block (W's read spread over the column)
            [(L["kCbWSlice"], "bfloat16", L["kBf16Bytes"], slice_pages(plan))]
            if spreads(plan)
            else []
        )
        + (
            # a receiver's tiles of an X block (X's read spread over the row)
            [(L["kCbXSlice"], "bfloat16", L["kBf16Bytes"], x_slice_pages(plan))]
            if x_spreads(plan)
            else []
        )
    )


def spreads(plan):
    """Whether W's read is spread over the column's ranks (the plan's, by
    with_spreads)."""
    return plan.w_spread


def spread_choices(mode, allowed):
    """The settings of one spread the planner tries: --spread / --xspread `mode`, and
    whether the plan has ranks to share it."""
    if not allowed or mode == "off":
        return (False,)
    return (True,) if mode == "on" else (False, True)


def with_spreads(plans, budget):
    """`plans` (one program's pieces) with the spreads the model (plan_ns, the slowest
    piece) prices lowest with every piece inside `budget`, the same in every piece (the
    program's buffers and defines are one); no spread on a tie, None when none fits. A
    spread needs the 2d layout, a broadcast of more than one rank and tiles to share."""
    two = all(plan.layout == "2d" for plan in plans)
    w_ok = two and all(plan.rows > 1 and plan.kb > 1 for plan in plans)
    x_ok = two and all(plan.cols > 1 and plan.bm * plan.kb > 1 for plan in plans)
    best = None
    for w_spread in spread_choices(SPREAD, w_ok):
        for x_spread in spread_choices(X_SPREAD, x_ok):
            options = [
                dataclasses.replace(plan, w_spread=w_spread, x_spread=x_spread)
                for plan in plans
            ]
            if any(o.l1_bytes() + extra_bytes(o) > budget for o in options):
                continue
            cost = max(plan_ns(option) for option in options)
            if best is None or cost < best[0]:
                best = (cost, options)
    return None if best is None else best[1]


def slice_pages(plan):
    """A column rank's K rows of a W block, the most: K block / ranks, bn wide."""
    return pl.cdiv(plan.kb, plan.rows) * plan.bn


def x_spreads(plan):
    """Whether X's read is spread over the row's ranks (the plan's, by
    with_spreads)."""
    return plan.x_spread


def x_slice_pages(plan):
    """A row rank's tiles of an X block, the most: bm x K block / the ranks of the
    fewest a row broadcast has (a rect from the grid's first column, cut by PANELS)."""
    return pl.cdiv(plan.bm * plan.kb, row_ranks_least(plan))


def row_ranks_least(plan):
    if PANELS is None:
        return plan.cols
    parts = [
        min(last, plan.cols - 1) - first + 1
        for first, last in PANELS
        if first < plan.cols
    ]
    return min(parts)


def extra_bytes(plan):
    return sum(page * pages for _, _, page, pages in extra_buffers(plan))


def l1_budget(session, islands, extra=0):
    """L1 for the buffers: the L1 the session gives buffers (Session.l1) less the
    other rows of the load's budget (on more islands also what the session keeps, with
    `extra` tensors more than it holds now, and the two tokens it counts with the
    buffers), at most the project's budget."""
    free = session.sessions[0].l1()[1]
    if islands == 1:  # the host transfer buffer, control block and a slot
        return min(pl.L1_BUDGET, free - L1_TRANSFER - L1_RUNTIME)
    tokens = 2 * abi.headers()["kTokenBytes"]
    held = session.held_bytes(extra=extra)
    return min(pl.L1_BUDGET, free - held - L1_TRANSFER - L1_RUNTIME_ISLANDS - tokens)


# The K-split plan (LINEAR_SPLIT) for an island's part of at most SPLIT_ROWS row tiles
# (a decode step: one tile row is up to 32 tokens): every rank reads W, both movers.
SPLIT_ROWS = 2
# the K-split model: the card's DRAM ceiling (ceilings.py, card 0, 2026-11-15) and a
# reduce's rate into a rank (an assumed 20 GB/s a step of the tree)
DRAM_BYTES_PER_S = 495.6e9
REDUCE_BYTES_PER_S = 20e9
# W's pages a K chunk at most: every mover's reads in small batches, one barrier each,
# as the ceiling's (deep batches leave the ranks far from the banks behind)
SPLIT_CHUNK_PAGES = 16
# products each followed by the all-reduce, timed one at a time (input dim sharded)
REDUCED_CALLS = 20


def split_buffers(plan, adds=False):
    """(index, format, page bytes, pages) of the K-split plan's buffers (mover A's
    chunks the plan's depth, mover B's two; `adds`: both partials of a block), T's
    slot last. The whole-K plan: X whole in kCbKX, W's halves both the plan's depth."""
    block_tiles = plan.oh * plan.ow
    x_chunk, w_chunk = plan.oh * plan.kc, plan.kc * plan.ow
    both = (
        [(L["kCbKAdd"], "bfloat16", L["kBf16Bytes"], 2 * block_tiles)] if adds else []
    )
    if plan.cast:  # X's halves unused: a page each
        x_whole = pl.cdiv(plan.kt, plan.kc) * x_chunk
        both += [(L["kCbKX"], "bfloat16", L["kBf16Bytes"], x_whole)]
    x_halves = (1, 1) if plan.cast else (plan.depth * x_chunk, 2 * x_chunk)
    w_b = (plan.depth if plan.cast else 2) * w_chunk  # whole chunks (km::move)
    return both + [
        (L["kCbKa"], "bfloat16", L["kBf16Bytes"], x_halves[0]),
        (L["kCbKaB"], "bfloat16", L["kBf16Bytes"], x_halves[1]),
        (L["kCbKw"], "bfloat16", L["kBf16Bytes"], plan.depth * w_chunk),
        (L["kCbKwB"], "bfloat16", L["kBf16Bytes"], w_b),
        (L["kCbKArrive"], "float32", L["kPartBytes"], plan.arrive_pages()),
        (L["kCbKSum"], "float32", L["kPartBytes"], block_tiles),
        (L["kCbKOut"], "bfloat16", L["kBf16Bytes"], block_tiles),
        (L["kCbStamp"], "uint32", L["kStampBytes"], 1),
    ]


def pattern_word(letters):
    """A km::Pattern word: the length, and from bit 8 a bit set where mover B moves."""
    owners = sum(1 << at for at, letter in enumerate(letters) if letter == "B")
    return len(letters) | owners << 8


def split_ns(plan, ranks):
    """The K-split model's ns: a rank's W at its share of the card's DRAM rate, then
    the tree's steps of one block each."""
    w_bytes = pl.cdiv(plan.kt, plan.s) * plan.ow * L["kBf16Bytes"]
    steps = max(1, plan.s - 1).bit_length()
    reduce = steps * plan.oh * plan.ow * L["kPartBytes"] / REDUCE_BYTES_PER_S
    return 1e9 * (w_bytes * ranks / DRAM_BYTES_PER_S + reduce)


def split_choice(m_tiles, k_tiles, n_tiles, ranks, budget, adds=False, least_k=None):
    """The K-split plan of the island's part with the least split_ns on `ranks` ranks
    inside `budget`: one block height (all the rows), every block width, the largest
    group the ranks allow with K for each rank (`least_k`: the least island's K part,
    default k_tiles); K chunks of at most SPLIT_CHUNK_PAGES pages of W."""
    best = None
    # at least 8 columns where N has them: a K row of a chunk on every bank
    for width in range(min(8, n_tiles), n_tiles + 1):
        groups = pl.cdiv(n_tiles, width)
        size = min(ranks // groups, k_tiles if least_k is None else least_k)
        if size < 1:
            continue
        chunk = max(1, min(pl.cdiv(k_tiles, size), SPLIT_CHUNK_PAGES // width))
        sh, sw = pl.subblock(m_tiles, width)
        plan = pl.Split(
            m_tiles,
            k_tiles,
            n_tiles,
            size,
            m_tiles,
            width,
            chunk,
            sh,
            sw,
            pattern=SPLIT_PATTERN,
            depth=SPLIT_DEPTH[SPLIT_PATTERN],
        )
        buffers = split_buffers(plan, adds)
        if sum(page * pages for _, _, page, pages in buffers) > budget:
            continue
        cost = (split_ns(plan, ranks), groups)
        if best is None or cost < best[0]:
            best = (cost, plan)
    return None if best is None else best[1]


# The whole-K plan (S 1, kKsXCast; Kohaku-TT's form): a core all of K for its columns,
# X cast a panel by its first core's writer; a receiver's W on both movers.
CAST_X_TILES = 32  # X's tiles a multicast (Kohaku-TT's)
CAST_W_PAGES = (32, 16, 8)  # W's pages a chunk (32: Kohaku-TT's 64 KB blocks)
CAST_DEPTHS = (4, 3, 2)  # W's chunks a half: the most bytes in flight that fit
# a receiver's W chunks, their movers: 5 of 8 on the writer (NoC 0), Kohaku-TT's 5:3
CAST_PATTERN = "BABABBAB"
CAST_PATTERNS = ("BABABBAB", "AB", "ABABAABA")
# the reader's and the writer's NoC (--cast-nocs): X cast on NoC 0, the sender's W on
# NoC 1, as Kohaku-TT's (0,1: X on NoC 1, slower by the account)
CAST_NOCS = (1, 0)
# The tails (ns): a tree hop (fixed, fp32 at 80 GB/s, a tile's combine), the root's pack
# and Y; the whole-K plan's first cast and its tail past the last core's W.
HOP_NS = 1_350
HOP_BYTES_PER_S = 80e9
COMBINE_TILE_NS = 100
ROOT_NS = 1_000
ROOT_TILE_NS = 170
CAST_FIRST_NS = 9_000
# The reads' rates fitted to the 16 decode loops (README): the split's W and X again;
# the whole-K plan's W by (islands, ow 1 or wider), at most a core's rate each core.
SPLIT_READ_BYTES_PER_S = {1: 470e9, 2: 470e9}
CAST_READ_BYTES_PER_S = {(1, 1): 400e9, (1, 2): 465e9, (2, 1): 390e9, (2, 2): 415e9}
CAST_CORE_BYTES_PER_S = 7.2e9
# the decode plan (--decode): "priced" the cheaper of the two by their terms, or one
DECODE = "priced"


def root_ns(plan):
    """The root's term: its sum packed to bf16 and written to Y."""
    return ROOT_NS + plan.oh * plan.ow * ROOT_TILE_NS


def split_terms_ns(plan, islands=1):
    """The K-split plan's time by its terms: every rank's W and its group's X again at
    the rate measured, the tree's rounds, the root."""
    tiles = pl.cdiv(plan.kt, plan.s) * (plan.ow + plan.oh)
    read = tiles * L["kBf16Bytes"] * plan.cores() / SPLIT_READ_BYTES_PER_S[islands]
    rounds = max(1, plan.s - 1).bit_length() if plan.s > 1 else 0
    part = plan.oh * plan.ow
    hop = (
        HOP_NS + 1e9 * part * L["kPartBytes"] / HOP_BYTES_PER_S + part * COMBINE_TILE_NS
    )
    return 1e9 * read + rounds * hop + root_ns(plan)


def cast_ns(plan, islands=1):
    """The whole-K plan's time by its terms: W at the rate measured for its width (a
    core's at most on each), the first multicast and the tail, the sum packed and
    written."""
    rate = min(
        CAST_READ_BYTES_PER_S[islands, min(plan.ow, 2)],
        plan.cores() * CAST_CORE_BYTES_PER_S,
    )
    read = plan.kt * plan.ow * L["kBf16Bytes"] * plan.cores() / rate
    return 1e9 * read + CAST_FIRST_NS + root_ns(plan)


def cast_rects(nocs, used, ranks, noc=1):
    """The whole-K plan's multicasts over the first `used` cores, a panel's columns
    to its last used row each: (sender, its cores, its used cores, start, end: the
    translated corners as x | y << 16, high first on NoC 1, low first on NoC 0); None
    where a rect holds a core of no rank or is no NoC0 box with its corners the box's.
    """
    columns, rows = nocs.grid
    rects = []
    for first, last in nocs.panels:
        cores = [y * columns + x for y in range(rows) for x in range(first, last + 1)]
        mine = [core for core in cores if core < used]
        if not mine:
            continue
        bottom = max(core // columns for core in mine)
        cores = [core for core in cores if core // columns <= bottom]
        if max(cores) >= ranks:
            return None
        places = [(core % columns, core // columns) for core in cores]
        points = {nocs.noc0[place] for place in places}
        xs, ys = [x for x, _ in points], [y for _, y in points]
        box = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
        if len(points) != len(places) or box != len(places):
            return None
        noc0 = {nocs.noc_xy[place]: nocs.noc0[place] for place in places}
        low = (min(x for x, _ in noc0), min(y for _, y in noc0))
        high = (max(x for x, _ in noc0), max(y for _, y in noc0))
        if (noc0.get(low), noc0.get(high)) != ((min(xs), min(ys)), (max(xs), max(ys))):
            return None
        first, last = (high, low) if noc == 1 else (low, high)
        start, end = first[0] | first[1] << 16, last[0] | last[1] << 16
        rects.append((cores[0], cores, mine, start, end))
    return rects


def cast_choice(m_tiles, k_tiles, n_tiles, ranks, budget, nocs, adds=False):
    """The whole-K plan on the most of the first `ranks` ranks inside `budget` (the
    narrowest block, then the most W bytes in flight), its multicasts; or
    None where none fits (X whole and W's halves in L1, the rects)."""
    best = None
    for width in range(1, n_tiles + 1):
        used = pl.cdiv(n_tiles, width)
        if used > ranks:
            continue
        rects = cast_rects(nocs, used, ranks, CAST_NOCS[1])
        if rects is None:
            continue
        sh, sw = pl.subblock(m_tiles, width)
        for pages in CAST_W_PAGES:
            chunk = max(1, min(k_tiles, pages // width))
            for depth in CAST_DEPTHS:
                plan = pl.Split(
                    m_tiles,
                    k_tiles,
                    n_tiles,
                    1,
                    m_tiles,
                    width,
                    chunk,
                    sh,
                    sw,
                    pattern=CAST_PATTERN,
                    depth=depth,
                    cast=True,
                )
                buffers = split_buffers(plan, adds)
                if sum(page * count for _, _, page, count in buffers) > budget:
                    continue
                cost = (width, -depth * chunk, -chunk)
                if best is None or cost < best[0]:
                    best = (cost, plan, rects)
    return None if best is None else best[1:]


def fitting_plan(m_tiles, k_tiles, n_tiles, grid, ranks, budget):
    """The grid plan's rule on the island's grid with the extra buffers inside
    `budget`, with fewer rows of cores until its rect lies inside the island's ranks
    (link cores take the last cores)."""
    columns, rows = grid
    for used_rows in range(rows, 0, -1):
        room = budget
        while room > 0:
            plan = pl.grid_plan(m_tiles, k_tiles, n_tiles, (columns, used_rows), room)
            if plan is None:
                break
            over = plan.l1_bytes() + extra_bytes(plan) - budget
            if over <= 0:
                if plan.group(grid)[1] <= ranks:
                    return with_spreads([plan], budget)[0]
                break
            room -= over
    return None


@dataclasses.dataclass(frozen=True)
class Piece:
    """A rectangle of the island's cores from core column x0, row y0, making the
    window's tiles from row m0 and column n0 on in `plan`'s bands (2d: whole blocks)."""

    plan: pl.Grid
    x0: int = 0
    y0: int = 0
    m0: int = 0
    n0: int = 0

    @property
    def layout(self):
        return self.plan.layout

    def rect(self, grid):
        x, y, columns, rows = self.plan.rect(grid)
        return self.x0 + x, self.y0 + y, columns, rows

    def places(self, grid):
        """{core index: (first row tile, first column tile)} of the piece's cores."""
        shift = self.y0 * grid[0] + self.x0
        return {
            core + shift: (row + self.m0, column + self.n0)
            for core, (row, column) in self.plan.places(grid).items()
        }


# A K block's ns on a core: its products and subblock turns, or its slower feed (a fixed
# cost, then tiles read and multicast), fitted to the one-card calls (see the README).
PRODUCT_NS = 24.5
SUBBLOCK_NS = 51
FEED_BLOCK_NS = 4_750
FEED_TILE_NS = 203
# A spread block's feed: the ranks' tiles into the sender, then its cast; fitted to the
# rows a spread bounds, W's (13 rows) and X's (5; the row's hand-over is the slower)
SPREAD_BLOCK_NS = {"w": 3_250, "x": 0}
SPREAD_TILE_NS = {"w": 116, "x": 180}


def k_block_ns(bm, bn, kb, x_spread=False, w_spread=False):
    """The larger of a K block's compute and its two feeds (X's bm x kb tiles, W's kb x
    bn), each read by its sender alone or spread."""
    sh, sw = pl.subblock(bm, bn)
    compute = bm * bn * kb * PRODUCT_NS + bm * bn // (sh * sw) * SUBBLOCK_NS
    feeds = [
        (
            SPREAD_BLOCK_NS[name] + tiles * SPREAD_TILE_NS[name]
            if spread
            else FEED_BLOCK_NS + tiles * FEED_TILE_NS
        )
        for name, tiles, spread in (("x", bm * kb, x_spread), ("w", kb * bn, w_spread))
    ]
    return max(compute, *feeds)


def plan_ns(plan):
    """A core's time for its blocks of `plan` by k_block_ns."""
    return (
        plan.nbh
        * plan.nbw
        * plan.nk
        * k_block_ns(plan.bm, plan.bn, plan.kb, plan.x_spread, plan.w_spread)
    )


def piece_plan(m_tiles, k_tiles, n_tiles, rect_size, block):
    """A 2d Grid of `rect_size` (columns, rows) cores for m x n tiles in bands of whole
    blocks `block` (bm, bn, kb, nbh, nbw), or None when its rect does not fit."""
    bm, bn, kb, nbh, nbw = block
    if m_tiles <= 0 or n_tiles <= 0:
        return None
    pcm, pcn = nbh * bm, nbw * bn
    plan = pl.Grid(
        m_tiles,
        k_tiles,
        n_tiles,
        pl.cdiv(m_tiles, pcm),
        pl.cdiv(n_tiles, pcn),
        pcm,
        pcn,
        bm,
        bn,
        kb,
        *pl.subblock(bm, bn),
    )
    try:
        plan.check(rect_size, budget=1 << 40)  # L1: the caller's
    except ValueError:
        return None
    return plan


def two_rects(grid, ranks):
    """The island's ranks as rectangles: the whole grid where every core is a rank
    {"whole": (A, None)}; two when its link cores take the end of the last row:
    {"beside": (A, B), "below": (A, B)}, each (x0, y0, columns, rows)."""
    columns, rows = grid
    links = columns * rows - ranks
    if links == 0:
        return {"whole": ((0, 0, columns, rows), None)}
    if not 0 < links < columns:
        return {}
    return {
        "beside": (
            (0, 0, columns - links, rows),
            (columns - links, 0, links, rows - 1),
        ),
        "below": ((0, 0, columns, rows - 1), (0, rows - 1, columns - links, 1)),
    }


def island_pieces(m_tiles, k_tiles, n_tiles, grid, ranks, budget, kb):
    """The pieces of the island's ranks: the old single rect (fitting_plan), or the
    whole grid, or two rectangles sharing one block, in bands of whole blocks: the
    cheapest by plan_ns (then the larger block, the larger K block), any K block up to
    `kb`."""
    single = fitting_plan(m_tiles, k_tiles, n_tiles, grid, ranks, budget)
    best = None
    if single is not None:
        cost = plan_ns(single)
        best = ((cost, -single.bm * single.bn, -single.kb), [Piece(single)])
    rects = two_rects(grid, ranks)
    if not rects or single is None or single.layout != "2d":
        return None if best is None else best[1]
    for k_block in range(kb, 0, -1):
        for bm in range(1, m_tiles + 1):
            for bn in range(1, n_tiles + 1):
                sh, sw = pl.subblock(bm, bn)
                if sh * sw < 4 or bm * bn * G["kPartBytes"] > budget:
                    continue
                probe = piece_plan(bm, k_tiles, bn, (1, 1), (bm, bn, k_block, 1, 1))
                if probe is None:
                    continue
                if probe.l1_bytes() + extra_bytes(probe) > budget:
                    continue
                for form, (a, b) in rects.items():
                    for pieces in split_pieces(
                        form, a, b, m_tiles, k_tiles, n_tiles, (bm, bn, k_block)
                    ):
                        # the pieces' own buffers, not the probe's
                        plans = with_spreads([p.plan for p in pieces], budget)
                        if plans is None:
                            continue
                        pieces = [
                            dataclasses.replace(p, plan=plan)
                            for p, plan in zip(pieces, plans, strict=True)
                        ]
                        cost = (
                            max(plan_ns(p.plan) for p in pieces),
                            -bm * bn,
                            -k_block,
                        )
                        if cost < best[0]:
                            best = (cost, pieces)
    return best[1]


def split_pieces(form, a, b, m_tiles, k_tiles, n_tiles, block):
    """Each way rects a and b share the window: "beside" a takes the first columns,
    b the rest; "below" a the first rows. Bands of whole blocks a rect."""
    bm, bn, kb = block
    if form == "whole":
        nbh = pl.cdiv(pl.cdiv(m_tiles, a[3]), bm)
        nbw = pl.cdiv(pl.cdiv(n_tiles, a[2]), bn)
        plan = piece_plan(m_tiles, k_tiles, n_tiles, a[2:], (bm, bn, kb, nbh, nbw))
        if plan is not None:
            yield [Piece(plan, a[0], a[1])]
        return
    if form == "beside":
        nbh_a = pl.cdiv(pl.cdiv(m_tiles, a[3]), bm)
        nbh_b = pl.cdiv(pl.cdiv(m_tiles, b[3]), bm)
        for nbw_a in range(1, pl.cdiv(n_tiles, a[2] * bn) + 1):
            n_a = min(n_tiles, a[2] * nbw_a * bn)
            plan_a = piece_plan(
                m_tiles, k_tiles, n_a, a[2:], (bm, bn, kb, nbh_a, nbw_a)
            )
            if plan_a is None:
                continue
            n_b = n_tiles - n_a
            if n_b == 0:
                yield [Piece(plan_a, a[0], a[1])]
                continue
            nbw_b = pl.cdiv(pl.cdiv(n_b, b[2]), bn)
            plan_b = piece_plan(
                m_tiles, k_tiles, n_b, b[2:], (bm, bn, kb, nbh_b, nbw_b)
            )
            if plan_b is not None:
                yield [Piece(plan_a, a[0], a[1]), Piece(plan_b, b[0], b[1], 0, n_a)]
        return
    nbw_a = pl.cdiv(pl.cdiv(n_tiles, a[2]), bn)
    nbw_b = pl.cdiv(pl.cdiv(n_tiles, b[2]), bn)
    for nbh_a in range(1, pl.cdiv(m_tiles, a[3] * bm) + 1):
        m_a = min(m_tiles, a[3] * nbh_a * bm)
        plan_a = piece_plan(m_a, k_tiles, n_tiles, a[2:], (bm, bn, kb, nbh_a, nbw_a))
        if plan_a is None:
            continue
        m_b = m_tiles - m_a
        if m_b == 0:
            yield [Piece(plan_a, a[0], a[1])]
            continue
        nbh_b = pl.cdiv(m_b, bm)
        plan_b = piece_plan(m_b, k_tiles, n_tiles, b[2:], (bm, bn, kb, nbh_b, nbw_b))
        if plan_b is not None:
            yield [Piece(plan_a, a[0], a[1]), Piece(plan_b, b[0], b[1], m_a, 0)]


def sent_pages(pieces, grid):
    """S's pages a set (LINEAR_ADD: an island's partial): the output blocks of every
    piece's cores, in core order."""
    return sum(
        len(piece.places(grid))
        * piece.plan.nbh
        * piece.plan.nbw
        * piece.plan.bm
        * piece.plan.bn
        for piece in pieces
    )


def pieces_args(pieces, nocs, ranks, m_tiles, group_word, movers=(0, 1)):
    """Each core's argument function: its piece's broadcast words (row on movers[0],
    the reader's NoC, column on movers[1], the writer's; `group_word` the command's),
    then its core words (first row and column, first page of a set of S, a set's pages,
    plan words with M whole); compute: core words, then the broadcast windows."""
    group = layout.Group(0, ranks)
    set_pages = sent_pages(pieces, nocs.grid)
    words, first = {}, 0
    for piece in pieces:
        plan = piece.plan
        casts = [
            card.grid_broadcasts(
                piece,
                nocs,
                group,
                along,
                noc,
                (G[broadcast], G[cb], window, pages),
            )
            for along, noc, broadcast, cb, window, pages in (
                (
                    "row",
                    movers[0],
                    "kBroadcastRow",
                    "kCbRow",
                    plan.wa,
                    plan.bm * plan.kb,
                ),
                (
                    "column",
                    movers[1],
                    "kBroadcastCol",
                    "kCbCol",
                    plan.ww,
                    plan.kb * plan.bn,
                ),
            )
        ]
        plan_words = [m_tiles, *plan.words()[1:]]
        for core, (row, column) in sorted(piece.places(nocs.grid).items()):
            mine = [1, row, column, first, set_pages, *plan_words]
            words[core] = (
                [
                    [
                        (cast[core] | {"group": group_word})[field]
                        for field in card.BROADCAST_FIELDS
                    ]
                    for cast in casts
                ],
                mine,
                (plan.wa, plan.ww),
            )
            first += plan.nbh * plan.nbw * plan.bm * plan.bn
    windows = (pieces[0].plan.wa, pieces[0].plan.ww)
    # a core of no piece: no broadcast, its rank word 0, and M
    idle_core = [0] * L["kCoreWords"]
    idle_core[L["kCorePlan"] + L["kPlMt"]] = m_tiles
    idle = [0] * len(card.BROADCAST_FIELDS) + idle_core

    def args(core):
        if core not in words:
            return idle, idle, idle_core + list(windows)
        (row, column), mine, window = words[core]
        return row + mine, column + mine, mine + list(window)

    return args


class Linear:
    """The linear kernels loaded on every island of `session`: one LINEAR command a
    call on all its ranks, exclusive (the broadcasts inside each island)."""

    def __init__(self, session, dimensions, tensors, wait=2.0, deadline=5.0):
        graph = session.graph
        m_tiles, k_tiles, n_tiles = (value // pl.TILE for value in dimensions)
        self.k_tiles, self.m_tiles = k_tiles, m_tiles
        counts = {graph.count(i) for i in range(len(graph.islands))}
        if len(counts) != 1:
            raise SystemExit(f"one program needs islands of one size: {counts}")
        ranks = counts.pop()
        self.islands = len(graph.islands)
        self.nocs = session.sessions[0].nocs()
        global PANELS
        PANELS = list(self.nocs.panels)
        self.n_tiles = n_tiles
        self.k_parts = k_spans(tensors["w"].placement, k_tiles)
        self.w_columns = w_column_spans(tensors["w"].placement, n_tiles)
        # Y a copy on several islands: each island's partial (W split along K), or,
        # with W split along N, each island's own columns written into its copy
        w_place = tensors["w"].placement
        self.y_copy = self.islands > 1 and tensors["y"].placement.copy
        self.tensors_y_copy = self.y_copy and (w_place.copy or w_place.split != 1)
        # the one addition across made by the kernels (LINEAR_ADD), two islands
        self.adds = self.tensors_y_copy and ADD == "kernel"
        # the islands' columns of a Y copy gathered by the kernels (LINEAR_GATHER)
        self.gathers = self.y_copy and not self.tensors_y_copy and GATHER == "kernel"
        if self.adds and self.islands != 2:
            raise SystemExit(
                f"the kernels' addition is for two islands: {self.islands}"
            )
        self.made = made_spans(tensors["x"].placement, m_tiles)
        self.kept = kept_spans(tensors["y"].placement, m_tiles, n_tiles)
        if not all(count for _, count in self.k_parts):
            raise SystemExit(f"an island holds no K of W: {self.k_parts}")
        part_m = max(count for _, count in self.made)
        part_k = max(count for _, count in self.k_parts)
        part_n = max(
            self.window(here, to)[1]
            for here in range(self.islands)
            for to in range(self.islands)
        )
        if any(  # an island's columns kept by another (W split along K, Y along N)
            self.window(here, to)[1]
            for here in range(self.islands)
            for to in range(self.islands)
            if to != here
        ):
            raise SystemExit(
                "W split along K takes Y a copy (each island's partial of all of Y)"
            )
        # S and T; R, the all-reduce's scratch, when Y is each island's partial and
        # the all-reduce adds
        reduces = self.tensors_y_copy and not self.adds
        self.budget = l1_budget(session, self.islands, extra=2 + reduces)
        grid, planned = self.nocs.grid, ranks  # the grid and ranks planned on
        if COLUMNS is not None:
            if self.islands != 1 or not 0 < COLUMNS <= grid[0]:
                raise SystemExit(f"--columns {COLUMNS}: one island, 1 to {grid[0]}")
            grid, planned = (COLUMNS, grid[1]), COLUMNS * grid[1]
        # a few rows: the K-split plan
        self.split, self.casts, self.priced = None, None, {}
        if part_m <= SPLIT_ROWS and COLUMNS is None:
            # every rank of a group on every island gets K: the least island's part
            least_k = min((count for _, count in self.k_parts if count), default=1)
            self.split = split_choice(
                part_m, part_k, part_n, planned, self.budget, self.adds, least_k
            )
            whole = cast_choice(
                part_m, part_k, part_n, planned, self.budget, self.nocs, self.adds
            )
            # the cheaper by their terms (us), or the one --decode names
            if self.split is not None:
                self.priced["split"] = split_terms_ns(self.split, self.islands) / 1e3
            if whole is not None:
                self.priced["whole"] = cast_ns(whole[0], self.islands) / 1e3
            pick = DECODE
            if pick == "priced" and self.priced:
                pick = min(self.priced, key=self.priced.get)
            if pick == "whole":
                if whole is None:
                    raise SystemExit(f"no whole-K plan fits: {dimensions}")
                self.split, self.casts = whole
            log(f"decode plans priced (us) {self.priced}: {pick}")
        self.pieces = (
            []
            if self.split is not None
            else island_pieces(
                part_m,
                part_k,
                part_n,
                grid,
                planned,
                self.budget,
                pl.K_BLOCKS[0],
            )
        )
        if self.split is None and self.pieces is None:
            raise SystemExit(f"no grid plan fits L1 and {ranks} ranks: {dimensions}")
        if self.gathers and (self.split is None or m_tiles != 1):
            raise SystemExit(f"the kernels' gather is the K-split plan's: {dimensions}")
        if SUBBLOCK is not None:  # the script's --subblock, in every piece
            sh, sw = SUBBLOCK
            if any(p.plan.bm % sh or p.plan.bn % sw for p in self.pieces):
                raise SystemExit(f"subblock {sh} x {sw} of {self.pieces[0].plan}")
            self.pieces = [
                dataclasses.replace(p, plan=dataclasses.replace(p.plan, sh=sh, sw=sw))
                for p in self.pieces
            ]
        # the block, K block and buffers of all
        self.plan = self.split or self.pieces[0].plan
        for name in ("kCbRow", "kCbCol", "kCbPart"):
            assert L[name] == G[name], name
        assert L["kCbYOut"] == G["kCbGridOut"]
        pages = 1  # S: no partial added
        if self.adds:  # a set an island a command parity: each island's partial
            pages = 2 * self.islands * self.set_pages()
        tensors["s"] = session.empty(
            (pages * pl.TILE, pl.TILE), "bfloat16", "tile", copy=True
        )
        tensors["t"] = session.empty(
            (ranks, L["kStampBytes"] // 4), "uint32", "row", copy=True
        )
        if reduces:  # the all-reduce's scratch: twice Y's pages
            m_rows, _, n_columns = dimensions
            tensors["r"] = session.empty(
                (2 * m_rows, n_columns), "bfloat16", "tile", copy=True
            )
        self.tensors = tensors
        self.session = session
        self.group = (0, session.ranks)
        x, w, y, s = (tensors[name] for name in "xwys")
        if self.split is not None:
            self.program = self.split_program(tensors, part_n)
        else:
            self.program = self.grid_program(tensors, ranks, m_tiles)
        self.run = session.load(self.program, deadline=deadline, wait=wait)
        # what each island's session keeps in every core, by its own count (the
        # plan's budget took held_bytes before the load made it)
        self.l1_held = [
            sum(size for _, size in one.l1_held()) for one in session.sessions
        ]
        planned = session.held_bytes() if self.islands > 1 else 0
        if self.islands > 1 and max(self.l1_held) > planned:
            log(f"the session keeps {self.l1_held} B a core, the plan took {planned}")

    def split_program(self, tensors, part_n):
        """The linear program built for the K-split plan (LINEAR_SPLIT): its words on
        every CPU, X and W read on both movers, Y and T by the writer."""
        plan = self.split
        words = [0] * L["kKsWords"]
        cores = {}  # the whole-K plan: each core's multicast words
        for sender, rect, mine, start, end in self.casts or []:
            common = {"kKsXCast": 1, "kKsXStart": start, "kKsXEnd": end}
            common["kKsXChunks"] = max(1, CAST_X_TILES // (plan.oh * plan.kc))
            for core in rect:
                cores[core] = dict(common)
                if core in mine:
                    cores[core] |= {"kKsXRole": L["kKXReceiver"], "kKsXSender": sender}
            cores[sender] = common | {
                "kKsPattern": pattern_word("A"),  # its writer casts X
                "kKsXRole": L["kKXSender"],
                "kKsXWaits": len(mine) - 1,
                "kKsXDests": len(rect) - 1,
            }
        for name, value in (
            ("kKsS", plan.s),
            ("kKsOh", plan.oh),
            ("kKsOw", plan.ow),
            ("kKsKc", plan.kc),
            ("kKsSh", plan.sh),
            ("kKsSw", plan.sw),
            ("kKsNbw", pl.cdiv(part_n, plan.ow)),
            ("kKsArrive", plan.arrive_pages()),
            ("kKsForm", pl.REDUCE_FORMS[SPLIT_REDUCE]),
            ("kKsMt", plan.mt if self.islands == 1 else self.m_tiles),
            ("kKsSetPages", self.set_pages() if self.adds else 0),
            ("kKsPattern", pattern_word(plan.pattern)),
        ):
            words[L[name]] = value

        def mine(core):
            own = list(words)
            for name, value in cores.get(core, {}).items():
                own[L[name]] = value
            return own

        fp32 = "UnpackToDestFp32"
        x, w, y = (tensors[name] for name in "xwy")
        return km.Program(
            "dense_linear",
            kernels=KERNELS,
            buffers=[km.Buffer(*buffer) for buffer in split_buffers(plan, self.adds)],
            args=lambda core: (mine(core),) * 3,
            compile_args={
                "reader": x.tensor_args() + w.tensor_args(),
                "writer": y.tensor_args()
                + x.tensor_args()
                + w.tensor_args()
                + tensors["t"].accessor_args()
                + tensors["s"].tensor_args(),
                "compute": [],
            },
            compute={
                "math_fidelity": "HiFi2",
                "fp32_dest_acc_en": True,
                "unpack_to_dest_mode": {L["kCbKArrive"]: fp32, L["kCbKSum"]: fp32},
            },
            handlers=LINEAR_H,
            defines={"LINEAR_SPLIT": 1}
            | ({"LINEAR_ADD": 1} if self.adds else {})
            | ({"LINEAR_GATHER": 1} if self.gathers else {})
            | ({"LINEAR_ACCOUNT": 1} if ACCOUNT else {}),
            **(
                {"noc": dict(zip(("reader", "writer"), CAST_NOCS, strict=True))}
                if self.casts and CAST_NOCS != (0, 1)
                else {}
            ),
        )

    def grid_program(self, tensors, ranks, m_tiles):
        """The linear program of the grid plan: the pieces' broadcasts and blocks."""
        x, w, y, s = (tensors[name] for name in "xwys")
        return km.Program(
            "dense_linear",
            kernels=KERNELS,
            buffers=[
                km.Buffer(*buffer)
                for buffer in self.plan.buffers()[:4] + extra_buffers(self.plan)
            ],
            args=pieces_args(
                self.pieces,
                self.nocs,
                ranks,
                m_tiles,
                self.group[0] | self.group[1] << 16,
                NOCS,
            ),
            compile_args={
                "reader": x.accessor_args() + s.accessor_args(),
                "writer": w.accessor_args()
                + y.accessor_args()
                + tensors["t"].accessor_args()
                + s.accessor_args(),
                "compute": [],
            },
            compute={
                "math_fidelity": "HiFi2",
                "fp32_dest_acc_en": True,
                "unpack_to_dest_mode": {L["kCbPart"]: "UnpackToDestFp32"},
            },
            handlers=LINEAR_H,
            noc=dict(zip(("reader", "writer"), NOCS, strict=True)),
            defines=({"LINEAR_ADD": 1} if self.adds else {})
            | (
                {"LINEAR_SPREAD": 1}
                if any(spreads(p.plan) for p in self.pieces)
                else {}
            )
            | (
                {"LINEAR_XSPREAD": 1}
                if any(x_spreads(p.plan) for p in self.pieces)
                else {}
            )
            | ({"LINEAR_VALID_SENT": 1} if VALID_SENT else {})
            | ({"LINEAR_ACCOUNT": 1} if ACCOUNT else {}),
        )

    def set_pages(self):
        """S's pages a set (LINEAR_ADD): every core's output blocks of the grid plan;
        a block a group of the K split."""
        if self.split is None:
            return sent_pages(self.pieces, self.nocs.grid)
        plan = self.split
        return pl.cdiv(plan.mt, plan.oh) * pl.cdiv(plan.nt, plan.ow) * plan.oh * plan.ow

    def words(self):
        addresses = [self.tensors[name].address for name in "xwys"]
        return addresses + [self.k_tiles, self.n_tiles, self.tensors["t"].address]

    def passes(self, mhz=1350.0):
        """The last call's passes on each island from the writers' wall clocks (AICLK
        cycles, and microseconds at `mhz`): to the crossing (the far partial or the
        gather), the crossing, after it."""
        out = []
        for island, one in enumerate(self.session.sessions):
            (raw,) = one.read(self.tensors["t"].parts[island])
            words = (
                raw.view(np.uint32).reshape(-1, L["kStampBytes"] // 4).astype(np.int64)
            )
            ran = words[words[:, L["kStampEnd"]] != 0]
            account = {}
            names = SPLIT_ACCOUNT_WORDS if self.split is not None else ACCOUNT_WORDS
            clocks = SPLIT_ACCOUNT_CLOCKS if self.split is not None else ACCOUNT_CLOCKS
            clocks = [(name, L[w] if isinstance(w, str) else w) for name, w in clocks]
            names = [(name, L[w]) for name, w in names]
            if ACCOUNT and len(ran):  # the slowest core's phases, and each one's most
                span = (ran[:, L["kStampEnd"]] - ran[:, L["kStampStart"]]) % 2**32
                slow = int(np.argmax(span))
                start = ran[slow, L["kStampStart"]]
                account = {
                    "core": slow,
                    "span": int(span[slow]),
                    "slowest": {name: int(ran[slow, w]) for name, w in names}
                    | {name: int((ran[slow, w] - start) % 2**32) for name, w in clocks},
                    "most": {name: int(ran[:, w].max()) for name, w in names},
                    # every core's, in core order (the spread of each phase), and its
                    # clocks from its own start (-1: not stamped)
                    "cores": {name: ran[:, w].tolist() for name, w in names}
                    | {
                        name: np.where(
                            ran[:, w] == 0,
                            -1,
                            (ran[:, w] - ran[:, L["kStampStart"]]) % 2**32,
                        ).tolist()
                        for name, w in clocks
                    },
                }
            # the cores that stamped a crossing: the K split's first ranks (every rank
            # with the gather), the grid's ranks with blocks
            has = ran[:, L["kStampPut"]] != 0
            stamped = bool(has.any())
            # the 32-bit clock as signed cycles from the first core's start
            ran = (ran - ran[0, L["kStampStart"]] + 2**31) % 2**32 - 2**31
            start = ran[:, L["kStampStart"]].min()
            put = ran[has, L["kStampPut"]].max() if stamped else start
            synced = ran[has, L["kStampSynced"]].max() if put != start else start
            end = ran[:, L["kStampEnd"]].max()
            cycles = {"put": put - start, "sync": synced - put, "own": end - synced}
            # the cores' spread: the first to the last taking the command, ending
            cycles["span"] = end - start
            cycles["start_spread"] = ran[:, L["kStampStart"]].max() - start
            cycles["end_spread"] = end - ran[:, L["kStampEnd"]].min()
            if stamped:  # inside the crossing: the last core's steps, from its start
                at = {
                    name: ran[has, L[stamp]].max()
                    for name, stamp in (
                        ("settled", "kStampSettled"),
                        ("up", "kStampUp"),
                        ("down", "kStampDown"),
                    )
                }
                cycles |= {
                    "settle": at["settled"] - put,
                    "up": at["up"] - at["settled"],
                    "down": at["down"] - at["up"],
                    "go": synced - at["down"],
                }
                # each core's own wait for the crossing
                waits = ran[has, L["kStampSettled"]] - ran[has, L["kStampPut"]]
                cycles["settle_median"] = int(np.median(waits))
            out.append(
                {
                    "account": account,
                    "cores": len(ran),
                    "cycles": {key: int(value) for key, value in cycles.items()},
                    "microseconds_at_mhz": {
                        key: round(float(value) / mhz, 1)
                        for key, value in cycles.items()
                    },
                }
            )
        return out

    def calls(self, commands, timeout=60.0):
        """`commands` LINEAR commands back to back: (host seconds from the first
        submit to the last event, [commands, islands] microseconds of each part)."""
        words = np.tile(np.array(self.words(), np.int64), (commands, 1))
        started = time.perf_counter()
        handles = self.run.submit_many("linear", *self.group, words, exclusive=True)
        events = self.run.wait(list(handles), timeout=timeout)
        host_s = time.perf_counter() - started
        unfinished = [part for event in events for part in event if part.kind != "done"]
        if unfinished:
            raise km.KomorebiError(f"{len(unfinished)} parts not done: {unfinished[0]}")
        return host_s, np.array(
            [part.microseconds for event in events for part in event]
        ).reshape(commands, -1)

    def reduced(self, commands):
        """`commands` calls each followed by the all-reduce of Y's partials (the
        input dim sharded form, whole): per call the host's microseconds of the
        product, of the reduce, and each island's part of the reduce."""
        out = []
        for _ in range(commands):
            started = time.perf_counter()
            self.calls(1, timeout=10.0)
            between = time.perf_counter()
            events = self.run.all_reduce(self.tensors["y"], self.tensors["r"])
            ended = time.perf_counter()
            if any(part.kind != "done" for part in events):
                raise km.KomorebiError(f"the all-reduce not done: {events}")
            out.append(
                {
                    "product_us": round((between - started) * 1e6, 1),
                    "reduce_us": round((ended - between) * 1e6, 1),
                    "reduce_parts_us": [
                        round(float(part.microseconds), 1) for part in events
                    ],
                }
            )
        return out

    def gathered(self, commands):
        """`commands` calls each followed by the all-gather of Y (the output dim
        sharded form into a copy of Y): per call the host's microseconds of the
        product, of the gather, and each island's part of the gather."""
        out = []
        for _ in range(commands):
            started = time.perf_counter()
            self.calls(1, timeout=10.0)
            between = time.perf_counter()
            events = self.run.all_gather(self.tensors["y"])
            ended = time.perf_counter()
            if any(part.kind != "done" for part in events):
                raise km.KomorebiError(f"the all-gather not done: {events}")
            out.append(
                {
                    "product_us": round((between - started) * 1e6, 1),
                    "gather_us": round((ended - between) * 1e6, 1),
                    "gather_parts_us": [
                        round(float(part.microseconds), 1) for part in events
                    ],
                }
            )
        return out

    def single(self, commands):
        """`commands` calls one at a time: the slowest part's microseconds each; each
        call's host time, its parts' (an island's, by its dispatcher's ticks) and its
        passes in `single_detail`."""
        out, self.single_detail = [], []
        for _ in range(commands):
            host_s, parts = self.calls(1, timeout=10.0)
            out.append(parts.max(axis=1))
            self.single_detail.append(
                {
                    "host_us": round(host_s * 1e6, 1),
                    "parts_us": [round(float(value), 1) for value in parts[0]],
                    "passes": self.passes(),
                }
            )
        return np.concatenate(out)

    def window(self, here, to):
        """(first, count) of the columns island `to` keeps of the rows island `here`
        makes and holds W's columns of, as the kernels' window() finds them."""
        made, kept = self.made[here], self.kept[to]
        if to != here and self.y_copy:  # each island its partial, or its columns
            return (0, 0)
        if not made[1] or made[0] < kept[0][0] or sum(made) > sum(kept[0]):
            return (0, 0)
        held = self.w_columns[here]
        if kept[1][0] < held[0] or sum(kept[1]) > sum(held):
            return held if self.y_copy else (0, 0)
        return kept[1]

    def describe(self):
        """Each piece: its rect on the island's grid, first row and column, bands; the
        decode plans' prices and the whole-K plan's multicasts."""
        if self.split is not None:
            return {
                "priced_us": self.priced,
                "casts": [
                    {"sender": sender, "cores": len(rect), "used": len(used)}
                    for sender, rect, used, _, _ in self.casts or []
                ],
            }
        return [
            {
                "rect": piece.rect(self.nocs.grid),
                "m0": piece.m0,
                "n0": piece.n0,
                "bands": [piece.plan.pcm, piece.plan.pcn],
                "blocks": [piece.plan.nbh, piece.plan.nbw],
                "cores": len(piece.places(self.nocs.grid)),
            }
            for piece in self.pieces
        ]

    def stop(self):
        """Stops the run: each island's kernel result and the cores that gave up."""
        self.run.stop()
        results = self.run.result()
        states = self.run.state()
        return {
            "cores": len(states),
            "kernel": [
                (result or {}).get("error_name")
                for result in (results if isinstance(results, list) else [results])
            ],
            "gave_up": [
                {key: state[key] for key in ("core", "reader", "writer", "compute")}
                for state in states
                if state["reader"] or state["writer"] or state["compute"]
            ][:8],
            "broadcast_gave_up": [
                {"core": state["core"]} | area["gave_up"]
                for state in states
                for area in state["broadcasts"]
                if area["gave_up"]
            ][:8],
            # each core's AICLK cycles waited in the row and the column broadcast since
            # the load (a sender for its receivers, a receiver for VALID), and blocks
            "broadcast_waits": [
                [state["core"]]
                + [
                    [area["wait_cycles"], area["blocks"]]
                    for area in state["broadcasts"][:2]
                ]
                for state in states
            ],
        }


class Grid:
    """The dense project's own grid kernels on a one-island session, beside: its
    plan for the whole product, one GRID command a call on its rect."""

    def __init__(self, session, dimensions, tensors, wait=2.0, deadline=5.0):
        m_tiles, k_tiles, n_tiles = (value // pl.TILE for value in dimensions)
        nocs = session.sessions[0].nocs()
        self.plan = pl.grid_plan(m_tiles, k_tiles, n_tiles, grid=nocs.grid)
        self.group = self.plan.group(nocs.grid)
        self.tensors = {"a": tensors["x"], "w": tensors["w"], "y": tensors["y"]}
        self.program = card.grid_program(self.plan, nocs, self.tensors)
        self.run = session.load(self.program, deadline=deadline, wait=wait)

    def calls(self, commands, timeout=60.0):
        addresses = [self.tensors[name].address for name in ("a", "w", "y")]
        words = np.tile(
            np.array(pl.grid_args(self.plan, *addresses, G["kGridFull"]), np.int64),
            (commands, 1),
        )
        started = time.perf_counter()
        handles = self.run.submit_many("grid", *self.group, words, exclusive=True)
        events = self.run.wait(list(handles), timeout=timeout)
        host_s = time.perf_counter() - started
        unfinished = [part for event in events for part in event if part.kind != "done"]
        if unfinished:
            raise km.KomorebiError(f"{len(unfinished)} parts not done: {unfinished[0]}")
        return host_s, np.array(
            [part.microseconds for event in events for part in event]
        ).reshape(commands, -1)

    def single(self, commands):
        return np.concatenate(
            [self.calls(1, timeout=10.0)[1].max(axis=1) for _ in range(commands)]
        )

    stop = Linear.stop


def measure(product, cards, seconds):
    """Three single calls, then calls back to back for about `seconds` with every
    card's AICLK sampled."""
    before = product.run.crossings()
    single = product.single(3)
    one_microseconds = float(np.median(single))
    log(f"single call {one_microseconds:.1f} us")
    if seconds <= 0:  # the single calls alone, each waited for
        crossed = (product.run.crossings() - before).total()
        return {
            "single_microseconds": [float(value) for value in single],
            "reps": 0,
            "microseconds_per_call": round(one_microseconds, 3),
            "crossed_a_call": {field: count / 3 for field, count in crossed.items()},
            "aiclk_mhz": {},
        }
    reps = max(2, math.ceil(seconds * 1e6 / one_microseconds))
    before = product.run.crossings()
    with contextlib.ExitStack() as stack:
        clocks = [stack.enter_context(bench.Clock(index)) for index in cards]
        log(f"loop of {reps} calls")
        host_s, microseconds = product.calls(
            reps, timeout=3 * reps * max(one_microseconds, 50.0) / 1e6 + 10
        )
    crossed = (product.run.crossings() - before).total()
    report = {
        "single_microseconds": [float(value) for value in single],
        "reps": reps,
        "seconds": round(host_s, 3),
        "microseconds_per_call": round(host_s * 1e6 / reps, 3),
        "part_microseconds_median": np.median(microseconds, axis=0).tolist(),
        "crossed_a_call": {field: count / reps for field, count in crossed.items()},
        "aiclk_mhz": {
            card_index: clock.summary() for card_index, clock in zip(cards, clocks)
        },
    }
    log(f"loop: {report['microseconds_per_call']} us per call")
    if getattr(product, "tensors_y_copy", False) and not product.adds:  # Y the sum
        report["reduced"] = product.reduced(REDUCED_CALLS)
        log(f"with the all-reduce: {report['reduced'][-1]}")
    elif getattr(product, "y_copy", False) and not product.gathers:  # then gathered
        report["gathered"] = product.gathered(REDUCED_CALLS)
        log(f"with the all-gather: {report['gathered'][-1]}")
    return report


def clean(stopped):
    gave_up = stopped["gave_up"] or stopped["broadcast_gave_up"]
    return all(kernel == "none" for kernel in stopped["kernel"]) and not gave_up


def checked(session, tensors, dimensions, matrices, kind):
    """Y's sampled rows read back and checked against the float64 product; when the
    check fails on several islands, each sampled row tile's error and its holder."""
    m_rows, _, n_columns = dimensions
    rows = ref.sampled_rows(m_rows)
    started = time.perf_counter()
    y = tensors["y"]
    if len(session.sessions) > 1 and y.placement.copy:  # after the all-reduce
        copies = [
            card.read_rows(one, part, rows, log)
            for one, part in zip(session.sessions, y.parts, strict=True)
        ]
        report = run.check(copies[0][:, :n_columns], rows, *matrices, kind)
        report["copies_equal"] = all(np.array_equal(copies[0], one) for one in copies)
        report["ok"] = report["ok"] and report["copies_equal"]
        got = copies[0]
    else:
        got = card.read_rows(session, y, rows, log)
        report = run.check(got[:, :n_columns], rows, *matrices, kind)
    report["readback_s"] = round(time.perf_counter() - started, 3)
    if not report["ok"] and len(session.sessions) > 1:
        want = ref.matmul(matrices[0][rows], matrices[1])
        y = tensors["y"]
        report["row_tiles"] = [
            {
                "row_tile": int(rows[at]) // pl.TILE,
                "holder": y.holders(
                    int(rows[at]) // pl.TILE * y.shape[1] // pl.TILE, 1
                )[0][0],
                **ref.errors(
                    got[at : at + pl.TILE, :n_columns], want[at : at + pl.TILE]
                ),
            }
            for at in range(0, len(rows), pl.TILE)
        ]
    return report


def sizes(program):
    found = getattr(program, "sizes", None)
    return None if found is None else {"text": found.text, "total": found.total}


def at_error(product):
    """After a call gave up, while the run goes on: what crossed the links, and every
    core of each island as its dispatcher reads it (link cores too)."""
    out = {}
    runs = getattr(product.run, "runs", [product.run])
    try:
        out["crossings"] = {
            str(key): row for key, row in product.run.crossings().rows.items()
        }
    except (km.KomorebiError, AttributeError) as error:
        out["crossings"] = f"{type(error).__name__}: {error}"
    for island, one in enumerate(runs):
        try:
            out[f"island {island}"] = one.inspect()
        except km.KomorebiError as error:
            out[f"island {island}"] = f"{type(error).__name__}: {error}"
    plain = json.dumps(
        out, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)
    )
    return json.loads(plain)


def attempt(product_class, session, dimensions, tensors, cards, args, matrices):
    """Loads, measures, stops and checks one product; its report."""
    flops = 2 * math.prod(dimensions)
    report = {}
    try:
        product = product_class(session, dimensions, tensors, wait=args.wait)
    except km.KomorebiError as error:
        return {"error": f"{type(error).__name__}: {error}", "ok": False}
    report["plan"] = run.describe(product.plan)
    report["kernel_sizes"] = sizes(product.program)
    if isinstance(product, Linear):
        report["pieces"] = product.describe()
        report["l1_budget"] = product.budget
        report["l1_held"] = product.l1_held
    log(f"{product_class.__name__} loaded, plan {report['plan']}")
    try:
        report["loop"] = measure(product, cards, args.seconds)
    except km.KomorebiError as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["at_error"] = at_error(product)
        with contextlib.suppress(km.KomorebiError):
            product.run.stop()
        report["ok"] = False
        return report
    report["single_detail"] = getattr(product, "single_detail", None)
    report["run"] = product.stop()
    if isinstance(product, Linear):
        report["passes"] = product.passes()
        log(f"passes: {report['passes']}")
    report["check"] = checked(session, tensors, dimensions, matrices, args.inputs)
    per_call = report["loop"]["microseconds_per_call"]
    report["tflops"] = round(flops / per_call / 1e6, 2)
    crossed = report["loop"]["crossed_a_call"]["bytes"]
    if crossed:
        report["crossed_gb_per_s"] = round(crossed / per_call / 1e3, 2)
    report["ok"] = report["check"]["ok"] and clean(report["run"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shape", choices=run.SHAPES, default=None)
    parser.add_argument("--cube", type=int, default=None, help="M = K = N")
    parser.add_argument("--dims", default=None, help="M,K,N")
    for name in ("m", "k", "n"):
        parser.add_argument(f"--{name}", type=int, default=None)
    parser.add_argument(
        "--place",
        choices=list(PLACES),
        default="out",
        help="out: W, Y split along N, X a copy; in: X, W split along K, Y a copy (a "
        "partial an island); rows: X, Y split along M, W a copy on every island (W "
        "gathered)",
    )
    parser.add_argument("--cards", default="0")
    parser.add_argument("--inputs", choices=["uniform", "int"], default="uniform")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--wait", type=float, default=2.0, help="wait limit, s")
    parser.add_argument("--links", type=int, default=0, help="page links, two cards")
    parser.add_argument(
        "--kernel-bytes", type=int, default=None, help="kernel buffer (default: own)"
    )
    parser.add_argument(
        "--beside", action="store_true", help="then the project's grid kernels"
    )
    parser.add_argument(
        "--subblock", default=None, help="HxW: the subblock in place of the plan's"
    )
    parser.add_argument(
        "--columns", type=int, default=None, help="one card: the grid's first columns"
    )
    parser.add_argument(
        "--nocs", default="0,1", help="R,W: the reader's and the writer's NoC"
    )
    parser.add_argument(
        "--valid",
        choices=["acked", "sent"],
        default="acked",
        help="a broadcast block is valid once acknowledged, or once sent",
    )
    parser.add_argument(
        "--account", action="store_true", help="the kernels' account of each phase"
    )
    parser.add_argument(
        "--reduce",
        choices=list(pl.REDUCE_FORMS),
        default="tree",
        help="the K-split plan's reduce in a group",
    )
    parser.add_argument(
        "--pattern",
        choices=list(SPLIT_DEPTH),
        default="AB",
        help="the K-split plan's chunks: their movers (mover A's half 3 deep but AB's)",
    )
    parser.add_argument(
        "--decode",
        choices=["priced", "split", "whole"],
        default="priced",
        help="a few rows: the K-split or the whole-K plan, or the cheaper by its terms",
    )
    parser.add_argument(
        "--cast-depth",
        type=int,
        default=None,
        help="the whole-K plan: W's chunks a mover's half, this depth alone",
    )
    parser.add_argument(
        "--cast-pattern",
        choices=list(CAST_PATTERNS),
        default=CAST_PATTERNS[0],
        help="the whole-K plan: a receiver's W chunks, their movers",
    )
    parser.add_argument(
        "--cast-nocs",
        default="1,0",
        help="R,W: the whole-K plan's reader's and writer's NoC (1,0: X cast on NoC 0)",
    )
    parser.add_argument(
        "--add",
        choices=["kernel", "reduce"],
        default="kernel",
        help="input dim sharded: the kernels add the other island's partial, or the "
        "all-reduce of Y after the product",
    )
    parser.add_argument(
        "--gather",
        choices=["kernel", "host"],
        default="kernel",
        help="a chain's first layer: the kernels gather Y's columns at the call's end, "
        "or the host's all-gather after it",
    )
    parser.add_argument(
        "--spread",
        choices=["auto", "on", "off"],
        default="auto",
        help="the grid plan: W's read spread over the column's ranks (auto: where the "
        "plan's model prices it lower), or the column's sender reads it all",
    )
    parser.add_argument(
        "--xspread",
        choices=["auto", "on", "off"],
        default="auto",
        help="the grid plan: X's read spread over the row's ranks (auto: where the "
        "plan's model prices it lower), or the row's sender reads it all",
    )
    parser.add_argument(
        "--shapes",
        default=None,
        help='several products in one session: "M,K,N[:place];..."',
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    if args.shapes:
        dimensions = None
    elif args.dims:
        dimensions = tuple(int(value) for value in args.dims.split(","))
    elif args.cube:
        dimensions = (args.cube,) * 3
    elif args.shape:
        dimensions = run.SHAPES[args.shape]
    elif None in (args.m, args.k, args.n):
        raise SystemExit("name --shape, --cube, --dims, or all of --m, --k and --n")
    else:
        dimensions = (args.m, args.k, args.n)
    if dimensions and any(value % pl.TILE for value in dimensions):
        raise SystemExit(f"M, K and N are whole tiles of {pl.TILE}: {dimensions}")
    cards = [int(value) for value in args.cards.split(",")]
    global SUBBLOCK, COLUMNS, NOCS, VALID_SENT, ACCOUNT, SPLIT_REDUCE, ADD, GATHER
    global SPREAD, X_SPREAD, SPLIT_PATTERN, DECODE, CAST_DEPTHS, CAST_PATTERN, CAST_NOCS
    SPLIT_PATTERN, DECODE, CAST_PATTERN = args.pattern, args.decode, args.cast_pattern
    CAST_NOCS = tuple(int(value) for value in args.cast_nocs.split(","))
    if sorted(CAST_NOCS) != [0, 1]:
        raise SystemExit(f"--cast-nocs {args.cast_nocs}: 0,1 or 1,0")
    if args.cast_depth:
        CAST_DEPTHS = (args.cast_depth,)
    ACCOUNT, SPLIT_REDUCE, ADD = args.account, args.reduce, args.add
    GATHER = args.gather
    SPREAD = args.spread
    X_SPREAD = args.xspread
    NOCS = tuple(int(value) for value in args.nocs.split(","))
    VALID_SENT = args.valid == "sent"
    if args.subblock:
        SUBBLOCK = tuple(int(value) for value in args.subblock.split("x"))
    COLUMNS = args.columns
    # several products in one session (--shapes): each "M,K,N[:place]"
    products = [(dimensions, args.place)]
    if args.shapes:
        products = []
        for item in args.shapes.split(";"):
            dims, _, place = item.partition(":")
            products.append(
                (tuple(int(value) for value in dims.split(",")), place or args.place)
            )
    for dims, place in products:
        if place not in [*PLACES, "chain"] or any(value % pl.TILE for value in dims):
            raise SystemExit(f"{dims}:{place}: whole tiles, a place of {list(PLACES)}")
        if place == "chain" and len(cards) != 2:
            raise SystemExit("a chain is two layers on two cards: --cards A,B")
    # two cards: the larger build (an island's build 96,240 B at prefill, 105,680 at
    # decode); the kernels' addition or gather larger still
    two = len(cards) > 1
    adds = len(cards) > 1 and (
        ADD == "kernel"
        and any(p == "in" for _, p in products)
        or GATHER == "kernel"
        and any(p == "chain" for _, p in products)
    )
    kernel_bytes = KERNEL_BYTES["add" if adds else "two"] if two else None
    if two and any(dims[0] <= SPLIT_ROWS * pl.TILE for dims, _ in products):
        kernel_bytes = max(kernel_bytes, KERNEL_BYTES["decode"])
    kernel_bytes = args.kernel_bytes or kernel_bytes
    options = {"links": args.links, "kernel_bytes": kernel_bytes}
    options = {key: value for key, value in options.items() if value}
    reports = []
    with km.Session(cards=cards, log=log, **options) as session:
        for dims, place in products:
            one = chain_report if place == "chain" else product_report
            reports.append(one(session, dims, place, cards, args))
            if args.json:  # each product's report kept as it ends
                Path(args.json).write_text(
                    json.dumps({"products": reports}, indent=1, default=float)
                )
    report = reports[0] if not args.shapes else {"products": reports}
    report["ok"] = all(one["ok"] for one in reports)
    report["load_avg"] = list(Path("/proc/loadavg").read_text().split()[:3])
    print(json.dumps(power.clocked(report), default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))
    sys.exit(0 if report["ok"] else 1)


def product_report(session, dimensions, place, cards, args):
    """One product on the open session: its inputs placed by `place`, the linear
    kernels loaded, measured, stopped and checked (and the grid kernels beside)."""
    report = {"m": dimensions[0], "k": dimensions[1], "n": dimensions[2]}
    report |= {"cards": cards, "links": args.links, "place": place}
    report["nocs"], report["valid"] = list(NOCS), args.valid
    log(f"building the inputs of {dimensions}, {place}")
    x_matrix, w_matrix = ref.inputs(*dimensions, kind=args.inputs)
    log("inputs built")
    started = time.perf_counter()
    m_rows, _, n_columns = dimensions
    # by the placement, each island half of what is split (S, the tiles other
    # islands put, made with the kernels)
    halves = {"group": even_group(session.graph)}
    x_hint, w_hint, y_hint = (
        {**halves, **hint} if "split" in hint else hint for hint in PLACES[place]
    )
    tensors = {
        "x": session.tensor(x_matrix, "bfloat16", "tile", **x_hint),
        "w": session.tensor(w_matrix, "bfloat16", "tile", **w_hint),
        "y": session.empty((m_rows, n_columns), "bfloat16", "tile", **y_hint),
    }
    report["upload_s"] = round(time.perf_counter() - started, 3)
    report["placements"] = {
        name: [list(part) for part in tensor.placement.parts]
        for name, tensor in tensors.items()
    }
    log(f"operands placed in {report['upload_s']} s: {report['placements']}")
    report["x280_mhz"] = [one.pll_mhz for one in session.sessions]
    matrices = (x_matrix, w_matrix)
    report["linear"] = attempt(
        Linear, session, dimensions, tensors, cards, args, matrices
    )
    if args.beside and report["linear"]["ok"]:
        log(f"cool-down {args.seconds} s")
        time.sleep(args.seconds)
        release_runs(session)
        report["grid"] = attempt(
            Grid, session, dimensions, tensors, cards, args, matrices
        )
    report["ok"] = report["linear"]["ok"] and report.get("grid", {}).get("ok", True)
    for tensor in tensors.values():  # the next product's room (the tensor table)
        tensor.free()
    release_runs(session)
    return report


def chain_report(session, dimensions, place, cards, args):
    """Two layers, M x H -> F -> H, both output dim sharded: the first writes each
    island's columns into a copy of Y1, the all-gather completes it, the second
    reads it as its X. Each layer loaded, measured, stopped and checked in turn."""
    m_rows, hidden, mlp = dimensions
    report = {"m": m_rows, "hidden": hidden, "mlp": mlp, "cards": cards}
    report |= {"place": place, "nocs": list(NOCS), "valid": args.valid}
    log(f"building the chain {dimensions}")
    x_matrix, w1_matrix = ref.inputs(m_rows, hidden, mlp, kind=args.inputs)
    _, w2_matrix = ref.inputs(m_rows, mlp, hidden, kind=args.inputs)
    halves = {"group": even_group(session.graph), "split": 1}
    y1 = session.empty((m_rows, mlp), "bfloat16", "tile", copy=True)
    first = {
        "x": session.tensor(x_matrix, "bfloat16", "tile", copy=True),
        "w": session.tensor(w1_matrix, "bfloat16", "tile", **halves),
        "y": y1,
    }
    # the all-gather's shares (pages of Y1's one tile row) are the islands' columns
    shares = [
        (part.start, len(part))
        for part in (layout.partition(range(mlp // pl.TILE), c, 2) for c in range(2))
    ]
    if m_rows > pl.TILE or w_column_spans(first["w"].placement, mlp // pl.TILE) != [
        tuple(share) for share in shares
    ]:
        raise SystemExit(f"a chain is one tile row, W1's columns the shares: {shares}")
    report["first"] = attempt(
        Linear,
        session,
        (m_rows, hidden, mlp),
        first,
        cards,
        args,
        (x_matrix, w1_matrix),
    )
    report["first"]["gather_shares"] = shares
    for name in ("x", "w", "s", "t"):
        first[name].free()
    release_runs(session)
    if not report["first"]["ok"]:
        y1.free()
        report["ok"] = False
        return report
    rows = np.arange(m_rows)
    got = card.read_rows(session.sessions[0], y1.parts[0], rows, log)[:, :mlp]
    second = {
        "x": y1,
        "w": session.tensor(w2_matrix, "bfloat16", "tile", **halves),
        "y": session.empty((m_rows, hidden), "bfloat16", "tile", **halves),
    }
    report["second"] = attempt(
        Linear,
        session,
        (m_rows, mlp, hidden),
        second,
        cards,
        args,
        (got, w2_matrix),
    )
    for tensor in second.values():
        tensor.free()
    release_runs(session)
    report["ok"] = report["second"]["ok"]
    if report["ok"]:
        loop = [report[layer]["loop"] for layer in ("first", "second")]
        report["chain_us"] = {
            "first": loop[0]["microseconds_per_call"],
            "second": loop[1]["microseconds_per_call"],
        }
        if "gathered" in loop[0]:  # the host's all-gather between the layers
            gathered = loop[0]["gathered"]
            report["chain_us"] |= {
                "gather": float(np.median([one["gather_us"] for one in gathered])),
                "first_then_gather": float(
                    np.median(
                        [one["product_us"] + one["gather_us"] for one in gathered]
                    )
                ),
            }
        else:  # inside the first layer's call: its stamps' gather, each island
            report["chain_us"]["gather_in_first"] = [
                one["microseconds_at_mhz"]["sync"] for one in report["first"]["passes"]
            ]
        log(f"chain: {report['chain_us']}")
    return report


def release_runs(session):
    """Drops the session's stopped runs, so that their L1 (each core's control block
    and transfer buffer) is free before the next product's load: a session keeps its
    last run until a load replaces it, and the load places its own L1 first."""
    session.run = None
    for one in session.sessions:
        one.run = None
    gc.collect()


if __name__ == "__main__":
    main()
