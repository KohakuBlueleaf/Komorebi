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
from komorebi import abi, layout  # noqa: E402
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
# The kernel buffer the linear kernels' code takes (mock builds): 79,936 B without puts;
# K form or K split on two cards up to 108,240 B; the kernels' addition 118,000 B.
KERNEL_BYTES = {"one": 88 * 1024, "two": 108 * 1024, "add": 116 * 1024}
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
# input dim sharded: the addition across in the kernels (kernel) or the all-reduce of Y
# after the product (reduce) (--add)
ADD = "kernel"
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
ACCOUNT_CLOCKS = (
    ("reader_end", "kAcReaderEnd"),
    ("compute_start", "kAcComputeStart"),
    ("compute_end", "kAcComputeEnd"),
)
# Placements by hint (X, W, Y): "k" (K and N split, puts); "rows" (W gathered: X, Y
# split along M); "out" (W, Y split along N, X a copy); "in" (X, W along K, Y partials).
PLACES = {
    "k": ({"split": 1}, {"split": 0}, {"split": 1}),
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


def chunk(plan):
    """Tiles a put: the smallest whole number of subblocks at or over kPutTiles."""
    subblock = plan.sh * plan.sw
    return pl.cdiv(L["kPutTiles"], subblock) * subblock


def extra_buffers(plan, exchange):
    """The linear kernels' buffers past the grid plan's: two chunks of the tiles put
    across (a subblock on one island, where nothing is put), another island's subblock
    put here, the token and its slot."""
    subblock = plan.sh * plan.sw
    send = 2 * chunk(plan) if exchange else subblock
    return [
        (L["kCbSend"], "bfloat16", L["kBf16Bytes"], send),
        (L["kCbSent"], "bfloat16", L["kBf16Bytes"], 2 * subblock),
        (L["kCbToken"], "uint32", L["kSignalBytes"], 1),
        (L["kCbArrive"], "uint32", L["kSignalBytes"], 1),
        (L["kCbStamp"], "uint32", L["kStampBytes"], 1),
    ]


def extra_bytes(plan, exchange):
    return sum(page * pages for _, _, page, pages in extra_buffers(plan, exchange))


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
    return min(pl.L1_BUDGET, free - held - L1_TRANSFER - L1_RUNTIME - tokens)


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
    """(index, format, page bytes, pages) of the K-split plan's buffers (two chunks a
    mover; `adds`: both partials of a block), T's slot last."""
    block_tiles = plan.oh * plan.ow
    both = (
        [(L["kCbKAdd"], "bfloat16", L["kBf16Bytes"], 2 * block_tiles)] if adds else []
    )
    return both + [
        (L["kCbKa"], "bfloat16", L["kBf16Bytes"], 2 * plan.oh * plan.kc),
        (L["kCbKaB"], "bfloat16", L["kBf16Bytes"], 2 * plan.oh * plan.kc),
        (L["kCbKw"], "bfloat16", L["kBf16Bytes"], 2 * plan.kc * plan.ow),
        (L["kCbKwB"], "bfloat16", L["kBf16Bytes"], 2 * plan.kc * plan.ow),
        (L["kCbKArrive"], "float32", L["kPartBytes"], plan.arrive_pages()),
        (L["kCbKSum"], "float32", L["kPartBytes"], block_tiles),
        (L["kCbKOut"], "bfloat16", L["kBf16Bytes"], block_tiles),
        (L["kCbStamp"], "uint32", L["kStampBytes"], 1),
    ]


def split_ns(plan, ranks):
    """The K-split model's ns: a rank's W at its share of the card's DRAM rate, then
    the tree's steps of one block each."""
    w_bytes = pl.cdiv(plan.kt, plan.s) * plan.ow * L["kBf16Bytes"]
    steps = max(1, plan.s - 1).bit_length()
    reduce = steps * plan.oh * plan.ow * L["kPartBytes"] / REDUCE_BYTES_PER_S
    return 1e9 * (w_bytes * ranks / DRAM_BYTES_PER_S + reduce)


def split_choice(m_tiles, k_tiles, n_tiles, ranks, budget, adds=False):
    """The K-split plan of the island's part with the least split_ns on `ranks` ranks
    inside `budget`: one block height (all the rows), every block width, the largest
    group the ranks allow; K chunks of at most SPLIT_CHUNK_PAGES pages of W."""
    best = None
    # at least 8 columns where N has them: a K row of a chunk on every bank
    for width in range(min(8, n_tiles), n_tiles + 1):
        groups = pl.cdiv(n_tiles, width)
        size = min(ranks // groups, k_tiles)
        if size < 1:
            continue
        chunk = max(1, min(pl.cdiv(k_tiles, size), SPLIT_CHUNK_PAGES // width))
        sh, sw = pl.subblock(m_tiles, width)
        plan = pl.Split(m_tiles, k_tiles, n_tiles, size, m_tiles, width, chunk, sh, sw)
        buffers = split_buffers(plan, adds)
        if sum(page * pages for _, _, page, pages in buffers) > budget:
            continue
        cost = (split_ns(plan, ranks), groups)
        if best is None or cost < best[0]:
            best = (cost, plan)
    return None if best is None else best[1]


# The writers' put counters in T (kernels/linear/linear.h kStampFed ..), by name
PUT_COUNTS = (
    ("fed", "kStampFed"),
    ("line", "kStampLine"),
    ("piece", "kStampPiece"),
    ("turn", "kStampTurn"),
    ("write", "kStampWrite"),
    ("pieces", "kStampPieces"),
)


def fitting_plan(m_tiles, k_tiles, n_tiles, grid, ranks, budget, exchange):
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
            over = plan.l1_bytes() + extra_bytes(plan, exchange) - budget
            if over <= 0:
                if plan.group(grid)[1] <= ranks:
                    return plan
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


def k_block_ns(bm, bn, kb):
    sh, sw = pl.subblock(bm, bn)
    compute = bm * bn * kb * PRODUCT_NS + bm * bn // (sh * sw) * SUBBLOCK_NS
    feed = FEED_BLOCK_NS + max(bm, bn) * kb * FEED_TILE_NS
    return max(compute, feed)


def plan_ns(plan):
    """A core's time for its blocks of `plan` by k_block_ns."""
    return plan.nbh * plan.nbw * plan.nk * k_block_ns(plan.bm, plan.bn, plan.kb)


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


def island_pieces(
    m_tiles, k_tiles, n_tiles, grid, ranks, budget, exchange, kb, every=False
):
    """The pieces of the island's ranks: the old single rect (fitting_plan), or the
    whole grid, or two rectangles sharing one block, in bands of whole blocks: the
    cheapest by plan_ns (then the larger block, the larger K block). The K block is
    `kb`, lower only when nothing fits; `every`: any K block size up to `kb`."""
    single = fitting_plan(m_tiles, k_tiles, n_tiles, grid, ranks, budget, exchange)
    best = None
    if single is not None:
        cost = plan_ns(single)
        best = ((cost, -single.bm * single.bn, -single.kb), [Piece(single)])
    rects = two_rects(grid, ranks)
    if not rects or single is None or single.layout != "2d":
        return None if best is None else best[1]
    k_blocks = range(kb, 0, -1) if every else pl.K_BLOCKS
    for k_block in [value for value in k_blocks if value <= kb] or [1]:
        found = False
        for bm in range(1, m_tiles + 1):
            for bn in range(1, n_tiles + 1):
                sh, sw = pl.subblock(bm, bn)
                if sh * sw < 4 or bm * bn * G["kPartBytes"] > budget:
                    continue
                probe = piece_plan(bm, k_tiles, bn, (1, 1), (bm, bn, k_block, 1, 1))
                if probe is None:
                    continue
                if probe.l1_bytes() + extra_bytes(probe, exchange) > budget:
                    continue
                for form, (a, b) in rects.items():
                    for pieces in split_pieces(
                        form, a, b, m_tiles, k_tiles, n_tiles, (bm, bn, k_block)
                    ):
                        found = True
                        cost = (
                            max(plan_ns(p.plan) for p in pieces),
                            -bm * bn,
                            -k_block,
                        )
                        if cost < best[0]:
                            best = (cost, pieces)
        if found and not every:
            break
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
    """S's pages a set (a set each other island): the output blocks of every piece's
    cores, in core order."""
    return sum(
        len(piece.places(grid))
        * piece.plan.nbh
        * piece.plan.nbw
        * piece.plan.bm
        * piece.plan.bn
        for piece in pieces
    )


def piece_blocks(piece, grid, width):
    """The output blocks of a piece's cores holding some of the window's tiles (M x
    `width`): those a core puts."""
    plan = piece.plan
    blocks = 0
    for first_row, first_column in piece.places(grid).values():
        row_end = min(first_row + plan.pcm, plan.mt + piece.m0)
        column_end = min(first_column + plan.pcn, width)
        for block in range(plan.nbh * plan.nbw):
            row = first_row + block // plan.nbw * plan.bm
            column = first_column + block % plan.nbw * plan.bn
            blocks += row < row_end and column < column_end
    return blocks


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
    # a core of no piece: no broadcast, its rank word 0, and M, so that it finds the
    # exchange its island's pieces find and joins the sync's tree with them
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
        self.n_tiles = n_tiles
        self.k_parts = k_spans(tensors["w"].placement, k_tiles)
        self.w_columns = w_column_spans(tensors["w"].placement, n_tiles)
        self.tensors_y_copy = self.islands > 1 and tensors["y"].placement.copy
        # the one addition across made by the kernels (LINEAR_ADD), two islands
        self.adds = self.tensors_y_copy and ADD == "kernel"
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
        self.exchange = any(
            self.window(here, to)[1]
            for here in range(self.islands)
            for to in range(self.islands)
            if to != here
        )
        one_card = pl.grid_plan(m_tiles, k_tiles, n_tiles, grid=self.nocs.grid)
        # S and T; R, the all-reduce's scratch, when Y is each island's partial and
        # the all-reduce adds
        reduces = self.tensors_y_copy and not self.adds
        self.budget = l1_budget(session, self.islands, extra=2 + reduces)
        grid, planned = self.nocs.grid, ranks  # the grid and ranks planned on
        if COLUMNS is not None:
            if self.islands != 1 or not 0 < COLUMNS <= grid[0]:
                raise SystemExit(f"--columns {COLUMNS}: one island, 1 to {grid[0]}")
            grid, planned = (COLUMNS, grid[1]), COLUMNS * grid[1]
        # a few rows and nothing put: the K-split plan
        self.split = None
        if part_m <= SPLIT_ROWS and not self.exchange and COLUMNS is None:
            self.split = split_choice(
                part_m, part_k, part_n, planned, self.budget, self.adds
            )
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
                self.exchange,
                # nothing put: any K block the model likes; puts: the one-card plan's
                (
                    pl.K_BLOCKS[0]
                    if not self.exchange or one_card is None
                    else one_card.kb
                ),
                every=not self.exchange,
            )
        )
        if self.split is None and self.pieces is None:
            raise SystemExit(f"no grid plan fits L1 and {ranks} ranks: {dimensions}")
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
        pages = (
            max(1, self.islands - 1) * sent_pages(self.pieces, self.nocs.grid)
            if self.exchange and self.split is None
            else 1  # nothing put
        )
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
        ):
            words[L[name]] = value
        fp32 = "UnpackToDestFp32"
        x, w, y = (tensors[name] for name in "xwy")
        return km.Program(
            "dense_linear",
            kernels=KERNELS,
            buffers=[km.Buffer(*buffer) for buffer in split_buffers(plan, self.adds)],
            args=lambda core: (words, words, words),
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
            defines={"LINEAR_SPLIT": 1, "LINEAR_OWN": 1}
            | ({"LINEAR_ADD": 1} if self.adds else {})
            | ({"LINEAR_ACCOUNT": 1} if ACCOUNT else {}),
        )

    def grid_program(self, tensors, ranks, m_tiles):
        """The linear program of the grid plan: the pieces' broadcasts and blocks."""
        x, w, y, s = (tensors[name] for name in "xwys")
        return km.Program(
            "dense_linear",
            kernels=KERNELS,
            buffers=[
                km.Buffer(*buffer)
                for buffer in self.plan.buffers()[:4]
                + extra_buffers(self.plan, self.exchange)
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
            # nothing exchanged: kernels without the puts and the sync (smaller)
            defines=({} if self.exchange else {"LINEAR_OWN": 1})
            | ({"LINEAR_ADD": 1} if self.adds else {})
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
        cycles, and microseconds at `mhz`): the puts sent, the sync, its own columns."""
        out = []
        for island, one in enumerate(self.session.sessions):
            (raw,) = one.read(self.tensors["t"].parts[island])
            words = (
                raw.view(np.uint32).reshape(-1, L["kStampBytes"] // 4).astype(np.int64)
            )
            ran = words[words[:, L["kStampEnd"]] != 0]
            counts = {  # the puts' cycles (pieces: a count): mean and most a core
                name: ran[:, L[stamp]]
                for name, stamp in PUT_COUNTS
                if ran[:, L["kStampPieces"]].any()
            }
            account = {}
            names = SPLIT_ACCOUNT_WORDS if self.split is not None else ACCOUNT_WORDS
            clocks = () if self.split is not None else ACCOUNT_CLOCKS
            if ACCOUNT and len(ran):  # the slowest core's phases, and each one's most
                span = (ran[:, L["kStampEnd"]] - ran[:, L["kStampStart"]]) % 2**32
                slow = int(np.argmax(span))
                start = ran[slow, L["kStampStart"]]
                account = {
                    "core": slow,
                    "span": int(span[slow]),
                    "slowest": {name: int(ran[slow, L[w]]) for name, w in names}
                    | {
                        name: int((ran[slow, L[w]] - start) % 2**32)
                        for name, w in clocks
                    },
                    "most": {name: int(ran[:, L[w]].max()) for name, w in names},
                    # every core's, in core order (the spread of each phase)
                    "cores": {name: ran[:, L[w]].tolist() for name, w in names},
                }
            # the cores that stamped the puts (or the addition's gather): all of the
            # K form's; the K split's first ranks, the grid's ranks with blocks
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
            if stamped:  # inside the sync: the last core's each step, from the last put
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
                # each core's own wait for its last put's confirmation
                waits = ran[has, L["kStampSettled"]] - ran[has, L["kStampPut"]]
                cycles["settle_median"] = int(np.median(waits))
            out.append(
                {
                    "account": account,
                    "put_counts": {
                        name: [round(float(value.mean()), 1), int(value.max())]
                        for name, value in counts.items()
                    },
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

    def crossed(self):
        """Bytes a call should cross, from the placements and the pieces: each island
        holding part of K puts, for every other island keeping columns, each core's
        output blocks that hold some of them (whole blocks, bf16); and the ideal: every
        other holder's M x columns tiles."""
        plan = self.plan
        page = self.tensors["s"].aligned_page_bytes
        total = {"blocks": 0, "tiles": 0}
        for sender, (_, k_part) in enumerate(self.k_parts):
            for to in range(self.islands):
                width = self.window(sender, to)[1]
                if to == sender or not k_part or not width:
                    continue
                total["tiles"] += self.made[sender][1] * width * page
                for piece in self.pieces:
                    total["blocks"] += piece_blocks(piece, self.nocs.grid, width) * (
                        plan.bm * plan.bn * page
                    )
        total["put_bytes"] = chunk(plan) * page if self.exchange else 0
        return total

    def window(self, here, to):
        """(first, count) of the columns island `to` keeps of the rows island `here`
        makes and holds W's columns of, as the kernels' window() finds them."""
        made, kept = self.made[here], self.kept[to]
        if to != here and self.tensors_y_copy:  # each island its own partial
            return (0, 0)
        if not made[1] or made[0] < kept[0][0] or sum(made) > sum(kept[0]):
            return (0, 0)
        held = self.w_columns[here]
        if kept[1][0] < held[0] or sum(kept[1]) > sum(held):
            return (0, 0)
        return kept[1]

    def describe(self):
        """Each piece: its rect on the island's grid, first row and column, bands."""
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
        report["exchange"] = product.exchange
        report["l1_budget"] = product.budget
        report["l1_held"] = product.l1_held
        report["crossed_bytes"] = product.crossed()
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
        default="k",
        help="k: X, W split along K, Y along N (W sharded); rows: X, Y split along "
        "M, W a copy on every island (W gathered); out: W, Y split along N, X a "
        "copy; in: X, W split along K, Y a copy (a partial an island)",
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
        "--add",
        choices=["kernel", "reduce"],
        default="kernel",
        help="input dim sharded: the kernels add the other island's partial, or the "
        "all-reduce of Y after the product",
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
    global SUBBLOCK, COLUMNS, NOCS, VALID_SENT, ACCOUNT, SPLIT_REDUCE, ADD
    ACCOUNT, SPLIT_REDUCE, ADD = args.account, args.reduce, args.add
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
        if place not in PLACES or any(value % pl.TILE for value in dims):
            raise SystemExit(f"{dims}:{place}: whole tiles, a place of {list(PLACES)}")
    # the larger build: the K form's puts, or the K-split plan's page calls (mock
    # 105,376 B), on two cards; the rows, out and in forms otherwise fit the one-card's
    puts = len(cards) > 1 and any(
        place == "k" or dims[0] <= SPLIT_ROWS * pl.TILE for dims, place in products
    )
    adds = len(cards) > 1 and ADD == "kernel" and any(p == "in" for _, p in products)
    kernel_bytes = (
        args.kernel_bytes or KERNEL_BYTES["add" if adds else "two" if puts else "one"]
    )
    options = {"links": args.links, "kernel_bytes": kernel_bytes}
    options = {key: value for key, value in options.items() if value}
    reports = []
    with km.Session(cards=cards, log=log, **options) as session:
        for dims, place in products:
            reports.append(product_report(session, dims, place, cards, args))
    report = reports[0] if not args.shapes else {"products": reports}
    report["ok"] = all(one["ok"] for one in reports)
    report["load_avg"] = list(Path("/proc/loadavg").read_text().split()[:3])
    print(json.dumps(report, default=float), flush=True)
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
