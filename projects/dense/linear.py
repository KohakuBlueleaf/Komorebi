"""A linear layer Y = X x W written for islands (kernels/linear): one command a call on
every rank; X and W split along K alike, Y along N, by hints. Each island puts the
columns others keep and adds what they put. Prints one JSON line; exit 1: a failed
check."""

import argparse
import contextlib
import dataclasses
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
# Two cards, the rows the load's L1 budget adds to the buffers (its table at a load):
# what the session keeps in every core (link area, relay, tensor table: known only
# once a load made them), the host transfer buffer, the control block and overlay
# area, the reader's slot.
L1_HELD = 10_752 + 37_440 + 1_536
L1_TRANSFER = 32_848
L1_RUNTIME = 17_024 + 21_568
# The kernel buffer the linear kernels' code takes (all five CPUs): one card 83,008 B;
# two cards (a rank's kernels: the page calls' client side) 94,224 B.
KERNEL_BYTES = {"one": 88 * 1024, "two": 96 * 1024}
log = run.log


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


def chunk(plan):
    """Tiles a put: the smallest whole number of subblocks at or over kPutTiles."""
    subblock = plan.sh * plan.sw
    return pl.cdiv(L["kPutTiles"], subblock) * subblock


def extra_buffers(plan, islands):
    """The linear kernels' buffers past the grid plan's: two chunks of the tiles put
    across (a subblock on one island, where nothing is put), another island's subblock
    put here, the token and its slot."""
    subblock = plan.sh * plan.sw
    send = 2 * chunk(plan) if islands > 1 else subblock
    return [
        (L["kCbSend"], "bfloat16", L["kBf16Bytes"], send),
        (L["kCbSent"], "bfloat16", L["kBf16Bytes"], 2 * subblock),
        (L["kCbToken"], "uint32", L["kSignalBytes"], 1),
        (L["kCbArrive"], "uint32", L["kSignalBytes"], 1),
        (L["kCbStamp"], "uint32", L["kStampBytes"], 1),
    ]


def extra_bytes(plan, islands):
    return sum(page * pages for _, _, page, pages in extra_buffers(plan, islands))


def l1_budget(session, islands):
    """L1 for the buffers: the project's budget on one island; on more, the L1 the
    session gives buffers (Session.l1) less the other rows of the load's budget, and
    the two tokens it counts with the buffers."""
    if islands == 1:
        return pl.L1_BUDGET
    free = session.sessions[0].l1()[1]
    tokens = 2 * abi.headers()["kTokenBytes"]
    return min(pl.L1_BUDGET, free - L1_HELD - L1_TRANSFER - L1_RUNTIME - tokens)


def fitting_plan(m_tiles, k_tiles, n_tiles, grid, ranks, budget, islands):
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
            over = plan.l1_bytes() + extra_bytes(plan, islands) - budget
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


# ns a tile product a core: the compute's own, or the broadcasts' feed over the block's
# intensity bm bn / (bm + bn); fitted to the one-card plans measured
TILE_NS = 37
FED_NS = 225


def block_ns(bm, bn):
    return max(TILE_NS, FED_NS * (bm + bn) / (bm * bn))


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
    m_tiles, k_tiles, n_tiles, grid, ranks, budget, islands, kb, every=False
):
    """The pieces of the island's ranks: the old single rect (fitting_plan), or the
    whole grid, or two rectangles sharing one block, in bands of whole blocks: the
    cheapest by block_ns (then the larger block, the larger K block). The K block is
    `kb`, lower only when nothing fits; `every`: any K block up to `kb`."""
    single = fitting_plan(m_tiles, k_tiles, n_tiles, grid, ranks, budget, islands)
    best = None
    if single is not None:
        blocks = single.nbh * single.nbw
        cost = blocks * single.bm * single.bn * block_ns(single.bm, single.bn)
        best = ((cost, -single.bm * single.bn, -single.kb), [Piece(single)])
    rects = two_rects(grid, ranks)
    if not rects or single is None or single.layout != "2d":
        return None if best is None else best[1]
    for k_block in [value for value in pl.K_BLOCKS if value <= kb] or [1]:
        found = False
        for bm in range(1, m_tiles + 1):
            for bn in range(1, n_tiles + 1):
                sh, sw = pl.subblock(bm, bn)
                if sh * sw < 4 or bm * bn * G["kPartBytes"] > budget:
                    continue
                probe = piece_plan(bm, k_tiles, bn, (1, 1), (bm, bn, k_block, 1, 1))
                if probe is None:
                    continue
                if probe.l1_bytes() + extra_bytes(probe, islands) > budget:
                    continue
                for form, (a, b) in rects.items():
                    for pieces in split_pieces(
                        form, a, b, m_tiles, k_tiles, n_tiles, (bm, bn, k_block)
                    ):
                        found = True
                        blocks = max(p.plan.nbh * p.plan.nbw for p in pieces)
                        cost = (
                            blocks * bm * bn * block_ns(bm, bn),
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


def pieces_args(pieces, nocs, ranks, m_tiles, group_word):
    """Each core's argument function: the broadcast words of its piece (row broadcast
    on NoC0, column broadcast on NoC1, `group_word` the command's), then its core
    words (in a piece, first row and column, its first page of a set of S, a set's
    pages, plan words with M whole); compute: core words, then the broadcast windows."""
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
                ("row", 0, "kBroadcastRow", "kCbRow", plan.wa, plan.bm * plan.kb),
                ("column", 1, "kBroadcastCol", "kCbCol", plan.ww, plan.kb * plan.bn),
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
    idle = [0] * (len(card.BROADCAST_FIELDS) + L["kCoreWords"])

    def args(core):
        if core not in words:
            return idle, idle, idle[: L["kCoreWords"]] + list(windows)
        (row, column), mine, window = words[core]
        return row + mine, column + mine, mine + list(window)

    return args


class Linear:
    """The linear kernels loaded on every island of `session`: one LINEAR command a
    call on all its ranks, exclusive (the broadcasts inside each island)."""

    def __init__(self, session, dimensions, tensors, wait=2.0, deadline=5.0):
        graph = session.graph
        m_tiles, k_tiles, n_tiles = (value // pl.TILE for value in dimensions)
        self.k_tiles = k_tiles
        counts = {graph.count(i) for i in range(len(graph.islands))}
        if len(counts) != 1:
            raise SystemExit(f"one program needs islands of one size: {counts}")
        ranks = counts.pop()
        self.islands = len(graph.islands)
        self.nocs = session.sessions[0].nocs()
        self.n_tiles = n_tiles
        self.k_parts = spans(tensors["w"].placement, 0)
        self.columns = spans(tensors["y"].placement, 1)
        if not all(count for _, count in self.k_parts):
            raise SystemExit(f"an island holds no K of W: {self.k_parts}")
        part_k = max(count for _, count in self.k_parts)
        part_n = max(count for _, count in self.columns)
        one_card = pl.grid_plan(m_tiles, k_tiles, n_tiles, grid=self.nocs.grid)
        self.pieces = island_pieces(
            m_tiles,
            part_k,
            part_n,
            self.nocs.grid,
            ranks,
            l1_budget(session, self.islands),
            self.islands,
            # one island: any K block the model likes; more: the one-card plan's
            pl.K_BLOCKS[0] if self.islands == 1 or one_card is None else one_card.kb,
            every=self.islands == 1,
        )
        if self.pieces is None:
            raise SystemExit(f"no grid plan fits L1 and {ranks} ranks: {dimensions}")
        self.plan = self.pieces[0].plan  # the block, K block and buffers of all
        for name in ("kCbRow", "kCbCol", "kCbPart"):
            assert L[name] == G[name], name
        assert L["kCbYOut"] == G["kCbGridOut"]
        pages = max(1, self.islands - 1) * sent_pages(self.pieces, self.nocs.grid)
        tensors["s"] = session.empty(
            (pages * pl.TILE, pl.TILE), "bfloat16", "tile", copy=True
        )
        tensors["t"] = session.empty(
            (ranks, L["kStampBytes"] // 4), "uint32", "row", copy=True
        )
        self.tensors = tensors
        self.session = session
        self.group = (0, session.ranks)
        x, w, y, s = (tensors[name] for name in "xwys")
        self.program = km.Program(
            "dense_linear",
            kernels=KERNELS,
            buffers=[
                km.Buffer(*buffer)
                for buffer in self.plan.buffers()[:4]
                + extra_buffers(self.plan, self.islands)
            ],
            args=pieces_args(
                self.pieces,
                self.nocs,
                ranks,
                m_tiles,
                self.group[0] | self.group[1] << 16,
            ),
            compile_args={
                "reader": x.accessor_args() + s.accessor_args(),
                "writer": w.accessor_args()
                + y.accessor_args()
                + tensors["t"].accessor_args(),
                "compute": [],
            },
            compute={
                "math_fidelity": "HiFi2",
                "fp32_dest_acc_en": True,
                "unpack_to_dest_mode": {L["kCbPart"]: "UnpackToDestFp32"},
            },
            handlers=LINEAR_H,
        )
        self.run = session.load(self.program, deadline=deadline, wait=wait)
        # what each island's session keeps in every core, by its own count (the
        # plan's budget took L1_HELD before the load made it)
        self.l1_held = [
            sum(size for _, size in one.l1_held()) for one in session.sessions
        ]
        if self.islands > 1 and max(self.l1_held) > L1_HELD:
            log(f"the session keeps {self.l1_held} B a core, the plan took {L1_HELD}")

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
            stamped = ran[:, L["kStampPut"]].all()
            # the 32-bit clock as signed cycles from the first core's start
            ran = (ran - ran[0, L["kStampStart"]] + 2**31) % 2**32 - 2**31
            start = ran[:, L["kStampStart"]].min()
            put = ran[:, L["kStampPut"]].max() if stamped else start
            synced = ran[:, L["kStampSynced"]].max() if put != start else start
            end = ran[:, L["kStampEnd"]].max()
            cycles = {"put": put - start, "sync": synced - put, "own": end - synced}
            if stamped:  # inside the sync: the last core's each step, from the last put
                at = {
                    name: ran[:, L[stamp]].max()
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
                waits = ran[:, L["kStampSettled"]] - ran[:, L["kStampPut"]]
                cycles["settle_median"] = int(np.median(waits))
            out.append(
                {
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

    def single(self, commands):
        """`commands` calls one at a time: the slowest part's microseconds each."""
        return np.concatenate(
            [self.calls(1, timeout=10.0)[1].max(axis=1) for _ in range(commands)]
        )

    def crossed(self):
        """Bytes a call should cross, from the placements and the pieces: each island
        holding part of K puts, for every other island keeping columns, each core's
        output blocks that hold some of them (whole blocks, bf16); and the ideal: every
        other holder's M x columns tiles."""
        plan = self.plan
        page = self.tensors["s"].aligned_page_bytes
        total = {"blocks": 0, "tiles": 0}
        for sender, (_, k_part) in enumerate(self.k_parts):
            for to, (_, width) in enumerate(self.columns):
                if to == sender or not k_part or not width:
                    continue
                total["tiles"] += plan.mt * width * page
                for piece in self.pieces:
                    total["blocks"] += piece_blocks(piece, self.nocs.grid, width) * (
                        plan.bm * plan.bn * page
                    )
        total["put_bytes"] = chunk(plan) * page
        return total

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
    got = card.read_rows(session, tensors["y"], rows, log)
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
        report["l1_held"] = product.l1_held
        report["crossed_bytes"] = product.crossed()
    log(f"{product_class.__name__} loaded, plan {report['plan']}")
    try:
        report["loop"] = measure(product, cards, args.seconds)
    except km.KomorebiError as error:
        report["error"] = f"{type(error).__name__}: {error}"
        with contextlib.suppress(km.KomorebiError):
            product.run.stop()
        report["ok"] = False
        return report
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
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    dimensions = (args.cube,) * 3 if args.cube else run.SHAPES[args.shape]
    if any(value % pl.TILE for value in dimensions):
        raise SystemExit(f"M, K and N are whole tiles of {pl.TILE}: {dimensions}")
    cards = [int(value) for value in args.cards.split(",")]
    report = {"m": dimensions[0], "k": dimensions[1], "n": dimensions[2]}
    report |= {"cards": cards, "links": args.links}
    log("building the inputs")
    x_matrix, w_matrix = ref.inputs(*dimensions, kind=args.inputs)
    log("inputs built")
    kernel_bytes = args.kernel_bytes or KERNEL_BYTES["two" if len(cards) > 1 else "one"]
    options = {"links": args.links, "kernel_bytes": kernel_bytes}
    options = {key: value for key, value in options.items() if value}
    with km.Session(cards=cards, log=log, **options) as session:
        started = time.perf_counter()
        m_rows, _, n_columns = dimensions
        # X and W split along K alike, Y along N, each island half (S, the tiles
        # other islands put, made with the kernels)
        halves = {"group": even_group(session.graph)}
        tensors = {
            "x": session.tensor(x_matrix, "bfloat16", "tile", split=1, **halves),
            "w": session.tensor(w_matrix, "bfloat16", "tile", split=0, **halves),
            "y": session.empty(
                (m_rows, n_columns), "bfloat16", "tile", split=1, **halves
            ),
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
            report["grid"] = attempt(
                Grid, session, dimensions, tensors, cards, args, matrices
            )
    report["ok"] = report["linear"]["ok"] and report.get("grid", {}).get("ok", True)
    report["load_avg"] = list(Path("/proc/loadavg").read_text().split()[:3])
    print(json.dumps(report, default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
