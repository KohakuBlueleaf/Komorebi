"""The plan of a matrix product on persistent kernels, as pure functions: which form,
the K chunk and the column block, the buffers per core, and the commands.

Sizes are in 32 x 32 tiles of bf16 (2048 bytes)."""

from dataclasses import dataclass
from pathlib import Path

from komorebi import abi

HERE = Path(__file__).resolve().parent
DENSE_H = HERE / "kernels" / "dense.h"
GRID_H = HERE / "kernels" / "grid" / "grid.h"
SPLIT_H = HERE / "kernels" / "split" / "split.h"
K = abi.constants(DENSE_H, known=abi.headers().constants)
G = abi.constants(GRID_H, known=abi.headers().constants)
S = abi.constants(SPLIT_H, known=abi.headers().constants)

TILE = 32
TILE_BYTES = K["kTileBytes"]
# L1 bytes per core the program's own buffers may take: 1,381,376 ran on the card,
# 1,459,200 was rejected when the program was built.
L1_BUDGET = 1_380_000
MAX_CHUNK = 32  # K tiles per chunk, as Kohaku-TT's matmul
MAX_BLOCK = 4  # output tiles per block: half of DST in fp32
# The benchmark shapes (M, K, N), from the workloads of the tensor library compared.
SHAPES = {
    "t1": (32, 32, 32),
    "t2": (64, 128, 256),
    "t3": (256, 1024, 128),
    "kvproj": (128, 896, 128),
    "lin384": (16384, 384, 384),
    "sq128": (128, 128, 128),
    "sq256": (256, 256, 256),
    "sq512": (512, 512, 512),
    "sq1024": (1024, 1024, 1024),
    "sq2048": (2048, 2048, 2048),
    "sq4096": (4096, 4096, 4096),
    "lin1024": (16384, 1024, 1024),
    "sq8192": (8192, 8192, 8192),
    "sq16384": (16384, 16384, 16384),
    "dec4096": (32, 4096, 4096),
}


@dataclass(frozen=True)
class Plan:
    """How [mt, kt] @ [kt, nt] runs: form, K chunk, column block, buffer pages."""

    mt: int
    kt: int
    nt: int
    form: int  # K["kFormResident"] or K["kFormStream"]
    kc: int
    ct: int
    a_pages: int
    w_pages: int  # resident W (0 when streamed)
    ws_pages: int  # streamed W chunks (0 when resident)
    out_pages: int

    @property
    def resident(self):
        return self.form == K["kFormResident"]

    @property
    def row_item(self):
        """A's whole row is one buffer item (one chunk, or the streamed form)."""
        return not self.resident or self.kc == self.kt

    def l1_bytes(self):
        pages = self.a_pages + self.w_pages + self.ws_pages + self.out_pages
        return pages * TILE_BYTES + K["kProfBytes"]

    def buffers(self):
        """(index, pages) of the program's buffers."""
        buffers = [(K["kCbA"], self.a_pages), (K["kCbOut"], self.out_pages)]
        if self.w_pages:
            buffers.append((K["kCbW"], self.w_pages))
        if self.ws_pages:
            buffers.append((K["kCbWs"], self.ws_pages))
        return buffers


def chunk(k_tiles):
    """The largest divisor of k_tiles that is at most MAX_CHUNK."""
    return max(
        divisor
        for divisor in range(1, min(k_tiles, MAX_CHUNK) + 1)
        if k_tiles % divisor == 0
    )


def block(n_tiles):
    """The largest column block of at most MAX_BLOCK tiles that divides n_tiles."""
    return max(divisor for divisor in range(1, MAX_BLOCK + 1) if n_tiles % divisor == 0)


def plan(m_tiles, k_tiles, n_tiles, budget=L1_BUDGET, form=None):
    """The plan for [m_tiles, k_tiles] @ [k_tiles, n_tiles] tiles: W resident when it
    fits `budget` with the other buffers (or as `form` says), else streamed; None when
    neither fits."""
    chunk_tiles, block_width = chunk(k_tiles), block(n_tiles)
    item = k_tiles if chunk_tiles == k_tiles else chunk_tiles
    resident = Plan(
        m_tiles,
        k_tiles,
        n_tiles,
        K["kFormResident"],
        chunk_tiles,
        block_width,
        2 * item,
        k_tiles * n_tiles,
        0,
        2 * block_width,
    )
    stream = Plan(
        m_tiles,
        k_tiles,
        n_tiles,
        K["kFormStream"],
        chunk_tiles,
        block_width,
        2 * k_tiles,
        0,
        2 * chunk_tiles * block_width,
        2 * block_width,
    )
    choices = {K["kFormResident"]: resident, K["kFormStream"]: stream}
    if form is not None:
        chosen = choices[form]
        return chosen if chosen.l1_bytes() <= budget else None
    for candidate in (resident, stream):
        if candidate.l1_bytes() <= budget:
            return candidate
    return None


def group_size(plan, workers):
    """Cores of a PRODUCT group: one per row tile, at most every worker."""
    return min(plan.mt, workers)


def load_args(plan, w_address, cb=K["kCbW"]):
    """Command words a0.. of LOAD into W buffer `cb`."""
    return [w_address, cb, 0, plan.kt, plan.nt]


def product_args(plan, a_address, y_address, w_address=0, mode=0):
    """Command words a0..a9 of PRODUCT."""
    return [
        a_address,
        y_address,
        plan.mt,
        plan.kt,
        plan.nt,
        plan.kc,
        plan.ct,
        mode,
        plan.form,
        w_address,
    ]


# The grid form. Subblocks sh x sw of at most 4 fp32 tiles (half of DST), the first
# that divides the block; K blocks of the largest of these that fits.
SUBBLOCKS = ((1, 4), (4, 1), (2, 2), (1, 3), (3, 1), (1, 2), (2, 1), (1, 1))
K_BLOCKS = (16, 8, 4, 2, 1)
BROADCAST_BLOCKS_MAX = 32767  # km::kBroadcastBlocksMax


def cdiv(numerator, denominator):
    return -(-numerator // denominator)


LAYOUTS = ("2d", "rows", "cols")


@dataclass(frozen=True)
class Grid:
    """[mt, kt] @ [kt, nt] on a rectangle of cores: `rows` x `cols` output bands of pcm
    x pcn tiles in bm x bn blocks, K blocks of kb, broadcast windows wa and ww. Layout
    "2d": A along the rows, W down the columns; "rows": a row band per core, W down the
    columns; "cols": a column band per core, A along the rows."""

    mt: int
    kt: int
    nt: int
    rows: int
    cols: int
    pcm: int
    pcn: int
    bm: int
    bn: int
    kb: int
    sh: int
    sw: int
    wa: int = 2
    ww: int = 2
    layout: str = "2d"
    # the linear kernels' reads of W and of A spread over a column's or a row's ranks
    w_spread: bool = False
    x_spread: bool = False

    @property
    def nk(self):
        return cdiv(self.kt, self.kb)

    @property
    def kb_last(self):
        return self.kt - (self.nk - 1) * self.kb

    @property
    def nbh(self):
        return cdiv(self.pcm, self.bm)

    @property
    def nbw(self):
        return cdiv(self.pcn, self.bn)

    def blocks(self):
        """Blocks each broadcast delivers in one call."""
        return self.nbh * self.nbw * self.nk

    def words(self):
        """The plan words of the kernels' arguments, in grid.h's order."""
        return [
            self.mt,
            self.kt,
            self.nt,
            self.pcm,
            self.pcn,
            self.bm,
            self.bn,
            self.kb,
            self.nk,
            self.kb_last,
            self.nbh,
            self.nbw,
            self.sh,
            self.sw,
        ]

    def buffers(self):
        """(index, format, page bytes, pages) of the program's buffers."""
        part = self.bm * self.bn if self.nk > 1 else 1
        return [
            (G["kCbRow"], "bfloat16", TILE_BYTES, self.wa * self.bm * self.kb),
            (G["kCbCol"], "bfloat16", TILE_BYTES, self.ww * self.kb * self.bn),
            (G["kCbGridOut"], "bfloat16", TILE_BYTES, 2 * self.sh * self.sw),
            (G["kCbPart"], "float32", G["kPartBytes"], part),
            (G["kCbGridProf"], "uint32", K["kProfBytes"], 1),
        ]

    def l1_bytes(self):
        return sum(page * pages for _, _, page, pages in self.buffers())

    def rect(self, grid):
        """The rectangle of cores (x0, y0, columns, rows) the bands lie on."""
        grid_columns, grid_rows = grid
        if self.layout == "rows":
            return (0, 0, cdiv(self.rows, grid_rows), grid_rows)
        if self.layout == "cols":
            return (0, 0, grid_columns, cdiv(self.cols, grid_columns))
        return (0, 0, self.cols, self.rows)

    def places(self, grid):
        """{core index: (first row tile, first column tile)} of every rect core."""
        grid_columns, grid_rows = grid
        _, _, core_columns, core_rows = self.rect(grid)
        placed = {}
        for y in range(core_rows):
            for x in range(core_columns):
                if self.layout == "rows":
                    band = (x * grid_rows + y, 0)
                elif self.layout == "cols":
                    band = (0, y * grid_columns + x)
                else:
                    band = (y, x)
                placed[y * grid_columns + x] = (band[0] * self.pcm, band[1] * self.pcn)
        return placed

    def group(self, grid):
        """(first, count): the smallest range of the core order covering the rect."""
        _, _, core_columns, core_rows = self.rect(grid)
        return 0, (core_rows - 1) * grid[0] + core_columns

    def check(self, grid, budget=L1_BUDGET, ranks=None):
        """Raises ValueError when the plan does not fit the grid, L1 or the
        broadcasts, or (`ranks`: an island's) its group reaches past the ranks."""
        grid_columns, grid_rows = grid
        _, _, core_columns, core_rows = self.rect(grid)
        if ranks is not None and self.group(grid)[1] > ranks:
            raise ValueError(f"its group {self.group(grid)} reaches past {ranks} ranks")
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout {self.layout!r}")
        if self.layout == "rows" and (self.cols != 1 or self.rows % grid_rows):
            raise ValueError("layout rows: one column band, whole columns of cores")
        if self.layout == "cols" and (self.rows != 1 or self.cols % grid_columns):
            raise ValueError("layout cols: one row band, whole rows of cores")
        if not (1 <= core_rows <= grid_rows and 1 <= core_columns <= grid_columns):
            raise ValueError(
                f"{core_rows} x {core_columns} cores on a "
                f"{grid_columns} x {grid_rows} grid"
            )
        if self.rows * self.pcm < self.mt or self.cols * self.pcn < self.nt:
            raise ValueError("the bands do not cover the output")
        if self.bm % self.sh or self.bn % self.sw or self.sh * self.sw > 4:
            raise ValueError(f"subblock {self.sh} x {self.sw} of {self.bm} x {self.bn}")
        if self.blocks() > BROADCAST_BLOCKS_MAX:
            raise ValueError(f"{self.blocks()} broadcast blocks in one call")
        if self.l1_bytes() > budget:
            raise ValueError(f"{self.l1_bytes()} bytes of L1 > {budget}")


def subblock(block_height, block_width):
    return next(
        (height, width)
        for height, width in SUBBLOCKS
        if block_height % height == 0 and block_width % width == 0
    )


def layout_of(m_tiles, n_tiles, grid):
    """The rule's layout: "cols" for fewer row tiles than grid rows; "rows" when a
    band of rows per core reuses W more than 2d reuses either operand; else "2d"."""
    grid_columns, grid_rows = grid
    if m_tiles < grid_rows:
        return "cols"
    rows_per_core = cdiv(m_tiles, grid_columns * grid_rows)
    return (
        "rows"
        if rows_per_core >= 3
        and rows_per_core > min(cdiv(m_tiles, grid_rows), cdiv(n_tiles, grid_columns))
        else "2d"
    )


def bands(m_tiles, n_tiles, layout, grid):
    """(rows, cols, pcm, pcn) of a layout: bands over the grid, whole lines of cores
    for "rows" and "cols"; 2d column bands rounded up to whole subblocks of 4."""
    grid_columns, grid_rows = grid
    if layout == "rows":
        band_height = cdiv(m_tiles, grid_columns * grid_rows)
        return (
            cdiv(cdiv(m_tiles, band_height), grid_rows) * grid_rows,
            1,
            band_height,
            n_tiles,
        )
    if layout == "cols":
        band_width = cdiv(n_tiles, grid_columns * grid_rows)
        return (
            1,
            cdiv(cdiv(n_tiles, band_width), grid_columns) * grid_columns,
            m_tiles,
            band_width,
        )
    band_height = cdiv(m_tiles, min(grid_rows, m_tiles))
    band_width = cdiv(n_tiles, min(grid_columns, n_tiles))
    if band_width > 2 and band_width % 4:
        band_width = min(cdiv(band_width, 4) * 4, n_tiles)
    return (
        cdiv(m_tiles, band_height),
        cdiv(n_tiles, band_width),
        band_height,
        band_width,
    )


def grid_plan(
    m_tiles, k_tiles, n_tiles, grid=(12, 10), budget=L1_BUDGET, layout=None, ranks=None
):
    """The rule for the grid form: the layout, its bands, then the largest blocks that
    fit L1 (2d and cols halve rows first, rows halves columns first, starting from two
    column blocks when n_tiles >= 8), then the largest K block. None when nothing
    fits; `ranks` (an island's): its group within them, else the next layout tried."""
    if ranks is not None and layout is None:
        first = layout_of(m_tiles, n_tiles, grid)
        for each in (first, *(other for other in LAYOUTS if other != first)):
            found = grid_plan(m_tiles, k_tiles, n_tiles, grid, budget, each, ranks)
            if found is not None:
                return found
        return None
    layout = layout or layout_of(m_tiles, n_tiles, grid)
    rows, cols, band_height, band_width = bands(m_tiles, n_tiles, layout, grid)
    block_height, block_width = band_height, band_width
    if layout == "rows" and band_width >= 8:
        block_width = cdiv(band_width, 2)
    while block_height >= 1 and block_width >= 1:
        subblock_height, subblock_width = subblock(block_height, block_width)
        for k_block_tiles in K_BLOCKS:
            if k_block_tiles > k_tiles and k_block_tiles > 1:
                continue
            candidate = Grid(
                m_tiles,
                k_tiles,
                n_tiles,
                rows,
                cols,
                band_height,
                band_width,
                block_height,
                block_width,
                k_block_tiles,
                subblock_height,
                subblock_width,
                layout=layout,
            )
            try:
                candidate.check(grid, budget, ranks)
            except ValueError:
                continue
            return candidate
        if layout == "rows" and block_width > 1:
            block_width = cdiv(block_width, 2)
        else:
            block_height = cdiv(block_height, 2) if block_height > 1 else 0
    return None


REDUCE_FORMS = {"owner": 0, "tree": 1}
PATTERNS = {"AB": S["kSplitAB"], "AAAB": S["kSplitAAAB"]}  # the chunks' movers


@dataclass(frozen=True)
class Split:
    """[mt, kt] @ [kt, nt] with K split: groups of s cores, an oh x ow block a group,
    each core 1/s of K in chunks of kc (pattern; depth, depth_b: mover A's and B's W
    halves), parts summed by owner or tree. Subblock sh x sw; arrive: arrival pages (0:
    either form's). cast: the whole-K plan (s 1), X multicast once a panel."""

    mt: int
    kt: int
    nt: int
    s: int
    oh: int
    ow: int
    kc: int
    sh: int
    sw: int
    arrive: int = 0
    pattern: str = "AB"
    depth: int = 1
    cast: bool = False
    depth_b: int = 1

    def nbw(self):
        return cdiv(self.nt, self.ow)

    def groups(self):
        return cdiv(self.mt, self.oh) * self.nbw()

    def cores(self):
        return self.groups() * self.s

    def arrive_pages(self):
        """Arrival slots of P tiles for either form: s - 1 (owner) covers the tree's;
        or the plan's own count."""
        return self.arrive or max(1, self.s - 1) * self.oh * self.ow

    def words(self):
        """The kernels' arguments, in split.h's order."""
        return [
            self.mt,
            self.kt,
            self.nt,
            self.s,
            self.oh,
            self.ow,
            self.kc,
            self.sh,
            self.sw,
            self.nbw(),
            self.arrive_pages(),
            PATTERNS[self.pattern],
        ]

    def buffers(self):
        """(index, format, page bytes, pages) of the program's buffers."""
        block_tiles = self.oh * self.ow
        return [
            (S["kCbSa"], "bfloat16", TILE_BYTES, self.depth * self.oh * self.kc),
            (S["kCbSaMoverB"], "bfloat16", TILE_BYTES, self.oh * self.kc),
            (S["kCbSw"], "bfloat16", TILE_BYTES, self.depth * self.kc * self.ow),
            (
                S["kCbSwMoverB"],
                "bfloat16",
                TILE_BYTES,
                self.depth_b * self.kc * self.ow,
            ),
            (S["kCbArrive"], "float32", S["kSumBytes"], self.arrive_pages()),
            (S["kCbSum"], "float32", S["kSumBytes"], block_tiles),
            (S["kCbSplitOut"], "bfloat16", TILE_BYTES, block_tiles),
        ]

    def l1_bytes(self):
        return sum(page * pages for _, _, page, pages in self.buffers())

    def check(self, cores, budget=L1_BUDGET):
        if not 1 <= self.s <= self.kt:
            raise ValueError(f"group of {self.s} for {self.kt} K tiles")
        if self.cores() > cores:
            raise ValueError(f"{self.cores()} cores > {cores}")
        if self.oh % self.sh or self.ow % self.sw or self.sh * self.sw > 4:
            raise ValueError(f"subblock {self.sh} x {self.sw} of {self.oh} x {self.ow}")
        if self.l1_bytes() > budget:
            raise ValueError(f"{self.l1_bytes()} bytes of L1 > {budget}")
        if self.pattern not in PATTERNS or min(self.depth, self.depth_b) < 1:
            raise ValueError(f"pattern {self.pattern!r}, depth {self.depth}")
        if self.arrive % (self.oh * self.ow):
            raise ValueError(f"{self.arrive} arrival pages: not whole slots")


def split_plan(
    m_tiles,
    k_tiles,
    n_tiles,
    group_cores,
    block_height,
    block_width,
    chunk_tiles=None,
    cores=120,
    budget=L1_BUDGET,
    arrive=0,
    pattern="AB",
    depth=1,
    depth_b=1,
):
    """A K-split plan: group_cores cores a group, output block block_height x
    block_width, K chunk chunk_tiles (default: the whole partition of K, at most
    MAX_CHUNK), arrive arrival pages (0: enough for either form), the chunks' pattern
    and the movers' W halves in chunks (depth A's, depth_b B's)."""
    chunk_tiles = chunk_tiles or min(cdiv(k_tiles, group_cores), MAX_CHUNK)
    subblock_height, subblock_width = subblock(block_height, block_width)
    plan = Split(
        m_tiles,
        k_tiles,
        n_tiles,
        group_cores,
        block_height,
        block_width,
        chunk_tiles,
        subblock_height,
        subblock_width,
        arrive,
        pattern,
        depth,
        depth_b=depth_b,
    )
    plan.check(cores, budget)
    return plan


def split_table(plans):
    """The kernels' arguments of several plans that one loaded program takes, plan p
    at word p * kSplitWords (a command's a5 picks it); they share every buffer."""
    plans = list(plans)
    if not 1 <= len(plans) <= S["kSplitPlans"]:
        raise ValueError(f"{len(plans)} plans: 1 to {S['kSplitPlans']} a program")
    if any(plan.buffers() != plans[0].buffers() for plan in plans):
        raise ValueError("the plans of one program share their buffers")
    return [word for plan in plans for word in plan.words()]


def split_args(
    plan, a_address, w_address, y_address, form="owner", mode=0, index=0, plans=1
):
    """Command words a0..a4 of SPLIT, and a5 the plan's index in a table of
    `plans` when it is not 0."""
    if not 0 <= index < plans:
        raise ValueError(f"plan {index} of a table of {plans}")
    words = [a_address, w_address, y_address, REDUCE_FORMS[form], mode]
    return words + [index] if index else words


def grid_args(plan, a_address, w_address, y_address, mode=0):
    """Command words a0..a3 of GRID."""
    return [a_address, w_address, y_address, mode]


def flops(plan):
    """Floating-point operations of one product (multiply and add counted)."""
    return 2 * plan.mt * plan.kt * plan.nt * TILE**3


def tile_products(plan):
    return plan.mt * plan.kt * plan.nt
