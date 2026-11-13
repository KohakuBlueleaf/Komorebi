"""The plan of a matrix product on resident workers, as pure functions: which form,
the K chunk and the column block, the buffers per core, and the records.

Sizes are in 32 x 32 tiles of bf16 (2048 bytes)."""

from dataclasses import dataclass
from pathlib import Path

from komorebi import abi

HERE = Path(__file__).resolve().parent
DENSE_H = HERE / "kernels" / "dense.h"
GRID_H = HERE / "kernels" / "grid" / "grid.h"
K = abi.constants(DENSE_H, known=abi.headers().constants)
G = abi.constants(GRID_H, known=abi.headers().constants)

TILE = 32
TILE_BYTES = K["kTileBytes"]
# L1 bytes per core the program's own buffers may take: 1,381,376 ran on the card,
# 1,459,200 was refused when the program was built.
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
        out = [(K["kCbA"], self.a_pages), (K["kCbOut"], self.out_pages)]
        if self.w_pages:
            out.append((K["kCbW"], self.w_pages))
        if self.ws_pages:
            out.append((K["kCbWs"], self.ws_pages))
        return out


def chunk(kt):
    """The largest divisor of kt that is at most MAX_CHUNK."""
    return max(d for d in range(1, min(kt, MAX_CHUNK) + 1) if kt % d == 0)


def block(nt):
    """The largest column block of at most MAX_BLOCK tiles that divides nt."""
    return max(d for d in range(1, MAX_BLOCK + 1) if nt % d == 0)


def plan(mt, kt, nt, budget=L1_BUDGET, form=None):
    """The plan for [mt, kt] @ [kt, nt] tiles: W resident when it fits `budget` with
    the other buffers (or as `form` says), else streamed; None when neither fits."""
    kc, ct = chunk(kt), block(nt)
    item = kt if kc == kt else kc
    resident = Plan(
        mt, kt, nt, K["kFormResident"], kc, ct, 2 * item, kt * nt, 0, 2 * ct
    )
    stream = Plan(mt, kt, nt, K["kFormStream"], kc, ct, 2 * kt, 0, 2 * kc * ct, 2 * ct)
    choices = {K["kFormResident"]: resident, K["kFormStream"]: stream}
    if form is not None:
        p = choices[form]
        return p if p.l1_bytes() <= budget else None
    for p in (resident, stream):
        if p.l1_bytes() <= budget:
            return p
    return None


def team_size(p, workers):
    """Members of a PRODUCT team: one per row tile, at most every worker."""
    return min(p.mt, workers)


def load_args(p, w_addr, cb=K["kCbW"]):
    """Record words a0.. of LOAD into W buffer `cb`."""
    return [w_addr, cb, 0, p.kt, p.nt]


def product_args(p, a_addr, y_addr, w_addr=0, mode=0):
    """Record words a0..a9 of PRODUCT."""
    return [a_addr, y_addr, p.mt, p.kt, p.nt, p.kc, p.ct, mode, p.form, w_addr]


# The grid form. Subblocks sh x sw of at most 4 fp32 tiles (half of DST), the first
# that divides the block; K blocks of the largest of these that fits.
SUBBLOCKS = ((1, 4), (4, 1), (2, 2), (1, 3), (3, 1), (1, 2), (2, 1), (1, 1))
K_BLOCKS = (16, 8, 4, 2, 1)
LANE_BLOCKS_MAX = 32767  # km::kLaneBlocksMax


def cdiv(a, b):
    return -(-a // b)


LAYOUTS = ("2d", "rows", "cols")


@dataclass(frozen=True)
class Grid:
    """[mt, kt] @ [kt, nt] on a rectangle of cores: `rows` x `cols` output bands of
    pcm x pcn tiles in bm x bn blocks, K blocks of kb, lane windows wa and ww. Layout
    "2d": A along the rows, W down the columns; "rows": a row band per core, W down
    the columns; "cols": a column band per core, A along the rows."""

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
        """Blocks each lane hands over in one call."""
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
        gx, gy = grid
        if self.layout == "rows":
            return (0, 0, cdiv(self.rows, gy), gy)
        if self.layout == "cols":
            return (0, 0, gx, cdiv(self.cols, gx))
        return (0, 0, self.cols, self.rows)

    def places(self, grid):
        """{core index: (first row tile, first column tile)} of every rect core."""
        gx, gy = grid
        _, _, nx, ny = self.rect(grid)
        out = {}
        for y in range(ny):
            for x in range(nx):
                if self.layout == "rows":
                    band = (x * gy + y, 0)
                elif self.layout == "cols":
                    band = (0, y * gx + x)
                else:
                    band = (y, x)
                out[y * gx + x] = (band[0] * self.pcm, band[1] * self.pcn)
        return out

    def team(self, grid):
        """(first, count): the smallest range of the core order covering the rect."""
        _, _, nx, ny = self.rect(grid)
        return 0, (ny - 1) * grid[0] + nx

    def check(self, grid, budget=L1_BUDGET):
        """Raises ValueError when the plan does not fit the grid, L1 or the lanes."""
        gx, gy = grid
        _, _, nx, ny = self.rect(grid)
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout {self.layout!r}")
        if self.layout == "rows" and (self.cols != 1 or self.rows % gy):
            raise ValueError("layout rows: one column band, whole columns of cores")
        if self.layout == "cols" and (self.rows != 1 or self.cols % gx):
            raise ValueError("layout cols: one row band, whole rows of cores")
        if not (1 <= ny <= gy and 1 <= nx <= gx):
            raise ValueError(f"{ny} x {nx} cores on a {gx} x {gy} grid")
        if self.rows * self.pcm < self.mt or self.cols * self.pcn < self.nt:
            raise ValueError("the bands do not cover the output")
        if self.bm % self.sh or self.bn % self.sw or self.sh * self.sw > 4:
            raise ValueError(f"subblock {self.sh} x {self.sw} of {self.bm} x {self.bn}")
        if self.blocks() > LANE_BLOCKS_MAX:
            raise ValueError(f"{self.blocks()} lane blocks in one call")
        if self.l1_bytes() > budget:
            raise ValueError(f"{self.l1_bytes()} bytes of L1 > {budget}")


def subblock(bm, bn):
    return next((h, w) for h, w in SUBBLOCKS if bm % h == 0 and bn % w == 0)


def layout_of(mt, nt, grid):
    """The rule's layout: "cols" for fewer row tiles than grid rows; "rows" when a
    band of rows per core reuses W more than 2d reuses either operand; else "2d"."""
    gx, gy = grid
    if mt < gy:
        return "cols"
    rows1 = cdiv(mt, gx * gy)
    return "rows" if rows1 >= 3 and rows1 > min(cdiv(mt, gy), cdiv(nt, gx)) else "2d"


def bands(mt, nt, layout, grid):
    """(rows, cols, pcm, pcn) of a layout: bands over the grid, whole lines of cores
    for "rows" and "cols"; 2d column bands rounded up to whole subblocks of 4."""
    gx, gy = grid
    if layout == "rows":
        pcm = cdiv(mt, gx * gy)
        return cdiv(cdiv(mt, pcm), gy) * gy, 1, pcm, nt
    if layout == "cols":
        pcn = cdiv(nt, gx * gy)
        return 1, cdiv(cdiv(nt, pcn), gx) * gx, mt, pcn
    pcm = cdiv(mt, min(gy, mt))
    pcn = cdiv(nt, min(gx, nt))
    if pcn > 2 and pcn % 4:
        pcn = min(cdiv(pcn, 4) * 4, nt)
    return cdiv(mt, pcm), cdiv(nt, pcn), pcm, pcn


def grid_plan(mt, kt, nt, grid=(12, 10), budget=L1_BUDGET, layout=None):
    """The rule for the grid form: the layout, its bands, then the largest blocks that
    fit L1 (2d and cols halve rows first, rows halves columns first, starting from two
    column blocks when nt >= 8), then the largest K block. None when nothing fits."""
    layout = layout or layout_of(mt, nt, grid)
    rows, cols, pcm, pcn = bands(mt, nt, layout, grid)
    bm, bn = pcm, pcn
    if layout == "rows" and pcn >= 8:
        bn = cdiv(pcn, 2)
    while bm >= 1 and bn >= 1:
        sh, sw = subblock(bm, bn)
        for kb in K_BLOCKS:
            if kb > kt and kb > 1:
                continue
            g = Grid(
                mt, kt, nt, rows, cols, pcm, pcn, bm, bn, kb, sh, sw, layout=layout
            )
            try:
                g.check(grid, budget)
            except ValueError:
                continue
            return g
        if layout == "rows" and bn > 1:
            bn = cdiv(bn, 2)
        else:
            bm = cdiv(bm, 2) if bm > 1 else 0
    return None


def grid_args(g, a_addr, w_addr, y_addr, mode=0):
    """Record words a0..a3 of GRID."""
    return [a_addr, w_addr, y_addr, mode]


def flops(p):
    """Floating-point operations of one product (multiply and add counted)."""
    return 2 * p.mt * p.kt * p.nt * TILE**3


def tile_products(p):
    return p.mt * p.kt * p.nt
