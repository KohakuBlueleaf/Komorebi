"""The plan of a matrix product on resident workers, as pure functions: which form,
the K chunk and the column block, the buffers per core, and the records.

Sizes are in 32 x 32 tiles of bf16 (2048 bytes)."""

from dataclasses import dataclass
from pathlib import Path

from komorebi import abi

HERE = Path(__file__).resolve().parent
DENSE_H = HERE / "kernels" / "dense.h"
K = abi.constants(DENSE_H, known=abi.headers().constants)

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


def flops(p):
    """Floating-point operations of one product (multiply and add counted)."""
    return 2 * p.mt * p.kt * p.nt * TILE**3


def tile_products(p):
    return p.mt * p.kt * p.nt
