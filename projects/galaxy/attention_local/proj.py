"""The attention projections of a DeepSeek-V4.1-Flash layer on one Wormhole chip: wq_a,
q_norm, wq_b, wkv, kv_norm, wo_a (block-diagonal over 8 groups), wo_b on one row tile
[32, K] bf16 (a decode token or a block of 32). Products are bounds' K split, norms the
row form on one core. README.md."""

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi.dsl.tile import row_operand

TILE = 32
DIM, Q_RANK, HEADS, HEAD_DIM = 5120, 1280, 64, 512
O_GROUPS, O_RANK = 8, 1024
EPS = 1e-20  # rms_norm_eps of the config
CHUNK = 2  # X tiles a block, as bounds.py
WIDE, NARROW = 8, 16  # the two group widths (tiles of Y)
Q_TILES, KV_TILES = Q_RANK // TILE, HEAD_DIM // TILE  # the normed rows: 40 and 16 tiles


@km.tile_function
def rms_q(row, weight):
    ms = km.reduce_sum(row * row, exact=True) * (1.0 / Q_RANK)
    return row * km.rsqrt(ms + EPS).col() * weight.row()


@km.tile_function
def rms_kv(row, weight):
    ms = km.reduce_sum(row * row, exact=True) * (1.0 / HEAD_DIM)
    return row * km.rsqrt(ms + EPS).col() * weight.row()


class Projections(km.Program):
    """A product a handler: Y[:, first:first + WIDTH] = X[:, K range] W[K range, ...]
    for the group's block of Y columns (wo_a's W stacked by `wo_a_stacked`, K in the
    block's group); norm_q / norm_kv: x rsqrt(mean(x^2) + eps) g on one core."""

    CHUNK = km.constant(CHUNK)
    # each product's K tiles (wo_a: a group's) and K split (ranks a group of WIDTH
    # columns): at most 79 ranks, the trees' arrival slots inside L1
    QA_K, QA_SPLIT = km.constant(DIM // TILE), km.constant(15)
    KV_K, KV_SPLIT = km.constant(DIM // TILE), km.constant(39)
    QB_K, QB_SPLIT = km.constant(Q_RANK // TILE), km.constant(1)
    OA_K, OA_SPLIT = km.constant(HEADS * HEAD_DIM // O_GROUPS // TILE), km.constant(4)
    OA_COLS = km.constant(O_RANK // TILE)
    OB_K, OB_SPLIT = km.constant(O_GROUPS * O_RANK // TILE), km.constant(3)
    wide_sums: km.L1[km.fp32, 1, WIDE, 32, 32]
    narrow_sums: km.L1[km.fp32, 1, NARROW, 32, 32]
    q_weight: km.L1[km.fp32, Q_TILES // 2, 32, 32]
    kv_weight: km.L1[km.fp32, KV_TILES // 2, 32, 32]

    @km.tile_function(fidelity="HiFi3")
    def project(x, w):
        return x @ w

    @km.handler
    def wq_a(self, x: km.Tensor, w_qa: km.Tensor, y: km.Tensor):
        share = self.rank % self.QA_SPLIT
        first = self.rank // self.QA_SPLIT * WIDE
        k0 = share * self.QA_K // self.QA_SPLIT
        k1 = (share + 1) * self.QA_K // self.QA_SPLIT
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(w_qa.tiles[k0:k1, first : first + WIDE], block=self.CHUNK * WIDE),
        ):
            self.wide_sums += self.project(xb, wb)
        whole = km.reduce(self.wide_sums, form="tree", group=self.QA_SPLIT)
        if self.rank == self.leader:
            y.pages(first, WIDE).write(whole)

    @km.handler
    def wkv(self, x: km.Tensor, w_kv: km.Tensor, y: km.Tensor):
        share = self.rank % self.KV_SPLIT
        first = self.rank // self.KV_SPLIT * WIDE
        k0 = share * self.KV_K // self.KV_SPLIT
        k1 = (share + 1) * self.KV_K // self.KV_SPLIT
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(w_kv.tiles[k0:k1, first : first + WIDE], block=self.CHUNK * WIDE),
        ):
            self.wide_sums += self.project(xb, wb)
        whole = km.reduce(self.wide_sums, form="tree", group=self.KV_SPLIT)
        if self.rank == self.leader:
            y.pages(first, WIDE).write(whole)

    @km.handler
    def wq_b(self, qr: km.Tensor, w_qb: km.Tensor, y: km.Tensor):
        share = self.rank % self.QB_SPLIT
        first = self.rank // self.QB_SPLIT * NARROW
        k0 = share * self.QB_K // self.QB_SPLIT
        k1 = (share + 1) * self.QB_K // self.QB_SPLIT
        for xb, wb in zip(
            km.Stream(qr.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(
                w_qb.tiles[k0:k1, first : first + NARROW], block=self.CHUNK * NARROW
            ),
        ):
            self.narrow_sums += self.project(xb, wb)
        whole = km.reduce(self.narrow_sums, form="tree", group=self.QB_SPLIT)
        if self.rank == self.leader:
            y.pages(first, NARROW).write(whole)

    @km.handler
    def wo_a(self, o: km.Tensor, w_oa: km.Tensor, y: km.Tensor):
        share = self.rank % self.OA_SPLIT
        first = self.rank // self.OA_SPLIT * NARROW
        group = first // self.OA_COLS
        koff = group * self.OA_K
        wcol = first - group * self.OA_COLS
        k0 = koff + share * self.OA_K // self.OA_SPLIT
        k1 = koff + (share + 1) * self.OA_K // self.OA_SPLIT
        for xb, wb in zip(
            km.Stream(o.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(
                w_oa.tiles[k0:k1, wcol : wcol + NARROW], block=self.CHUNK * NARROW
            ),
        ):
            self.narrow_sums += self.project(xb, wb)
        whole = km.reduce(self.narrow_sums, form="tree", group=self.OA_SPLIT)
        if self.rank == self.leader:
            y.pages(first, NARROW).write(whole)

    @km.handler
    def wo_b(self, oa: km.Tensor, w_ob: km.Tensor, y: km.Tensor):
        share = self.rank % self.OB_SPLIT
        first = self.rank // self.OB_SPLIT * WIDE
        k0 = share * self.OB_K // self.OB_SPLIT
        k1 = (share + 1) * self.OB_K // self.OB_SPLIT
        for xb, wb in zip(
            km.Stream(oa.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(w_ob.tiles[k0:k1, first : first + WIDE], block=self.CHUNK * WIDE),
        ):
            self.wide_sums += self.project(xb, wb)
        whole = km.reduce(self.wide_sums, form="tree", group=self.OB_SPLIT)
        if self.rank == self.leader:
            y.pages(first, WIDE).write(whole)

    @km.handler
    def norm_q(self, x: km.Tensor, g: km.Tensor, out: km.Tensor):
        self.q_weight.fill(km.Stream(g.pages(0, Q_TILES // 2)))
        results = km.Stream(out.pages(0, Q_TILES), write=True, block=8)
        for row in km.Stream(x.pages(0, Q_TILES), block=8, row=Q_TILES):
            results.push(rms_q(row, self.q_weight))

    @km.handler
    def norm_kv(self, x: km.Tensor, g: km.Tensor, out: km.Tensor):
        self.kv_weight.fill(km.Stream(g.pages(0, KV_TILES // 2)))
        results = km.Stream(out.pages(0, KV_TILES), write=True, block=8)
        for row in km.Stream(x.pages(0, KV_TILES), block=8, row=KV_TILES):
            results.push(rms_kv(row, self.kv_weight))


# Each product: (WIDTH, K tiles a group, split, columns a group, Y tiles), as the
# class's constants (wo_a: 8 groups of 32 columns, 4 ranks a K split).
PRODUCTS = {
    "wq_a": (WIDE, DIM // TILE, 15, Q_RANK // TILE, Q_RANK // TILE),
    "wq_b": (
        NARROW,
        Q_RANK // TILE,
        1,
        HEADS * HEAD_DIM // TILE,
        HEADS * HEAD_DIM // TILE,
    ),
    "wkv": (WIDE, DIM // TILE, 39, HEAD_DIM // TILE, HEAD_DIM // TILE),
    "wo_a": (
        NARROW,
        HEADS * HEAD_DIM // O_GROUPS // TILE,
        4,
        O_RANK // TILE,
        O_GROUPS * O_RANK // TILE,
    ),
    "wo_b": (WIDE, O_GROUPS * O_RANK // TILE, 3, DIM // TILE, DIM // TILE),
}


def cores(name):
    """The ranks of a product's command: a group a block of WIDTH columns."""
    width, _, split, _, ntiles = PRODUCTS[name]
    return ntiles // width * split


def norm_pages(weight):
    """A norm weight [32 n] as the row form's field pages: chunk c's per-column tile."""
    weight = np.asarray(weight, np.float32)
    n = weight.size // TILE
    pages = [row_operand(weight[64 * c : 64 * c + 64], 2) for c in range(n // 2)]
    return np.asarray(pages, np.float32)


def stored(values, fmt):
    """The values a [K, N] matrix holds on the chip in `fmt` (its pages decoded)."""
    values = np.asarray(values, np.float32)
    raw, _ = formats.to_bytes(values, fmt, "tile")
    return np.asarray(formats.from_bytes(raw, values.shape, fmt, "tile"), np.float32)


def wo_a_stacked(wo_a):
    """wo_a [8 x 1024, 4096] (out, in) as the stacked [8 x 4096, 1024]: group g's
    W_g = wo_a[g x 1024 : (g + 1) x 1024].T at rows g x 4096."""
    wo_a = np.asarray(wo_a, np.float32).reshape(O_GROUPS, O_RANK, -1)
    return np.ascontiguousarray(wo_a.transpose(0, 2, 1).reshape(-1, O_RANK))
