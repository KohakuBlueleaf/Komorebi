"""The global attention of DeepSeek-V4.1-Flash as a piece of the driver's interface:
the sites `compressor`, `indexer`, `sparse_attn` on one Wormhole chip (README.md).
`make` serves all three; `make_compressor`, `make_indexer`, `make_sparse_attn` one each.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import komorebi as km
from komorebi import formats
from komorebi.dsl.tile import row_operand

DIM, QR, HEAD = 5120, 1280, 512
IHEADS, IDX = 32, 128  # index heads, index head dim
HEADS = 64  # query heads
D = HEAD // 32  # tiles of a head
QX = D + 2  # q's tiles a row: the head, the mask selector, one zero tile
KEYS = 20  # the decode's key row tiles: 128 window + 512 selected
COLS = KEYS + 2  # a score row's tiles: the keys, the sink tile, a masked tile
ROWS = 64  # q's row tiles at prefill: 32 tokens x 64 heads
SCALE = HEAD**-0.5
EPS = 1e-20
MASKED = -1.0e30
KV_SOURCES = {2: 2, 8: 2, 14: 2, 20: 1}  # layer: ratio
INDEX_SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)
CANDIDATE_SOURCE = 20
MAX_LEN = 4096
PAGE = {"bfloat16": 2048, "float32": 4096}
# the one set of compute options of every class on the box
WEIGHTS = "checkpoint"  # the driver's W for make: the Checkpoint (driver/interface.py)
COMPUTE = {"math_fidelity": "HiFi2", "fp32_dest_acc_en": True, "dst_full_sync_en": True}
# the attention's key list: tile 0 the sink, 1-4 the window, 5 a prefill block's own
# keys, from tile ENTRY0 the selected entries (entry j in row 32 ENTRY0 + j)
ENTRY0, ENTRY_TILES = 6, 16
KEY_TILES = ENTRY0 + ENTRY_TILES
CACHE_TILES = (MAX_LEN + 128) // 32  # the gather's source rows: window + entries


@km.tile_function(fidelity="HiFi2")
def product(x, w):
    return x @ w


@km.tile_function
def rectified(a):
    return km.relu(a)


@km.tile_function
def gated(k0, k1, s0, s1):
    m = km.maximum(s0, s1)
    e0 = km.exp(s0 - m)
    e1 = km.exp(s1 - m)
    return (k0 * e0 + k1 * e1) / (e0 + e1)


@km.tile_function
def same(k):
    return k * 1.0


@km.tile_function
def rms(row, weight):
    ms = km.reduce_sum(row * row, exact=True) * (1.0 / HEAD)
    return row * km.rsqrt(ms + EPS).col() * weight.row()


@km.tile_function
def softmax_row(row):
    x = row * SCALE
    m = km.reduce_max(x, exact=True)
    e = km.exp(x - m.col())
    total = km.reduce_sum(e, exact=True)
    return e / total.col()


class Linear(km.Program, compute=COMPUTE):
    """linear_x: Y = X W for one row tile of X (K = 160 tiles), groups of `split` cores,
    a group `width` tiles of Y (float32) at pages y0 + group x width."""

    @km.handler
    def linear_x(
        self,
        xh: km.Tensor,
        wx: km.Tensor,
        yx: km.Tensor,
        y0: int,
        k_tiles: km.constant[int] = 160,
        split: km.constant[int] = 4,
        width: km.constant[int] = 8,
    ):
        share = self.rank % split
        first = self.rank // split * width
        k0 = share * k_tiles // split
        k1 = (share + 1) * k_tiles // split
        acc = km.tile.zeros(km.fp32, width)
        for xb, wb in zip(
            km.Stream(xh.tiles[0:1, k0:k1], block=2),
            km.Stream(wx.tiles[k0:k1, first : first + width], block=2 * width),
        ):
            acc += product(xb, wb)
        whole = km.reduce(acc, form="tree", group=split, local=True)
        if self.rank == self.leader:
            yx.pages(y0 + first, width).write(whole)


class Compress(Linear, compute=COMPUTE):
    """The compressor: linear_x the projection; pool (ratio 2) the softmax gate of the
    state's two slots, rounded (ratio 1) its slot, into the bf16 pooled latent; norm
    RMSNorm of its rows (README.md)."""

    weight: km.L1[km.fp32, D // 2, 32, 32]

    @km.handler
    def pool(self, state: km.Tensor, pooled: km.Tensor):
        c0 = self.rank
        g0 = D + self.rank
        results = km.Stream(pooled.tiles[0:1, c0 : c0 + 1], write=True, block=1)
        for k0, k1, s0, s1 in zip(
            km.Stream(state.tiles[0:1, c0 : c0 + 1], block=1),
            km.Stream(state.tiles[1:2, c0 : c0 + 1], block=1),
            km.Stream(state.tiles[0:1, g0 : g0 + 1], block=1),
            km.Stream(state.tiles[1:2, g0 : g0 + 1], block=1),
        ):
            results.push(gated(k0, k1, s0, s1))

    @km.handler
    def rounded(self, state: km.Tensor, pooled: km.Tensor):
        c0 = self.rank
        results = km.Stream(pooled.tiles[0:1, c0 : c0 + 1], write=True, block=1)
        for k0 in km.Stream(state.tiles[0:1, c0 : c0 + 1], block=1):
            results.push(same(k0))

    @km.handler
    def norm(self, pooled: km.Tensor, wn: km.Tensor, out: km.Tensor):
        self.weight.fill(km.Stream(wn.pages(0, D // 2)))
        results = km.Stream(out.pages(0, D), write=True, block=2)
        for row in km.Stream(pooled.pages(0, D), block=2, row=D):
            results.push(rms(row, self.weight))


class Index(Linear, compute=COMPUTE):
    """The indexer: linear_x for wq_b (X padded to K = 160) and weights_proj (its
    weight padded to 4,096 columns); column, score, compact the decode's index scores
    (README.md)."""

    h1: km.L1[km.bf16, 1, 32, 32]

    @km.handler
    def column(self, sel: km.Tensor, wv: km.Tensor, wc: km.Tensor):
        own = km.Broadcast(self, along="column", length=1, mover="A")
        if own.sender():  # noqa: SIM108 (the class language takes a statement)
            blk = own.cast(wv.tiles[0:1, 0:1].T)
        else:
            blk = own.arrive()
        acc = km.tile.zeros(km.fp32, 1)
        for xb in km.Stream(sel.tiles[0:1, 0:1], block=1):
            acc += product(xb, blk)
        wc.pages(self.rank, 1).write(acc)

    @km.handler
    def score(
        self, kc: km.Tensor, qv: km.Tensor, wc: km.Tensor, scores: km.Tensor, tt: int
    ):
        own = km.Broadcast(self, along="column", length=1, mover="A")
        if own.sender():
            blk = own.cast(qv.tiles[0:1, 0:4].T, wc.tiles[0:1, 0:1])
        else:
            blk = own.arrive()
        n = (tt + self.size - 1 - self.rank) // self.size
        for i in range(n):
            r = self.rank + i * self.size
            acc = km.tile.zeros(km.fp32, 1)
            for kb in km.Stream(kc.tiles[r : r + 1, 0:4], block=4):
                acc += product(kb, blk[0])
            self.h1.write(rectified(acc))
            out = km.tile.zeros(km.fp32, 1)
            out += product(self.h1, blk[1][0:1, :])
            scores.pages(r, 1).write(out)

    @km.handler
    def compact(self, scores: km.Tensor, ep: km.Tensor, packed: km.Tensor, groups: int):
        n = (groups + self.size - 1 - self.rank) // self.size
        for i in range(n):
            g = self.rank + i * self.size
            k0 = 32 * g
            acc = km.tile.zeros(km.fp32, 1)
            for xb, wb in zip(
                km.Stream(scores.tiles[0:1, k0 : k0 + 32], block=2),
                km.Stream(ep.tiles[0:32, 0:1], block=2),
            ):
                acc += product(xb, wb)
            packed.pages(g, 1).write(acc)


class Attend(km.Program, compute=COMPUTE):
    """The attention: qk S = q K^T a tile; softmax a row of keys, sink and a masked
    tile; pv O = P K; gather the chosen cache rows into a key list from row tile e0 by
    a 0/1 selection product (README.md)."""

    @km.handler
    def gather(
        self,
        gsel: km.Tensor,
        ckv: km.Tensor,
        kall: km.Tensor,
        tiles: int,
        kt: int,
        e0: int,
    ):
        n = (tiles * 4 + self.size - 1 - self.rank) // self.size
        for i in range(n):
            t = self.rank + i * self.size
            b = t // 4
            c0 = t % 4 * 4
            acc = km.tile.zeros(km.fp32, 4)
            for xb, wb in zip(
                km.Stream(gsel.tiles[b : b + 1, 0:kt], block=2),
                km.Stream(ckv.tiles[0:kt, c0 : c0 + 4], block=8),
            ):
                acc += product(xb, wb)
            kall.pages((e0 + b) * D + c0, 4).write(acc)

    @km.handler
    def qk(
        self,
        qx: km.Tensor,
        kt: km.Tensor,
        s: km.Tensor,
        rows: int,
        keys: km.constant[int] = KEYS,
    ):
        n = (rows * keys + self.size - 1 - self.rank) // self.size
        for i in range(n):
            t = self.rank + i * self.size
            b = t // keys
            k = t % keys
            acc = km.tile.zeros(km.fp32, 1)
            for xb, wb in zip(
                km.Stream(qx.tiles[b : b + 1, 0:QX], block=2),
                km.Stream(kt.tiles[0:QX, k : k + 1], block=2),
            ):
                acc += product(xb, wb)
            s.pages(b * COLS + k, 1).write(acc)

    @km.handler
    def softmax(
        self, s: km.Tensor, p: km.Tensor, rows: int, keys: km.constant[int] = KEYS
    ):
        n = (rows + self.size - 1 - self.rank) // self.size
        for i in range(n):
            b = self.rank + i * self.size
            results = km.Stream(p.pages(b * COLS, keys + 2), write=True, block=2)
            for row in km.Stream(s.pages(b * COLS, keys + 2), block=2, row=keys + 2):
                results.push(softmax_row(row))

    @km.handler
    def pv(
        self,
        p: km.Tensor,
        kall: km.Tensor,
        o: km.Tensor,
        rows: int,
        keys: km.constant[int] = KEYS,
    ):
        n = (rows * D + self.size - 1 - self.rank) // self.size
        for i in range(n):
            t = self.rank + i * self.size
            b = t // D
            c = t % D
            acc = km.tile.zeros(km.fp32, 1)
            for xb, wb in zip(
                km.Stream(p.tiles[b : b + 1, 0:keys], block=2),
                km.Stream(kall.tiles[0:keys, c : c + 1], block=2),
            ):
                acc += product(xb, wb)
            o.pages(b * D + c, 1).write(acc)


def plan(ranks, kt, nt, chunk=2, widest=4):
    """Of the widths w dividing N's tiles, groups g = nt / w, split s = min(ranks // g,
    K tiles // chunk); the most cores, ties to the wider w (fewer reads of X). At most 4
    wide: the sums stay in Dst (8 tiles) beside the arriving tile."""
    best = None
    for w in range(1, min(nt, widest) + 1):
        if nt % w:
            continue
        g = nt // w
        s = min(ranks // g, kt // chunk)
        if s >= 1 and (best is None or g * s >= best[0] * best[1]):
            best = (g, s, w)
    g, s, w = best
    return {"k_tiles": kt, "split": s, "width": w}, g * s


def eprime():
    """E': 32 tiles stacked, tile b a 1 at (0, b)."""
    e = np.zeros((32 * 32, 32), np.float32)
    for b in range(32):
        e[32 * b, b] = 1.0
    return e


PACKED = np.array(
    [
        ((a // 16) * 2 + b // 16) * 256 + (a % 16) * 16 + b % 16
        for b in range(32)
        for a in range(32)
    ]
)  # entry 32 b + a of a packed tile: (a, b) in the tile's faces


def f32(t):
    """A torch tensor as float32 numpy."""
    return t.detach().float().numpy()


class Piece:
    sites = ("compressor", "indexer", "sparse_attn")

    def __init__(self, session, ckpt, layers, sites=None, log=None, run=None):
        self.session = session
        self.sites = tuple(sites or Piece.sites)
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.times = []  # one dict a call: site, layer, box us by handler, host s
        self._reference(ckpt)
        want = set(li for li in layers if li is not None)
        indexer = "indexer" in self.sites
        self.kv_layers = [
            li
            for li in KV_SOURCES
            if ("compressor" in self.sites and li in want)
            or (indexer and any(m >= li for m in want if m in INDEX_SOURCES))
        ]
        self.index_layers = [li for li in INDEX_SOURCES if indexer and li in want]
        # a command's ranks: the first island's (joined islands: 4 are link cores)
        graph = getattr(session, "graph", None)
        self.ranks = getattr(session, "ranks", 79) if graph is None else graph.count(0)
        t0 = time.monotonic()
        self._tensors(ckpt)
        self.log(
            f"piece: tensors in {time.monotonic() - t0:.1f} s (kv {self.kv_layers}, "
            f"index {self.index_layers})"
        )
        self._load(run)
        self.state = {
            li: {"k": torch.zeros(MAX_LEN // KV_SOURCES[li], IDX, dtype=torch.bfloat16)}
            for li in self.kv_layers
        }
        self.candidates = None

    # the reference's own functions for the host's part
    def _reference(self, ckpt):
        from galaxy.reference import cpu_kernels  # noqa: PLC0415 (the driver's package)
        from galaxy.reference.model_loader import (  # noqa: PLC0415
            import_reference_code,
        )

        self.ck = cpu_kernels
        self.mod = import_reference_code(ckpt.dir)
        with open(os.path.join(ckpt.dir, "inference", "config.json")) as f:
            args = self.mod.ModelArgs(**json.load(f))
        pre = self.mod.precompute_freqs_cis.__wrapped__
        dt = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)  # as the reference
        try:
            self.freqs = pre(
                args.rope_head_dim,
                MAX_LEN,
                args.original_seq_len,
                args.compress_rope_theta,
                args.rope_factor,
                args.beta_fast,
                args.beta_slow,
            )
        finally:
            torch.set_default_dtype(dt)
        self.args = args

    def _put(self, tensor, values, fmt, first=0):
        raw, _ = formats.to_bytes(np.ascontiguousarray(values, np.float32), fmt, "tile")
        self.session.write(
            tensor, np.asarray(raw, np.uint8).reshape(-1, PAGE[fmt]), first=first
        )

    def _get(self, tensor, rows, cols, fmt, first=0):
        pages = rows * cols // 1024
        out = self.session.read(tensor, first=first, pages=pages)
        raw = out[0] if isinstance(out, (tuple, list)) else out
        return np.asarray(
            formats.from_bytes(np.asarray(raw).reshape(-1), (rows, cols), fmt, "tile"),
            np.float32,
        )

    def _tensor(self, values, fmt):
        return self.session.tensor(
            np.ascontiguousarray(values, np.float32), fmt, "tile"
        )

    def _tensors(self, ckpt):
        s, ck = self.session, self.ck
        compressor, indexer = "compressor" in self.sites, "indexer" in self.sites
        self.w = {}
        if compressor or indexer:
            self.tx = s.empty((32, DIM), "bfloat16", "tile")
        for li in self.kv_layers:
            p = f"layers.{li}.attn.compressor."
            w = self.w.setdefault(li, {})
            if compressor:
                wc = np.zeros((DIM, 1024), np.float32)
                wc[:, :HEAD] = f32(ckpt.get(p + "wkv.weight")).T
                if KV_SOURCES[li] == 2:
                    wc[:, HEAD:] = f32(ckpt.get(p + "wgate.weight")).T
                wn = f32(ckpt.get(p + "norm.weight"))
                pages = np.asarray(
                    [row_operand(wn[64 * c : 64 * c + 64], 2) for c in range(D // 2)],
                    np.float32,
                )
                twn = s.empty((D // 2, 1024), "float32")
                s.write(twn, pages.view(np.uint8).reshape(D // 2, -1))
                w.update(
                    wc=self._tensor(wc, "bfloat16"),
                    wn=twn,
                    state=s.empty((64, 1024), "float32", "tile"),
                )
            if indexer:
                cap = MAX_LEN // KV_SOURCES[li] // 32 + 1
                q = f"layers.{li}.attn.indexer."
                w.update(
                    kc=s.empty((32 * cap, IDX), "bfloat16", "tile"),
                    wk=ckpt.get(q + "wk.weight").to(torch.bfloat16).clone(),
                    k_norm=ckpt.get(q + "k_norm.weight").to(torch.bfloat16).clone(),
                )
        if compressor:
            self.pooled = s.empty((32, HEAD), "bfloat16", "tile")
            self.out = s.empty((32, HEAD), "bfloat16", "tile")
        if indexer:
            for li in self.index_layers:
                p = f"layers.{li}.attn.indexer."
                wq = ck.dequant_fp8_weight(
                    ckpt.get(p + "wq_b.weight"), ckpt.get(p + "wq_b.scale")
                )
                wp = np.zeros((DIM, IHEADS * IDX), np.float32)
                wp[:, :IHEADS] = f32(ckpt.get(p + "weights_proj.weight")).T
                self.w.setdefault(li, {}).update(
                    wq=self._tensor(f32(wq).T, "bfloat16"),
                    wp=self._tensor(wp, "bfloat16"),
                )
            self.tqr = s.empty((32, DIM), "bfloat16", "tile")  # qr in columns < 1280
            self.tq = s.empty((32, IHEADS * IDX), "float32", "tile")
            self.tyx = s.empty((32, 1024), "float32", "tile")
            self.tw = s.empty((32, 32), "bfloat16", "tile")
            self.twc = s.empty((32, 32 * 72), "bfloat16", "tile")
            self.tsel = self._tensor(np.eye(32), "bfloat16")
            self.tqv = s.empty((32, IDX), "bfloat16", "tile")
            groups = -(-(MAX_LEN // 32) // 32)
            self.tsc = s.empty((32, 32 * 32 * groups), "float32", "tile")
            self.tep = self._tensor(eprime(), "float32")
            self.tpk = s.empty((32, 32 * groups), "float32", "tile")
        if "sparse_attn" in self.sites:
            self.qx = s.empty((32 * ROWS, 32 * QX), "bfloat16", "tile")
            self.kt = s.empty((32 * QX, 32 * KEYS), "bfloat16", "tile")
            self.kall = s.empty((32 * KEY_TILES, HEAD), "bfloat16", "tile")
            self.gsel = s.empty((32 * KEYS, 32 * CACHE_TILES), "bfloat16", "tile")
            self.ckv = s.empty((32 * CACHE_TILES, HEAD), "bfloat16", "tile")
            self.ts = s.empty((32 * ROWS, 32 * COLS), "float32", "tile")
            self.tp = s.empty((32 * ROWS, 32 * COLS), "bfloat16", "tile")
            self.to = s.empty((32 * ROWS, HEAD), "bfloat16", "tile")

    def _load(self, run=None):
        common = dict(session=self.session, log=self.log, form="runtime")
        t0 = time.monotonic()
        programs = []  # (program, ranks it starts on)
        plans = {}
        if "indexer" in self.sites:
            self.p_wq, self.c_wq = plan(self.ranks, QR // 32, IHEADS * IDX // 32)
            self.p_wp, self.c_wp = plan(self.ranks, DIM // 32, 1)
            il = self.w[self.index_layers[0]]
            self.index = Index(
                xh=self.tx,
                wx=il["wp"],
                yx=self.tyx,
                sel=self.tsel,
                wv=self.tw,
                wc=self.twc,
                kc=self.w[self.kv_layers[-1]]["kc"],
                qv=self.tqv,
                scores=self.tsc,
                ep=self.tep,
                packed=self.tpk,
                handlers={"linear_x": [self.p_wq, self.p_wp]},
                **common,
                **self._rect(),
            )
            programs.append(self.index)  # first: its broadcasts' rectangle (0, 72)
            plans.update(wq_b=self.p_wq, weights_proj=self.p_wp)
        if "compressor" in self.sites:
            self.p_c = {r: plan(self.ranks, DIM // 32, r * HEAD // 32) for r in (1, 2)}
            kv0 = self.w[self.kv_layers[-1]]
            self.kv_program = Compress(
                xh=self.tx,
                wx=kv0["wc"],
                yx=kv0["state"],
                state=kv0["state"],
                pooled=self.pooled,
                wn=kv0["wn"],
                out=self.out,
                handlers={"linear_x": [self.p_c[1][0], self.p_c[2][0]]},
                **common,
            )
            programs.append(self.kv_program)
            plans.update(compressor=self.p_c)
        if "sparse_attn" in self.sites:
            keys = [{"keys": KEYS}, {"keys": 2}]
            self.attend = Attend(
                gsel=self.gsel,
                ckv=self.ckv,
                qx=self.qx,
                kt=self.kt,
                s=self.ts,
                p=self.tp,
                kall=self.kall,
                o=self.to,
                handlers={"qk": keys, "softmax": keys, "pv": keys},
                **common,
            )
            programs.append(self.attend)
        for program in programs:  # the one load's place for the service overlay
            program.slots(count=1, service="shared")
        self.held = None  # the first submit loads its program on every rank
        if run is not None:  # the shared load (projects/galaxy/load): handed to it
            run.take({p: (0, self.session.ranks) for p in programs})
            self.run = run
            return
        # one run, switched as a site needs another program: the first on the island.s
        # ranks the others leave, each other one rank, the last on the later islands too
        r, every = self.ranks, self.session.ranks
        first = r - (len(programs) - 1)
        groups = {programs[0]: (0, first)}
        groups.update({p: (first + i, 1) for i, p in enumerate(programs[1:])})
        last = programs[-1]
        groups[last] = (groups[last][0], every - groups[last][0])
        self.run = self.session.load(groups)
        self.log(
            f"piece: {[type(p).__name__ for p in programs]} built and loaded in "
            f"{time.monotonic() - t0:.1f} s; plans {plans}"
        )

    def _rect(self):
        """A broadcast class's rectangle on joined islands: the first island's whole
        rows (the grid is not its ranks); one chip: the grid, given by nothing."""
        if getattr(self.session, "graph", None) is None:
            return {}
        columns = self.session.nocs().grid[0]
        return {"rect": (0, 0, columns, self.ranks // columns)}

    def _holds(self, program):
        """Whether the ranks hold `program`: a shared load says (another piece may
        have switched them); a load of its own, the last one loaded."""
        if hasattr(self.run, "take"):
            return self.run.holds(program, (0, self.ranks))
        return program is self.held

    def _submit(self, program, name, group, words, us):
        # a load to the cores' program sets its buffers up again (empty)
        if not self._holds(program):
            t0 = time.monotonic()
            self.run.wait(self.run.load(program, group=(0, self.ranks)), timeout=30.0)
            self.held = program
            us["switch_host"] = round(
                us.get("switch_host", 0.0) + (time.monotonic() - t0) * 1e6, 1
            )
        event = self.run.wait(
            self.run.submit(getattr(program, name), group, args=words), timeout=30.0
        )
        if event.kind != "done":
            raise km.KomorebiError(f"{name}: {event}")
        us[name] = round(us.get(name, 0.0) + event.microseconds, 2)

    def __call__(self, site, layer, *inputs):
        t0 = time.monotonic()
        us = {}
        out = getattr(self, "_" + site)(layer, us, *inputs)
        self.times.append(
            {
                "site": site,
                "layer": layer,
                "us": us,
                "s": round(time.monotonic() - t0, 4),
            }
        )
        self.log("piece: " + json.dumps(self.times[-1]))
        return out

    # the compressor
    def _compressor(self, layer, us, x, start_pos):
        ratio, w = KV_SOURCES[layer], self.w[layer]
        seqlen = x.size(1)
        xs = f32(x[0])
        sets = self.p_c[ratio]
        group = (0, sets[1])

        def words(tx, slot):
            return (tx, w["wc"], w["state"], 32 * slot, *sets[0].values())

        if start_pos == 0:
            assert (
                seqlen % ratio == 0
            ), "a trailing partial group at prefill is not handled"
            n = seqlen // ratio
            for slot in range(ratio):
                rows = np.zeros((32, DIM), np.float32)
                rows[:n] = xs[slot::ratio]
                self._put(self.tx, rows, "bfloat16")
                self._submit(
                    self.kv_program, "linear_x", group, words(self.tx, slot), us
                )
        else:
            n = 1
            rows = np.zeros((32, DIM), np.float32)
            rows[0] = xs[0]
            self._put(self.tx, rows, "bfloat16")
            slot = start_pos % ratio
            self._submit(self.kv_program, "linear_x", group, words(self.tx, slot), us)
            if (start_pos + 1) % ratio:
                return None
        # the gated pool of the slots (model.py 473-475, 482) or the bf16 rounding
        # (462), then the norm
        self._submit(
            self.kv_program,
            "pool" if ratio == 2 else "rounded",
            (0, D),
            (w["state"], self.pooled),
            us,
        )
        self._submit(
            self.kv_program, "norm", (0, 1), (self.pooled, w["wn"], self.out), us
        )
        got = self._get(self.out, 32, HEAD, "bfloat16")[:n]
        return torch.from_numpy(got).to(torch.bfloat16).unsqueeze(0)

    # the indexer
    def _keys(self, layer, latent, start_pos):
        """The index keys of new latents (the reference's lines 537-547), into the
        host's copy and the chip's cache."""
        ratio, w, mod, ck = KV_SOURCES[layer], self.w[layer], self.mod, self.ck
        seqlen = latent.size(1)
        freqs = (
            self.freqs[: seqlen * ratio : ratio]
            if start_pos == 0
            else self.freqs[start_pos + 1 - ratio].unsqueeze(0)
        )
        k = F.linear(latent, w["wk"])
        kf = k.float()
        k = (
            w["k_norm"] * (kf * torch.rsqrt(kf.square().mean(-1, keepdim=True) + EPS))
        ).to(torch.bfloat16)
        mod.apply_rotary_emb(k[..., -self.args.rope_head_dim :], freqs)
        ck.fp4_act_quant(k, 32, True)
        e0 = start_pos // ratio
        cache = self.state[layer]["k"]
        cache[e0 : e0 + k.size(1)] = k[0]
        r0, r1 = e0 // 32, (e0 + k.size(1) - 1) // 32
        self._put(
            w["kc"], f32(cache[32 * r0 : 32 * (r1 + 1)]), "bfloat16", first=4 * r0
        )

    def _indexer(self, layer, us, x, qr, latent, start_pos, offset):
        mod, ck = self.mod, self.ck
        seqlen = x.size(1)
        if layer in KV_SOURCES and latent is not None:
            self._keys(layer, latent, start_pos)
        owner = max(li for li in KV_SOURCES if li <= layer)
        ratio = KV_SOURCES[owner]
        end_pos = start_pos + seqlen
        w = self.w[layer]
        # q = wq_b(E4M3(qr)) on the chip, its rotary and FP4 on the host
        qq = qr.clone()
        ck.act_quant(qq, 32, "ue8m0", torch.float8_e8m0fnu, True)
        rows = np.zeros((32, DIM), np.float32)
        rows[:seqlen, :QR] = f32(qq[0])
        self._put(self.tqr, rows, "bfloat16")
        self._submit(
            self.index,
            "linear_x",
            (0, self.c_wq),
            (self.tqr, w["wq"], self.tq, 0, *self.p_wq.values()),
            us,
        )
        q = torch.from_numpy(
            self._get(self.tq, 32, IHEADS * IDX, "float32")[:seqlen]
        ).to(torch.bfloat16)
        q = q.view(1, seqlen, IHEADS, IDX)
        mod.apply_rotary_emb(
            q[..., -self.args.rope_head_dim :], self.freqs[start_pos:end_pos]
        )
        ck.fp4_act_quant(q, 32, True)
        # weights_proj on the chip
        rows = np.zeros((32, DIM), np.float32)
        rows[:seqlen] = f32(x[0])
        self._put(self.tx, rows, "bfloat16")
        self._submit(
            self.index,
            "linear_x",
            (0, self.c_wp),
            (self.tx, w["wp"], self.tyx, 0, *self.p_wp.values()),
            us,
        )
        wsum = torch.from_numpy(self._get(self.tyx, 32, 32, "float32")[:seqlen]).to(
            torch.bfloat16
        )
        weights = (wsum * (IDX**-0.5 * IHEADS**-0.5)).unsqueeze(
            0
        )  # [1, s, 32] bf16, as the reference
        t = end_pos // ratio
        if start_pos == 0:
            # prefill: the scores on the host (the reference's lines 557-558)
            index_k = self.state[owner]["k"][:t].unsqueeze(0)
            score = torch.einsum("bshd,btd->bsht", q, index_k)
            score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
            compress_lens = (torch.arange(1, seqlen + 1) // ratio).unsqueeze(-1)
            score.masked_fill_(
                torch.arange(seqlen // ratio) >= compress_lens, -torch.inf
            )
        else:
            self._put(self.tqv, f32(q[0, 0]), "bfloat16")
            wrow = np.zeros((32, 32), np.float32)
            wrow[0] = f32(weights[0, 0])
            self._put(self.tw, wrow, "bfloat16")
            tt = -(-t // 32)
            groups = -(-tt // 32)
            self._submit(
                self.index, "column", (0, 72), (self.tsel, self.tw, self.twc), us
            )
            self._submit(
                self.index,
                "score",
                (0, 72),
                (self.w[owner]["kc"], self.tqv, self.twc, self.tsc, tt),
                us,
            )
            self._submit(
                self.index,
                "compact",
                (0, 72),
                (self.tsc, self.tep, self.tpk, groups),
                us,
            )
            out = self.session.read(self.tpk, first=0, pages=groups)
            raw = out[0] if isinstance(out, (tuple, list)) else out
            words = np.asarray(raw).view(np.float32).reshape(groups, 1024)
            score = torch.from_numpy(words[:, PACKED].reshape(-1)[:t].copy()).view(
                1, 1, t
            )
            compress_lens = t
        if layer == CANDIDATE_SOURCE:
            self.candidates = mod.select_candidate_blocks(
                score,
                compress_lens,
                self.args.candidate_topk_blocks,
                self.args.candidate_block_size,
            )
        elif layer > CANDIDATE_SOURCE:
            score = score.masked_fill(~self.candidates, -torch.inf)
        topk = min(self.args.index_topk, t)
        idxs = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()

    # the attention
    def _sparse_attn(self, layer, us, q, kv, attn_sink, topk_idxs, softmax_scale):
        assert abs(softmax_scale - SCALE) < 1e-12
        seqlen, n = q.size(1), kv.size(1)
        idx = topk_idxs[0].long().numpy()  # [s, K]
        kvs = f32(kv[0])
        if seqlen == 1:
            keys, rows_t = KEYS, 2
            assert idx.shape[1] <= 32 * KEYS
            valid = idx[0] >= 0
            got = self.gather(us, kvs, idx[0], self.kall, 0)
            kall = np.zeros((32 * KEYS, HEAD), np.float32)
            kall[: len(got)] = got
            if len(got) < 32 * KEYS:  # the key tiles past the gather's: zeros
                self._put(
                    self.kall, kall[len(got) :], "bfloat16", first=len(got) // 32 * D
                )
            mask = np.full((1, 32 * KEYS), MASKED, np.float32)
            mask[0, : idx.shape[1]][valid] = 0.0
        else:
            keys, rows_t = 2, 2 * seqlen
            assert n <= 64 and seqlen <= 32
            kall = np.zeros((64, HEAD), np.float32)
            kall[:n] = kvs
            mask = np.full((seqlen, 64), MASKED, np.float32)
            for i in range(seqlen):
                j = idx[i][idx[i] >= 0]
                mask[i, j] = 0.0
        m = 32 * rows_t
        qx = np.zeros((m, 32 * QX), np.float32)
        qx[:, :HEAD] = f32(q[0]).reshape(m, HEAD)
        qx[np.arange(m), HEAD + np.arange(m) // HEADS] = (
            1.0  # row (s, h) selects mask row s
        )
        kt = np.zeros((32 * QX, 32 * keys), np.float32)
        kt[:HEAD] = kall[: 32 * keys].T
        kt[HEAD : HEAD + mask.shape[0]] = mask
        sinit = np.zeros((m, 32 * COLS), np.float32)
        sinit[:, 32 * keys : 32 * keys + 64] = MASKED
        sinit[:, 32 * keys] = np.tile(f32(attn_sink), m // HEADS) / SCALE
        self._put(self.qx, qx, "bfloat16")
        self._put(
            self.kt,
            kt if keys == KEYS else np.pad(kt, ((0, 0), (0, 32 * (KEYS - keys)))),
            "bfloat16",
        )
        if seqlen > 1:
            self._put(
                self.kall, np.pad(kall, ((0, 32 * KEYS - 64), (0, 0))), "bfloat16"
            )
        self._put(self.ts, sinit, "float32")
        self._submit(
            self.attend, "qk", (0, 72), (self.qx, self.kt, self.ts, rows_t, keys), us
        )
        self._submit(
            self.attend,
            "softmax",
            (0, min(rows_t, 72)),
            (self.ts, self.tp, rows_t, keys),
            us,
        )
        self._submit(
            self.attend, "pv", (0, 72), (self.tp, self.kall, self.to, rows_t, keys), us
        )
        o = self._get(self.to, m, HEAD, "bfloat16")
        return torch.from_numpy(o).to(torch.bfloat16).view(1, seqlen, HEADS, HEAD)

    def gather(self, us, cache, chosen, keys, e0):
        """The rows `cache[chosen]` (a -1: an unreachable choice, a zero row) into the
        key list `keys` from row tile e0 (ENTRY0: the attention's key list), on the
        chip; returns the rows read back. The cache goes up whole (version 1: the chip
        keeps no copy of the compressed cache between calls)."""
        m = len(chosen)
        tiles = -(-m // 32)
        kt = -(-len(cache) // 64) * 2  # whole blocks of 2
        assert tiles <= KEYS and kt <= CACHE_TILES
        sel = np.zeros((32 * tiles, 32 * kt), np.float32)
        ok = np.asarray(chosen) >= 0
        sel[np.arange(m)[ok], np.asarray(chosen)[ok]] = 1.0
        self._put(
            self.ckv, np.pad(cache, ((0, 32 * kt - len(cache)), (0, 0))), "bfloat16"
        )
        # the selection's rows as the tensor's whole row tiles (its width CACHE_TILES)
        self._put(
            self.gsel,
            np.pad(sel, ((0, 0), (0, 32 * (CACHE_TILES - kt)))),
            "bfloat16",
        )
        self._submit(
            self.attend,
            "gather",
            (0, min(4 * tiles, self.ranks)),
            (self.gsel, self.ckv, keys, tiles, kt, e0),
            us,
        )
        return self._get(keys, 32 * tiles, HEAD, "bfloat16", first=e0 * D)


def entry_mask(seen):
    """The key list's entries mask [16 x 32, 32] of seen [32, n <= 512] (query row,
    entry): tile t, row i, column c 0 if row i sees entry 32 t + c, else MASKED."""
    full = np.zeros((32, 32 * ENTRY_TILES), bool)
    full[:, : seen.shape[1]] = seen
    tiles = full.reshape(32, ENTRY_TILES, 32).transpose(1, 0, 2)
    return np.where(tiles, 0.0, MASKED).astype(np.float32).reshape(-1, 32)


def make(session, weights, layers, sites=None, run=None):
    """The piece for `sites` (default all three) of `layers`, on `session`'s chip;
    `run`: the shared load (projects/galaxy/load) its classes go to, else its own."""
    from galaxy.reference.model_loader import Checkpoint  # noqa: PLC0415

    assert isinstance(
        weights, Checkpoint
    ), "version 1 converts from the checkpoint (verify --ckpt)"
    return Piece(session, weights, layers, sites, run=run)


def make_compressor(session, weights, layers, run=None):
    """The site `compressor` alone (its class alone in the run)."""
    return make(session, weights, layers, ("compressor",), run=run)


def make_indexer(session, weights, layers, run=None):
    """The site `indexer` alone (its class alone in the run)."""
    return make(session, weights, layers, ("indexer",), run=run)


def make_sparse_attn(session, weights, layers, run=None):
    """The site `sparse_attn` alone (its class alone in the run)."""
    return make(session, weights, layers, ("sparse_attn",), run=run)


def make_compress_index(session, weights, layers, run=None):
    """The sites `compressor` and `indexer` (their two classes, not the attention's)."""
    return make(session, weights, layers, ("compressor", "indexer"), run=run)
