"""Step 2: the indexer of the global attention (DeepSeek-V4.1-Flash), decode shape, on
one Wormhole chip, and the host's choice of the best entries, against float64
(README.md). Example: python indexer.py --entries 1000"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi import formats

STARTED = time.monotonic()
DIM, QR, HEADS, IDX = 5120, 1280, 32, 128  # hidden, q rank, index heads, index head dim
TOPK = 512


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function(fidelity="HiFi2")
def product(x, w):
    return x @ w


@km.tile_function
def rectified(a):
    return km.relu(a)


class Indexer(
    km.Program,
    compute={
        "math_fidelity": "HiFi2",
        "fp32_dest_acc_en": True,
        "dst_full_sync_en": True,
    },
):
    """linear: a K-split product; qrel: q's 32 index heads as rows, rotated; column,
    score, compact: the scores relu(K Q^T) w^T packed a row; append: a new row into a
    cache's row tile (README.md)."""

    APPEND = km.constant(4)
    hq: km.L1[km.bf16, 4, 32, 32]
    h1: km.L1[km.bf16, 1, 32, 32]

    @km.handler
    def linear(
        self,
        x: km.Tensor,
        w: km.Tensor,
        y: km.Tensor,
        y0: int,
        k_tiles: km.constant[int] = 40,
        split: km.constant[int] = 4,
        width: km.constant[int] = 8,
    ):
        share = self.rank % split
        first = self.rank // split * width
        k0 = share * k_tiles // split
        k1 = (share + 1) * k_tiles // split
        acc = km.tile.zeros(km.fp32, width)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0:k1], block=2),
            km.Stream(w.tiles[k0:k1, first : first + width], block=2 * width),
        ):
            acc += product(xb, wb)
        whole = km.reduce(acc, form="tree", group=split)
        if self.rank == self.leader:
            y.pages(y0 + first, width).write(whole)

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
        whole = km.reduce(acc, form="tree", group=split)
        if self.rank == self.leader:
            yx.pages(y0 + first, width).write(whole)

    @km.handler(overlay=True)
    def qrel(self, q: km.Tensor, erow: km.Tensor, rot: km.Tensor, qv: km.Tensor):
        j = self.rank
        h = km.tile.zeros(km.fp32, 4)
        for xb, wb in zip(
            km.Stream(erow.tiles[0:1, 0:HEADS], block=2),
            km.Stream(q.tiles[0:HEADS, 0:4], block=8),
        ):
            h += product(xb, wb)
        self.hq.write(h)
        out = km.tile.zeros(km.fp32, 1)
        for wb in km.Stream(rot.tiles[0:4, j : j + 1], block=4):
            out += product(self.hq, wb)
        qv.pages(j, 1).write(out)

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

    @km.handler(overlay=True)
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

    @km.handler(overlay=True)
    def append(self, cache: km.Tensor, row: km.Tensor, sel: km.Tensor, r: int, m: int):
        c0 = self.rank * self.APPEND
        e = 2 * m + 1
        acc = km.tile.zeros(km.fp32, self.APPEND)
        for xb, wb in zip(
            km.Stream(sel.tiles[0:1, 0:1], block=1),
            km.Stream(cache.tiles[r : r + 1, c0 : c0 + self.APPEND], block=self.APPEND),
        ):
            acc += product(xb, wb)
        for xb, wb in zip(
            km.Stream(sel.tiles[0:1, e : e + 1], block=1),
            km.Stream(row.tiles[0:1, c0 : c0 + self.APPEND], block=self.APPEND),
        ):
            acc += product(xb, wb)
        cache.tiles[r : r + 1, c0 : c0 + self.APPEND].write(acc)


def plan(ranks, kt, nt, chunk=2, widest=8):
    """Of the widths w dividing N's tiles, groups g = nt / w, split s = min(ranks // g,
    K tiles // chunk); the most cores, ties to the narrower w (fewer cores a sum)."""
    best = None
    for w in range(1, min(nt, widest) + 1):
        if nt % w:
            continue
        g = nt // w
        s = min(ranks // g, kt // chunk)
        if s >= 1 and (best is None or g * s > best[0] * best[1]):
            best = (g, s, w)
    g, s, w = best
    return {"k_tiles": kt, "split": s, "width": w, "cores": g * s}


def bf16(values):
    """float32 values rounded to bfloat16 (nearest even), as float32."""
    bits = np.ascontiguousarray(values, np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)


def stored(values, fmt):
    """The values a tensor of `fmt` holds after the host's encode."""
    values = np.ascontiguousarray(values, np.float32)
    raw, _ = formats.to_bytes(values, fmt, "tile")
    return np.asarray(formats.from_bytes(raw, values.shape, fmt, "tile"), np.float32)


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


def packed_scores(session, tensor, groups):
    """The scores from `groups` packed tiles: entry 1024 g + 32 b + a at (a, b) of tile
    g."""
    (raw,) = session.read(tensor, first=0, pages=groups)
    words = np.asarray(raw).view(np.float32).reshape(groups, 1024)
    return words[:, PACKED].reshape(-1)


def erow():
    """E: a row of 32 tiles, tile h a 1 at (h, 0)."""
    e = np.zeros((32, 32 * HEADS), np.float32)
    for h in range(HEADS):
        e[h, 32 * h] = 1.0
    return e


def selectors():
    """[I, E_m] for m = 0 .. 31: 64 tiles, E_m a 1 at (m, 0)."""
    s = np.zeros((32, 64 * 32), np.float32)
    for m in range(32):
        s[:, 64 * m : 64 * m + 32] = np.eye(32, dtype=np.float32)
        s[m, 64 * m + 32] = 1.0
    return s


def rotation(
    pos,
    dim=128,
    rope=64,
    theta=160000.0,
    factor=16,
    beta_fast=32,
    beta_slow=1,
    original=65536,
):
    """blockdiag(I, R(pos)) of the reference's rotary (YaRN with these parameters, as
    precompute_freqs_cis): x_rope @ R rotates each adjacent pair (re, im) by pos x freq.
    """
    freqs = 1.0 / (theta ** (np.arange(0, rope, 2, dtype=np.float64) / rope))
    if original > 0:

        def corrected(rotations):
            return (
                rope * np.log(original / (rotations * 2 * np.pi)) / (2 * np.log(theta))
            )

        low = max(np.floor(corrected(beta_fast)), 0)
        high = min(np.ceil(corrected(beta_slow)), rope - 1)
        ramp = np.clip((np.arange(rope // 2) - low) / max(high - low, 1e-3), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = freqs.astype(np.float32).astype(np.float64)  # the reference's float32 table
    angle = pos * freqs
    c, s = np.cos(angle), np.sin(angle)
    r = np.eye(dim)
    base = dim - rope
    for i in range(rope // 2):
        a = base + 2 * i
        # (re, im) -> (re c - im s, re s + im c): row vector times R
        r[a, a], r[a, a + 1] = c[i], s[i]
        r[a + 1, a], r[a + 1, a + 1] = -s[i], c[i]
    return r


def errors(got, want):
    """Relative RMS error, largest error over the largest value."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    scale = np.sqrt((want**2).mean())
    return {
        "rel_rms": float(np.sqrt(((got - want) ** 2).mean()) / scale),
        "max_over_max": float(np.abs(got - want).max() / np.abs(want).max()),
    }


def read_tiles(session, tensor, first, pages, fmt, cols=32):
    """Pages first .. first + pages of a tile tensor as [32 x pages, cols] float64 rows
    (each page a tile)."""
    (raw,) = session.read(tensor, first=first, pages=pages)
    raw = np.asarray(raw).reshape(pages, -1)
    tiles = [
        np.asarray(formats.from_bytes(raw[p], (32, 32), fmt, "tile"), np.float64)
        for p in range(pages)
    ]
    return np.concatenate(tiles)[:, :cols]


COLUMN0 = np.concatenate(
    [np.arange(16) * 16, 512 + np.arange(16) * 16]
)  # (r, 0) in a tile


def column0(session, tensor, pages):
    """Column 0 of the first `pages` tiles of a float32 tile tensor, in row order: each
    tile's rows 0-15 are face 0 (column 0 at 16 r), rows 16-31 face 2 (at 512 + 16 r).
    """
    (raw,) = session.read(tensor, first=0, pages=pages)
    words = np.asarray(raw).view(np.float32).reshape(pages, 1024)
    return words[:, COLUMN0].reshape(-1)


def topk(scores, k):
    """The best k entries, sorted by position (the reference's top-k then sort)."""
    k = min(k, len(scores))
    return np.sort(np.argpartition(-scores, k - 1)[:k])


def submit(run, handler, group, words, repeat):
    times = []
    for _ in range(repeat):
        event = run.wait(run.submit(handler, group, args=words), timeout=30.0)
        if event.kind != "done":
            raise km.KomorebiError(f"{getattr(handler, '__name__', handler)}: {event}")
        times.append(event.microseconds)
    return round(float(np.median(times)), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--entries", type=int, default=1000, help="index keys in the cache"
    )
    parser.add_argument("--pos", type=int, default=2001, help="the query's position")
    parser.add_argument("--card", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dump", help="an .npz of the scores, got and wanted")
    parser.add_argument("--form", default="runtime", choices=("kernel", "runtime"))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    t, tt = args.entries, -(-args.entries // 32)
    cap = tt + 1  # the cache's row tiles: one more for the appended entry
    qr = np.zeros((32, QR), np.float32)
    qr[0] = bf16(rng.standard_normal(QR))
    x = np.zeros((32, DIM), np.float32)
    x[0] = bf16(rng.standard_normal(DIM))
    wb = stored(rng.standard_normal((QR, HEADS * IDX)) / np.sqrt(QR), "bfp8_b")
    wp = bf16(rng.standard_normal((DIM, HEADS)) / np.sqrt(DIM) / 64.0)
    kc = np.zeros((32 * cap, IDX), np.float32)
    kc[:t] = bf16(rng.standard_normal((t, IDX)))
    newk = np.zeros((32, IDX), np.float32)
    newk[0] = bf16(rng.standard_normal(IDX))
    rot = rotation(args.pos)
    rot_b = bf16(rot.astype(np.float32))
    report = {"entries": t, "pos": args.pos}
    with km.Session(card=args.card, log=log) as session:
        # weights_proj's tree on the 7 ranks past the score's rectangle (72 .. 78): no
        # leader of a tree is a core of the score
        pq, pw = plan(session.ranks, QR // 32, HEADS * IDX // 32), plan(
            session.ranks - 72, DIM // 32, 1
        )
        report["plans"] = {"wq_b": pq, "weights_proj": pw}
        tq = session.tensor(qr, "bfloat16", "tile")
        tx = session.tensor(x, "bfloat16", "tile")
        twb = session.tensor(wb, "bfp8_b", "tile")
        twp = session.tensor(wp, "bfloat16", "tile")
        tqh = session.empty((32 * HEADS, IDX), "bfloat16", "tile")  # q, page h x 4 + j
        tw = session.empty((32, 32), "bfloat16", "tile")
        twc = session.empty(
            (32, 32 * 72), "bfloat16", "tile"
        )  # a page a core of the column's group
        te = session.tensor(erow(), "bfloat16", "tile")
        trot = session.tensor(rot_b, "bfloat16", "tile")
        tqv = session.empty((32, IDX), "bfloat16", "tile")
        tkc = session.tensor(kc, "bfloat16", "tile")
        tsel = session.tensor(selectors(), "bfloat16", "tile")
        tnew = session.tensor(newk, "bfloat16", "tile")
        groups = -(-tt // 32)
        tsc = session.empty(
            (32, 32 * 32 * groups), "float32", "tile"
        )  # page r: entry row tile r
        tep = session.tensor(eprime(), "float32", "tile")
        tpk = session.empty((32, 32 * groups), "float32", "tile")
        sets = [{k: p[k] for k in ("k_tiles", "split", "width")} for p in (pq, pw)]
        program = Indexer(
            x=tq,
            w=twb,
            y=tqh,
            q=tqh,
            erow=te,
            rot=trot,
            qv=tqv,
            kc=tkc,
            wv=tw,
            wc=twc,
            scores=tsc,
            ep=tep,
            packed=tpk,
            cache=tkc,
            row=tnew,
            sel=tsel,
            session=session,
            log=log,
            xh=tx,
            wx=twp,
            yx=tw,
            handlers={"linear": [sets[0]], "linear_x": [sets[1]]},
            form=args.form,
        )
        run = session.load(program)
        us = {}
        try:
            us["wq_b"] = submit(
                run,
                program.linear,
                (0, pq["cores"]),
                (tq, twb, tqh, 0, *sets[0].values()),
                args.repeat,
            )
            us["weights_proj"] = submit(
                run,
                program.linear_x,
                (72, pw["cores"]),
                (tx, twp, tw, 0, *sets[1].values()),
                args.repeat,
            )
            us["qrel"] = submit(
                run, program.qrel, (0, 4), (tqh, te, trot, tqv), args.repeat
            )
            cores = (
                72  # the broadcasts' rectangle: the grid's whole rows of ranks (8 x 9)
            )
            us["column"] = submit(
                run, program.column, (0, cores), (tsel, tw, twc), args.repeat
            )
            us["score"] = submit(
                run, program.score, (0, cores), (tkc, tqv, twc, tsc, tt), args.repeat
            )
            got_c0 = column0(session, tsc, tt)[:t].astype(np.float64)
            us["compact"] = submit(
                run, program.compact, (0, cores), (tsc, tep, tpk, groups), args.repeat
            )
            t0 = time.monotonic()
            got_s = packed_scores(session, tpk, groups)[:t].astype(np.float64)
            t1 = time.monotonic()
            chosen = topk(got_s, TOPK)
            t2 = time.monotonic()
            us["host_read"] = round((t1 - t0) * 1e6, 1)
            us["host_topk"] = round((t2 - t1) * 1e6, 1)
            # append: the new key (row 0 of tnew) into entry t
            us["append"] = submit(
                run, program.append, (0, 1), (tkc, tnew, tsel, t // 32, t % 32), 1
            )
        finally:
            run.stop()
        got_qh = read_tiles(session, tqh, 0, 4 * HEADS, "bfloat16")
        got_w = read_tiles(session, tw, 0, 1, "bfloat16")[0]
        got_wc = read_tiles(session, twc, 0, 1, "bfloat16")[:, 0]
        got_qv = np.concatenate(
            [read_tiles(session, tqv, j, 1, "bfloat16") for j in range(4)], axis=1
        )
        got_kc = read_tiles(session, tkc, (t // 32) * 4, 4, "bfloat16")
    got_kc = np.concatenate([got_kc[32 * j : 32 * j + 32] for j in range(4)], axis=1)
    # float64 of the stored values, stage by stage
    q = qr[0].astype(np.float64) @ wb.astype(np.float64)
    gq = np.stack(
        [
            np.concatenate([got_qh[32 * (4 * h + j)] for j in range(4)])
            for h in range(HEADS)
        ]
    )
    report["wq_b"] = errors(gq, q.reshape(HEADS, IDX))
    w = x[0].astype(np.float64) @ wp.astype(np.float64)
    report["weights_proj"] = errors(got_w, w)
    report["column_exact"] = bool(np.array_equal(got_wc, got_w))
    want_qv = bf16(gq.astype(np.float32)).astype(np.float64) @ rot_b.astype(np.float64)
    report["qrel"] = errors(got_qv, want_qv)
    want_s = np.maximum(kc[:t].astype(np.float64) @ got_qv.T, 0) @ got_w
    report["score"] = errors(got_c0, want_s)
    report["packed"] = errors(got_s, got_c0)
    if args.dump:
        np.savez(args.dump, got=got_c0, packed=got_s, want=want_s)
    want_set = topk(want_s, TOPK)
    report["topk_same"] = int(len(np.intersect1d(chosen, want_set)))
    report["topk_of"] = int(len(want_set))
    # the whole chain from the stored inputs, float64 (rotation exact)
    s_chain = (
        np.maximum(kc[:t].astype(np.float64) @ (q.reshape(HEADS, IDX) @ rot).T, 0) @ w
    )
    report["score_chain"] = errors(got_s, s_chain)
    report["topk_same_chain"] = int(len(np.intersect1d(chosen, topk(s_chain, TOPK))))
    want_tile = kc[32 * (t // 32) : 32 * (t // 32) + 32].copy()
    want_tile[t % 32] = newk[0]
    report["append_exact"] = bool(np.array_equal(got_kc, want_tile))
    report["us"] = us
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
