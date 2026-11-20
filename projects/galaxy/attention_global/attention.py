"""Step 3: the decode attention of the 64 query heads over 128 window keys and 512
gathered entries in one softmax with the sink (DeepSeek-V4.1-Flash), on one Wormhole
chip, against float64 (README.md). Example: python attention.py --entries 1000"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi.dsl.tile import row_operand

STARTED = time.monotonic()
HEADS, HEAD = 64, 512
D = HEAD // 32  # tiles of a head
WINDOW = 4  # row tiles of the window's keys (128)
SELECTED = 16  # row tiles of the selected keys (512)
KEYS = WINDOW + SELECTED
COLS = KEYS + 2  # a score row: the keys, the sink column, one masked column
SCALE = HEAD**-0.5
MASKED = -1.0e30  # a key not to attend: its score, after the mask


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function(fidelity="HiFi2")
def product(x, w):
    return x @ w


@km.tile_function
def softmax_row(row, mask):
    x = row * SCALE + mask.row()
    m = km.reduce_max(x, exact=True)
    e = km.exp(x - m.col())
    total = km.reduce_sum(e, exact=True)
    return e / total.col()


class Pair(km.Entry):
    r: int  # the cache's row tile
    g: int  # the selection tile's page


class Attend(
    km.Program,
    compute={
        "math_fidelity": "HiFi2",
        "fp32_dest_acc_en": True,
        "dst_full_sync_en": True,
    },
):
    """gather: the selected entries into the key rows by 0/1 selection tiles; scores:
    S = Q K^T; softmax: a head row's scale, masks and sink; pv: O = P K; unrope: o's
    last 64 dims rotated back (README.md)."""

    mask: km.L1[km.fp32, COLS // 2, 32, 32]

    @km.handler(overlay=True)
    def gather(
        self,
        kc: km.Tensor,
        kwin: km.Tensor,
        gsel: km.Tensor,
        table: km.Tensor,
        kall: km.Tensor,
    ):
        if self.rank < 64:
            i = self.rank // 4
            c0 = self.rank % 4 * 4
            acc = km.tile.zeros(km.fp32, 4)
            for e in table.pages(i, 1).entries(Pair, first=8, size=2, count=0):
                for xb, wb in zip(
                    km.Stream(gsel.tiles[0:1, e.g : e.g + 1], block=1),
                    km.Stream(kc.tiles[e.r : e.r + 1, c0 : c0 + 4], block=4),
                ):
                    acc += product(xb, wb)
            kall.tiles[WINDOW + i : WINDOW + i + 1, c0 : c0 + 4].write(acc)
        else:
            r = (self.rank - 64) // 2
            c1 = (self.rank - 64) % 2 * 8
            win = km.tile.zeros(km.fp32, 8)
            for xb, wb in zip(
                km.Stream(gsel.tiles[0:1, 0:1], block=1),
                km.Stream(kwin.tiles[r : r + 1, c1 : c1 + 8], block=8),
            ):
                win += product(xb, wb)
            kall.tiles[r : r + 1, c1 : c1 + 8].write(win)

    @km.handler
    def scores(self, q: km.Tensor, kall: km.Tensor, s: km.Tensor):
        t = self.rank % (2 * KEYS)
        b = t // KEYS
        k = t % KEYS
        own = km.Broadcast(self, along="column", length=1, mover="A")
        if own.sender():  # noqa: SIM108 (the class language takes a statement)
            blk = own.cast(kall.tiles[k : k + 1, 0:D].T)
        else:
            blk = own.arrive()
        acc = km.tile.zeros(km.fp32, 1)
        for xb in km.Stream(q.tiles[b : b + 1, 0:D], block=4):
            acc += product(xb, blk)
        s.pages(b * COLS + k, 1).write(acc)

    @km.handler(overlay=True)
    def softmax(self, s: km.Tensor, wm: km.Tensor, p: km.Tensor):
        b = self.rank
        self.mask.fill(km.Stream(wm.pages(0, COLS // 2)))
        results = km.Stream(p.pages(b * COLS, COLS), write=True, block=2)
        for row in km.Stream(s.pages(b * COLS, COLS), block=2, row=COLS):
            results.push(softmax_row(row, self.mask))

    @km.handler(overlay=True)
    def pv(self, p: km.Tensor, kall: km.Tensor, o: km.Tensor):
        b = self.rank // D
        c = self.rank % D
        acc = km.tile.zeros(km.fp32, 1)
        for xb, wb in zip(
            km.Stream(p.tiles[b : b + 1, 0:KEYS], block=4),
            km.Stream(kall.tiles[0:KEYS, c : c + 1], block=4),
        ):
            acc += product(xb, wb)
        o.pages(b * D + c, 1).write(acc)

    @km.handler(overlay=True)
    def unrope(self, o: km.Tensor, rinv: km.Tensor):
        b = self.rank
        acc = km.tile.zeros(km.fp32, 2)
        for xb, wb in zip(
            km.Stream(o.tiles[b : b + 1, D - 2 : D], block=2),
            km.Stream(rinv.tiles[0:2, 0:2], block=4),
        ):
            acc += product(xb, wb)
        o.tiles[b : b + 1, D - 2 : D].write(acc)


def bf16(values):
    """float32 values rounded to bfloat16 (nearest even), as float32."""
    bits = np.ascontiguousarray(values, np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)


def rope_matrix(
    pos, rope=64, theta=160000.0, factor=16, beta_fast=32, beta_slow=1, original=65536
):
    """R(pos), 64 x 64: a row vector of the rotary dims times R rotates each adjacent
    pair (re, im) by pos x freq (precompute_freqs_cis with YaRN, its float32 table)."""
    freqs = 1.0 / (theta ** (np.arange(0, rope, 2, dtype=np.float64) / rope))
    if original > 0:

        def corrected(rotations):
            return (
                rope * np.log(original / (rotations * 2 * np.pi)) / (2 * np.log(theta))
            )

        low = max(np.floor(corrected(beta_fast)), 0)
        high = min(np.ceil(corrected(beta_slow)), rope - 1)
        ramp = np.clip((np.arange(rope // 2) - low) / max(high - low, 1e-3), 0, 1)
        freqs = freqs / factor * ramp + freqs * (1 - ramp)
    freqs = freqs.astype(np.float32).astype(np.float64)
    angle = pos * freqs
    c, s = np.cos(angle), np.sin(angle)
    r = np.zeros((rope, rope))
    for i in range(rope // 2):
        r[2 * i, 2 * i], r[2 * i, 2 * i + 1] = c[i], s[i]
        r[2 * i + 1, 2 * i], r[2 * i + 1, 2 * i + 1] = -s[i], c[i]
    return r


def selection(chosen, gather_pages=None):
    """From the chosen entries (sorted): G tiles (page 0 the identity) and each row tile
    i's pairs (r, g): G_g row j has a 1 at column a when entry 32 i + j of the selection
    is row a of the cache's row tile r."""
    tiles = [np.eye(32, dtype=np.float32)]
    pairs = [[] for _ in range(SELECTED)]
    for i in range(SELECTED):
        part = chosen[32 * i : 32 * i + 32]
        for r in np.unique(part // 32):
            g = np.zeros((32, 32), np.float32)
            for j, e in enumerate(part):
                if e // 32 == r:
                    g[j, e % 32] = 1.0
            pairs[i].append((int(r), len(tiles)))
            tiles.append(g)
    return np.concatenate(tiles, axis=1), pairs


def table_pages(pairs, words=256):
    """A page of words a selected row tile: the count at word 0, pairs from word 8."""
    t = np.zeros((SELECTED, words), np.uint32)
    for i, ps in enumerate(pairs):
        t[i, 0] = len(ps)
        for n, (r, g) in enumerate(ps):
            t[i, 8 + 2 * n], t[i, 9 + 2 * n] = r, g
    return t


def mask_values(window_valid, selected_count):
    """A score row's additive mask: 0 where a key is attended, MASKED elsewhere; the
    sink column (KEYS x 32) 0 at its column 0."""
    m = np.full(COLS * 32, MASKED, np.float32)
    m[: 32 * WINDOW][window_valid] = 0.0
    m[32 * WINDOW : 32 * WINDOW + selected_count] = 0.0
    m[32 * KEYS] = 0.0
    return m


def mask_pages(values):
    """The mask field's pages: chunk c's per-column tile of its 64 values."""
    return np.asarray(
        [row_operand(values[64 * c : 64 * c + 64], 2) for c in range(COLS // 2)],
        np.float32,
    )


def errors(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    scale = np.sqrt((want**2).mean())
    return {
        "rel_rms": float(np.sqrt(((got - want) ** 2).mean()) / scale),
        "max_over_max": float(np.abs(got - want).max() / np.abs(want).max()),
    }


def tiles_of(session, tensor, rows, cols, fmt):
    (raw,) = session.read(tensor)
    return np.asarray(
        formats.from_bytes(np.asarray(raw).reshape(-1), (rows, cols), fmt, "tile"),
        np.float64,
    )


def submit(run, handler, group, words, repeat):
    times = []
    for _ in range(repeat):
        event = run.wait(run.submit(handler, group, args=words), timeout=30.0)
        if event.kind != "done":
            raise km.KomorebiError(f"{getattr(handler, '__name__', handler)}: {event}")
        times.append(event.microseconds)
    return round(float(np.median(times)), 2)


def reference(q, kwin, kc, chosen, window_valid, sink, rinv):
    """float64: one softmax over [window | chosen] and the sink, o = P K, o's last 64
    dims times R^T."""
    keys = np.concatenate([kwin[window_valid], kc[chosen]]).astype(np.float64)
    s = q.astype(np.float64) @ keys.T * SCALE
    m = np.maximum(s.max(1), sink)
    e = np.exp(s - m[:, None])
    p = e / (e.sum(1) + np.exp(sink - m))[:, None]
    o = p @ keys
    o[:, -64:] = o[:, -64:] @ rinv
    return o


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--entries", type=int, default=1000, help="entries in the compressed cache"
    )
    parser.add_argument("--pos", type=int, default=2001)
    parser.add_argument("--card", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    t = args.entries
    q = bf16(rng.standard_normal((HEADS, HEAD)) * 2.0)
    kwin = bf16(rng.standard_normal((32 * WINDOW, HEAD)))
    kc = np.zeros((32 * (-(-t // 32)), HEAD), np.float32)
    kc[:t] = bf16(rng.standard_normal((t, HEAD)))
    chosen = np.sort(rng.choice(t, size=min(512, t), replace=False))
    window_valid = np.ones(32 * WINDOW, bool)
    if args.pos < 32 * WINDOW:
        window_valid[args.pos + 1 :] = False
    sink = rng.standard_normal(HEADS).astype(np.float32)
    rinv = rope_matrix(args.pos).T
    rinv_b = bf16(rinv.astype(np.float32))
    gsel, pairs = selection(chosen)
    s_init = np.zeros((HEADS, COLS * 32), np.float32)
    s_init[:, 32 * KEYS] = sink / SCALE  # the sink column, written once a layer
    rinv_pad = np.zeros((64, 64), np.float32)
    rinv_pad[:] = rinv_b
    report = {"entries": t, "pos": args.pos, "pairs": int(sum(len(p) for p in pairs))}
    with km.Session(card=args.card, log=log) as session:
        tq = session.tensor(q, "bfloat16", "tile")
        tkw = session.tensor(kwin, "bfloat16", "tile")
        tkc = session.tensor(kc, "bfloat16", "tile")
        tg = session.tensor(gsel, "bfloat16", "tile")
        ttab = session.empty((SELECTED, 256), "uint32")
        session.write(ttab, table_pages(pairs).view(np.uint8).reshape(SELECTED, -1))
        tall = session.empty((32 * KEYS, HEAD), "bfloat16", "tile")
        ts = session.tensor(s_init, "float32", "tile")
        tm = session.empty((COLS // 2, 1024), "float32")
        session.write(
            tm,
            mask_pages(mask_values(window_valid, len(chosen)))
            .view(np.uint8)
            .reshape(COLS // 2, -1),
        )
        tp = session.empty((HEADS, COLS * 32), "bfloat16", "tile")
        to = session.empty((HEADS, HEAD), "bfloat16", "tile")
        tr = session.tensor(rinv_pad, "bfloat16", "tile")
        program = Attend(
            kc=tkc,
            kwin=tkw,
            gsel=tg,
            table=ttab,
            kall=tall,
            q=tq,
            s=ts,
            wm=tm,
            p=tp,
            o=to,
            rinv=tr,
            session=session,
            log=log,
            form="runtime",
        )
        run = session.load(program)
        us = {}
        try:
            us["gather"] = submit(
                run, Attend.gather, (0, 72), (tkc, tkw, tg, ttab, tall), args.repeat
            )
            us["scores"] = submit(
                run, Attend.scores, (0, 72), (tq, tall, ts), args.repeat
            )
            us["softmax"] = submit(
                run, Attend.softmax, (0, 2), (ts, tm, tp), args.repeat
            )
            us["pv"] = submit(run, Attend.pv, (0, 2 * D), (tp, tall, to), args.repeat)
            got_pv = tiles_of(session, to, HEADS, HEAD, "bfloat16")
            us["unrope"] = submit(run, Attend.unrope, (0, 2), (to, tr), 1)
        finally:
            run.stop()
        got_all = tiles_of(session, tall, 32 * KEYS, HEAD, "bfloat16")
        got_s = tiles_of(session, ts, HEADS, COLS * 32, "float32")
        got_p = tiles_of(session, tp, HEADS, COLS * 32, "bfloat16")
        got_o = tiles_of(session, to, HEADS, HEAD, "bfloat16")
    # stage by stage against float64 of the stored values
    want_all = np.zeros((32 * KEYS, HEAD))
    want_all[: 32 * WINDOW] = kwin
    want_all[32 * WINDOW : 32 * WINDOW + len(chosen)] = kc[chosen]
    report["gather_exact"] = bool(np.array_equal(got_all, want_all))
    want_s = q.astype(np.float64) @ got_all.T
    report["scores"] = errors(got_s[:, : 32 * KEYS], want_s)
    x = got_s * SCALE + mask_values(window_valid, len(chosen))
    e = np.exp(x - x.max(1, keepdims=True))
    want_p = e / e.sum(1, keepdims=True)
    report["softmax"] = errors(got_p, want_p)
    want_pv = got_p[:, : 32 * KEYS] @ got_all
    report["pv"] = errors(got_pv, want_pv)
    want_o = got_pv.copy()
    want_o[:, -64:] = got_pv[:, -64:] @ rinv_b.astype(np.float64)
    report["unrope"] = errors(got_o, want_o)
    report["whole"] = errors(
        got_o, reference(q, kwin, kc, chosen, window_valid, sink, rinv)
    )
    report["us"] = us
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
