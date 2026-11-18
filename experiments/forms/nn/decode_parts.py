"""Two parts of a decode layer written from the existing forms, against numpy: one
attention step of a small head a core (scores Q K^T, scaled and masked causally, a
row softmax, the weighted sum with V), as commands and with --chain as one chain; and
an RMS norm over a token's row of tiles (README.md)."""

import argparse
import sys

import numpy as np
from forms import (
    across_list,
    card_options,
    copied,
    error,
    finish,
    group,
    log,
    open_session,
    placed,
    random,
    read,
    tiles,
    timed,
)
from norms import CHUNKS, WIDE, field_pages

import komorebi as km

HEADS, DT, ST = 8, 2, 4  # heads, a head's 64 values in tiles, 128 keys in tiles
SCALE = 1.0 / np.sqrt(DT * 32)
RMS_ROWS = 4  # tile rows of the RMS norm's tokens (128), a core each
EPS = 1e-6


@km.tile_function(tolerance=1e-5)
def mask_scale(s, mask):
    return s * SCALE + mask


@km.tile_function(tolerance=1e-5)
def softmax(row):
    m = km.reduce_max(row, exact=True)
    e = km.exp(row - m.col())
    return e * km.reciprocal(km.reduce_sum(e, exact=True)).col()


@km.tile_function(tolerance=1e-5)
def rms_norm(row, weight):
    ms = km.reduce_sum(row * row, exact=True) * (1.0 / (32 * WIDE))
    return row * km.rsqrt(ms + EPS).col() * weight.row()


class Attention(km.Program):
    """Head h (rank h): Q's tile row h, K^T's tile rows h DT to h DT + DT - 1, V's
    tile rows h ST to h ST + ST - 1; scores, probabilities and O in tile row h."""

    @km.tile_function(fidelity="HiFi4")
    def product(a, b):
        return a @ b

    @km.handler
    def scores(self, q: km.Tensor, kt: km.Tensor, s: km.Tensor):
        h = self.rank
        acc = km.tile.zeros(km.fp32, ST)
        for qb, kb in zip(
            km.Stream(q.tiles[h : h + 1, 0:DT], block=1),
            km.Stream(kt.tiles[h * DT : h * DT + DT, 0:ST], block=ST),
        ):
            acc += self.product(qb, kb)
        s.tiles[h : h + 1, 0:ST].write(acc)

    @km.handler
    def masked(self, s: km.Tensor, mask: km.Tensor, m: km.Tensor):
        h = self.rank
        results = km.Stream(m.tiles[h : h + 1, 0:ST], write=True, block=ST)
        for a, b in zip(
            km.Stream(s.tiles[h : h + 1, 0:ST], block=ST),
            km.Stream(mask.tiles[0:1, 0:ST], block=ST),
        ):
            results.push(mask_scale(a, b))

    @km.handler
    def probabilities(self, m: km.Tensor, p: km.Tensor):
        h = self.rank
        results = km.Stream(p.tiles[h : h + 1, 0:ST], write=True, block=ST)
        for row in km.Stream(m.tiles[h : h + 1, 0:ST], block=ST, row=ST):
            results.push(softmax(row))

    @km.handler
    def weigh(self, p: km.Tensor, v: km.Tensor, o: km.Tensor):
        h = self.rank
        acc = km.tile.zeros(km.fp32, DT)
        for pb, vb in zip(
            km.Stream(p.tiles[h : h + 1, 0:ST], block=1),
            km.Stream(v.tiles[h * ST : h * ST + ST, 0:DT], block=DT),
        ):
            acc += self.product(pb, vb)
        o.tiles[h : h + 1, 0:DT].write(acc)


class RMSNorm(km.Program):
    """Each core normalises the 32 tokens of its tile row of WIDE tiles."""

    weight: km.L1[km.fp32, CHUNKS, 32, 32]

    @km.handler
    def rows(self, x: km.Tensor, w: km.Tensor, out: km.Tensor):
        first = self.rank * WIDE
        self.weight.fill(km.Stream(w.pages(0, CHUNKS)))
        results = km.Stream(out.pages(first, WIDE), write=True, block=8)
        for row in km.Stream(x.pages(first, WIDE), block=8, row=WIDE):
            results.push(rms_norm(row, self.weight))


def causal_mask():
    """[32, ST x 32]: query i at key position (ST - 1) 32 + i sees keys up to it."""
    keys = np.arange(ST * 32)[None, :]
    queries = (ST - 1) * 32 + np.arange(32)[:, None]
    return np.where(keys <= queries, 0.0, -np.inf)


def attention(q, kt, v, mask):
    """numpy's attention of every head, float64: O as [HEADS 32, DT 32]."""
    out = []
    for h in range(HEADS):
        qh = q[h * 32 : h * 32 + 32]
        kh = kt[h * DT * 32 : (h + 1) * DT * 32]
        vh = v[h * ST * 32 : (h + 1) * ST * 32]
        s = qh @ kh * SCALE + mask
        e = np.exp(s - s.max(axis=1, keepdims=True))
        out.append(e / e.sum(axis=1, keepdims=True) @ vh)
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--chain", action="store_true", help="also as one chain")
    card_options(parser)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    q = random(rng, (HEADS * 32, DT * 32))
    kt = random(rng, (HEADS * DT * 32, ST * 32))
    v = random(rng, (HEADS * ST * 32, DT * 32))
    mask = causal_mask()
    x = random(rng, (RMS_ROWS * 32, WIDE * 32))
    weight = random(rng, WIDE * 32)
    report = {"heads": HEADS, "head": DT * 32, "keys": ST * 32, "place": args.place}
    report["hidden"], report["tokens"] = WIDE * 32, RMS_ROWS * 32
    got = {}
    with open_session(args) as session:
        across = across_list(args)

        def at(count):
            return group(session, args, count)

        heads, own = placed(session, args, HEADS), placed(session, args, RMS_ROWS)
        tq, tkt, tv = (tiles(session, a, **heads) for a in (q, kt, v))
        tmask = tiles(session, mask, "float32", **copied(args))
        scores = np.zeros((HEADS * 32, ST * 32))

        def tensors():
            s, m = (tiles(session, scores, "float32", **heads) for _ in "ab")
            p = tiles(session, scores, **heads)
            o = tiles(session, np.zeros((HEADS * 32, DT * 32)), "float32", **heads)
            return {"s": s, "m": m, "p": p, "o": o}

        def calls(n):
            return [
                ("scores", Attention.scores, (tq, tkt, n["s"])),
                ("masked", Attention.masked, (n["s"], tmask, n["m"])),
                ("softmax", Attention.probabilities, (n["m"], n["p"])),
                ("weigh", Attention.weigh, (n["p"], tv, n["o"])),
            ]

        sets = {"commands": tensors()}
        if args.chain:
            sets["chain"] = tensors()
        made = dict(sets["commands"], q=tq, kt=tkt, v=tv, mask=tmask)
        run = session.load(Attention(**made, session=session, log=log))
        try:
            for name, handler, call in calls(sets["commands"]):
                us = timed(run, handler, at(HEADS), call, across=across)
                report[f"{name}_us"] = us
        finally:
            run.stop()
        got["attention"] = read(session, sets["commands"]["o"])
        got["p"] = read(session, sets["commands"]["p"])
        if args.chain:
            try:
                made = dict(sets["chain"], q=tq, kt=tkt, v=tv, mask=tmask, chain=True)
                run = session.load(Attention(**made, session=session, log=log))
                try:
                    steps = [
                        handler(*call, cores=at(HEADS))
                        for _, handler, call in calls(sets["chain"])
                    ]
                    chain = run.chain(steps, cores=at(HEADS))
                    event = run.wait(run.submit(chain), timeout=30.0)
                    report["chain_event"] = event.kind
                    report["chain_us"] = round(event.microseconds, 1)
                finally:
                    run.stop()
                got["chain"] = read(session, sets["chain"]["o"])
            except Exception as refused:  # noqa: BLE001 - the refusal is the result
                report["chain_refused"] = f"{type(refused).__name__}: {refused}"[:700]
                log(f"one chain: {report['chain_refused']}")
        tx = tiles(session, x, **own)
        tw = session.tensor(field_pages(weight), "float32", **copied(args))
        out = tiles(session, np.zeros_like(x), "float32", **own)
        run = session.load(RMSNorm(x=tx, w=tw, out=out, session=session, log=log))
        try:
            report["rms_norm_us"] = timed(
                run, RMSNorm.rows, at(RMS_ROWS), (tx, tw, out), across=across
            )
        finally:
            run.stop()
        got["rms_norm"] = read(session, out)
    if across is not None:
        report["across"] = across
    want = attention(q, kt, v, mask)
    sums = got["p"].sum(axis=1)  # P's rows (bfloat16) should each sum to 1
    report["p_row_sum"] = {"max": float(np.abs(sums - 1).max())}
    report["p_row_sum"]["rms"] = float(np.sqrt(((sums - 1) ** 2).mean()))
    checks = []
    for name in ("attention", "chain"):
        if name in got:
            report[name] = error(got[name], want)
            checks.append((name, "rms", 0.01))
    ms = (x * x).mean(axis=1, keepdims=True)
    report["rms_norm"] = error(got["rms_norm"], x / np.sqrt(ms + EPS) * weight)
    checks.append(("rms_norm", "rms", 0.01))
    for name, _, _ in checks:
        log(f"{name}: {report[name]}")
    finish(report, checks)


if __name__ == "__main__":
    sys.exit(main())
