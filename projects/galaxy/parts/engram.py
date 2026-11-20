"""Engram's projections of DeepSeek-V4.1-Flash on one chip (site engram_mix, layers 1
and 14): the key and value projection and the gate; the table lookup stays on the host
(README.md). `python engram.py card --card N` checks it against float64."""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi.formats import bf16_bits

DIM, HC = 5120, 4
ROWS_IN = 24 * 256  # 6,144: n_hash_cols x head_dim
KV = DIM * (HC + 1)  # 25,600
EPS = 1e-20  # config.json rms_norm_eps
CLAMP = 1e-6
K_TILES, KV_COLS = ROWS_IN // 32, KV // 32  # 192, 800
KV_CORES, WIDTH, PASS = 50, 16, 8  # kv: 50 cores x 16 tile columns, passes of 8
ROW_TILES = DIM // 32  # 160 tiles a stream's row
GROUP, EACH = 10, 16  # dots, norms, gate, mix: 10 cores a stream, 16 tiles a core
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function
def square_sums(row):
    return km.reduce_sum(row * row, exact=True).col()


@km.tile_function
def plain_sums(row):
    return km.reduce_sum(row, exact=True).col()


@km.tile_function
def weighted(h, q, kw, key):
    return h * q.row() * kw.row() * key


@km.tile_function
def row_max(row):
    return km.reduce_max(row, exact=True).col()


@km.tile_function
def norm_factor(t, a, b):
    norm = (a.col() * (1.0 / DIM) + EPS) * (b.col() * (1.0 / DIM) + EPS)
    return t * 0.0 + km.rsqrt(norm) * (DIM**-0.5)


@km.tile_function
def signed_root(t, d, f):
    dot = d.col() * f.col()
    m = km.maximum(abs(dot), CLAMP)
    return km.where(dot < 0.0, m * km.rsqrt(m) * -1.0, m * km.rsqrt(m)) + t * 0.0


@km.tile_function
def gated(h, v, g):
    return h + km.sigmoid(g.col()) * v


# one set of compute options for every class of a load
COMPUTE = {"math_fidelity": "HiFi2", "dst_full_sync_en": True, "fp32_dest_acc_en": True}


class Engram(km.Program, compute=COMPUTE):
    """key_value on cores [0, KV_CORES); dots, norms, gate and mix on groups of GROUP
    cores, one a stream c (module docstring). Built in the runtime's form: the kernel
    form's writer is over NCRISC's code region. Each handler opens its write stream
    first, so that every handler's write takes the same pair of buffer indices."""

    K_TILES = km.constant(K_TILES)
    WIDTH = km.constant(WIDTH)
    PASS = km.constant(PASS)
    KV_COLS = km.constant(KV_COLS)
    CHUNK = km.constant(2)

    EACH = km.constant(EACH)
    ROW = km.constant(ROW_TILES)
    VALUE = km.constant(HC * ROW_TILES)
    GROUP = km.constant(GROUP)
    s0: km.L1[km.fp32, 1, 32, 32]
    s1: km.L1[km.fp32, 1, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def key_value(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, at: int):
        for part in range(WIDTH // PASS):
            c0 = self.rank * self.WIDTH + part * self.PASS
            r0 = at + c0  # W: the pages from `at` of a one-column arena
            acc = km.tile.zeros(km.fp32, self.PASS)
            for xb, wb in zip(
                km.Stream(x.tiles[0:1, 0 : self.K_TILES], block=self.CHUNK),
                km.Stream(
                    w.tiles[
                        r0 : r0 + self.K_TILES * self.KV_COLS : self.KV_COLS,
                        0 : self.PASS,
                    ],
                    block=self.CHUNK * self.PASS,
                ),
            ):
                acc += self.project(xb, wb)
            y.tiles[0:1, c0 : c0 + self.PASS].write(acc)

    @km.handler
    def dots(
        self,
        xs: km.Tensor,
        kv: km.Tensor,
        q: km.Tensor,
        kw: km.Tensor,
        p: km.Tensor,
        c: int,
    ):
        t0 = self.rank * self.EACH
        k0 = c * self.ROW + t0
        out = km.Stream(p.tiles[c : c + 1, t0 : t0 + self.EACH], write=True, block=2)
        for h, qt, kt, key in zip(
            km.Stream(xs.tiles[c : c + 1, t0 : t0 + self.EACH], block=2),
            km.Stream(q.tiles[0:1, k0 : k0 + self.EACH], block=2),
            km.Stream(kw.tiles[0:1, k0 : k0 + self.EACH], block=2),
            km.Stream(kv.tiles[0:1, k0 : k0 + self.EACH], block=2),
        ):
            out.push(weighted(h, qt, kt, key))

    @km.handler
    def norms(self, xs: km.Tensor, kv: km.Tensor, a: km.Tensor, c: int):
        t0 = self.rank * self.EACH
        k0 = c * self.ROW + t0
        factor = km.Stream(a.pages(c * self.GROUP + self.rank, 1), write=True, block=1)
        for row in km.Stream(
            xs.tiles[c : c + 1, t0 : t0 + self.EACH], block=2, row=self.EACH
        ):
            self.s0.write(square_sums(row))
        for row in km.Stream(
            kv.tiles[0:1, k0 : k0 + self.EACH], block=2, row=self.EACH
        ):
            self.s1.write(square_sums(row))
        km.all_reduce(self.s0)
        km.all_reduce(self.s1)
        for t in km.Stream(xs.tiles[c : c + 1, t0 : t0 + 1], block=1):
            factor.push(norm_factor(t, self.s0, self.s1))

    @km.handler
    def gate(self, xs: km.Tensor, p: km.Tensor, a: km.Tensor, g: km.Tensor, c: int):
        t0 = self.rank * self.EACH
        root = km.Stream(g.pages(c * self.GROUP + self.rank, 1), write=True, block=1)
        for row in km.Stream(
            p.tiles[c : c + 1, t0 : t0 + self.EACH], block=2, row=self.EACH
        ):
            self.s0.write(plain_sums(row))
        km.all_reduce(self.s0)
        acc = km.tile.zeros(km.fp32)
        for t in km.Stream(a.pages(c * self.GROUP + self.rank, 1), block=1):
            acc += t
        self.s1.write(acc)
        for t in km.Stream(xs.tiles[c : c + 1, t0 : t0 + 1], block=1):
            root.push(signed_root(t, self.s0, self.s1))

    @km.handler
    def mix(self, xs: km.Tensor, kv: km.Tensor, g: km.Tensor, out: km.Tensor, c: int):
        t0 = self.rank * self.EACH
        v0 = self.VALUE + t0
        result = km.Stream(
            out.tiles[c : c + 1, t0 : t0 + self.EACH], write=True, block=2
        )
        acc = km.tile.zeros(km.fp32)
        for t in km.Stream(g.pages(c * self.GROUP + self.rank, 1), block=1):
            acc += t
        self.s0.write(acc)
        for h, v in zip(
            km.Stream(xs.tiles[c : c + 1, t0 : t0 + self.EACH], block=2),
            km.Stream(kv.tiles[0:1, v0 : v0 + self.EACH], block=2),
        ):
            result.push(gated(h, v, self.s0))


def as_bf16(values):
    """float32 values rounded to bf16, as float32."""
    values = np.asarray(values, np.float32)
    bits = bf16_bits(values).astype(np.uint32) << 16
    return bits.view(np.float32).reshape(values.shape)


class EngramMix:
    """Engram on a session, wkv the pages from `at` of the arena `w` (the loader's
    layers.N.engram.wkv) or its own: `set_weights(wkv, q, k)` ([out, in] or None; q, k
    [4, 5120]); `__call__(x, rows)`, x [s, 4, 5120], rows [s, 24, 256] -> (out, kv,
    microseconds)."""

    def __init__(self, session, w=None, at=0):
        self.session = session
        s = session
        self.rows = s.empty((32, ROWS_IN), "bfloat16", "tile")
        self.own = w is None
        if self.own:  # an arena's form: one tile column, the held pages in order
            w = s.empty((K_TILES * KV_COLS * 32, 32), "bfloat16", "tile")
        self.w, self.at = w, at
        self.kv = s.empty((32, KV), "bfloat16", "tile")
        self.xs = s.empty((HC * 32, DIM), "bfloat16", "tile")
        self.q = s.empty((32, HC * DIM), "bfloat16", "tile")
        self.kw = s.empty((32, HC * DIM), "bfloat16", "tile")
        self.p = s.empty((HC * 32, DIM), "float32", "tile")
        self.out = s.empty((HC * 32, DIM), "float32", "tile")
        self.g = s.empty((HC * GROUP * 32, 32), "float32", "tile")
        self.a = s.empty((HC * GROUP * 32, 32), "float32", "tile")
        self.kp = self.gp = Engram(
            log=log,
            session=s,
            form="runtime",
            x=self.rows,
            w=self.w,
            y=self.kv,
            kv=self.kv,
            xs=self.xs,
            q=self.q,
            kw=self.kw,
            p=self.p,
            a=self.a,
            g=self.g,
            out=self.out,
        )
        self.run = s.load(self.kp)

    def put(self, tensor, values, dtype):
        raw, _ = formats.to_bytes(
            np.ascontiguousarray(values, np.float32), dtype, "tile"
        )
        self.session.write(
            tensor, np.asarray(raw).view(np.uint8).reshape(tensor.pages, -1)
        )

    def get(self, tensor, shape, dtype):
        (raw,) = self.session.read(tensor, first=0, pages=tensor.pages)
        raw = np.asarray(raw).reshape(-1).view(np.uint8)
        return np.asarray(formats.from_bytes(raw, shape, dtype, "tile"))

    def set_weights(self, wkv, q, k):
        if wkv is not None:
            raw, _ = formats.to_bytes(np.asarray(wkv, np.float32).T, "bfloat16", "tile")
            self.session.write(
                self.w, np.asarray(raw).view(np.uint8).reshape(self.w.pages, -1)
            )
        for tensor, rows in ((self.q, q), (self.kw, k)):
            v = np.zeros((32, HC * DIM), np.float32)
            v[0] = np.asarray(rows, np.float32).reshape(-1)
            self.put(tensor, v, "bfloat16")

    def _wait(self, tickets, name):
        us = []
        for ticket in tickets:
            event = self.run.wait(ticket, timeout=30.0)
            if event.kind != "done":
                raise km.KomorebiError(f"{name}: {event}")
            us.append(event.microseconds)
        return max(us)

    def __call__(self, x, rows):
        s = x.shape[0]
        r = np.zeros((32, ROWS_IN), np.float32)
        r[:s] = np.asarray(rows, np.float32).reshape(s, ROWS_IN)
        xs = np.zeros((HC * 32, DIM), np.float32)
        for c in range(HC):
            xs[c * 32 : c * 32 + s] = x[:, c]
        self.put(self.rows, r, "bfloat16")
        self.put(self.xs, xs, "bfloat16")
        run, kp, gp = self.run, self.kp, self.gp
        us = {
            "kv": self._wait(
                [
                    run.submit(
                        kp.key_value,
                        (0, KV_CORES),
                        args=(self.rows, self.w, self.kv, self.at),
                    )
                ],
                "kv",
            )
        }
        groups = [(c * GROUP, GROUP) for c in range(HC)]
        us["dots"] = self._wait(
            [
                run.submit(
                    kp.dots, g, args=(self.xs, self.kv, self.q, self.kw, self.p, c)
                )
                for c, g in enumerate(groups)
            ],
            "dots",
        )
        us["norms"] = self._wait(
            [
                run.submit(gp.norms, g, args=(self.xs, self.kv, self.a, c))
                for c, g in enumerate(groups)
            ],
            "norms",
        )
        us["gate"] = self._wait(
            [
                run.submit(gp.gate, g, args=(self.xs, self.p, self.a, self.g, c))
                for c, g in enumerate(groups)
            ],
            "gate",
        )
        us["mix"] = self._wait(
            [
                run.submit(gp.mix, g, args=(self.xs, self.kv, self.g, self.out, c))
                for c, g in enumerate(groups)
            ],
            "mix",
        )
        out = self.get(self.out, (HC * 32, DIM), "float32").reshape(HC, 32, DIM)[:, :s]
        out = as_bf16(out)  # the reference's .to(bfloat16) of its float32 sum
        kv = self.get(self.kv, (32, KV), "bfloat16")[:s]
        return out.transpose(1, 0, 2), kv, us

    def stop(self):
        self.run.stop()


def reference(x, rows, wkv, q, k):
    """float64 of the stored values, the reference's roundings kept (kv to bf16)."""
    s = x.shape[0]
    kv = as_bf16(rows.reshape(s, ROWS_IN).astype(np.float64) @ wkv.T.astype(np.float64))
    key = kv[:, : HC * DIM].reshape(s, HC, DIM).astype(np.float64)
    value = kv[:, HC * DIM :].astype(np.float64)
    weight = q.astype(np.float64) * k.astype(np.float64)
    h = x.astype(np.float64)
    rstd = 1 / np.sqrt((h**2).mean(-1) + EPS) / np.sqrt((key**2).mean(-1) + EPS)
    dot = (h * weight * key).sum(-1) * rstd * DIM**-0.5
    gate = 1 / (1 + np.exp(-np.copysign(np.sqrt(np.maximum(np.abs(dot), CLAMP)), dot)))
    return h + gate[..., None] * value[:, None, :], kv, gate


def errors(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return {
        "rel_rms": float(np.linalg.norm(got - want) / np.linalg.norm(want)),
        "max_err": float(np.abs(got - want).max()),
        "max_ref": float(np.abs(want).max()),
    }


def card(args):
    """Random bf16 streams (scale 1), rows of the table's magnitude, wkv of the released
    magnitude, q and k near 1."""
    rng = np.random.default_rng(args.seed)
    s = args.tokens
    x = as_bf16(rng.standard_normal((s, HC, DIM)))
    rows = as_bf16(rng.standard_normal((s, 24, 256)) * 0.1)
    wkv = as_bf16(rng.standard_normal((KV, ROWS_IN)) * 0.02)
    q = as_bf16(1 + 0.2 * rng.standard_normal((HC, DIM)))
    k = as_bf16(1 + 0.2 * rng.standard_normal((HC, DIM)))
    want, want_kv, gate = reference(x, rows, wkv, q, k)
    with km.Session(log=log, card=args.card) as session:
        em = EngramMix(session)
        try:
            em.set_weights(wkv, q, k)
            runs = [em(x, rows) for _ in range(args.repeat)]
        finally:
            em.stop()
    got, kv, _ = runs[-1]
    report = {
        "tokens": s,
        "kv": errors(kv, want_kv),
        "out": errors(got, want),
        "out_minus_h": errors(got - x, want - x),
        "bf16_of_ref": errors(as_bf16(want), want),
        "gate_range": [float(gate.min()), float(gate.max())],
        "us": [r[2] for r in runs],
    }
    print(json.dumps(report), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["card"])
    p.add_argument("--card", type=int, default=0)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=4)
    args = p.parse_args()
    return {"card": card}[args.what](args)


if __name__ == "__main__":
    sys.exit(main())
