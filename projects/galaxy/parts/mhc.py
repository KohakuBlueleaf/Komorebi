"""The four residual streams of DeepSeek-V4.1-Flash and their mixing (mHC): hc_mixes,
hc_pre and hc_post on one chip, s tokens (1 to 32) in one tile row (README.md).
`python mhc.py card --card N` checks the three against float64."""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi.formats import bf16_bits

DIM, HC = 5120, 4
FLAT = DIM * HC  # 20,480
MIX = (2 + HC) * HC  # 24
EPS = 1e-20  # config.json rms_norm_eps (hc_mixes' statistic)
HC_EPS = 1e-6
SINKHORN = 20
K_EACH, CORES = 16, 40  # mixes: tile rows of K a core (640 = 40 x 16)
COLS, EACH = DIM // 32, 4  # pre/post: tile columns, a core's (160 = 40 x 4)
STACK = HC + 1  # Z's tile rows: the 4 streams, then x
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function
def square_sums(row):
    return km.reduce_sum(row * row, exact=True).col()


@km.tile_function
def show_rsqrt(t, s):
    return t * 0.0 + km.rsqrt(s.col() * (1.0 / FLAT) + EPS)


@km.tile_function
def times(a, b):
    return a * b


# one set of compute options for every class of a load
COMPUTE = {"math_fidelity": "HiFi2", "dst_full_sync_en": True, "fp32_dest_acc_en": True}


class Hc(km.Program, compute=COMPUTE):
    """mixes, gather, pre and post (module docstring)."""

    K_EACH = km.constant(K_EACH)
    EACH = km.constant(EACH)
    STACK = km.constant(STACK)
    Z_BLOCK = STACK * EACH
    sq: km.L1[km.fp32, 1, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(a, b):
        return a @ b

    @km.handler
    def mixes(self, x: km.Tensor, w: km.Tensor, m: km.Tensor):
        first = self.rank * self.K_EACH
        for row in km.Stream(
            x.tiles[0:1, first : first + self.K_EACH], block=2, row=self.K_EACH
        ):
            self.sq.write(square_sums(row))
        km.all_reduce(self.sq)
        acc = km.tile.zeros(km.fp32, 1)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, first : first + self.K_EACH], block=2),
            km.Stream(w.tiles[first : first + self.K_EACH, 0:1], block=2),
        ):
            acc += self.project(xb, wb)
        m.pages(self.rank, 1).write(acc)
        if self.rank == self.leader:
            factor = km.Stream(m.pages(CORES, 1), write=True, block=1)
            for t in km.Stream(w.tiles[0:1, 0:1], block=1):
                factor.push(show_rsqrt(t, self.sq))

    @km.handler
    def gather(self, m: km.Tensor, mixed: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for t in km.Stream(m.pages(0, CORES), block=8):
            total += t
        factor = km.tile.zeros(km.fp32)
        for t in km.Stream(m.pages(CORES, 1), block=1):
            factor += t
        mixed.pages(0, 1).write(times(total, factor))

    @km.handler
    def pre(self, d: km.Tensor, z: km.Tensor, out: km.Tensor):
        c0 = self.rank * self.EACH
        acc = km.tile.zeros(km.fp32, self.EACH)
        for db, zb in zip(
            km.Stream(d.tiles[0:1, 0 : self.STACK], block=self.STACK),
            km.Stream(z.tiles[0 : self.STACK, c0 : c0 + self.EACH], block=self.Z_BLOCK),
        ):
            acc += self.project(db, zb)
        out.tiles[0:1, c0 : c0 + self.EACH].write(acc)

    @km.handler
    def post(self, d: km.Tensor, z: km.Tensor, out: km.Tensor):
        c0 = self.rank * self.EACH
        for k in range(HC):
            acc = km.tile.zeros(km.fp32, self.EACH)
            for db, zb in zip(
                km.Stream(d.tiles[k : k + 1, 0 : self.STACK], block=self.STACK),
                km.Stream(
                    z.tiles[0 : self.STACK, c0 : c0 + self.EACH], block=self.Z_BLOCK
                ),
            ):
                acc += self.project(db, zb)
            out.tiles[k : k + 1, c0 : c0 + self.EACH].write(acc)


def as_bf16(values):
    """float32 values rounded to bf16, as float32."""
    values = np.asarray(values, np.float32)
    bits = bf16_bits(values).astype(np.uint32) << 16
    return bits.view(np.float32).reshape(values.shape)


def split_sinkhorn(mixes, scale, base):
    """The reference kernel's hc_split_sinkhorn in float32: mixes [s, 24] -> pre [s, 4],
    post [s, 4], comb [s, 4, 4]."""
    m = np.asarray(mixes, np.float32)
    scale, base = np.asarray(scale, np.float32), np.asarray(base, np.float32)
    sig = lambda v: np.float32(1) / (np.float32(1) + np.exp(-v))  # noqa: E731
    pre = sig(m[:, :HC] * scale[0] + base[:HC]) + np.float32(HC_EPS)
    post = np.float32(2) * sig(m[:, HC : 2 * HC] * scale[1] + base[HC : 2 * HC])
    comb = (m[:, 2 * HC :] * scale[2] + base[2 * HC :]).reshape(-1, HC, HC)
    comb = np.exp(comb - comb.max(-1, keepdims=True))
    comb = comb / comb.sum(-1, keepdims=True) + np.float32(HC_EPS)
    comb = comb / (comb.sum(-2, keepdims=True) + np.float32(HC_EPS))
    for _ in range(SINKHORN - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + np.float32(HC_EPS))
        comb = comb / (comb.sum(-2, keepdims=True) + np.float32(HC_EPS))
    return pre.astype(np.float32), post.astype(np.float32), comb.astype(np.float32)


def diagonal(coefficients):
    """[s, n] coefficients -> [32, n x 32] float32: tile j diagonal, its row t
    coef[t, j]."""
    s, n = coefficients.shape
    d = np.zeros((32, n * 32), np.float32)
    t = np.arange(s)
    for j in range(n):
        d[t, j * 32 + t] = coefficients[:, j]
    return d


class Mhc:
    """mHC loaded on a session (cores [0, CORES)): `set_fn(fn)`, then `mixes(x)`,
    `pre(residual, pre_mix)`, `post(x, residual, post, comb)` with the reference's
    shapes of one tile row of tokens (x [s, 4, 5120] etc.); each returns its result and
    the box commands' microseconds.
    """

    def __init__(self, session):
        self.session = session
        self.x = session.empty((32, FLAT), "bfloat16", "tile")
        self.w = session.empty((FLAT, 32), "float32", "tile")
        self.m = session.empty((32 * (CORES + 1), 32), "float32", "tile")
        self.mixed = session.empty((32, 32), "float32", "tile")
        self.d = session.empty((HC * 32, STACK * 32), "float32", "tile")
        self.z = session.empty((STACK * 32, DIM), "bfloat16", "tile")
        self.out = session.empty((HC * 32, DIM), "bfloat16", "tile")
        t = dict(
            x=self.x,
            w=self.w,
            m=self.m,
            mixed=self.mixed,
            d=self.d,
            z=self.z,
            out=self.out,
        )
        # built for size: the default kernel buffer of a session holds them then
        self.mp = self.xp = Hc(log=log, session=session, form="runtime", **t)
        self.run = session.load(self.mp)
        self.group = (0, CORES)
        self.fn = None

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

    def _go(self, handler, group, *args):
        event = self.run.wait(self.run.submit(handler, group, args=args), timeout=30.0)
        if event.kind != "done":
            raise km.KomorebiError(f"{handler}: {event}")
        return event.microseconds

    def set_fn(self, fn, tensor=None):
        """hc_fn [24, 20480] float32 as the product's W [20480, 32] (into self.w)."""
        w = np.zeros((FLAT, 32), np.float32)
        w[:, :MIX] = np.asarray(fn, np.float32).T
        self.put(self.w if tensor is None else tensor, w, "float32")

    def fn_tensor(self, fn):
        """A W tensor of its own holding hc_fn, for `mixes(x, w)`."""
        tensor = self.session.empty((FLAT, 32), "float32", "tile")
        self.set_fn(fn, tensor)
        return tensor

    def mixes(self, x, w=None):
        """x [s, 4, 5120] (bf16 values) -> mixes [s, 24] float32 (before the split),
        with W `w` (else self.w)."""
        s = x.shape[0]
        flat = np.zeros((32, FLAT), np.float32)
        flat[:s] = np.asarray(x, np.float32).reshape(s, FLAT)
        self.put(self.x, flat, "bfloat16")
        w = self.w if w is None else w
        us = [self._go(self.mp.mixes, self.group, self.x, w, self.m)]
        us.append(self._go(self.mp.gather, (0, 1), self.m, self.mixed))
        return self.get(self.mixed, (32, 32), "float32")[:s, :MIX], us

    def _mix(self, handler, coefficients, rows, x, residual):
        s = residual.shape[0]
        z = np.zeros((STACK * 32, DIM), np.float32)
        for m in range(HC):
            z[m * 32 : m * 32 + s] = residual[:, m]
        if x is not None:
            z[HC * 32 : HC * 32 + s] = x
        d = np.zeros((HC * 32, STACK * 32), np.float32)
        for k, c in enumerate(coefficients):
            d[k * 32 : (k + 1) * 32] = diagonal(c)
        self.put(self.z, z, "bfloat16")
        self.put(self.d, d, "float32")
        us = self._go(handler, self.group, self.d, self.z, self.out)
        out = self.get(self.out, (HC * 32, DIM), "bfloat16").reshape(HC, 32, DIM)[
            :rows, :s
        ]
        return out, us

    def pre(self, residual, pre_mix):
        """residual [s, 4, 5120], pre_mix [s, 4] -> [s, 5120] (bf16 values)."""
        c = np.concatenate([pre_mix, np.zeros((len(pre_mix), 1), np.float32)], 1)
        out, us = self._mix(self.xp.pre, [c], 1, None, residual)
        return out[0], us

    def post(self, x, residual, post, comb):
        """x [s, 5120], residual [s, 4, 5120], post [s, 4], comb [s, 4, 4] ->
        [s, 4, 5120]."""
        rows = [
            np.concatenate([comb[:, :, k], post[:, k : k + 1]], 1) for k in range(HC)
        ]
        out, us = self._mix(self.xp.post, rows, HC, x, residual)
        return out.transpose(1, 0, 2), us

    def stop(self):
        self.run.stop()


def errors(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return {
        "rel_rms": float(np.linalg.norm(got - want) / np.linalg.norm(want)),
        "max_err": float(np.abs(got - want).max()),
        "max_ref": float(np.abs(want).max()),
    }


def reference(x, y, fn, scale, base):
    """float64 of the stored values: hc_mixes' mixes and split, hc_pre and hc_post."""
    s = x.shape[0]
    flat = x.reshape(s, FLAT).astype(np.float64)
    r = 1.0 / np.sqrt((flat**2).mean(1, keepdims=True) + EPS)
    mixes = flat @ np.asarray(fn, np.float64).T * r
    pre, post, comb = split_sinkhorn(mixes, scale, base)
    hpre = (pre[:, :, None].astype(np.float64) * x).sum(1)
    hpost = post[:, :, None] * y[:, None, :].astype(np.float64) + (
        comb[:, :, :, None].astype(np.float64) * x[:, :, None, :]
    ).sum(1)
    return mixes, (pre, post, comb), hpre, hpost


def card(args):
    """Random streams (scale 1, as the residual), hc_fn of the released magnitude,
    s rows."""
    rng = np.random.default_rng(args.seed)
    s = args.tokens
    x = as_bf16(rng.standard_normal((s, HC, DIM)))
    y = as_bf16(rng.standard_normal((s, DIM)))
    fn = (rng.standard_normal((MIX, FLAT)) * 0.02).astype(np.float32)
    scale = np.float32([0.5, 0.5, 1.0])
    base = (rng.standard_normal(MIX) * 0.5).astype(np.float32)
    want_m, (pre, post, comb), want_pre, want_post = reference(x, y, fn, scale, base)
    with km.Session(log=log, card=args.card) as session:
        mhc = Mhc(session)
        try:
            mhc.set_fn(fn)
            got_m, us_m = zip(*[mhc.mixes(x) for _ in range(args.repeat)])
            got_pre, us_pre = zip(*[mhc.pre(x, pre) for _ in range(args.repeat)])
            got_post, us_post = zip(
                *[mhc.post(y, x, post, comb) for _ in range(args.repeat)]
            )
        finally:
            mhc.stop()
    split = split_sinkhorn(got_m[-1], scale, base)
    report = {
        "tokens": s,
        "mixes": errors(got_m[-1], want_m),
        "pre_mix": errors(split[0], pre),
        "post_mix": errors(split[1], post),
        "comb": errors(split[2], comb),
        "hc_pre": errors(got_pre[-1], want_pre),
        "hc_post": errors(got_post[-1], want_post),
        "bf16_of_ref_pre": errors(as_bf16(want_pre), want_pre),
        "us_mixes_gather": list(us_m),
        "us_pre": list(us_pre),
        "us_post": list(us_post),
    }
    print(json.dumps(report), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["card"])
    p.add_argument("--card", type=int, default=0)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()
    return {"card": card}[args.what](args)


if __name__ == "__main__":
    sys.exit(main())
