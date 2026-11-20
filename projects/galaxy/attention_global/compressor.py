"""Step 1: the compressor of the global cache (DeepSeek-V4.1-Flash) on one Wormhole
chip, synthetic weights, against float64: `project`, `pool`, `norm` (README.md).
Example: python compressor.py --ratio 2"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi.dsl.tile import row_operand

STARTED = time.monotonic()
DIM, HEAD = 5120, 512  # hidden, the latent
K_TILES, HEAD_TILES = DIM // 32, HEAD // 32
EPS = 1e-20  # the config's rms_norm_eps


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


class Compressor(
    km.Program,
    compute={
        "math_fidelity": "HiFi2",
        "fp32_dest_acc_en": True,
        "dst_full_sync_en": True,
    },
):
    """project: the K-split product into the token's state slot; pool: the softmax
    gate of a group's two slots; norm: RMSNorm of a row of 512 (README.md)."""

    SPLIT = km.constant(19)
    WIDTH = km.constant(8)
    CHUNK = km.constant(2)
    W_BLOCK = CHUNK * WIDTH
    sums: km.L1[km.fp32, 1, WIDTH, 32, 32]
    weight: km.L1[km.fp32, HEAD_TILES // 2, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def product(x, w):
        return x @ w

    @km.tile_function
    def gated(k0, k1, s0, s1):
        m = km.maximum(s0, s1)
        e0 = km.exp(s0 - m)
        e1 = km.exp(s1 - m)
        return (k0 * e0 + k1 * e1) / (e0 + e1)

    @km.tile_function
    def rms(row, weight):
        ms = km.reduce_sum(row * row, exact=True) * (1.0 / HEAD)
        return row * km.rsqrt(ms + EPS).col() * weight.row()

    @km.handler
    def project(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, slot: int):
        share = self.rank % self.SPLIT
        first = self.rank // self.SPLIT * self.WIDTH
        k0 = share * K_TILES // self.SPLIT
        k1 = (share + 1) * K_TILES // self.SPLIT
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(w.tiles[k0:k1, first : first + self.WIDTH], block=self.W_BLOCK),
        ):
            self.sums += self.product(xb, wb)
        whole = km.reduce(self.sums, form="tree", group=self.SPLIT)
        if self.rank == self.leader:
            y.tiles[slot : slot + 1, first : first + self.WIDTH].write(whole)

    @km.handler
    def pool(self, state: km.Tensor, pooled: km.Tensor):
        c0 = self.rank
        g0 = HEAD_TILES + self.rank
        results = km.Stream(pooled.tiles[0:1, c0 : c0 + 1], write=True, block=1)
        for k0, k1, s0, s1 in zip(
            km.Stream(state.tiles[0:1, c0 : c0 + 1], block=1),
            km.Stream(state.tiles[1:2, c0 : c0 + 1], block=1),
            km.Stream(state.tiles[0:1, g0 : g0 + 1], block=1),
            km.Stream(state.tiles[1:2, g0 : g0 + 1], block=1),
        ):
            results.push(self.gated(k0, k1, s0, s1))

    @km.handler
    def norm(self, src: km.Tensor, wn: km.Tensor, out: km.Tensor):
        self.weight.fill(km.Stream(wn.pages(0, HEAD_TILES // 2)))
        results = km.Stream(out.pages(0, HEAD_TILES), write=True, block=HEAD_TILES)
        for row in km.Stream(
            src.pages(0, HEAD_TILES), block=HEAD_TILES, row=HEAD_TILES
        ):
            results.push(self.rms(row, self.weight))


def plan(ranks, nt, chunk=2, widest=8):
    """Of the widths w dividing N's tiles, groups g = nt / w, split s = min(ranks // g,
    K tiles // chunk); the most cores, ties to the narrower w (fewer cores a sum)."""
    best = None
    for w in range(1, min(nt, widest) + 1):
        if nt % w:
            continue
        g = nt // w
        s = min(ranks // g, K_TILES // chunk)
        if s >= 1 and (best is None or g * s > best[0] * best[1]):
            best = (g, s, w)
    g, s, w = best
    return {"groups": g, "split": s, "width": w, "cores": g * s}


def bf16(values):
    """float32 values rounded to bfloat16 (nearest even), as float32."""
    bits = np.ascontiguousarray(values, np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)


def weight_pages(values):
    """The norm weight's field pages: chunk c's per-column tile of its 64 values."""
    pages = [
        row_operand(values[64 * c : 64 * c + 64], 2) for c in range(len(values) // 64)
    ]
    return np.asarray(pages, np.float32)


def errors(got, want):
    """Relative RMS error, largest error over the largest value."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    scale = np.sqrt((want**2).mean())
    return {
        "rel_rms": float(np.sqrt(((got - want) ** 2).mean()) / scale),
        "max_over_max": float(np.abs(got - want).max() / np.abs(want).max()),
    }


def tile_values(session, tensor, shape, fmt):
    (raw,) = session.read(tensor)
    return np.asarray(
        formats.from_bytes(np.asarray(raw).reshape(-1), shape, fmt, "tile"), np.float64
    )


def timed(run, handler, group, words, repeat):
    times = []
    for _ in range(repeat + 1):
        event = run.wait(run.submit(handler, group, args=words), timeout=30.0)
        if event.kind != "done":
            raise km.KomorebiError(f"{handler.__name__}: {event}")
        times.append(event.microseconds)
    return round(float(np.median(times[1:])), 2), times[1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ratio", type=int, default=2, choices=(1, 2))
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    n = 2 * HEAD if args.ratio == 2 else HEAD
    # x: two tokens (row 0 of each row tile), bfloat16 values; W bf16 values of the
    # scale of the checkpoint's (normal / sqrt(K)); the norm weight near 1.
    xs = [np.zeros((32, DIM), np.float32) for _ in range(args.ratio)]
    for x in xs:
        x[0] = bf16(rng.standard_normal(DIM).astype(np.float32))
    w = bf16((rng.standard_normal((DIM, n)) / np.sqrt(DIM)).astype(np.float32))
    gw = bf16((1.0 + 0.1 * rng.standard_normal(HEAD)).astype(np.float32))
    report = {"ratio": args.ratio}
    with km.Session(card=args.card, log=log) as session:
        p = plan(session.ranks, n // 32)
        report["plan"] = p
        log(f"open: {session.ranks} ranks, plan {p}")
        tx = [session.tensor(x, "bfloat16", "tile") for x in xs]
        tw = session.tensor(w, "bfloat16", "tile")
        # ratio 1: the projection is the reference's bf16 linear, so its sums are
        # packed to bfloat16 and are the norm's input
        state = session.empty(
            (32 * args.ratio, n), "float32" if args.ratio == 2 else "bfloat16", "tile"
        )
        pooled = session.empty((32, HEAD), "bfloat16", "tile")
        out = session.empty((32, HEAD), "bfloat16", "tile")
        wn = session.empty((HEAD_TILES // 2, 1024), "float32")
        session.write(wn, weight_pages(gw).view(np.uint8).reshape(HEAD_TILES // 2, -1))
        program = Compressor(
            x=tx[0],
            w=tw,
            y=state,
            state=state,
            pooled=pooled,
            src=pooled if args.ratio == 2 else state,
            wn=wn,
            out=out,
            SPLIT=p["split"],
            WIDTH=p["width"],
            session=session,
            log=log,
        )
        run = session.load(program)
        try:
            us = {}
            for slot in range(args.ratio):
                us[f"project{slot}"], _ = timed(
                    run,
                    Compressor.project,
                    (0, p["cores"]),
                    (tx[slot], tw, state, slot),
                    args.repeat,
                )
            src = state
            if args.ratio == 2:
                us["pool"], _ = timed(
                    run, Compressor.pool, (0, 16), (state, pooled), args.repeat
                )
                src = pooled
            got_state = tile_values(
                session,
                state,
                (32 * args.ratio, n),
                "float32" if args.ratio == 2 else "bfloat16",
            )
            us["norm"], _ = timed(
                run, Compressor.norm, (0, 1), (src, wn, out), args.repeat
            )
        finally:
            run.stop()
        got_pooled = tile_values(
            session, pooled if args.ratio == 2 else state, (32, HEAD), "bfloat16"
        )
        got_out = tile_values(session, out, (32, HEAD), "bfloat16")
    # float64 of the stored values, stage by stage, and the whole chain
    want_y = [x.astype(np.float64)[0] @ w.astype(np.float64) for x in xs]
    report["project"] = errors(
        np.stack([got_state[32 * s] for s in range(args.ratio)]), np.stack(want_y)
    )
    if (
        args.ratio == 1
    ):  # against float64 rounded to bf16, as the reference's linear gives
        report["project_bf16"] = errors(
            got_state[0], bf16(want_y[0].astype(np.float32))
        )
    ys = np.stack([got_state[32 * s] for s in range(args.ratio)])
    if args.ratio == 2:
        kv, sc = ys[:, :HEAD], ys[:, HEAD:]
        e = np.exp(sc - sc.max(0))
        want_pool = (kv * e / e.sum(0)).sum(0)
        report["pool"] = errors(got_pooled[0], bf16(want_pool.astype(np.float32)))
    pin = got_pooled[0]
    want_out = pin / np.sqrt((pin * pin).mean() + EPS) * gw
    report["norm"] = errors(got_out[0], want_out)
    report["norm_bf16"] = errors(got_out[0], bf16(want_out.astype(np.float32)))
    # the whole chain against float64 from x and W (the reference's order, no bf16 cast)
    yw = np.stack(want_y)
    if args.ratio == 2:
        kv, sc = yw[:, :HEAD], yw[:, HEAD:]
        e = np.exp(sc - sc.max(0))
        lat = (kv * e / e.sum(0)).sum(0)
    else:
        lat = yw[0]
    lat = bf16(lat.astype(np.float32)).astype(np.float64)
    report["chain"] = errors(got_out[0], lat / np.sqrt((lat * lat).mean() + EPS) * gw)
    report["us"] = us
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
