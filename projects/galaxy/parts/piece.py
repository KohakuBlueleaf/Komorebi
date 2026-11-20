"""Pieces 4, 5, 6, 17 and 18 in the driver's call interface (driver/interface.py):
`make(session, weights, layers)` gives the piece, `piece(site, layer, *inputs)` the
site's outputs as the reference's host tensors, on the session's one chip (README.md).
"""

import time

import numpy as np
import torch

import komorebi as km
from komorebi import formats
from projects.galaxy.parts import engram as en
from projects.galaxy.parts import head as hd
from projects.galaxy.parts import mhc
from projects.galaxy.parts import norms as nm
from projects.galaxy.weights import host as cache

DIM, HC = 5120, 4
NORMS = {  # site: (width, the weight's name in the loader's cache; {L} the layer)
    "attn_norm": (DIM, "layers.{L}.attn_norm@bfloat16"),
    "ffn_norm": (DIM, "layers.{L}.ffn_norm@bfloat16"),
    "q_norm": (1280, "layers.{L}.attn.q_norm@bfloat16"),
    "kv_norm": (512, "layers.{L}.attn.kv_norm@bfloat16"),
    "final_norm": (DIM, "norm@bfloat16"),
}
MIXES = {"hc_attn_mixes": "hc_attn", "hc_ffn_mixes": "hc_ffn"}
PRES = ("hc_attn_pre", "hc_ffn_pre", "hc_head_pre")
POSTS = ("hc_attn_post", "hc_ffn_post")
SITES = (*NORMS, *MIXES, *PRES, *POSTS, "engram_mix", "head", "choose")
HEAD = [f"head.s{q}@bfloat16" for q in range(4)]  # the loader's four slices
ENGRAM = (1, 14)  # the layers with Engram
GLOBAL = ("hc_head_pre", "final_norm", "head", "choose")  # the sites of the layer None


def serves(site, layer):
    """Whether `site` is at `layer` (None: the global sites after the last layer)."""
    if site in GLOBAL:
        return layer is None
    return layer in ENGRAM if site == "engram_mix" else layer is not None


def refuse(site, layer):
    """The refusal by name of a call to `site` at a layer that has none."""
    at = {"engram_mix": f"layers {ENGRAM}", **dict.fromkeys(GLOBAL, "layer None")}
    only = at.get(site, "the numbered layers")
    raise KeyError(f"{site}: layer {layer} has no {site} (it is at {only})")


def host(values, dtype):
    """A float32 numpy array as a torch tensor of `dtype`."""
    return torch.from_numpy(np.ascontiguousarray(values, np.float32)).to(dtype)


def numpy(tensor):
    return tensor.detach().float().numpy()


def put(session, tensor, values, dtype):
    raw, _ = formats.to_bytes(np.ascontiguousarray(values, np.float32), dtype, "tile")
    session.write(tensor, np.asarray(raw).view(np.uint8).reshape(tensor.pages, -1))


def get(session, tensor, shape, dtype):
    (raw,) = session.read(tensor, first=0, pages=tensor.pages)
    raw = np.asarray(raw).reshape(-1).view(np.uint8)
    return np.asarray(formats.from_bytes(raw, shape, dtype, "tile"))


def done(run, ticket, name):
    event = run.wait(ticket, timeout=60.0)
    if event.kind != "done":
        raise km.KomorebiError(f"{name}: {event}")
    return event.microseconds


def hand(session, programs, run):
    """The run the programs take commands on: `run` (the shared load,
    projects/galaxy/load), the programs handed to it; None: a load of their own."""
    if run is None:
        return session.load(programs[0] if len(programs) == 1 else programs)
    run.take({program: (0, session.ranks) for program in programs})
    return run


def ready(run, program, group):
    """In a shared load, `program` loaded on `group` first when another piece left
    those ranks holding another program; a load of its own: nothing to do."""
    if hasattr(run, "take") and not run.holds(program, group):
        done(run, run.load(program, group), f"switch to {program.name}")


class NormPart:
    """The norms of every width (one class, norms.RmsNorm) on the chip, every weight of
    `names` written once here: `__call__(x, name)` with x [s, width] -> [s, width]
    (bf16 values), the width x's."""

    WIDTHS = (DIM, 1280, 512)

    def __init__(self, session, names, run=None):
        self.session, self.parts = session, {}
        for width in self.WIDTHS:
            e = session.empty((32, 32 * nm.EACH), "float32", "tile")
            put(session, e, nm.eps_tiles(width), "float32")
            self.parts[width] = (
                session.empty((32, width), "bfloat16", "tile"),
                session.empty((32, width), "float32", "tile"),
                e,
                session.empty((32, width), "bfloat16", "tile"),
            )
        x, w, e, out = self.parts[DIM]
        self.program = nm.RmsNorm(
            log=nm.log, session=session, form="runtime", x=x, w=w, e=e, out=out
        )
        self.run, self.own = hand(session, [self.program], run), run is None
        self.weights = {}
        for name in names:  # float32: the norm reads it by .row()
            weight = cache.held(name).reshape(-1)
            w = session.empty((32, len(weight)), "float32", "tile")
            put(session, w, nm.weight_rows(weight), "float32")
            self.weights[name] = w

    def __call__(self, x, name):
        s, width = x.shape
        xt, _, e, out = self.parts[width]
        w = self.weights[name]
        rows = np.zeros((32, width), np.float32)
        rows[:s] = x
        put(self.session, xt, rows, "bfloat16")
        group = (0, nm.cores(width))
        ready(self.run, self.program, group)
        us = done(
            self.run,
            self.run.submit(nm.RmsNorm.norm, group, args=(xt, w, e, out)),
            "norm",
        )
        return get(self.session, out, (32, width), "bfloat16")[:s], {"norm": us}

    def close(self):
        if self.own:
            self.run.stop()
        for t in (
            *[t for ts in self.parts.values() for t in ts],
            *self.weights.values(),
        ):
            t.free()


class MhcPart:
    """mhc.Mhc on the chip, the hc_fn of every prefix (layers.N.hc_attn, ...) written
    once here; scale and base kept on the host for the split."""

    def __init__(self, session, prefixes, run=None):
        self.m = mhc.Mhc(session, run)
        if run is not None:
            hand(session, [self.m.mp], run)
        self.fn, self.split = {}, {}
        for prefix in prefixes:
            # held [20,480, 24]: the [24, in] of the reference
            self.fn[prefix] = self.m.fn_tensor(cache.held(f"{prefix}_fn@float32").T)
            self.split[prefix] = [
                torch.from_numpy(cache.held(f"{prefix}_{n}@float32").reshape(-1))
                for n in ("scale", "base")
            ]

    def ready(self):
        ready(self.m.run, self.m.mp, self.m.group)
        return self.m

    def mixes(self, x, prefix):
        mixes, us = self.ready().mixes(x, self.fn[prefix])
        return mixes, {"mixes": us[0], "gather": us[1]}

    def close(self):
        self.m.stop()
        for t in (
            self.m.x,
            self.m.w,
            self.m.m,
            self.m.mixed,
            self.m.d,
            self.m.z,
            self.m.out,
            *self.fn.values(),
        ):
            t.free()


class EngramPart:
    """engram.EngramMix on the chip, each layer's wkv the loader's arena pages and its
    q_weight and k_weight rows written once here; `use(layer)` points the commands at
    that layer's (no write)."""

    def __init__(self, session, weights, layers, run=None):
        first = weights[f"layers.{layers[0]}.engram.wkv@bfloat16"]
        self.e = e = en.EngramMix(session, first.arena, first.first, run)
        if run is not None:
            hand(session, [e.kp], run)
        self.own = (e.q, e.kw)
        self.layers = {}
        for n in layers:
            prefix = f"layers.{n}.engram"
            wkv = weights[f"{prefix}.wkv@bfloat16"]
            e.q, e.kw = (session.empty(t.shape, "bfloat16", "tile") for t in self.own)
            q = cache.held(f"{prefix}.q_weight@bfloat16")
            e.set_weights(None, q, cache.held(f"{prefix}.k_weight@bfloat16"))
            self.layers[n] = (wkv.arena, wkv.first, e.q, e.kw)

    def use(self, layer):
        e = self.e
        ready(e.run, e.kp, (0, max(en.KV_CORES, en.HC * en.GROUP)))
        e.w, e.at, e.q, e.kw = self.layers[layer]

    def close(self):
        self.e.stop()
        e = self.e
        mine = [t for _, _, q, k in self.layers.values() for t in (q, k)]
        for t in (e.rows, e.kv, e.xs, *self.own, *mine, e.p, e.g, e.a, e.out):
            t.free()


class HeadPart:
    """The head on one chip: the loader's four slices (head.s0 .. s3, each an island's
    on the box) and each one's ids beside the shared x, y, c, g, r; the logits and the
    choice command by command, a slice after another."""

    def __init__(self, session, slices, run=None):
        self.session = s = session
        n = hd.CORES * hd.WIDTH * 32
        self.ts = ts = hd.Tensors(session, False)
        self.slices = [(h.arena, h.first) for h in slices]
        self.ids = [ts.ids] + [s.empty((32, n), "float32", "tile") for _ in range(3)]
        for q in range(4):
            pages = np.ascontiguousarray(hd.slot_pages(q)).view(np.uint8)
            s.write(self.ids[q], pages.reshape(self.ids[q].pages, -1))
        self.cp = hd.HeadChoice(
            log=hd.log,
            session=s,
            form="runtime",
            y=ts.y,
            ids=ts.ids,
            c=ts.c,
            g=ts.g,
            r=ts.r,
        )
        # its own load is a list (one layout: beside); a shared load switches them
        beside = {"beside": self.cp} if run is None else {}
        self.lp = hd.HeadLogits(
            log=hd.log,
            session=s,
            form="runtime",
            **beside,
            x=ts.x,
            w=self.slices[0][0],
            y=ts.y,
        )
        self.run, self.own = hand(s, [self.cp, self.lp], run), run is None
        self.group = (0, hd.CORES)

    def logits(self, x):
        """x [5120] (bf16 values) -> logits [129280] float32."""
        ts, run = self.ts, self.run
        rows = np.zeros((32, DIM), np.float32)
        rows[0] = x
        ts.write_x(0, rows)
        out, us = [], []
        ready(run, self.lp, self.group)
        for w, at in self.slices:
            us.append(
                done(
                    run,
                    run.submit(self.lp.logits, self.group, args=(ts.x, w, ts.y, at)),
                    "logits",
                )
            )
            bits = ts.get(ts.y, 0, 0, ts.y.pages)
            y = formats.from_bytes(bits, ts.y.shape, "float32", "tile")
            out.append(hd.slot_logits(y)[0])
        return np.concatenate(out), {"logits": us}

    def choose(self, logits):
        """logits [129280] float32 -> the id of the largest (of equal ones the
        first)."""
        ts, run = self.ts, self.run
        n = hd.CORES * hd.WIDTH * 32
        cols = hd.slot_columns()
        best_v, best_i, us = -np.inf, -1, {"choose": [], "least": []}
        span = hd.SLICE * 32
        for q in range(4):
            part = np.asarray(logits[q * span : (q + 1) * span], np.float32)
            slots = part.reshape(hd.SLICE, 32)[cols]
            y = np.zeros((32, n), np.float32)
            y[0] = slots.reshape(-1)
            raw, _ = formats.to_bytes(y, "float32", "tile")
            ts.put(ts.y, 0, raw)
            ready(run, self.cp, self.group)
            us["choose"].append(
                done(
                    run,
                    run.submit(
                        self.cp.choose, self.group, args=(ts.y, self.ids[q], ts.c, ts.g)
                    ),
                    "choose",
                )
            )
            us["least"].append(
                done(
                    run,
                    run.submit(self.cp.least, self.group, args=(ts.c, ts.r)),
                    "least",
                )
            )
            v = float(ts.tile_values(ts.g, 0, 0)[0, 0])
            i = int(ts.tile_values(ts.r, 0, 0)[0, 0])
            if v > best_v:  # slices in id order: an equal later one keeps the earlier
                best_v, best_i = v, i
        return best_i, us

    def close(self):
        if self.own:
            self.run.stop()
        ts = self.ts
        for t in (ts.x, ts.w, ts.y, ts.c, ts.g, ts.r, *self.ids):
            t.free()


class PartsPiece:

    def __init__(self, session, weights, layers, sites=SITES, run=None):
        self.session, self.W, self.given = session, weights, list(layers)
        self.layers = [n for n in layers if n is not None]
        self.sites, self.run = sites, run
        self.part, self.part_key = None, None
        self.spent = {}
        self.made = {}  # a shared load: every part made here, its classes handed to it
        for key, part in PARTS.items():  # one part's instance: its part made here
            if not self._has(part):  # none of its sites at a layer given: not made
                continue
            if run is not None:
                self.made[key] = getattr(self, f"_make_{key}")()
            elif tuple(sites) == part:
                self._use(key, getattr(self, f"_make_{key}"))

    def _has(self, part):
        """Whether a site of `part` asked of this instance is at a layer given."""
        return any(serves(s, n) for s in part if s in self.sites for n in self.given)

    def _make_norms(self):
        held = set(cache.select(formats=["bfloat16"]))
        names = {
            pattern.format(L=n)
            for site, (_, pattern) in NORMS.items()
            for n in self.given
            if serves(site, n)
        }
        return NormPart(self.session, sorted(names & held), self.run)

    def _make_mixing(self):
        held = set(cache.select(formats=["float32"]))
        prefixes = [
            f"layers.{n}.{kind}"
            for n in self.layers
            for kind in MIXES.values()
            if f"layers.{n}.{kind}_fn@float32" in held
        ]
        return MhcPart(self.session, prefixes, self.run)

    def _make_engram(self):
        layers = [n for n in ENGRAM if n in self.layers]
        return EngramPart(self.session, self.W, layers, self.run)

    def _make_head(self):
        return HeadPart(self.session, [self.W[n] for n in HEAD], self.run)

    def _use(self, key, make):
        if key in self.made:  # a shared load: every part kept, no load between
            return self.made[key]
        if self.part_key != key:
            if self.part is not None:
                self.part.close()
                self.part, self.part_key = None, None
            self.part, self.part_key = make(), key
        return self.part

    def __call__(self, site, layer, *inputs):
        if site not in self.sites:
            raise KeyError(f"{site}: not a site of this instance ({self.sites})")
        if not serves(site, layer):
            refuse(site, layer)
        if layer not in self.given:
            given = self.given
            raise KeyError(f"{site}: layer {layer} is not one of this piece's {given}")
        t0 = time.monotonic()
        out = getattr(self, f"_{site}" if hasattr(self, f"_{site}") else "_site")(
            site, layer, *inputs
        )
        self.spent["wall_us"] = (time.monotonic() - t0) * 1e6
        return out

    def _site(self, site, layer, *inputs):
        if site in NORMS:
            return self._norm(site, layer, *inputs)
        if site in MIXES:
            return self._mixes(site, layer, *inputs)
        if site in PRES:
            return self._pre(site, layer, *inputs)
        if site in POSTS:
            return self._post(site, layer, *inputs)
        raise KeyError(f"{site}: not a site of this piece")

    def _norm(self, site, layer, x):
        width, name = NORMS[site]
        name = name.format(L=layer)
        part = self._use("norms", self._make_norms)
        if name not in part.weights:
            refuse(site, layer)
        y, self.spent = part(numpy(x[0]), name)
        return host(y[None], x.dtype)

    def _mhc(self):
        return self._use("mixing", self._make_mixing)

    def _mixes(self, site, layer, x):
        prefix = f"layers.{layer}.{MIXES[site]}"
        part = self._mhc()
        if prefix not in part.fn:
            refuse(site, layer)
        mixes, self.spent = part.mixes(numpy(x[0]), prefix)
        # the reference's own split (the driver's tree; not this one's)
        from galaxy.reference.cpu_kernels import hc_split_sinkhorn  # noqa: PLC0415

        pre, post, comb = hc_split_sinkhorn(
            torch.from_numpy(mixes[None]), *part.split[prefix]
        )
        return pre, post, comb

    def _pre(self, site, layer, x, pre_mix):
        part = self._mhc()
        y, us = part.ready().pre(numpy(x[0]), numpy(pre_mix[0]))
        self.spent = {"pre": us}
        return host(y[None], x.dtype)

    def _post(self, site, layer, x, residual, post, comb):
        part = self._mhc()
        y, us = part.ready().post(
            numpy(x[0]), numpy(residual[0]), numpy(post[0]), numpy(comb[0])
        )
        self.spent = {"post": us}
        return host(y[None], x.dtype)

    def _engram_mix(self, site, layer, x, rows):
        part = self._use("engram", self._make_engram)
        part.use(layer)
        y, _, self.spent = part.e(numpy(x[0]), numpy(rows[0]))
        return host(y[None], x.dtype)

    def _head_part(self):
        return self._use("head", self._make_head)

    def _head(self, site, layer, x):
        logits, self.spent = self._head_part().logits(numpy(x[0, -1]))
        return torch.from_numpy(logits[None].astype(np.float32))

    def _choose(self, site, layer, logits):
        token, self.spent = self._head_part().choose(numpy(logits[0]))
        return torch.tensor([token], dtype=torch.int64)

    def close(self):
        for part in self.made.values():
            part.close()
        if self.part is not None:
            self.part.close()
            self.part, self.part_key = None, None


PARTS = {  # one instance a part (its own process and chip): its sites
    "norms": tuple(NORMS),
    "mixing": (*MIXES, *PRES, *POSTS),
    "engram": ("engram_mix",),
    "head": ("head", "choose"),
}


def names(layers, sites=SITES):
    """The loader's names of the large weights the sites read: the head's slices (the
    layer None) and the Engram layers' wkv (bf16, an arena's pages each)."""
    head = HEAD if "head" in sites and None in layers else []
    engram = [f"layers.{n}.engram.wkv@bfloat16" for n in ENGRAM if n in layers]
    return head + (engram if "engram_mix" in sites else [])


def make(session, weights, layers, sites=SITES, run=None):
    """The piece for `sites` (all by default; make_<part>: one part's). `weights`: the
    loader's W when it holds the sites' large weights; else loaded here. `run`: the
    shared load (projects/galaxy/load) every part's classes go to; None: a part's own
    load, made again when another part is called (none with no site at `layers`)."""
    wanted = names(layers, sites)
    if wanted and not (
        hasattr(weights, "arenas") and all(n in weights for n in wanted)
    ):
        weights = cache.loader.load(session, wanted)
    return PartsPiece(session, weights, layers, sites, run)


def make_norms(session, weights, layers):
    return make(session, weights, layers, PARTS["norms"])


def make_mixing(session, weights, layers):
    return make(session, weights, layers, PARTS["mixing"])


def make_engram(session, weights, layers):
    return make(session, weights, layers, PARTS["engram"])


def make_head(session, weights, layers):
    return make(session, weights, layers, PARTS["head"])
