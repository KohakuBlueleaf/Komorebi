"""The token's path as a chain of sites, each site one call of the reference code
(README.md, `driver/path.py`)."""

import time
from dataclasses import dataclass, field

import torch

# site name -> piece number (the pieces of README.md)
SITE_PIECE = {
    "embed": 3,
    "engram_hash": 4,
    "engram_rows": 4,
    "engram_mix": 4,
    "hc_attn_mixes": 5,
    "hc_attn_pre": 5,
    "hc_attn_post": 5,
    "hc_ffn_mixes": 5,
    "hc_ffn_pre": 5,
    "hc_ffn_post": 5,
    "hc_head_pre": 5,
    "attn_norm": 6,
    "ffn_norm": 6,
    "q_norm": 6,
    "kv_norm": 6,
    "attn": 7,
    "wq_a": 7,
    "wq_b": 7,
    "wkv": 7,
    "wo_a": 7,
    "wo_b": 7,
    "rope_q": 8,
    "rope_kv": 8,
    "rope_o": 8,
    "rope_ckv": 8,
    "window_kv": 16,
    "window_store": 16,
    "sparse_attn": 9,
    "compress_kv": 10,
    "compressor": 10,
    "compress_store": 16,
    "indexer": 11,
    "ffn": 13,
    "gate": 13,
    "shared": 14,
    "routed": 15,
    "final_norm": 17,
    "head": 18,
    "choose": 18,
}


@dataclass
class Recorder:
    """Records the inputs and outputs of the sites of chosen layers in chosen steps."""

    layers: set  # layer numbers to record; None in the set records the global sites
    store: dict = field(
        default_factory=dict
    )  # (step, site, layer) -> (inputs, outputs)
    step: str | None = None  # the current step's name, None = do not record

    def want(self, layer) -> bool:
        return self.step is not None and layer in self.layers

    def put(self, site, layer, inputs, outputs):
        self.store[(self.step, site, layer)] = (inputs, outputs)


def _snap(v):
    """A copy of a site argument or result that later in-place updates cannot change."""
    if isinstance(v, torch.Tensor):
        return v.detach().clone()
    if isinstance(v, (tuple, list)):
        return type(v)(_snap(x) for x in v)
    return v


class TokenPath:
    def __init__(self, ref):
        self.ref = ref
        self.model = ref.model
        self.mod = ref.mod
        self.args = ref.args
        self.box: dict = {}  # (site, layer) -> callable; layer None = every layer
        self.recorder: Recorder | None = None
        self.calls: dict = {}  # (site, layer) -> number of box calls (for the report)
        # (site, "host" | "box") -> seconds in the site itself, not in sites inside it
        self.seconds: dict = {}
        self._inner: list = []

    # --- switch
    # -------------------------------------------------------------------------
    def put(self, site: str, impl, layers=None):
        """Run `site` on the box: `impl(layer, *inputs)` returns what the reference call
        returns (the same structure, shapes and dtypes, as host torch tensors). layers:
        an iterable of layer numbers, or None for every layer (and for the global
        sites)."""
        assert site in SITE_PIECE, site
        if layers is None:
            self.box[(site, None)] = impl
        else:
            for i in layers:
                self.box[(site, i)] = impl

    def where(self, site, layer):
        return self.box.get((site, layer)) or self.box.get((site, None))

    def site(self, name, layer, ref_fn, *inputs, names=None):
        impl = self.where(name, layer)
        rec = self.recorder is not None and self.recorder.want(layer)
        snap_in = _snap(inputs) if rec else None
        self._inner.append(0.0)
        t0 = time.perf_counter()
        if impl is not None:
            self.calls[(name, layer)] = self.calls.get((name, layer), 0) + 1
            out = impl(layer, *inputs)
        else:
            out = ref_fn(*inputs)
        dt = time.perf_counter() - t0
        own = dt - self._inner.pop()  # less the sites inside this one
        if self._inner:
            self._inner[-1] += dt
        key = (name, "box" if impl is not None else "host")
        self.seconds[key] = self.seconds.get(key, 0.0) + own
        if rec:
            self.recorder.put(name, layer, (names, snap_in), _snap(out))
        return out

    # --- the reference, written out with sites
    # ------------------------------------------
    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
        """Transformer.forward, text only, greedy. Returns (next token [b], logits [b,
        vocab])."""
        m, S = self.model, self.site
        hashes = S(
            "engram_hash",
            None,
            lambda ids, sp: m.engram_hash(ids, sp, None),
            input_ids,
            start_pos,
            names=("input_ids", "start_pos"),
        )
        h = S("embed", None, m.embed, input_ids, names=("input_ids",))
        h = h.unsqueeze(2).repeat(1, 1, self.args.hc_mult, 1)
        pre_mix = self.mod.make_identity_pre_mix(h, self.args.hc_mult)
        for i, layer in enumerate(m.layers):
            if layer.engram is not None:
                h = self.engram(
                    i, layer.engram, h, hashes[:, :, layer.engram.layer_hash_index, :]
                )
            h, pre_mix = self.block(i, layer, h, start_pos, pre_mix)
        last = m.layers[-1]
        h = S("hc_head_pre", None, last.hc_pre, h, pre_mix, names=("x", "pre_mix"))
        h = S("final_norm", None, m.norm, h, names=("x",))
        logits = S("head", None, m.head, h, names=("x",))
        token = S(
            "choose", None, lambda lg: lg.argmax(dim=-1), logits, names=("logits",)
        )
        return token, logits

    def engram(self, i, eng, x, hash_ids):
        """Engram.forward: the rows on the host, the projection and the gate as one
        site."""
        rows = self.site("engram_rows", i, eng.embed, hash_ids, names=("hash_ids",))
        return self.site(
            "engram_mix",
            i,
            lambda x_, r_: self.engram_mix_ref(eng, x_, r_),
            x,
            rows,
            names=("x", "rows"),
        )

    @staticmethod
    def engram_mix_ref(eng, x, rows):
        # model.py Engram.forward after `self.embed(hash_ids)` (lines 353-365), no token
        # mask
        kv = eng.wkv(rows.flatten(-2))
        key, value = kv.split([eng.hc_mult * eng.dim, eng.dim], dim=-1)
        key = key.float().unflatten(-1, (eng.hc_mult, eng.dim))
        weight = eng.q_weight.float() * eng.k_weight.float()
        h, eps = x.float(), eng.eps
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
            key.square().mean(-1) + eps
        )
        dot = (h * weight * key).sum(-1) * rstd * eng.dim**-0.5
        gate = torch.sigmoid(
            torch.copysign(dot.abs().clamp_min(eng.clamp_value).sqrt(), dot)
        )
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)

    def block(self, i, blk, x, start_pos, pre_mix):
        """Block.forward."""
        S = self.site
        residual = x
        attn_pre, attn_post, attn_comb = S(
            "hc_attn_mixes",
            i,
            lambda x_: blk.hc_mixes(
                x_, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base
            ),
            x,
            names=("x",),
        )
        x = S("hc_attn_pre", i, blk.hc_pre, x, pre_mix, names=("x", "pre_mix"))
        x = S("attn_norm", i, blk.attn_norm, x, names=("x",))
        x = S(
            "attn",
            i,
            lambda x_, sp: self.attention(i, blk.attn, x_, sp),
            x,
            start_pos,
            names=("x", "start_pos"),
        )
        x = S(
            "hc_attn_post",
            i,
            blk.hc_post,
            x,
            residual,
            attn_post,
            attn_comb,
            names=("x", "residual", "post", "comb"),
        )
        residual = x
        ffn_pre, ffn_post, ffn_comb = S(
            "hc_ffn_mixes",
            i,
            lambda x_: blk.hc_mixes(
                x_, blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base
            ),
            x,
            names=("x",),
        )
        x = S("hc_ffn_pre", i, blk.hc_pre, x, attn_pre, names=("x", "pre_mix"))
        x = S("ffn_norm", i, blk.ffn_norm, x, names=("x",))
        x = S("ffn", i, lambda x_: self.moe(i, blk.ffn, x_), x, names=("x",))
        x = S(
            "hc_ffn_post",
            i,
            blk.hc_post,
            x,
            residual,
            ffn_post,
            ffn_comb,
            names=("x", "residual", "post", "comb"),
        )
        return x, ffn_pre

    def _rope(self, name, i, t, freqs, inverse=False):
        """apply_rotary_emb(t, freqs) in place, as a site that returns the rotated
        values."""
        rot = self.site(
            name,
            i,
            lambda t_, f_: self.mod.apply_rotary_emb(t_.clone(), f_, inverse),
            t,
            freqs,
            names=("x", "freqs_cis"),
        )
        t.copy_(rot)

    def attention(self, i, attn, x, start_pos):
        """Attention.forward."""
        S, mod = self.site, self.mod
        bsz, seqlen, _ = x.size()
        freqs_cis = attn.freqs_cis[start_pos : start_pos + seqlen]
        rd = attn.rope_head_dim
        qr = S(
            "q_norm",
            i,
            attn.q_norm,
            S("wq_a", i, attn.wq_a, x, names=("x",)),
            names=("x",),
        )
        q = S("wq_b", i, attn.wq_b, qr, names=("x",)).unflatten(
            -1, (attn.n_local_heads, attn.head_dim)
        )
        self._rope("rope_q", i, q[..., -rd:], freqs_cis)

        kv, topk_idxs = S(
            "window_kv",
            i,
            lambda x_, f_, sp: self.window_kv(i, attn, x_, f_, sp),
            x,
            freqs_cis,
            start_pos,
            names=("x", "freqs_cis", "start_pos"),
        )
        if attn.compress_ratio:
            compress_kv, compress_idxs = S(
                "compress_kv",
                i,
                lambda x_, qr_, sp, off: self.compress_kv(i, attn, x_, qr_, sp, off),
                x,
                qr,
                start_pos,
                kv.size(1),
                names=("x", "qr", "start_pos", "offset"),
            )
            kv = torch.cat([kv, compress_kv], dim=1)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

        o = S(
            "sparse_attn",
            i,
            mod.sparse_attn,
            q,
            kv,
            attn.attn_sink,
            topk_idxs,
            attn.softmax_scale,
            names=("q", "kv", "attn_sink", "topk_idxs", "softmax_scale"),
        )
        self._rope("rope_o", i, o[..., -rd:], freqs_cis, inverse=True)

        o = o.view(bsz, seqlen, attn.n_local_groups, -1)
        o = S("wo_a", i, lambda o_: self.wo_a_ref(attn, o_), o, names=("x",))
        return S("wo_b", i, attn.wo_b, o.flatten(2), names=("x",))

    def wo_a_ref(self, attn, o):
        # model.py lines 787-789: quantize o to E4M3 and back, then the grouped (block-
        # diagonal) wo_a
        o = o.clone()
        self.mod.act_quant(
            o, self.mod.fp8_block_size, self.mod.scale_fmt, self.mod.scale_dtype, True
        )
        wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
        return torch.einsum("bsgd,grd->bsgr", o, wo_a)

    def window_kv(self, i, attn, x, freqs_cis, start_pos):
        """Attention._window_kv: the layer's own K (= V) of this token, into the
        128-slot ring."""
        S = self.site
        bsz, seqlen, _ = x.size()
        kv = S(
            "kv_norm",
            i,
            attn.kv_norm,
            S("wkv", i, attn.wkv, x, names=("x",)),
            names=("x",),
        )
        self._rope("rope_kv", i, kv[..., -attn.rope_head_dim :], freqs_cis)
        return S(
            "window_store",
            i,
            lambda kv_, sp: self._window_store(attn, kv_, sp, bsz, seqlen),
            kv,
            start_pos,
            names=("kv", "start_pos"),
        )

    def _window_store(self, attn, kv, start_pos, bsz, seqlen):
        # model.py lines 708-721: E4M3 storage (quantize and back), the ring write, the
        # slots to read
        mod, win = self.mod, attn.window_size
        kv = kv.clone()
        mod.act_quant(kv, mod.fp8_block_size, mod.scale_fmt, mod.scale_dtype, True)
        if start_pos == 0:
            if seqlen <= win:
                attn.window_kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                (
                    attn.window_kv_cache[:bsz, cutoff:win],
                    attn.window_kv_cache[:bsz, :cutoff],
                ) = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            window_kv = kv
        else:
            attn.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            window_kv = attn.window_kv_cache[:bsz]
        return window_kv, mod.get_window_topk_idxs(win, bsz, seqlen, start_pos)

    def compress_kv(self, i, attn, x, qr, start_pos, offset):
        """Attention._compress_kv: the shared compressed cache and the positions this
        query reads."""
        S, mod = self.site, self.mod
        bsz, seqlen, _ = x.size()
        ratio = attn.compress_ratio
        compress_len = (start_pos + seqlen) // ratio
        latent = None
        if attn.is_kv_source:
            latent = S(
                "compressor", i, attn.compressor, x, start_pos, names=("x", "start_pos")
            )
            mod.shared_attn.compress_kv = attn.compress_kv_cache
        # Attention._compress_topk_idxs
        if not attn.is_index_source:
            idxs = mod.shared_attn.topk_idxs
        else:
            if compress_len == 0:
                idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32)
            else:
                if attn.indexer.freqs_cis is None:
                    attn.indexer.freqs_cis = attn.freqs_cis
                idxs = S(
                    "indexer",
                    i,
                    attn.indexer,
                    x,
                    qr,
                    latent,
                    start_pos,
                    offset,
                    names=("x", "qr", "latent", "start_pos", "offset"),
                )
            mod.shared_attn.topk_idxs = idxs
        if latent is not None:
            freqs = (
                attn.freqs_cis[: seqlen - seqlen % ratio : ratio]
                if start_pos == 0
                else attn.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            )
            self._rope("rope_ckv", i, latent[..., -attn.rope_head_dim :], freqs)
            S(
                "compress_store",
                i,
                lambda l_, sp: self._compress_store(attn, l_, sp, bsz),
                latent,
                start_pos,
                names=("latent", "start_pos"),
            )
        return mod.shared_attn.compress_kv[:bsz, :compress_len], idxs

    def _compress_store(self, attn, latent, start_pos, bsz):
        # model.py lines 761-762: FP4 storage with E4M3 scales a 16 (quantize and back),
        # the cache write
        latent = latent.clone()
        self.mod.fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
        ratio = attn.compress_ratio
        attn.compress_kv_cache[
            :bsz, start_pos // ratio : start_pos // ratio + latent.size(1)
        ] = latent
        return None

    def moe(self, i, ffn, x):
        """MoE.forward (text: no image mask)."""
        S = self.site
        shape = x.size()
        x = x.view(-1, ffn.dim)
        weights, indices = S("gate", i, ffn.gate, x, names=("x",))
        y = S(
            "routed",
            i,
            lambda x_, w_, i_: self.routed_ref(ffn, x_, w_, i_),
            x,
            weights,
            indices,
            names=("x", "weights", "indices"),
        )
        y = y + S("shared", i, ffn.shared_experts, x, names=("x",))
        return y.type_as(x).view(shape)

    @staticmethod
    def routed_ref(ffn, x, weights, indices):
        # model.py lines 895-902: float32 sum of weight * expert(x) over each token's
        # chosen experts
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(
            indices.flatten(), minlength=ffn.n_routed_experts
        ).tolist()
        for e in range(ffn.n_routed_experts):
            if counts[e] == 0:
                continue
            idx, top = torch.where(indices == e)
            y[idx] += ffn.experts[e](x[idx], weights[idx, top, None])
        return y
