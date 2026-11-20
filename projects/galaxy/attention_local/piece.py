"""Pieces 7, 8, 9 in the galaxy driver's call interface:
`make(session, weights, layers)`, then `piece(site, layer, *inputs)`. Sites: wq_a, wq_b,
wkv, wo_a, wo_b, rope_q, rope_kv, rope_o, sparse_attn (any layer), and attn (layers of
compress ratio 0, the window kept on the chip). README.md."""

import os

import numpy as np
import torch

from projects.galaxy.attention_local import attend as at
from projects.galaxy.attention_local import layer as ly
from projects.galaxy.attention_local import proj as pj
from projects.galaxy.attention_local import window as wn

TILE = 32
COMPRESS_RATIOS = (0, 0) + (2,) * 18 + (1,) * 20  # config.json's, layers 0-39
POSITIONS = 4096  # the rope tables' positions (the reference's max_seq_len)
SITES = (
    "attn",
    "wq_a",
    "wq_b",
    "wkv",
    "wo_a",
    "wo_b",
    "rope_q",
    "rope_kv",
    "rope_o",
    "sparse_attn",
)


def dequantized(checkpoint, name):
    """A released E4M3 weight [out, in] times its E8M0 32 x 32 block scales, float32."""
    weight = checkpoint.get(f"{name}.weight")
    scale = checkpoint.get(f"{name}.scale").float()
    out_block = weight.size(0) // scale.size(0)
    in_block = weight.size(1) // scale.size(1)
    w = weight.float().unflatten(0, (-1, out_block)).unflatten(-1, (-1, in_block))
    return (w * scale[:, None, :, None]).flatten(0, 1).flatten(-2).numpy()


def layer_weights(checkpoint, layer):
    """A layer's attention weights as LayerTensors takes them (held as (in, out))."""
    prefix = f"layers.{layer}.attn"
    w = {
        name: dequantized(checkpoint, f"{prefix}.{name}").T
        for name in ("wq_a", "wq_b", "wkv", "wo_b")
    }
    w["wo_a"] = pj.wo_a_stacked(dequantized(checkpoint, f"{prefix}.wo_a"))
    for name in ("q_norm", "kv_norm"):
        w[name] = checkpoint.get(f"{prefix}.{name}.weight").float().numpy()
    w["attn_sink"] = checkpoint.get(f"{prefix}.attn_sink").float().numpy()
    return w


def host(values, like):
    """A float32 numpy array as a torch tensor of `like`'s dtype."""
    return torch.from_numpy(np.ascontiguousarray(values, np.float32)).to(like.dtype)


def rope_tables(freqs_cis, inverse):
    """The rope's c and s tiles [32, 64] of the call's positions (freqs_cis [s, 32]
    complex64: row t the angles of token t), s signed as window.tables's."""
    angles = torch.view_as_real(freqs_cis).double().numpy()  # [s, 32, 2]: cos, sin
    tokens = angles.shape[0]
    cos = np.zeros((TILE, wn.ROPE_DIM))
    sin = np.zeros((TILE, wn.ROPE_DIM))
    cos[:tokens] = np.repeat(angles[..., 0], 2, axis=1)
    sin[:tokens] = np.repeat(angles[..., 1], 2, axis=1)
    sin[:, 0::2] *= -1.0
    return cos, -sin if inverse else sin


class AttentionPiece:
    sites = SITES

    def __init__(self, session, checkpoint, layers, log=print):
        self.session, self.log = session, log
        layers = sorted(layer for layer in layers if layer is not None and layer < 40)
        self.shared = ly.Shared(session, POSITIONS)
        self.layers = {
            layer: ly.LayerTensors(
                session,
                self.shared,
                layer_weights(checkpoint, layer),
                COMPRESS_RATIOS[layer] != 0,
            )
            for layer in layers
        }
        s = session

        def tensor(shape, dtype="bfloat16"):
            return s.empty(shape, dtype, "tile")

        # the sites' own tensors: rope tables of a call, a window, masks, sinks
        self.site = {
            "cos": tensor((TILE, wn.ROPE_DIM)),
            "sin": tensor((TILE, wn.ROPE_DIM)),
            "window": tensor((at.KEY_TILES * TILE, pj.HEAD_DIM)),
            "masks": tensor((at.MASKS * TILE, TILE)),
            "entries": tensor((at.ENTRY_TILES * TILE, TILE)),
            "sinks": tensor((pj.HEADS * TILE, TILE)),
        }
        self.stack = ly.Stack(s, self.shared, self.layers[layers[0]], log=log)
        self.stack.load()
        self.spent = {}

    def __call__(self, site, layer, *x):
        method = getattr(self, f"site_{site}")
        return method(self.layers[layer], layer, *x)

    def run(self, *commands):
        self.spent = self.stack.submit(commands)
        return self.spent

    # the projections: x's rows into the command's row tile
    def product(self, name, source, result, x, inputs):
        st, w = self.stack, inputs.w
        rows = x.reshape(-1, x.shape[-1]).float().numpy()
        st.write(source, rows)
        handler = getattr(st.proj, name)
        weight = {
            "wq_a": "wq_a",
            "wq_b": "wq_b",
            "wkv": "wkv",
            "wo_a": "wo_a",
            "wo_b": "wo_b",
        }
        self.run(
            (
                name,
                st.proj,
                handler,
                (0, pj.cores(name)),
                (st.a[source], w[weight[name]], st.a[result]),
            )
        )
        return st.read(result)[: rows.shape[0]]

    def site_wq_a(self, inputs, layer, x):
        y = self.product("wq_a", "x", "qa", x, inputs)
        return host(y.reshape(*x.shape[:-1], -1), x)

    def site_wq_b(self, inputs, layer, x):
        y = self.product("wq_b", "qr", "q", x, inputs)
        return host(y.reshape(*x.shape[:-1], -1), x)

    def site_wkv(self, inputs, layer, x):
        y = self.product("wkv", "x", "kv0", x, inputs)
        return host(y.reshape(*x.shape[:-1], -1), x)

    def site_wo_a(self, inputs, layer, x):
        y = self.product("wo_a", "o_flat", "oa", x.flatten(-2), inputs)
        return host(y.reshape(*x.shape[:-1], pj.O_RANK), x)

    def site_wo_b(self, inputs, layer, x):
        y = self.product("wo_b", "oa", "y", x, inputs)
        return host(y.reshape(*x.shape[:-1], -1), x)

    # the rotations: x's 64 in the last 64 of the 512 of each head's tile row
    def rotate(self, x, freqs_cis, heads, inverse):
        st, sh = self.stack, self.shared
        tokens = x.shape[1]
        values = (
            x.float().numpy().reshape(tokens, heads, wn.ROPE_DIM).transpose(1, 0, 2)
        )
        target = st.a["q_heads"] if heads > 1 else st.a["kv"]
        full = np.zeros((heads * TILE, pj.HEAD_DIM), np.float32)
        full.reshape(heads, TILE, pj.HEAD_DIM)[:, :tokens, -wn.ROPE_DIM :] = values
        st.write(target, full)
        cos, sin = rope_tables(freqs_cis, inverse)
        st.write(self.site["cos"], cos)
        st.write(self.site["sin"], sin)
        self.run(
            (
                "swap",
                st.window,
                st.window.rope_swap,
                (0, heads),
                (target, sh.swap, st.a["turned"]),
            ),
            (
                "rope",
                st.window,
                st.window.rope,
                (0, heads),
                (target, st.a["turned"], self.site["cos"], self.site["sin"], 0),
            ),
        )
        got = st.read(target)[: heads * TILE].reshape(heads, TILE, pj.HEAD_DIM)
        out = got[:, :tokens, -wn.ROPE_DIM :].transpose(1, 0, 2)
        return host(out.reshape(x.shape), x)

    def site_rope_q(self, inputs, layer, x, freqs_cis):
        return self.rotate(x, freqs_cis, pj.HEADS, False)

    def site_rope_o(self, inputs, layer, x, freqs_cis):
        return self.rotate(x, freqs_cis, pj.HEADS, True)

    def site_rope_kv(self, inputs, layer, x, freqs_cis):
        return self.rotate(x, freqs_cis, 1, False)

    def site_sparse_attn(
        self, inputs, layer, q, kv, attn_sink, topk_idxs, softmax_scale
    ):
        """kv's n <= 672 rows from key 32 of the call's key list (key tile 0 the
        sink's), the masks of key tiles 1 .. 5 (variant 0) and the entries' mask of the
        later tiles from topk_idxs (-1: none)."""
        st, sh = self.stack, self.shared
        if abs(softmax_scale - at.SCALE) > 1e-12 * at.SCALE:
            raise ValueError(
                f"softmax_scale {softmax_scale}: the core is built for {at.SCALE}"
            )
        tokens, n = q.shape[1], kv.shape[1]
        if n > (at.KEY_TILES - 1) * TILE:
            raise ValueError(
                f"{n} key-values: the core takes {(at.KEY_TILES - 1) * TILE}"
            )
        heads = q[0].float().numpy().transpose(1, 0, 2)  # [64, tokens, 512]
        full = np.zeros((pj.HEADS, TILE, pj.HEAD_DIM), np.float32)
        full[:, :tokens] = heads
        st.write(st.a["q_heads"], full.reshape(-1, pj.HEAD_DIM))
        window = np.zeros((at.KEY_TILES * TILE, pj.HEAD_DIM), np.float32)
        window[TILE : TILE + n] = kv[0].float().numpy()
        st.write(self.site["window"], window)
        seen = np.zeros((TILE, (at.KEY_TILES - 1) * TILE), bool)
        idxs = topk_idxs[0].numpy()
        for i in range(tokens):
            chosen = idxs[i][idxs[i] >= 0]
            seen[i, chosen] = True
        block = at.MASKS * TILE
        masks = np.where(seen[:, :block], 0.0, at.MASKED).reshape(TILE, at.MASKS, TILE)
        st.write(self.site["masks"], masks.transpose(1, 0, 2).reshape(-1, TILE))
        st.write(self.site["entries"], at.entry_values(seen[:, block:]))
        st.write(self.site["sinks"], at.sink_values(attn_sink.float().numpy()))
        keys = 1 + (n + TILE - 1) // TILE
        self.run(
            (
                "attend",
                st.attend,
                st.attend.attend,
                at.group(self.session),
                at.words(
                    st.a["q_heads"],
                    self.site["window"],
                    st.a["o"],
                    sh.tables[max(keys, 5)],
                    self.site["sinks"],
                    self.site["masks"],
                    0,
                    self.site["entries"],
                ),
            )
        )
        o = st.read("o").reshape(pj.HEADS, TILE, pj.HEAD_DIM)[:, :tokens]
        return host(o.transpose(1, 0, 2)[None], q)

    def site_attn(self, inputs, layer, x, start_pos):
        """The whole attention of a layer of compress ratio 0 from x at start_pos: a
        prefill block (32 tokens, start_pos a multiple of 32) or a decode token."""
        if COMPRESS_RATIOS[layer] != 0:
            raise ValueError(f"layer {layer}: compressed entries are not this piece's")
        st = self.stack
        tokens = x.shape[1]
        if tokens not in (1, TILE):
            raise ValueError(f"{tokens} tokens: a decode token or a block of {TILE}")
        st.write_x(x[0].float().numpy())
        self.spent = st.step(inputs, start_pos, tokens == TILE)
        y = st.read("y")[:tokens]
        return host(y[None], x)


def make(session, weights, layers):
    """The piece on the session's chip. `weights`: the checkpoint (the reference's
    Checkpoint); any other (the weights loader's W) is not read yet: the checkpoint is
    opened from $GALAXY_LLM/ckpt in its place."""
    if type(weights).__name__ != "Checkpoint":
        from galaxy.reference.model_loader import Checkpoint  # noqa: PLC0415

        weights = Checkpoint(os.path.join(os.environ["GALAXY_LLM"], "ckpt"))
    return AttentionPiece(session, weights, layers)
