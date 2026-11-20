"""Pieces 7, 8, 9 in the driver's call interface (projects/galaxy/driver/interface.py):
`make(session, weights, layers)`, then `piece(site, layer, *inputs)`. Sites: wq_a, wq_b,
wkv, wo_a, wo_b, rope_q, rope_kv, rope_o, sparse_attn (any layer), and attn (layers of
compress ratio 0, the window kept on the chip). README.md."""

import numpy as np
import torch

from projects.galaxy.attention_local import attend as at
from projects.galaxy.attention_local import layer as ly
from projects.galaxy.attention_local import proj as pj
from projects.galaxy.attention_local import window as wn
from projects.galaxy.weights import host as cache

TILE = 32
COMPRESS_RATIOS = (0, 0) + (2,) * 18 + (1,) * 20  # config.json's, layers 0-39
POSITIONS = 4096  # the rope tables' positions (the reference's max_seq_len)
KERNEL_BYTES = 104_448  # joined islands: the Projections' 103,408 B, the L1 left theirs
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


LINEARS = ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")


class KernelBufferError(ValueError):
    """A joined session whose kernel buffer leaves the piece's programs too little."""


def names(layers):
    """The loader's names of the layers' linears (held in bfp8_b, [in, out]; wo_a
    [8, 4096, 1024], proj.wo_a_stacked's pages)."""
    return [f"layers.{n}.attn.{w}@{ly.WEIGHT_FORMAT}" for n in layers for w in LINEARS]


def layer_weights(loaded, layer):
    """A layer's weights as LayerTensors takes them: each linear its (arena, first
    page) in `loaded` (the loader's W on this session's chip), the norms and the sink
    their values in the same converted cache."""
    w = {}
    for name, held in zip(LINEARS, names([layer])):
        h = loaded[held]
        w[name] = (h.arena, h.first)
    prefix = f"layers.{layer}.attn"
    w["q_norm"] = cache.held(f"{prefix}.q_norm@bfloat16").reshape(-1)
    w["kv_norm"] = cache.held(f"{prefix}.kv_norm@bfloat16").reshape(-1)
    w["attn_sink"] = cache.held(f"{prefix}.attn_sink@float32").reshape(-1)
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

    def __init__(self, session, loaded, layers, log=print):
        self.session, self.log = session, log
        layers = sorted(layer for layer in layers if layer is not None and layer < 40)
        self.shared = ly.Shared(session, POSITIONS)
        self.layers = {
            layer: ly.LayerTensors(
                session,
                self.shared,
                layer_weights(loaded, layer),
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
        # (layer, x, qr, start_pos, offset) -> the reference's compress_kv outputs
        # (compressed entries [1, n, 512], their indices [1, s, k], offset + j or -1)
        self.compress = None

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
        words = (st.a[source], w[name], st.a[result], inputs.first[name])
        self.run((name, st.proj, handler, (0, pj.cores(name)), words))
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
        """The 4a key list from kv = [window rows | compressed entries]: the window's
        (the block's at a prefill, the 128 slots at a decode) from key 32, the entries
        topk_idxs chooses from key tile 6, each row's masks from topk_idxs."""
        st, sh = self.stack, self.shared
        if abs(softmax_scale - at.SCALE) > 1e-12 * at.SCALE:
            raise ValueError(
                f"softmax_scale {softmax_scale}: the core is built for {at.SCALE}"
            )
        tokens = q.shape[1]
        offset = tokens if tokens > 1 else wn.WINDOW  # the window's rows of kv
        if offset > at.MASKS * TILE or offset > kv.shape[1]:
            raise ValueError(f"{offset} window rows of {kv.shape[1]}: at most 160")
        heads = q[0].float().numpy().transpose(1, 0, 2)  # [64, tokens, 512]
        full = np.zeros((pj.HEADS, TILE, pj.HEAD_DIM), np.float32)
        full[:, :tokens] = heads
        st.write(st.a["q_heads"], full.reshape(-1, pj.HEAD_DIM))
        idxs = topk_idxs[0].numpy()
        rows, chosen = selected(kv[:, offset:], topk_idxs, offset)
        if len(rows) > at.ENTRY_TILES * TILE:
            raise ValueError(f"{len(rows)} entries chosen: the key list holds 512")
        window = np.zeros((at.KEY_TILES * TILE, pj.HEAD_DIM), np.float32)
        window[TILE : TILE + offset] = kv[0, :offset].float().numpy()
        first = (1 + at.MASKS) * TILE
        window[first : first + len(rows)] = rows
        st.write(self.site["window"], window)
        block = at.MASKS * TILE
        seen = np.zeros((TILE, block), bool)
        for i in range(tokens):
            seen[i, idxs[i][(idxs[i] >= 0) & (idxs[i] < offset)]] = True
        masks = np.where(seen, 0.0, at.MASKED).reshape(TILE, at.MASKS, TILE)
        st.write(self.site["masks"], masks.transpose(1, 0, 2).reshape(-1, TILE))
        st.write(self.site["entries"], at.entry_values(chosen))
        st.write(self.site["sinks"], at.sink_values(attn_sink.float().numpy()))
        keys = (
            1 + at.MASKS + (len(rows) + TILE - 1) // TILE
            if len(rows)
            else 1 + (offset + TILE - 1) // TILE
        )
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

    def set_window(self, layer, ring):
        """The layer's window ring [128, 512] (slot p mod 128 of position p) as the
        reference holds it: for a check that starts after a prefill it did not see."""
        st, tensor = self.stack, self.layers[layer].window
        full = st.read(tensor)
        full[TILE : TILE + wn.WINDOW] = np.asarray(ring, np.float32)
        st.write(tensor, full)

    def site_attn(self, inputs, layer, x, start_pos):
        """A layer's whole attention from x at start_pos: a prefill block (32 tokens,
        start_pos a multiple of 32) or a decode token; a layer of compressed entries
        takes them from `self.compress` (README.md, "Compressed layers")."""
        st = self.stack
        tokens = x.shape[1]
        if tokens not in (1, TILE):
            raise ValueError(f"{tokens} tokens: a decode token or a block of {TILE}")
        st.write_x(x[0].float().numpy())
        self.spent = st.step(
            inputs, start_pos, tokens == TILE, select=self.chooser(layer, x, start_pos)
        )
        y = st.read("y")[:tokens]
        return host(y[None], x)

    def attn_tensors(self, layer, x, start_pos, tokens, y, x_host=None):
        """The attn site on tensors of the chip: x's rows (`tokens` of them) in, y's
        out, both [32, 5120] bf16 tile; a compressed layer's hook takes x_host (x's
        values as the reference's tensor) or reads x back. {name: microseconds}."""
        st = self.stack
        if x_host is None and COMPRESS_RATIOS[layer] != 0 and self.compress:
            x_host = torch.from_numpy(st.read(x)[None, :tokens]).to(torch.bfloat16)
        select = self.chooser(layer, x_host, start_pos)
        self.spent = st.step(
            self.layers[layer], start_pos, tokens == TILE, select=select, x=x, y=y
        )
        return self.spent

    def chooser(self, layer, x, start_pos):
        """A compressed layer's select(qr) for Stack.step from `self.compress`; None
        for a layer of the window alone."""
        if COMPRESS_RATIOS[layer] == 0:
            return None
        if self.compress is None:
            raise ValueError(f"layer {layer}: piece.compress gives its entries")
        tokens = x.shape[1]
        offset = tokens if start_pos == 0 else wn.WINDOW  # the window_kv's rows

        def select(qr):
            kv, idxs = self.compress(
                layer, x, host(qr[None, :tokens], x), start_pos, offset
            )
            return selected(kv, idxs, offset)

        return select


def selected(kv, idxs, offset):
    """The entries the rows chose (rows [m, 512] in entry order) and seen [32, m]."""
    chosen = idxs[0].numpy().astype(np.int64) - offset  # [s, k], < 0 for none
    used = np.unique(chosen[chosen >= 0])
    seen = np.zeros((TILE, len(used)), bool)
    for i, row in enumerate(chosen):
        seen[i, np.searchsorted(used, row[row >= 0])] = True
    return kv[0].float().numpy()[used], seen


def make(session, weights, layers):
    """The piece on the session's chip. `weights`: the loader's W when it holds the
    layers' linears on this chip; else (None, a W of other islands) they are loaded
    here (the loader's subset load, projects/galaxy/weights)."""
    joined = at.island_ranks(session) < session.ranks
    if joined and (session.kernel_bytes or 1 << 30) > KERNEL_BYTES:
        raise KernelBufferError(
            f"a joined session for the attention piece needs "
            f"km.Session(..., kernel_bytes={KERNEL_BYTES:_}) or less (the Projections' "
            f"buffers and the islands' service slot); it has {session.kernel_bytes}"
        )
    layers = [n for n in layers if n is not None and n < 40]
    wanted = names(layers)
    if not (hasattr(weights, "arenas") and all(n in weights for n in wanted)):
        weights = cache.loader.load(session, wanted)
    return AttentionPiece(session, weights, layers)
