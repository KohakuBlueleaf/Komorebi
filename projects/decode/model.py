"""The model a decode step runs, as pure functions: its shapes from a `config.json`, the
K-split plans of its products, random weights as the card holds them (bf16), and one
layer in numpy (float64, each output rounded to bf16), the reference of the check."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from komorebi import abi, formats
from projects.dense import plan as pl

HERE = Path(__file__).resolve().parent
DECODE_H = HERE / "decode.h"
SPLIT_OP = pl.HERE / "kernels" / "split" / "split_op.hpp"
D = abi.constants(DECODE_H, known=abi.headers().constants)
D_MASKED = -1e30  # decode.h DC_MASKED
OP = abi.constants(SPLIT_OP, known=abi.headers().constants)
TILE = 32
CORES = 120
BLOCK_WIDTH, CHUNK = 8, 2  # every plan's output block (1 x 8 tiles) and K chunk
# The chunks' movers (dense/plan.py PATTERNS) by product, AB for the rest; mover A's
# half holds DEPTH chunks (AAAB gives it three of four).
PATTERNS, DEPTH = {"gate_up": "AAAB"}, 3
# The chain's halves (A's, B's chunks): the free L1 holds a record's next weight whole
# (but gate_up's) and the record after's first chunks behind it (stage 7).
CHAIN_HALVES = (5, 3)
GATE_TILES = BLOCK_WIDTH // 2  # a gate_up block: 4 tiles of the gate, then 4 of up
# A sum's tree takes at most half of a core's 8 ports (the other half is the set of
# the product in flight beside it): 4 rounds, groups of at most 16 cores.
SETS, GROUP_MOST = 2, 16
# The card's products of a layer, in order: q, k, v as one; gate and up as one. APART:
# q, k and v as three products (the overlapped form), q's plan o's.
PRODUCTS = ("qkv", "o", "gate_up", "down")
APART = ("q", "k", "v")
# The W's formats a product (the cache, activations bf16): bf16; bfp8, every W bfp8_b;
# official, gate and up bfp4_b and the rest bfp8_b (tt-metal's performance setting).
WEIGHTS = {
    "bf16": dict.fromkeys(PRODUCTS, "bfloat16"),
    "bfp8": dict.fromkeys(PRODUCTS, "bfp8_b"),
    "official": dict.fromkeys(PRODUCTS, "bfp8_b") | {"gate_up": "bfp4_b"},
}


@dataclass(frozen=True)
class Model:
    """A decoder layer's shapes: width, MLP width, heads and their width, layers."""

    name: str
    hidden: int
    mlp: int
    heads: int
    kv_heads: int
    head_dim: int
    layers: int
    eps: float
    rope_theta: float = 1e6

    def shapes(self):
        """{weight: (K, N)} of the layer's products."""
        q, kv = self.heads * self.head_dim, self.kv_heads * self.head_dim
        return {
            "q": (self.hidden, q),
            "k": (self.hidden, kv),
            "v": (self.hidden, kv),
            "o": (q, self.hidden),
            "up": (self.hidden, self.mlp),
            "gate": (self.hidden, self.mlp),
            "down": (self.mlp, self.hidden),
        }

    def products(self):
        """{product: (K, N)} of the card's products: qkv = [Wq | Wk | Wv], gate_up the
        gate's and up's columns in turns of GATE_TILES tiles each."""
        shapes = self.shapes()
        return {
            "qkv": (self.hidden, sum(shapes[name][1] for name in ("q", "k", "v"))),
            "o": shapes["o"],
            "gate_up": (self.hidden, 2 * self.mlp),
            "down": shapes["down"],
        }

    def layer_bytes(self, weights="bf16"):
        """Bytes a layer's weights take in DRAM, the products' in WEIGHTS[weights]'
        formats and the two norms' in bf16."""
        products = sum(
            k * n // (TILE * TILE) * formats.tile_bytes(WEIGHTS[weights][name])
            for name, (k, n) in self.products().items()
        )
        return products + 2 * 2 * self.hidden


# Qwen/Qwen3-1.7B's config.json: bf16, 2.8 GB of layer weights, one card's DRAM holds it
MODELS = {"qwen3-1.7b": Model("Qwen/Qwen3-1.7B", 2048, 6144, 16, 8, 128, 28, 1e-6)}


def from_config(path):
    """The Model of a Hugging Face `config.json` (a Llama-style decoder)."""
    config = json.loads(Path(path).read_text())
    heads = config["num_attention_heads"]
    return Model(
        Path(path)
        .parent.parent.parent.name.removeprefix("models--")
        .replace("--", "/"),
        config["hidden_size"],
        config["intermediate_size"],
        heads,
        config.get("num_key_value_heads", heads),
        config.get("head_dim") or config["hidden_size"] // heads,
        config["num_hidden_layers"],
        config.get("rms_norm_eps", 1e-6),
        config.get("rope_theta", 10000.0),
    )


def plans(model, cores=CORES, chunk=CHUNK, patterns=None, halves=(DEPTH, 1)):
    """{plan: Split} for the layer's shapes, sharing every buffer: 1 x 8 output blocks,
    `chunk` K tiles a chunk dealt by `patterns` (PATTERNS), W halves of `halves` chunks
    (A's, B's), groups as large as the cores allow up to GROUP_MOST, one arrival buffer
    of two sets, a slot per round."""
    patterns = PATTERNS if patterns is None else patterns
    named = {plan_of(model, name): kind for name, kind in patterns.items()}
    shapes = {}
    for name in PRODUCTS + APART:
        shapes.setdefault(plan_of(model, name), None)
    sizes = {
        shape: min(shape[0], cores // pl.cdiv(shape[1], BLOCK_WIDTH), GROUP_MOST)
        for shape in shapes
    }
    slots = max(math.ceil(math.log2(s)) for s in sizes.values())
    arrive = SETS * slots * BLOCK_WIDTH
    return {
        shape: pl.split_plan(
            1,
            *shape,
            sizes[shape],
            1,
            BLOCK_WIDTH,
            chunk,
            cores=cores,
            arrive=arrive,
            pattern=named.get(shape, "AB"),
            depth=halves[0],
            depth_b=halves[1],
        )
        for shape in shapes
    }


def plan_of(model, product):
    """The (K tiles, N tiles) key of `product`'s plan (a product or a weight)."""
    k, n = (model.products() | model.shapes())[product]
    return k // TILE, n // TILE


def bf16_random(rng, shape, bound):
    """Uniform values in [-bound, bound) cut to bf16: (bits as uint16, values)."""
    values = (rng.random(shape, dtype=np.float32) * 2 - 1) * np.float32(bound)
    bits = (values.view(np.uint32) >> 16).astype(np.uint16)
    return bits, formats.bf16_values(bits)


def tile_pages(bits):
    """bf16 bits [rows, cols] as the pages of a tile tensor: [tiles, 2048] uint8."""
    return np.ascontiguousarray(formats.tilize(bits)).view(np.uint8).reshape(-1, 2048)


def layer_weights(model, seed, layer):
    """A layer's weights: {name: (pages, values)}. W uniform with variance 1 / K (the
    output projections also / sqrt(2 x layers): the residual keeps its scale); each
    norm's weight uniform in [0.5, 1.5), in every row of a [32, hidden] tile row."""
    rng = np.random.default_rng([seed, layer])
    out = {}
    for name, (k, n) in model.shapes().items():
        bound = math.sqrt(3.0 / k)
        if name in ("o", "down"):
            bound /= math.sqrt(2 * model.layers)
        bits, values = bf16_random(rng, (k, n), bound)
        out[name] = (tile_pages(bits), values)
    for name in ("norm1", "norm2"):
        bits, values = bf16_random(rng, (1, model.hidden), 0.5)
        bits = formats.bf16_bits(values + 1.0)
        rows = np.repeat(bits, TILE, axis=0)
        out[name] = (tile_pages(rows), formats.bf16_values(bits)[0])
    for name in ("q_norm", "k_norm"):  # a head's norm weight, the qk tensor's (card)
        bits, values = bf16_random(rng, (model.head_dim,), 0.5)
        out[name] = (None, formats.bf16_values(formats.bf16_bits(values + 1.0)))
    return out


def fused_values(weights):
    """The card's W of a layer from layer_weights(): {name: values} of qkv, o, gate_up
    (the gate's GATE_TILES tiles, then up's, a block), down and APART; each W after a
    norm takes the norm's weight in its rows (W' = diag(weight) W, in bf16)."""
    values = {name: value for name, (_, value) in weights.items()}
    for name in ("q", "k", "v", "gate", "up"):
        norm_weight = values["norm1" if name in APART else "norm2"]
        values[name] = np.asarray(norm_weight, np.float32)[:, None] * values[name]
    qkv = np.concatenate([values[name] for name in APART], axis=1)
    k, n = values["gate"].shape
    turn = GATE_TILES * TILE
    pair = np.stack(
        [
            values["gate"].reshape(k, n // turn, turn),
            values["up"].reshape(k, n // turn, turn),
        ],
        axis=2,
    )
    out = {"qkv": qkv, "gate_up": pair.reshape(k, 2 * n)}
    out |= {name: values[name] for name in APART + ("o", "down")}
    return {name: formats.bf16_values(formats.bf16_bits(v)) for name, v in out.items()}


def format_of(weights, name):
    """The format of W `name` (a product or one of APART) in WEIGHTS[weights]."""
    return WEIGHTS[weights]["qkv" if name in APART else name]


def fused(weights, formats_name="bf16"):
    """The card's weights of a layer from layer_weights(): {name: pages} of
    fused_values() in WEIGHTS[formats_name]' formats ([tiles, tile bytes] uint8)."""
    out = {}
    for name, value in fused_values(weights).items():
        dtype = format_of(formats_name, name)
        raw, _ = formats.to_bytes(value, dtype, "tile")
        out[name] = raw.reshape(-1, formats.tile_bytes(dtype))
    return out


def quantized(model, weights, formats_name):
    """layer_weights() with the W the card holds in WEIGHTS[formats_name]' formats: each
    W decoded from fused()'s pages, a normed one divided by its norm's weight, so that
    layer()'s products are the card's (the same weights; float64)."""
    if formats_name == "bf16":
        return weights
    pages = fused(weights, formats_name)
    held = {}
    for name in ("qkv", "o", "gate_up", "down"):
        dtype = format_of(formats_name, name)
        held[name] = formats.from_bytes(
            pages[name].reshape(-1), model.products()[name], dtype, "tile"
        ).astype(np.float64)
    q, kv = model.shapes()["q"][1], model.shapes()["k"][1]
    split = dict(zip(APART, np.split(held["qkv"], [q, q + kv], axis=1)))
    k, n = model.shapes()["gate"]
    turn = GATE_TILES * TILE
    pair = held["gate_up"].reshape(k, n // turn, 2, turn)
    split |= {"gate": pair[:, :, 0].reshape(k, n), "up": pair[:, :, 1].reshape(k, n)}
    out = dict(weights)
    for name, value in split.items():
        norm_weight = np.asarray(
            weights["norm1" if name in APART else "norm2"][1], np.float64
        )[:, None]
        unnormed = np.divide(
            value, norm_weight, out=np.zeros_like(value), where=norm_weight != 0
        )
        out[name] = (None, unnormed)
    for name in ("o", "down"):
        out[name] = (None, held[name])
    return out


def squares(x):
    """x's rows' sums of squares as the card keeps them: DC_PARTS tiles side by side,
    the first with each row's sum on its diagonal, the others zero."""
    out = np.zeros((TILE, D["DC_PARTS"] * TILE), np.float32)
    rows = np.asarray(x, np.float64)
    out[:, :TILE] = np.diag((rows * rows).sum(axis=1))
    return out


def constants():
    """The norm's constant tiles side by side: the identity and the mean (1 / 2048 in
    every place: a row's sum over 2048 columns, times it, is its mean)."""
    mean = np.full((TILE, TILE), 1.0 / (D["DC_ROW_TILES"] * TILE), np.float32)
    return np.concatenate([np.eye(TILE, dtype=np.float32), mean], axis=1)


def rotary(model, count):
    """The rotation's table of positions 0..count-1 as the reference computes it
    (float32 frequencies, cos and sin cast to bf16): (cos, sin signed as rotate_half
    takes it: -sin in the first half), each [count, head width] float32."""
    dim = model.head_dim
    exponents = np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)
    inverse = np.float32(1.0) / np.power(np.float32(model.rope_theta), exponents)
    freqs = np.arange(count, dtype=np.float32)[:, None] * inverse[None, :]
    angles = np.concatenate([freqs, freqs], axis=1)
    cos, sin = bf16(np.cos(angles)), bf16(np.sin(angles))
    sin[:, : dim // 2] *= -1.0
    return cos, sin


def qk_weight_pages(q_norm, k_norm):
    """A layer's two head norm weights as the qk tensor's pages: each in every row of
    a [32, 128] tile row, q's 4 tiles then k's: [8, 2048] uint8."""
    rows = [
        np.repeat(np.asarray(w, np.float32)[None, :], TILE, 0) for w in (q_norm, k_norm)
    ]
    return np.concatenate([tile_pages(formats.bf16_bits(row)) for row in rows])


def qk_pages(model, rows):
    """The qk tensor (decode.h DC_QK_*) for `rows` positions: the identity, the mean,
    the head's mean (1 / head width), every layer's head norm weights 1, then the
    rotation's rows, 4 a page: [pages, 2048] uint8."""
    head = np.full((TILE, TILE), 1.0 / model.head_dim, np.float32)
    tiles = np.concatenate([constants(), head], axis=1)
    ones = np.ones(model.head_dim, np.float32)
    weights = np.concatenate([qk_weight_pages(ones, ones)] * D["DC_QK_LAYERS"])
    cos, sin = rotary(model, rows)
    table = formats.bf16_bits(np.concatenate([cos, sin], axis=1))
    per_page = pl.TILE_BYTES // D["DC_ROPE_ROW"]
    padded = np.zeros((-(-rows // per_page) * per_page, table.shape[1]), np.uint16)
    padded[:rows] = table
    pages = [tile_pages(formats.bf16_bits(tiles)), weights]
    return np.concatenate(pages + [padded.view(np.uint8).reshape(-1, pl.TILE_BYTES)])


def heads_norm_rotated(model, x, weight, positions):
    """The model's norm of each head of x ([rows, heads x width]), times `weight`, then
    the rotary embedding at `positions` (one a row), float64."""
    width = model.head_dim
    heads = np.asarray(x, np.float64).reshape(x.shape[0], -1, width)
    normed = heads / np.sqrt((heads * heads).mean(axis=2, keepdims=True) + model.eps)
    normed *= np.asarray(weight, np.float64)
    cos, sin = (
        table[positions][:, None, :] for table in rotary(model, positions[-1] + 1)
    )
    turned = np.concatenate(
        [normed[..., width // 2 :], normed[..., : width // 2]], axis=2
    )
    return (normed * cos + turned * sin).reshape(x.shape)


def start(seed, model):
    """x of the first layer: [32, hidden] uniform in [-1, 1), bf16 values."""
    rng = np.random.default_rng([seed, 1 << 20])
    return formats.bf16_values(
        formats.bf16_bits(rng.uniform(-1, 1, (TILE, model.hidden)))
    )


def bf16(x):
    return formats.bf16_values(formats.bf16_bits(np.asarray(x, np.float32)))


def norm(x, weight, eps):
    x = np.asarray(x, np.float64)
    return x / np.sqrt((x * x).mean(axis=1, keepdims=True) + eps) * weight


def silu(x):
    return x / (1.0 + np.exp(-x))


def cache(model, seed, layer, rows):
    """A layer's random cache of `rows` positions: (keys, values), each [KV heads,
    rows, head width] bf16 values, uniform with variance 1 (as k and v come out)."""
    rng = np.random.default_rng([seed, layer, 1 << 21])
    shape = (model.kv_heads, rows, model.head_dim)
    return tuple(bf16_random(rng, shape, math.sqrt(3.0))[1] for _ in range(2))


def cache_pages(values, rows):
    """A cache's [KV heads, n, width] values (n <= rows) as the pages of its tensor:
    [KV heads x rows, width] bf16 tiles, head after head, rows past n zero."""
    heads, count, width = values.shape
    full = np.zeros((heads, rows, width), np.float32)
    full[:, :count] = values
    return tile_pages(formats.bf16_bits(full.reshape(heads * rows, width)))


def attend_constants():
    """attend's constant tiles side by side: mask k (row i's columns past i + k - 31 at
    DC_MASKED, the rest 0) for k in 0..62, a tile of ones, one with ones in row 0 only,
    the identity."""
    rows, columns = np.arange(TILE)[:, None], np.arange(TILE)[None, :]
    masks = [
        np.where(columns > rows + k - D["DC_MASK_ZERO"], D_MASKED, 0.0)
        for k in range(D["DC_MASKS"])
    ]
    row = np.zeros((TILE, TILE))
    row[0] = 1.0
    return np.concatenate(masks + [np.ones((TILE, TILE)), row, np.eye(TILE)], axis=1)


def attention(model, q, keys, values, rows=None):
    """softmax(q k^T / sqrt(width)) v of every query head over its KV head's `keys` and
    `values` ([KV heads, n, width]), float64: [32, heads x width]. `rows`: the first
    rows are the last `rows` keys' tokens, row i over keys 0..n - rows + i (the rest 0);
    None: every row over all n."""
    width, group = model.head_dim, model.heads // model.kv_heads
    out = np.zeros((q.shape[0], model.heads * width))
    live = q.shape[0] if rows is None else rows
    count = keys.shape[1]
    mask = np.zeros((live, count))
    if rows is not None:
        mask[np.arange(count)[None, :] > count - rows + np.arange(rows)[:, None]] = (
            -np.inf
        )
    for head in range(model.heads):
        kv = head // group
        query = np.asarray(q[:live, head * width : (head + 1) * width], np.float64)
        scores = query @ np.asarray(keys[kv], np.float64).T / math.sqrt(width) + mask
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        out[:live, head * width : (head + 1) * width] = weights @ values[kv]
    return out


def layer(model, x, w, cached=None, rows=1, rotate=True):
    """One layer, float64, each output rounded to bf16 as the card stores it: {tensor:
    values}. `cached`: (keys, values) of positions 0..t-1; the tokens' k and v (rows 0
    to rows - 1) are positions t on, row i attending over 0..t + i (the other rows' a
    is 0), q and k normed by head and rotated (`rotate`). None: no attention, q for a.
    """
    f = {name: np.asarray(value[1], np.float64) for name, value in w.items()}
    out = {"h1": bf16(norm(x, f["norm1"], model.eps))}
    for name in ("q", "k", "v"):
        out[name] = bf16(out["h1"] @ f[name])
    if cached is not None and rotate:
        positions = cached[0].shape[1] + np.arange(rows)
        for name in ("q", "k"):
            turned = heads_norm_rotated(
                model, out[name][:rows], f[f"{name}_norm"], positions
            )
            out[name][:rows] = bf16(turned)
    out["a"] = out["q"]
    if cached is not None:
        width = model.head_dim
        tokens = [
            np.concatenate(
                [
                    cached[i],
                    out[name][:rows]
                    .reshape(rows, model.kv_heads, width)
                    .swapaxes(0, 1),
                ],
                axis=1,
            )
            for i, name in enumerate(("k", "v"))
        ]
        out["a"] = bf16(attention(model, out["q"], *tokens, rows=rows))
    out["x1"] = bf16(x + out["a"] @ f["o"])
    out["h"] = bf16(norm(out["x1"], f["norm2"], model.eps))
    out["up"] = bf16(out["h"] @ f["up"])
    out["gated"] = bf16(silu(out["h"] @ f["gate"]) * out["up"])
    out["x"] = bf16(out["x1"] + out["gated"] @ f["down"])
    return out


def errors(got, want):
    """Largest error over the largest |want|, RMS error over the RMS of want."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    error = np.abs(got - want)
    return {
        "max_rel": float(error.max() / np.abs(want).max()),
        "rms_rel": float(np.sqrt((error**2).mean() / (want**2).mean())),
    }
