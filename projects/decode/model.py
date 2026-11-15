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
OP = abi.constants(SPLIT_OP, known=abi.headers().constants)
TILE = 32
CORES = 120
BLOCK_WIDTH, CHUNK = 8, 2  # every plan's output block (1 x 8 tiles) and K chunk
GATE_TILES = BLOCK_WIDTH // 2  # a gate_up block: 4 tiles of the gate, then 4 of up
# A sum's tree takes at most half of a core's 8 ports (the other half is the set of
# the product in flight beside it): 4 rounds, groups of at most 16 cores.
SETS, GROUP_MOST = 2, 16
# The card's products of a layer, in order: q, k, v as one; gate and up as one. APART:
# q, k and v as three products (the overlapped form), q's plan o's.
PRODUCTS = ("qkv", "o", "gate_up", "down")
APART = ("q", "k", "v")


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

    def layer_bytes(self):
        """bf16 bytes a layer's weights take: the products' and the two norms'."""
        products = sum(k * n for k, n in self.shapes().values())
        return 2 * (products + 2 * self.hidden)


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
    )


def plans(model, cores=CORES):
    """{plan: Split} for the layer's shapes, sharing every buffer: 1 x 8 output blocks,
    chunks of 2 K tiles, groups as large as the cores allow up to GROUP_MOST, one
    arrival buffer of two sets (two products in flight), each a slot per round."""
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
            1, *shape, sizes[shape], 1, BLOCK_WIDTH, CHUNK, cores=cores, arrive=arrive
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
    return out


def fused(weights):
    """The card's weights of a layer from layer_weights(): {name: pages} of qkv, o,
    gate_up (the gate's GATE_TILES tiles, then up's, a block), down, the norms, APART.
    """
    values = {name: value for name, (_, value) in weights.items()}
    qkv = np.concatenate([values[name] for name in ("q", "k", "v")], axis=1)
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
    pages = {name: tile_pages(formats.bf16_bits(value)) for name, value in out.items()}
    kept = ("o", "down", "norm1", "norm2") + APART
    return pages | {name: weights[name][0] for name in kept}


def constants():
    """The norm's constant tiles side by side: the identity and the mean (1 / 2048 in
    every place: a row's sum over 2048 columns, times it, is its mean)."""
    mean = np.full((TILE, TILE), 1.0 / (D["DC_ROW_TILES"] * TILE), np.float32)
    return np.concatenate([np.eye(TILE, dtype=np.float32), mean], axis=1)


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


def layer(model, x, w):
    """One layer without attention (its output stands in as q), float64, each output
    rounded to bf16 as the card stores it: {tensor: values}."""
    f = {name: np.asarray(value[1], np.float64) for name, value in w.items()}
    out = {"h1": bf16(norm(x, f["norm1"], model.eps))}
    for name in ("q", "k", "v"):
        out[name] = bf16(out["h1"] @ f[name])
    out["x1"] = bf16(x + out["q"] @ f["o"])
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
