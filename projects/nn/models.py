"""The two torch modules Komorebi-nn runs: model A, the decode layer's MLP half with its
norm (random weights, the decode bench's), and model B, the whole decode layer with
the model's own weights of one layer, a random cache and its position. Needs torch."""

import numpy as np
import torch

from projects.decode import model as md
from projects.nn import front


def parameters(module, values):
    """{name: float values} as the module's bf16 parameters (no grad)."""
    for name, value in values.items():
        tensor = torch.from_numpy(np.ascontiguousarray(value, np.float32))
        module.register_parameter(
            name, torch.nn.Parameter(tensor.to(torch.bfloat16), requires_grad=False)
        )


class ModelA(torch.nn.Module):
    """y = x + (silu(h w1) * (h w3)) w2, h = norm(x) wn: [r, 2048] -> [r, 2048]."""

    def __init__(self, model, weights):
        super().__init__()
        self.eps = model.eps
        parameters(self, weights)

    def forward(self, x):
        return x + front.mlp(x, self.wn, self.w1, self.w3, self.eps) @ self.w2


def model_a_weights(model, seed):
    """Model A's weights: the decode bench's random layer 0 (gate, up, down, norm2)."""
    w = md.layer_weights(model, seed, 0)
    names = {"wn": "norm2", "w1": "gate", "w3": "up", "w2": "down"}
    return {name: w[key][1] for name, key in names.items()}


class ModelB(torch.nn.Module):
    """The decode layer for one token at `position`: norm, q k v, the heads' norms and
    rotation, komorebi::attend over the cache, o with its add, norm, the gate, down
    with its add: [1, 2048] -> [1, 2048]."""

    def __init__(self, model, weights, keys, values, position):
        super().__init__()
        self.eps, self.position = model.eps, position
        parameters(self, weights)
        cos, sin = md.rotary(model, position + 1)
        for name, value in (("cos", cos[position]), ("sin", sin[position])):
            self.register_buffer(name, torch.from_numpy(value).to(torch.bfloat16))
        for name, value in (("keys", keys), ("values", values)):
            self.register_buffer(name, torch.from_numpy(value).to(torch.bfloat16))

    def forward(self, x):
        qkv = front.qkv_attend(
            x,
            self.norm1,
            self.wq,
            self.wk,
            self.wv,
            self.q_norm,
            self.k_norm,
            self.cos,
            self.sin,
            self.keys,
            self.values,
            self.eps,
        )
        a = torch.ops.komorebi.attend(qkv, self.keys, self.values, self.position)
        x1 = x + a @ self.wo
        return x1 + front.mlp(x1, self.norm2, self.w_gate, self.w_up, self.eps) @ (
            self.w_down
        )


def model_b_weights(layer_weights):
    """Model B's weights from a layer's {name: (pages, values)} (decode's real.py)."""
    names = {
        "norm1": "norm1",
        "wq": "q",
        "wk": "k",
        "wv": "v",
        "q_norm": "q_norm",
        "k_norm": "k_norm",
        "wo": "o",
        "norm2": "norm2",
        "w_gate": "gate",
        "w_up": "up",
        "w_down": "down",
    }
    return {name: layer_weights[key][1] for name, key in names.items()}


def cache(model, seed, position, context):
    """A random cache (md.cache) of `position` rows in a cache of `context` rows a KV
    head, the rows past it 0: (keys, values), [KV heads, context, head width] each."""
    out = []
    for part in md.cache(model, seed, 0, position):
        full = np.zeros((model.kv_heads, context, model.head_dim), np.float32)
        full[:, :position] = part
        out.append(full)
    return tuple(out)
