"""How often each routed expert of each layer is picked for the fixed prompts
(README.md, `reference/expert_counts.py`)."""

import argparse
import json
import os
import time

import numpy as np
import torch

from ..driver import state
from ..driver.path import TokenPath
from .model_loader import load_reference
from .prompts import encode, prompt_text


def count_prompt(ref, ckpt, key, new_tokens):
    layers = len(ref.model.layers)
    experts = ref.args.n_routed_experts
    counts = np.zeros((layers, experts), np.int64)
    path = TokenPath(ref)

    def routed(layer, x, weights, indices):
        counts[layer] += np.bincount(indices.flatten().numpy(), minlength=experts)
        return path.routed_ref(ref.model.layers[layer].ffn, x, weights, indices)

    path.put("routed", routed, None)
    state.reset(ref)
    ids = encode(ref, ckpt, prompt_text(ref, ckpt, key))
    pos, inp, tokens = 0, torch.tensor([ids]), []
    for n in range(new_tokens):
        t0 = time.time()
        tok, _ = path.forward(inp, pos)
        tokens.append(int(tok[0]))
        print(f"{key} step {n} token {tokens[-1]} {time.time() - t0:.1f} s", flush=True)
        if n == new_tokens - 1:
            break  # the last token is not fed back: its picks are not made
        pos += inp.size(1)
        inp = tok.view(1, 1)
    return counts, ids, tokens


def share_line(counts):
    """The busiest and the median expert's share of a layer's picks, over the layers."""
    share = counts / counts.sum(1, keepdims=True)
    return {
        "busiest_share_max": float(share.max()),
        "busiest_share_mean": float(share.max(1).mean()),
        "median_share_mean": float(np.median(share, 1).mean()),
        "uniform_share": 1.0 / counts.shape[1],
        "unpicked_mean": float((counts == 0).sum(1).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default="en,zh")
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    ref = load_reference(a.ckpt, max_seq_len=4096, threads=a.threads)
    total = None
    for key in a.prompts.split(","):
        counts, ids, tokens = count_prompt(ref, a.ckpt, key, a.new)
        with open(os.path.join(a.out, key, "tokens.json")) as f:
            gold = json.load(f)["generated_ids"]
        same = tokens == gold[: len(tokens)]
        np.save(os.path.join(a.out, key, "expert_counts.npy"), counts)
        print(json.dumps({"prompt": key, "tokens_same": same, **share_line(counts)}))
        total = counts if total is None else total + counts
    np.save(os.path.join(a.out, "expert_counts.npy"), total)
    print(json.dumps({"prompt": "all", **share_line(total)}))


if __name__ == "__main__":
    main()
