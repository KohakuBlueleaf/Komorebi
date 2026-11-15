"""A real prompt on the card: the model's own weights (bf16), a token a step, the
embedding, final norm and head on the host, the largest logit taken greedily; against
torch_reference.py's tokens and hidden states (its tokenizer writes the prompt's
tokens). The checkpoint's safetensors are read with numpy alone."""

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import komorebi as km
from komorebi import formats
from projects.decode import card
from projects.decode import model as md

STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--snapshot", required=True, help="the model's snapshot folder")
    parser.add_argument("--reference", required=True, help="torch_reference.py's .npz")
    parser.add_argument("--context", type=int, default=256, help="the cache's rows")
    parser.add_argument("--json", default=None)
    parser.add_argument("--tokens", default=None, help=".npz of the card's tokens")
    return parser.parse_args()


class Checkpoint:
    """A safetensors checkpoint read by numpy: get(name) is a tensor's bf16 bits."""

    def __init__(self, folder):
        self.folder = Path(folder)
        index = json.loads((self.folder / "model.safetensors.index.json").read_text())
        self.where, self.headers = index["weight_map"], {}

    def get(self, name):
        shard = self.where[name]
        if shard not in self.headers:
            with open(self.folder / shard, "rb") as file:
                size = struct.unpack("<Q", file.read(8))[0]
                header = json.loads(file.read(size))
            self.headers[shard] = (8 + size, header)
        start, header = self.headers[shard]
        entry = header[name]
        if entry["dtype"] != "BF16":
            raise ValueError(f"{name}: {entry['dtype']}, not BF16")
        low, high = entry["data_offsets"]
        bits = np.memmap(
            self.folder / shard, np.uint16, "r", start + low, ((high - low) // 2,)
        )
        return np.asarray(bits).reshape(entry["shape"])


def layer_weights(checkpoint, layer):
    """Layer `layer`'s weights as md.layer_weights() gives them: {name: (pages,
    values)}, each W as [K, N] (the checkpoint's [N, K] transposed)."""
    prefix = f"model.layers.{layer}."
    names = {
        "q": "self_attn.q_proj",
        "k": "self_attn.k_proj",
        "v": "self_attn.v_proj",
        "o": "self_attn.o_proj",
        "gate": "mlp.gate_proj",
        "up": "mlp.up_proj",
        "down": "mlp.down_proj",
    }
    out = {}
    for name, key in names.items():
        bits = np.ascontiguousarray(checkpoint.get(prefix + key + ".weight").T)
        pages = md.tile_pages(bits) if name in ("o", "down") else None
        out[name] = (pages, formats.bf16_values(bits))
    norms = {
        "norm1": "input_layernorm",
        "norm2": "post_attention_layernorm",
        "q_norm": "self_attn.q_norm",
        "k_norm": "self_attn.k_norm",
    }
    for name, key in norms.items():
        out[name] = (
            None,
            formats.bf16_values(checkpoint.get(prefix + key + ".weight")),
        )
    return out


def final_norm(x, weight, eps):
    """The model's last norm as its library computes it: float32, cast to bf16, times
    the weight in bf16."""
    x = np.asarray(x, np.float32)
    rms = np.float32(1.0) / np.sqrt(
        (x * x).mean(axis=-1, keepdims=True) + np.float32(eps)
    )
    return md.bf16(np.asarray(weight, np.float32) * md.bf16(x * rms))


def relative_rms(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.sqrt(((got - want) ** 2).mean() / (want**2).mean()))


def main():
    args = parse()
    reference = np.load(args.reference)
    prompt, want = reference["prompt"].tolist(), reference["tokens"].tolist()
    hidden = reference["hidden"]  # [positions, layers, hidden]; the last one normed
    model = md.from_config(Path(args.snapshot) / "config.json")
    checkpoint = Checkpoint(args.snapshot)
    embedding = checkpoint.get("model.embed_tokens.weight")
    head = formats.bf16_values(checkpoint.get("lm_head.weight"))
    last_norm = formats.bf16_values(checkpoint.get("model.norm.weight"))
    steps = len(prompt) + len(want) - 1
    if steps > args.context:
        raise ValueError(f"{steps} positions past a cache of {args.context}")
    report = {"model": model.name, "prompt_tokens": len(prompt), "steps": steps}
    tokens, per_layer, step_us = list(prompt), [], []
    with km.Session(log=log) as session:
        decode = card.Decode(
            session, model, model.layers, 1.0, log, context=args.context
        )
        written = 0.0
        empty = np.zeros((model.kv_heads, 0, model.head_dim), np.float32)
        for layer in range(model.layers):
            written += decode.write_layer(layer, layer_weights(checkpoint, layer))
            written += decode.write_cache(layer, empty, empty)
        log(f"weights written ({written:.2f} s of writes)")
        try:
            for position in range(steps):
                decode.slot = 0  # one stats slot for every job (the thread's has 32)
                x = np.zeros((md.TILE, model.hidden), np.float32)
                x[0] = formats.bf16_values(embedding[tokens[position]])
                decode.write_activation("x", x)
                if position < hidden.shape[0]:  # layer by layer, each output kept
                    errors = []
                    for layer in range(model.layers):
                        decode.job(1, 1, first=layer, position=position)
                        decode.slot = 0
                        row = decode.read_activation("x")[0]
                        if layer == model.layers - 1:
                            row = final_norm(row, last_norm, model.eps)
                        errors.append(relative_rms(row, hidden[position, layer]))
                    per_layer.append(errors)
                    log(
                        f"position {position}: per layer {np.round(errors, 4).tolist()}"
                    )
                else:
                    ran = decode.job(1, model.layers, position=position)
                    step_us.append(ran["thread_ticks"] / card.TICKS_PER_MICROSECOND)
                row = decode.read_activation("x")[0]
                logits = head @ final_norm(row, last_norm, model.eps)
                if position >= len(prompt) - 1:
                    tokens.append(int(np.argmax(logits)))
                    log(f"position {position}: token {tokens[-1]}")
        except km.KomorebiError as error:
            report["failed"] = f"{type(error).__name__}: {error}"
            log(f"failed: {report['failed']}")
        decode.run.stop()
    got = tokens[len(prompt) :]
    same = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
    report |= {
        "card_tokens": got,
        "reference_tokens": want,
        "first_different": same,
        "per_layer_rms": per_layer,
        "step_us": step_us,
        "tokens_per_second": (
            round(1e6 / float(np.mean(step_us)), 1) if step_us else None
        ),
        "context": args.context,
    }
    if args.tokens:
        np.savez(args.tokens, card=np.array(got), reference=np.array(want))
    print(json.dumps(report), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    sys.exit(0 if "failed" not in report else 1)


if __name__ == "__main__":
    main()
