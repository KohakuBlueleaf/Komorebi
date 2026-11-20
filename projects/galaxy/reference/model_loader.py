"""Load the model's own reference code and its released checkpoint on a host CPU
(README.md, `reference/model_loader.py`)."""

import importlib
import json
import os
import struct
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoTokenizer

from . import cpu_kernels

_ST_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F16": torch.float16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I8": torch.int8,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
}


class Checkpoint:
    """The checkpoint's tensors by name, each a zero-copy view of a memory-mapped
    shard."""

    def __init__(self, ckpt_dir: str):
        self.dir = ckpt_dir
        with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self._maps: dict[str, np.memmap] = {}
        self._headers: dict[str, tuple[int, dict]] = {}

    def _shard(self, fname: str):
        if fname not in self._maps:
            path = os.path.join(self.dir, fname)
            with open(path, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(n))
            self._headers[fname] = (8 + n, header)
            self._maps[fname] = np.memmap(path, dtype=np.uint8, mode="r")
        return self._maps[fname], self._headers[fname]

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        fname = self.weight_map[name]
        mm, (base, header) = self._shard(fname)
        info = header[name]
        start, end = info["data_offsets"]
        raw = mm[base + start : base + end]
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # read-only memory map: torch warns, nobody writes
            t = torch.from_numpy(raw)
        return t.view(_ST_DTYPES[info["dtype"]]).view(info["shape"])


@dataclass
class Reference:
    model: torch.nn.Module  # the reference Transformer
    mod: object  # the reference `model` module (its functions and its shared_attn)
    args: object  # its ModelArgs
    tokenizer: object
    ckpt: Checkpoint


def import_reference_code(ckpt_dir: str):
    """Import the checkpoint's `inference/model.py` with the CPU kernels in place of
    `kernel`."""
    inf = os.path.join(ckpt_dir, "inference")
    sys.modules["kernel"] = cpu_kernels
    if inf not in sys.path:
        sys.path.insert(0, inf)
    return importlib.import_module("model")


def _dequant_wo_a(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """convert.py's wo_a step: E4M3 weight times its block scale, to bfloat16."""
    ob = weight.size(0) // scale.size(0)
    ib = weight.size(1) // scale.size(1)
    w = (
        weight.unflatten(0, (-1, ob)).unflatten(-1, (-1, ib)).float()
        * scale[:, None, :, None].float()
    )
    return w.flatten(2, 3).flatten(0, 1).bfloat16()


def load_reference(
    ckpt_dir: str, max_seq_len: int = 4096, threads: int | None = None
) -> Reference:
    """Build the reference model for one request (batch 1) of at most `max_seq_len`
    positions."""
    if threads:
        torch.set_num_threads(threads)
    torch.set_default_dtype(torch.bfloat16)  # as generate.py
    mod = import_reference_code(ckpt_dir)
    with open(os.path.join(ckpt_dir, "inference", "config.json")) as f:
        cfg = json.load(f)
    args = mod.ModelArgs(**cfg)
    args.max_batch_size = 1
    args.max_seq_len = max_seq_len
    args.temperature = 0.0  # greedy
    args.vision_n_layers = 0  # text only
    args.dspark_block_size = 0  # no speculative decoding
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)

    # Build on meta (no memory), the n-gram hash state for real on the CPU afterwards.
    real_hash_cls = mod.NgramHashState

    class _NoHash(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

    mod.NgramHashState = _NoHash
    try:
        with torch.device("meta"):
            model = mod.Transformer(args, tokenizer)
    finally:
        mod.NgramHashState = real_hash_cls
    model.engram_hash = real_hash_cls(args, model.engram_layout, tokenizer)

    ckpt = Checkpoint(ckpt_dir)
    for mname, module in model.named_modules():
        for pname, p in list(module._parameters.items()):
            if p is None:
                continue
            full = f"{mname}.{pname}" if mname else pname
            if full.endswith("attn.wo_a.weight"):
                t = _dequant_wo_a(
                    ckpt.get(full), ckpt.get(full.replace("weight", "scale"))
                )
            else:
                t = ckpt.get(full)
                if p.dtype == torch.float4_e2m1fn_x2:
                    assert t.dtype == torch.int8, full
                    t = t.view(torch.float4_e2m1fn_x2)
                elif t.dtype != p.dtype:
                    t = t.to(p.dtype)
            assert tuple(t.shape) == tuple(p.shape), (full, t.shape, p.shape)
            module._parameters[pname] = torch.nn.Parameter(t, requires_grad=False)
        if getattr(module, "scale", None) is not None and isinstance(
            module._parameters.get("weight"), torch.Tensor
        ):
            module.weight.scale = module.scale
        for bname, b in list(module._buffers.items()):
            if b is None or b.device.type != "meta":
                continue
            if bname == "freqs_cis":
                continue  # recomputed below
            fill = -torch.inf if bname == "score_state" else 0
            module._buffers[bname] = torch.full(b.shape, fill, dtype=b.dtype)

    # Rotary tables: the same arguments as Attention.__init__, computed on the CPU.
    pre = mod.precompute_freqs_cis.__wrapped__
    for layer in model.layers:
        attn = layer.attn
        if attn.compress_ratio:
            osl, theta = args.original_seq_len, args.compress_rope_theta
        else:
            osl, theta = 0, args.rope_theta
        attn._buffers["freqs_cis"] = pre(
            args.rope_head_dim,
            args.max_seq_len,
            osl,
            theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
    for n, t in list(model.named_parameters()) + list(model.named_buffers()):
        assert t.device.type == "cpu", n
    model.eval()
    return Reference(model=model, mod=mod, args=args, tokenizer=tokenizer, ckpt=ckpt)
