"""Check one box piece against the saved tensors of golden/ (README.md,
`driver/verify.py`)."""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch

from ..reference.model_loader import Checkpoint
from .interface import load_factory, load_weights
from .run import open_session, parse_layers

_DT = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
    "complex64": torch.complex64,
    "bool": torch.bool,
    "float8_e4m3fn": torch.float8_e4m3fn,
}


def to_value(arr, meta):
    dt = meta["dtype"]
    if dt == "int":
        return int(arr)
    if dt == "float":
        return float(arr)
    if dt == "bool":
        return bool(arr) if arr.ndim == 0 else torch.from_numpy(arr)
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if dt == "complex64":
        return torch.view_as_complex(t.float().contiguous())
    return t.to(_DT[dt])


def load_site(path, manifest_key, manifest):
    """(inputs in order, the reference outputs) of one golden file."""
    meta = manifest[manifest_key]
    d = np.load(path)
    ins, outs = [], {}
    for key, m in meta.items():  # manifest keeps the call's argument order
        if key.startswith("in_"):
            ins.append(None if m["dtype"] == "None" else to_value(d[key], m))
        else:
            outs[key] = None if m["dtype"] == "None" else to_value(d[key], m)
    if list(outs) == ["out"]:
        out = outs["out"]
    else:
        out = tuple(outs[k] for k in sorted(outs, key=lambda k: int(k[3:])))
    return ins, out


def compare(box, ref):
    """A list of per-output dicts."""
    if isinstance(ref, tuple):
        assert isinstance(box, (tuple, list)) and len(box) == len(ref), (
            type(box),
            len(ref),
        )
        return [compare(b, r)[0] for b, r in zip(box, ref)]
    if ref is None:
        return [{"none": box is None}]
    if not isinstance(box, torch.Tensor):
        box = torch.as_tensor(box)
    if tuple(box.shape) != tuple(ref.shape):
        return [{"shape": list(box.shape), "ref_shape": list(ref.shape)}]
    row = {
        "dtype": str(box.dtype).removeprefix("torch."),
        "ref_dtype": str(ref.dtype).removeprefix("torch."),
    }
    if ref.dtype.is_floating_point:
        a, b = box.double(), ref.double()
        nb = b.norm().item()
        row["rel_rms"] = (a - b).norm().item() / nb if nb else (a - b).norm().item()
        row["max_abs"] = (a - b).abs().max().item() if a.numel() else 0.0
        row["ref_max_abs"] = b.abs().max().item() if b.numel() else 0.0
    else:
        row["equal"] = bool(torch.equal(box.to(ref.dtype), ref))
    return [row]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--piece", required=True)
    ap.add_argument("--cards", default=None)
    ap.add_argument("--load", default=None, help="the loader's W (as driver.run)")
    ap.add_argument(
        "--ckpt",
        default=None,
        help="checkpoint dir: W = its Checkpoint when no --load",
    )
    ap.add_argument("--sites", default=None)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--prompts", default="en,zh")
    ap.add_argument("--steps", default="prefill,decode0,decode1")
    a = ap.parse_args()

    session = None
    if a.cards:
        import komorebi as km  # noqa: PLC0415 (only a run with box pieces needs it)

        session = open_session(km, parse_layers(a.cards, 0))
    if a.load:
        W = load_weights(session, a.load, parse_layers)
    elif a.ckpt:
        W = Checkpoint(a.ckpt)
    else:
        W = None
    layers = parse_layers(a.layers, 40) if a.layers else list(range(40)) + [None]
    piece = load_factory(a.piece)(session, W, layers)
    sites = a.sites.split(",") if a.sites else list(piece.sites)
    pat = re.compile(r"p\d\d_(.+?)(?:_L(\d\d))?\.npz$")
    for prompt in a.prompts.split(","):
        with open(os.path.join(a.golden, prompt, "manifest.json")) as f:
            manifest = json.load(f)
        for step in a.steps.split(","):
            for path in sorted(
                glob.glob(os.path.join(a.golden, prompt, step, "p*.npz"))
            ):
                m = pat.search(os.path.basename(path))
                site, layer = m.group(1), (int(m.group(2)) if m.group(2) else None)
                if site not in sites or layer not in layers:
                    continue
                ins, ref_out = load_site(
                    path, f"{step}/{os.path.basename(path)}", manifest
                )
                t0 = time.time()
                out = piece(site, layer, *ins)
                dt = time.time() - t0
                row = {
                    "prompt": prompt,
                    "step": step,
                    "site": site,
                    "layer": layer,
                    "seconds": round(dt, 6),
                    "outputs": compare(out, ref_out),
                }
                print(json.dumps(row), flush=True)
    if session is not None:
        session.__exit__(None, None, None)


if __name__ == "__main__":
    main()
