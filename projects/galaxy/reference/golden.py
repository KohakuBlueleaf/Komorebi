"""The saved inputs and outputs of every site for the fixed prompts (README.md,
`reference/golden.py`)."""

import argparse
import json
import os
import time

import numpy as np
import torch

from ..driver import state
from ..driver.path import SITE_PIECE, Recorder, TokenPath
from .model_loader import load_reference
from .prompts import encode, prompt_text

LAYERS = [0, 1, 2, 3, 14, 20, 21, 24]
STEPS = ["prefill", "decode0", "decode1"]


def _np(v):
    """A site value as a numpy array and the reference's dtype name."""
    if isinstance(v, torch.Tensor):
        dt = str(v.dtype).removeprefix("torch.")
        if v.is_complex():
            return torch.view_as_real(v).numpy().astype(np.float32), dt
        if v.dtype.is_floating_point:
            return v.float().numpy(), dt
        return v.numpy(), dt
    if isinstance(v, (bool, int, float)):
        return np.array(v), type(v).__name__
    raise TypeError(type(v))


def _flatten(prefix, v, out, meta):
    if v is None:
        meta[prefix] = {"dtype": "None"}
    elif isinstance(v, (tuple, list)):
        for k, x in enumerate(v):
            _flatten(f"{prefix}{k}", x, out, meta)
    else:
        arr, dt = _np(v)
        out[prefix] = arr
        meta[prefix] = {"dtype": dt, "shape": list(arr.shape)}


def save_step(rec, ref, step, path_dir, manifest, layers=LAYERS):
    os.makedirs(path_dir, exist_ok=True)
    for (st, site, layer), ((names, ins), outs) in rec.store.items():
        if st != step:
            continue
        arrays, meta = {}, {}
        for n, v in zip(names, ins):
            _flatten(f"in_{n}", v, arrays, meta)
        _flatten("out", outs, arrays, meta)
        lname = "" if layer is None else f"_L{layer:02d}"
        fname = f"p{SITE_PIECE[site]:02d}_{site}{lname}.npz"
        np.savez(os.path.join(path_dir, fname), **arrays)
        manifest[f"{step}/{fname}"] = meta
    for i in layers:  # piece 16: the state after the step
        arrays, meta = {}, {}
        for n, t in state.state_tensors(ref, i).items():
            _flatten(n, t, arrays, meta)
        fname = f"p16_state_L{i:02d}.npz"
        np.savez(os.path.join(path_dir, fname), **arrays)
        manifest[f"{step}/{fname}"] = meta


def run_prompt(
    path, ref, ckpt, key, out_dir, new_tokens, record=True, steps=STEPS, layers=LAYERS
):
    text = prompt_text(ref, ckpt, key)
    ids = encode(ref, ckpt, text)
    assert key not in ("en", "zh") or len(ids) == 32, len(ids)
    state.reset(ref)
    rec = Recorder(layers=set(layers) | {None})
    path.recorder = rec if record else None
    pdir = os.path.join(out_dir, key)
    os.makedirs(pdir, exist_ok=True)
    manifest, times, tokens, all_logits = {}, [], [], []
    pos, inp = 0, torch.tensor([ids])
    for n in range(new_tokens):
        step = STEPS[n] if (record and n < len(STEPS) and STEPS[n] in steps) else None
        rec.step = step
        t0 = time.time()
        tok, logits = path.forward(inp, pos)
        times.append(time.time() - t0)
        if step is not None:
            save_step(rec, ref, step, os.path.join(pdir, step), manifest, layers)
            rec.store.clear()
        tokens.append(int(tok[0]))
        all_logits.append(logits[0].float().numpy())
        print(
            f"{key} step {n} pos {pos} token {tokens[-1]} {times[-1]:.2f} s", flush=True
        )
        pos += inp.size(1)
        inp = tok.view(1, 1)
        if tokens[-1] == ref.tokenizer.eos_token_id:
            break
    path.recorder = None
    if record:
        with open(os.path.join(pdir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
    np.save(os.path.join(pdir, "logits.npy"), np.stack(all_logits))
    info = {
        "prompt": text,
        "prompt_ids": ids,
        "generated_ids": tokens,
        "generated_text": ref.tokenizer.decode(tokens),
        "seconds_per_step": [round(t, 3) for t in times],
        "steps_recorded": [st for st in STEPS if st in steps] if record else [],
        "layers_recorded": layers,
    }
    with open(os.path.join(pdir, "tokens.json"), "w") as f:
        json.dump(info, f, indent=1, ensure_ascii=False)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--prompts", default="en,zh")
    ap.add_argument(
        "--no-record",
        action="store_true",
        help="tokens and logits only (tokens.json, logits.npy)",
    )
    ap.add_argument("--steps", default=",".join(STEPS), help="the steps to record")
    ap.add_argument(
        "--layers", default=",".join(map(str, LAYERS)), help="the layers to record"
    )
    a = ap.parse_args()
    t0 = time.time()
    ref = load_reference(a.ckpt, max_seq_len=4096, threads=a.threads)
    print(f"load {time.time() - t0:.1f} s", flush=True)
    path = TokenPath(ref)
    for key in a.prompts.split(","):
        info = run_prompt(
            path,
            ref,
            a.ckpt,
            key,
            a.out,
            a.new,
            record=not a.no_record,
            steps=a.steps.split(","),
            layers=[int(x) for x in a.layers.split(",")],
        )
        print(key, info["generated_text"], flush=True)


if __name__ == "__main__":
    main()
