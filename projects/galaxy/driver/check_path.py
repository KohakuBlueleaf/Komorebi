"""Check that the driver's site chain, every site on the host, is the reference bit for
bit (README.md, `driver/check_path.py`)."""

import argparse
import time

import torch

from ..reference.model_loader import load_reference
from ..reference.prompts import PROMPTS, encode
from . import state
from .path import TokenPath


def run(fwd, ids, steps):
    out, pos, inp = [], 0, torch.tensor([ids])
    for _ in range(steps):
        t0 = time.time()
        tok, logits = fwd(inp, pos)
        out.append((int(tok[0]), logits.clone(), time.time() - t0))
        pos += inp.size(1)
        inp = tok.view(1, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    ref = load_reference(a.ckpt, max_seq_len=4096, threads=a.threads)
    ids = encode(ref, a.ckpt, PROMPTS["en"])
    state.reset(ref)
    r = run(lambda x, p: ref.model(x, p)[:2], ids, a.steps)
    state.reset(ref)
    path = TokenPath(ref)
    d = run(path.forward, ids, a.steps)
    ok = True
    for n, ((ta, la, sa), (tb, lb, sb)) in enumerate(zip(r, d)):
        same = torch.equal(la, lb)
        ok &= same and ta == tb
        diff = (la - lb).abs().max().item()
        print(
            f"step {n}: reference token {ta} ({sa:.1f} s), "
            f"path token {tb} ({sb:.1f} s), "
            f"logits equal {same}, largest difference {diff:.3g}"
        )
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
