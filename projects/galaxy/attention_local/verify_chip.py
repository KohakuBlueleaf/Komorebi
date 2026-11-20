"""galaxy.driver.verify's golden calls and comparison on a plain one-chip session or
on joined islands (`--cards`); verify's JSON line a call plus the chip's microseconds.
`python verify_chip.py --golden G --sites attn --layers 0,1` (weights: the loader's)
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from galaxy.driver.verify import compare, load_site, to_value  # noqa: E402

import komorebi as km  # noqa: E402
from projects.galaxy.attention_local import piece as pc  # noqa: E402


def golden_entries(path, step, manifest):
    """piece.compress from golden's compress_kv file of the attn call's layer, step."""
    name = (
        os.path.basename(path).replace("_attn_", "_compress_kv_").replace("p07", "p10")
    )

    def compress(layer, x, qr, start_pos, offset):
        ins, out = load_site(
            os.path.join(os.path.dirname(path), name), f"{step}/{name}", manifest
        )
        if int(ins[3]) != offset:
            raise ValueError(f"layer {layer}: offset {offset}, golden's {ins[3]}")
        return out

    return compress


def seed_window(piece, path, step, layer, manifest):
    """The layer's window from golden's state after this decode step: the step's own
    insert writes the one slot that differs from the state before it."""
    name = f"p16_state_L{layer:02d}.npz"
    meta = manifest[f"{step}/{name}"]["window_kv_cache"]
    data = np.load(os.path.join(os.path.dirname(path), name))
    piece.set_window(layer, to_value(data["window_kv_cache"], meta)[0].float().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--sites", default=",".join(pc.SITES))
    ap.add_argument("--layers", default="0,1")
    ap.add_argument("--prompts", default="en,zh")
    ap.add_argument("--steps", default="prefill,decode0,decode1")
    ap.add_argument("--cards", help="joined islands, one link, e.g. 0,1 (default: one)")
    a = ap.parse_args()
    layers = [int(v) for v in a.layers.split(",")]
    sites = a.sites.split(",")
    pat = re.compile(r"p\d\d_(.+?)(?:_L(\d\d))?\.npz$")
    cards = [int(v) for v in a.cards.split(",")] if a.cards else None
    joined = dict(cards=cards, links=1, kernel_bytes=pc.KERNEL_BYTES)
    with km.Session(**joined) if cards else km.Session() as session:
        piece = pc.make(session, None, layers)
        try:
            begun = set()  # (prompt, layer) whose attn chain has begun
            for prompt in a.prompts.split(","):
                with open(os.path.join(a.golden, prompt, "manifest.json")) as f:
                    manifest = json.load(f)
                for step in a.steps.split(","):
                    for path in sorted(
                        glob.glob(os.path.join(a.golden, prompt, step, "p*.npz"))
                    ):
                        m = pat.search(os.path.basename(path))
                        site, layer = m.group(1), (
                            int(m.group(2)) if m.group(2) else None
                        )
                        if site not in sites or layer not in layers:
                            continue
                        ins, ref_out = load_site(
                            path, f"{step}/{os.path.basename(path)}", manifest
                        )
                        if site == "attn" and (prompt, layer) not in begun:
                            begun.add((prompt, layer))
                            if step != "prefill":  # a prompt recorded from a decode
                                seed_window(piece, path, step, layer, manifest)
                        # a compressed layer's entries: golden's compress_kv
                        piece.compress = golden_entries(path, step, manifest)
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
                            "chip_us": round(sum(piece.spent.values()), 1),
                        }
                        print(json.dumps(row), flush=True)
        finally:
            piece.stack.stop()


if __name__ == "__main__":
    main()
