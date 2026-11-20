"""galaxy.driver.verify's golden calls and comparison on a plain one-chip session (its
own is of several cards); verify's JSON line a call plus the chip's microseconds.
`python verify_chip.py --golden G --ckpt C --sites attn --layers 0,1`
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from galaxy.driver.verify import compare, load_site  # noqa: E402
from galaxy.reference.model_loader import Checkpoint  # noqa: E402

import komorebi as km  # noqa: E402
from projects.galaxy.attention_local import piece as pc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sites", default=",".join(pc.SITES))
    ap.add_argument("--layers", default="0,1")
    ap.add_argument("--prompts", default="en,zh")
    ap.add_argument("--steps", default="prefill,decode0,decode1")
    a = ap.parse_args()
    layers = [int(v) for v in a.layers.split(",")]
    sites = a.sites.split(",")
    pat = re.compile(r"p\d\d_(.+?)(?:_L(\d\d))?\.npz$")
    with km.Session() as session:
        piece = pc.make(session, Checkpoint(a.ckpt), layers)
        try:
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
