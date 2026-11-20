"""What each conversion to a chip format does to the released values, on real tensors
(relative RMS, largest error, share kept exactly); README.md, "Formats".

python measure_formats.py CKPT_DIR [names...]
"""

import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ckpt  # noqa: E402


def _formats():
    """komorebi.formats by its file (pure numpy), without the package's native part."""
    root = os.environ.get("KOMOREBI_SRC", os.path.join(HERE, "..", "..", "..", "src"))
    path = os.path.join(root, "komorebi", "formats.py")
    spec = importlib.util.spec_from_file_location("km_formats", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


formats = _formats()

DEFAULT = [
    "layers.3.ffn.experts.0.w1.weight",
    "layers.3.ffn.experts.0.w2.weight",
    "layers.30.ffn.experts.200.w3.weight",
    "layers.3.attn.wq_a.weight",
    "layers.3.attn.wq_b.weight",
    "layers.3.attn.wkv.weight",
    "layers.3.attn.wo_a.weight",
    "layers.3.attn.wo_b.weight",
    "layers.3.ffn.shared_experts.w1.weight",
    "layers.3.ffn.shared_experts.w2.weight",
    "layers.3.ffn.gate.weight",
    "layers.3.hc_attn_fn",
    "layers.20.attn.compressor.wkv.weight",
    "layers.1.engram.wkv.weight",
    "head.weight",
]


def roundtrip(w_t, dtype, turned):
    raw, shape = formats.to_bytes(w_t, dtype, "tile", turned)
    back = formats.from_bytes(raw, shape, dtype, "tile", turned)
    return back[: w_t.shape[0], : w_t.shape[1]]


def stats(ref, got):
    ref = ref.astype(np.float64)
    err = got.astype(np.float64) - ref
    rms = np.sqrt(np.mean(err**2) / np.mean(ref**2))
    return rms, np.abs(err).max(), np.abs(ref).max(), np.mean(err == 0)


def main():
    root = sys.argv[1]
    names = sys.argv[2:] or DEFAULT
    c = ckpt.Checkpoint(root)
    print("name\tstored\tshape\tchip\tturned\trel_rms\tmax_err\tmax_abs\texact_share")
    for name in names:
        dtype, shape = c.info(name)[:2]
        t0 = time.time()
        w = c.values(name)  # [out, in] (hc_fn [24, 20480]: [out, in] too)
        rows = w.shape[0] if w.ndim == 2 else 1
        w_t = np.ascontiguousarray(w.reshape(rows, -1).T)  # [in, out]
        if name.endswith("head.weight"):
            w_t = w_t[:, :8192]  # a slice of the vocabulary: time
        chips = {
            "I8": ["bfp4_b", "bfp8_b"],
            "F8_E4M3": ["bfp8_b", "bfloat16"],
            "BF16": ["bfloat16", "bfp8_b"],
            "F32": ["float32", "bfloat16", "bfp8_b"],
        }[dtype]
        for chip in chips:
            for turned in (False, True):
                got = roundtrip(w_t, chip, turned)
                rms, mx, ma, ex = stats(w_t, got)
                print(
                    f"{name}\t{dtype}\t{'x'.join(map(str, shape))}\t{chip}\t"
                    f"{int(turned)}\t{rms:.3e}\t{mx:.3e}\t{ma:.3e}\t{ex:.4f}",
                    flush=True,
                )
        print(f"# {name}: {time.time() - t0:.1f} s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
