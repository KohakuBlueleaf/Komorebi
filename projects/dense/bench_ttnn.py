"""TT-NN's time for the matrix product (`ttnn.matmul`) and SwiGLU on the inputs of
bench.py and swiglu.py: one preset per call, every route and compute config in one
device open, timed as ttnn_reference.measure does and checked on sampled rows
against float64. The method is written out in README.md."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from projects import ttnn_reference as tr  # noqa: E402
from projects.dense import plan as pl  # noqa: E402
from projects.dense import swiglu_plan as sp  # noqa: E402
from ref import dense as ref  # noqa: E402

TILE = ref.TILE
# name -> compute kernel config arguments; None is TT-NN's own default, which for
# a bf16 matmul on this tree is HiFi2, bf16 accumulation in DST, packer L1 accumulation.
CONFIGS = {
    "default": None,
    "hifi2-fp32": dict(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    ),
}
ROUTES = ("composed", "fused-silu")  # SwiGLU: silu as an op, or inside the multiply


def upload(dev, x):
    """x (float32 array of bf16 values) as a bf16 tile-layout tensor on the device."""
    t = torch.from_numpy(x).bfloat16()
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def read_rows(y, rows):
    """Rows `rows` of the 2-D device tensor y as float32, one 32-row slice per read
    (at most 1 MB each, a progress line each)."""
    got = {}
    for t in np.unique(rows // TILE):
        r0 = int(t) * TILE
        piece = ttnn.slice(y, [r0, 0], [r0 + TILE, y.shape[-1]])
        got[r0] = ttnn.to_torch(piece).float().numpy()
        tr.log(f"rows {r0} to {r0 + TILE} read")
    return np.stack([got[r // TILE * TILE][r % TILE] for r in rows])


def matmul_call(ta, tw, cfg):
    """One `ttnn.matmul`."""
    return lambda: ttnn.matmul(ta, tw, compute_kernel_config=cfg)


def swiglu_call(route, tx, w1, w3, w2, cfg):
    """One layer, (silu(x W1) * (x W3)) W2, from `ttnn.matmul`; `composed` has silu and
    multiply as ops of their own, `fused-silu` puts silu on the multiply's first input.
    """

    def layer():
        a = ttnn.matmul(tx, w1, compute_kernel_config=cfg)
        b = ttnn.matmul(tx, w3, compute_kernel_config=cfg)
        if route == "composed":
            h = ttnn.multiply(ttnn.silu(a), b)
        else:
            h = ttnn.multiply(a, b, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return ttnn.matmul(h, w2, compute_kernel_config=cfg)

    return layer


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--op", choices=("matmul", "swiglu"), required=True)
    p.add_argument("--shape", required=True)
    p.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=CONFIGS)
    p.add_argument("--loops", type=int, default=tr.LOOPS)
    p.add_argument("--target", type=float, default=tr.TARGET_S)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    swiglu = args.op == "swiglu"
    shapes = sp.SHAPES if swiglu else pl.SHAPES
    if args.shape not in shapes:
        p.error(f"--shape: one of {', '.join(shapes)}")
    m, k, n = shapes[args.shape]
    out = {"op": args.op, "shape": args.shape, "m": m, "k": k, "n": n}
    out["command"] = tr.command(sys.argv)
    with tr.heartbeat("building the inputs"):
        if swiglu:
            host = ref.swiglu_inputs(m, k, n)
            rows = ref.sampled_rows(m)
            want = ref.swiglu(host[0][rows], *host[1:])
            out["d"], out["h"] = k, n
        else:
            host = ref.inputs(m, k, n)
            rows = ref.sampled_rows(m)
            want = ref.matmul(host[0][rows], host[1])
    tr.log("inputs built, float64 reference done")
    card = tr.visible_card()
    dev = ttnn.open_device(device_id=0)
    try:
        out["device"] = {
            "arch": str(dev.arch()),
            "compute_grid": [dev.compute_with_storage_grid_size().x]
            + [dev.compute_with_storage_grid_size().y],
        }
        with tr.heartbeat("upload"):
            tensors = [upload(dev, x) for x in host]
            ttnn.synchronize_device(dev)
        tr.log("uploaded")
        routes = ROUTES if swiglu else ("matmul",)
        out["results"] = {}
        for route in routes:
            for name in args.configs:
                cfg = CONFIGS[name]
                cfg = (
                    None
                    if cfg is None
                    else ttnn.init_device_compute_kernel_config(dev.arch(), None, **cfg)
                )
                key = f"{route}/{name}"
                tr.log(f"== {key}")
                call = (
                    swiglu_call(route, *tensors, cfg)
                    if swiglu
                    else matmul_call(*tensors, cfg)
                )
                res = tr.measure(
                    call,
                    lambda: ttnn.synchronize_device(dev),
                    card,
                    args.loops,
                    args.target,
                )
                y = call()
                ttnn.synchronize_device(dev)
                with tr.heartbeat("check"):
                    got = read_rows(y, rows)
                    res["check"] = ref.errors(got, want)
                tr.log(f"{key}: {res['us_per_call']:.2f} us, check {res['check']}")
                out["results"][key] = res
        out["program_cache_entries"] = dev.num_program_cache_entries()
    finally:
        ttnn.close_device(dev)
    out["ttnn"] = {
        "tt_metal": tr.tt_metal_version(os.environ["TT_METAL_HOME"]),
        "dispatch": "slow" if os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") else "fast",
    }
    out["layout"] = "bf16, TILE_LAYOUT, DRAM interleaved; output the same"
    out["load_avg"] = os.getloadavg()[0]
    out["date"] = time.strftime("%Y-%m-%d %H:%M")
    print(json.dumps(out), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
