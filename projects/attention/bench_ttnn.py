"""TT-NN's time for attention (`ttnn.transformer.scaled_dot_product_attention`) on
the inputs of run.py: one preset per call, every route in one device open, timed as
ttnn_reference.measure does, checked on sampled query rows against float64. The
method is written out in the dense project's README.md."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import ttnn

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402

from projects import ttnn_reference as tr  # noqa: E402
from projects.attention import plan as pl  # noqa: E402
from projects.attention.reference import sdpa  # noqa: E402

TILE = pl.TILE
# Routes: fp32 accumulation (None: the library's own compute config), tuned program
# config (whole grid, chunks of 64 below 2048 keys, else 256), exact exponential.
ROUTES = {
    "default": (None, False, False),
    "documented": (False, True, True),
    "documented-fp32": (True, True, True),
    "documented-approx": (False, True, False),
}


def upload(dev, x):
    """x (float32 array of bf16 values) as a bf16 tile-layout tensor on the device."""
    t = torch.from_numpy(x).bfloat16()
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def configs(dev, route, sq):
    """(compute kernel config, program config) of a route; None: the library's own."""
    fp32, tuned, exact = ROUTES[route]
    compute = program = None
    if fp32 is not None:
        compute = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            None,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32,
            packer_l1_acc=False,
        )
    if tuned:
        chunk = 256 if sq >= 2048 else 64
        grid = dev.compute_with_storage_grid_size()
        program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            exp_approx_mode=not exact,
        )
    return compute, program


def check(y, host, s, causal, count):
    """The sampled query tile rows of y (on the device) against float64."""
    q, k, v = host
    group = s.h // s.hk
    got, want = [], []
    for bi, hi, row in sdpa.sample_rows(s.b, s.h, s.sqt, count):
        r0 = row * TILE
        piece = ttnn.slice(y, [bi, hi, r0, 0], [bi + 1, hi + 1, r0 + TILE, y.shape[-1]])
        got.append(ttnn.to_torch(piece).float().numpy().reshape(TILE, -1))
        rows = np.arange(r0, r0 + TILE)
        want.append(
            sdpa.attention(
                q[bi, hi], k[bi, hi // group], v[bi, hi // group], rows, causal
            )
        )
        tr.log(f"batch {bi} head {hi} rows {r0} to {r0 + TILE} read")
    return sdpa.errors(np.concatenate(got), np.concatenate(want))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shape", choices=pl.SHAPES, required=True)
    p.add_argument("--routes", nargs="*", default=list(ROUTES), choices=ROUTES)
    p.add_argument("--rows", type=int, default=16, help="query tile rows checked")
    p.add_argument("--loops", type=int, default=tr.LOOPS)
    p.add_argument("--target", type=float, default=tr.TARGET_S)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    b, h, hk, sq, d, causal = pl.SHAPES[args.shape]
    s = pl.dims(b, h, hk, sq, sq, d, d, causal)
    out = {"shape": args.shape, "b": b, "h": h, "hk": hk, "s": sq, "d": d}
    out["causal"] = causal
    out["command"] = tr.command(sys.argv)
    with tr.heartbeat("building the inputs"):
        host = sdpa.inputs(b, h, hk, sq, sq, d, d)
    tr.log("inputs built")
    card = tr.visible_card()
    dev = ttnn.open_device(device_id=0)
    try:
        grid = dev.compute_with_storage_grid_size()
        out["device"] = {"arch": str(dev.arch()), "compute_grid": [grid.x, grid.y]}
        with tr.heartbeat("upload"):
            tq, tk, tv = (upload(dev, x) for x in host)
            ttnn.synchronize_device(dev)
        tr.log("uploaded")
        out["results"] = {}
        for route in args.routes:
            compute, program = configs(dev, route, sq)
            tr.log(f"== {route}")

            def call(compute=compute, program=program):
                return ttnn.transformer.scaled_dot_product_attention(
                    tq,
                    tk,
                    tv,
                    is_causal=causal,
                    program_config=program,
                    compute_kernel_config=compute,
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
                res["check"] = check(y, host, s, causal, args.rows)
            res["exact_exp"] = ROUTES[route][2]
            tr.log(
                f"{route}: {res['microseconds_per_call']:.2f} us, check {res['check']}"
            )
            out["results"][route] = res
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
