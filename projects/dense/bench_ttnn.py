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

from projects import ttnn_reference  # noqa: E402
from projects.dense import plan as pl  # noqa: E402
from projects.dense import swiglu_plan as sp  # noqa: E402
from reference import dense as ref  # noqa: E402

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


def upload(device, array):
    """`array` (float32, bf16 values) as a bf16 tile-layout tensor on the device."""
    host_tensor = torch.from_numpy(array).bfloat16()
    return ttnn.from_torch(
        host_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


def read_rows(y, rows):
    """Rows `rows` of the 2-D device tensor y as float32, one 32-row slice per read
    (at most 1 MB each, a progress line each)."""
    got = {}
    for tile in np.unique(rows // TILE):
        first_row = int(tile) * TILE
        piece = ttnn.slice(y, [first_row, 0], [first_row + TILE, y.shape[-1]])
        got[first_row] = ttnn.to_torch(piece).float().numpy()
        ttnn_reference.log(f"rows {first_row} to {first_row + TILE} read")
    return np.stack([got[row // TILE * TILE][row % TILE] for row in rows])


def matmul_call(a_tensor, w_tensor, config):
    """One `ttnn.matmul`."""
    return lambda: ttnn.matmul(a_tensor, w_tensor, compute_kernel_config=config)


def swiglu_call(route, x_tensor, w1, w3, w2, config):
    """One layer, (silu(x W1) * (x W3)) W2, from `ttnn.matmul`; `composed` has silu and
    multiply as ops of their own, `fused-silu` puts silu on the multiply's first input.
    """

    def layer():
        x_w1 = ttnn.matmul(x_tensor, w1, compute_kernel_config=config)
        x_w3 = ttnn.matmul(x_tensor, w3, compute_kernel_config=config)
        if route == "composed":
            hidden = ttnn.multiply(ttnn.silu(x_w1), x_w3)
        else:
            hidden = ttnn.multiply(
                x_w1, x_w3, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
            )
        return ttnn.matmul(hidden, w2, compute_kernel_config=config)

    return layer


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--operation", choices=("matmul", "swiglu"), required=True)
    parser.add_argument("--shape", required=True)
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=CONFIGS)
    parser.add_argument("--loops", type=int, default=ttnn_reference.LOOPS)
    parser.add_argument("--target", type=float, default=ttnn_reference.TARGET_S)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    swiglu = args.operation == "swiglu"
    shapes = sp.SHAPES if swiglu else pl.SHAPES
    if args.shape not in shapes:
        parser.error(f"--shape: one of {', '.join(shapes)}")
    m_rows, k_depth, n_columns = shapes[args.shape]
    report = {
        "op": args.operation,
        "shape": args.shape,
        "m": m_rows,
        "k": k_depth,
        "n": n_columns,
    }
    report["command"] = ttnn_reference.command(sys.argv)
    with ttnn_reference.heartbeat("building the inputs"):
        if swiglu:
            host = ref.swiglu_inputs(m_rows, k_depth, n_columns)
            rows = ref.sampled_rows(m_rows)
            want = ref.swiglu(host[0][rows], *host[1:])
            report["d"], report["h"] = k_depth, n_columns
        else:
            host = ref.inputs(m_rows, k_depth, n_columns)
            rows = ref.sampled_rows(m_rows)
            want = ref.matmul(host[0][rows], host[1])
    ttnn_reference.log("inputs built, float64 reference done")
    card = ttnn_reference.visible_card()
    device = ttnn.open_device(device_id=0)
    try:
        report["device"] = {
            "arch": str(device.arch()),
            "compute_grid": [device.compute_with_storage_grid_size().x]
            + [device.compute_with_storage_grid_size().y],
        }
        with ttnn_reference.heartbeat("upload"):
            tensors = [upload(device, array) for array in host]
            ttnn.synchronize_device(device)
        ttnn_reference.log("uploaded")
        routes = ROUTES if swiglu else ("matmul",)
        report["results"] = {}
        for route in routes:
            for name in args.configs:
                config = CONFIGS[name]
                config = (
                    None
                    if config is None
                    else ttnn.init_device_compute_kernel_config(
                        device.arch(), None, **config
                    )
                )
                key = f"{route}/{name}"
                ttnn_reference.log(f"== {key}")
                call = (
                    swiglu_call(route, *tensors, config)
                    if swiglu
                    else matmul_call(*tensors, config)
                )
                result = ttnn_reference.measure(
                    call,
                    lambda: ttnn.synchronize_device(device),
                    card,
                    args.loops,
                    args.target,
                )
                y = call()
                ttnn.synchronize_device(device)
                with ttnn_reference.heartbeat("check"):
                    got = read_rows(y, rows)
                    result["check"] = ref.errors(got, want)
                microseconds = result["microseconds_per_call"]
                ttnn_reference.log(
                    f"{key}: {microseconds:.2f} us, check {result['check']}"
                )
                report["results"][key] = result
        report["program_cache_entries"] = device.num_program_cache_entries()
    finally:
        ttnn.close_device(device)
    report["ttnn"] = {
        "tt_metal": ttnn_reference.tt_metal_version(os.environ["TT_METAL_HOME"]),
        "dispatch": "slow" if os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") else "fast",
    }
    report["layout"] = "bf16, TILE_LAYOUT, DRAM interleaved; output the same"
    report["load_avg"] = os.getloadavg()[0]
    report["date"] = time.strftime("%Y-%m-%d %H:%M")
    print(json.dumps(report), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report))


if __name__ == "__main__":
    main()
