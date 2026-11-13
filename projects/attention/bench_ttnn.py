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

from projects import ttnn_reference  # noqa: E402
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


def upload(device, values):
    """`values` (float32 array of bf16 values) as a bf16 tile-layout tensor on the
    device."""
    tensor = torch.from_numpy(values).bfloat16()
    return ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


def configs(device, route, query_length):
    """(compute kernel config, program config) of a route; None: the library's own."""
    fp32, tuned, exact = ROUTES[route]
    compute = program = None
    if fp32 is not None:
        compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            None,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32,
            packer_l1_acc=False,
        )
    if tuned:
        chunk = 256 if query_length >= 2048 else 64
        grid = device.compute_with_storage_grid_size()
        program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            exp_approx_mode=not exact,
        )
    return compute, program


def check(output, host, call, causal, count):
    """The sampled query tile rows of y (on the device) against float64."""
    queries, keys, values = host
    group = call.h // call.hk
    got, want = [], []
    for batch, head, row in sdpa.sample_rows(call.b, call.h, call.sqt, count):
        first_row = row * TILE
        piece = ttnn.slice(
            output,
            [batch, head, first_row, 0],
            [batch + 1, head + 1, first_row + TILE, output.shape[-1]],
        )
        got.append(ttnn.to_torch(piece).float().numpy().reshape(TILE, -1))
        rows = np.arange(first_row, first_row + TILE)
        want.append(
            sdpa.attention(
                queries[batch, head],
                keys[batch, head // group],
                values[batch, head // group],
                rows,
                causal,
            )
        )
        ttnn_reference.log(
            f"batch {batch} head {head} rows {first_row} to {first_row + TILE} read"
        )
    return sdpa.errors(np.concatenate(got), np.concatenate(want))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shape", choices=pl.SHAPES, required=True)
    parser.add_argument("--routes", nargs="*", default=list(ROUTES), choices=ROUTES)
    parser.add_argument("--rows", type=int, default=16, help="query tile rows checked")
    parser.add_argument("--loops", type=int, default=ttnn_reference.LOOPS)
    parser.add_argument("--target", type=float, default=ttnn_reference.TARGET_S)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    batches, heads, kv_heads, sequence_length, head_dim, causal = pl.SHAPES[
        arguments.shape
    ]
    call = pl.dims(
        batches,
        heads,
        kv_heads,
        sequence_length,
        sequence_length,
        head_dim,
        head_dim,
        causal,
    )
    report = {
        "shape": arguments.shape,
        "b": batches,
        "h": heads,
        "hk": kv_heads,
        "s": sequence_length,
        "d": head_dim,
    }
    report["causal"] = causal
    report["command"] = ttnn_reference.command(sys.argv)
    with ttnn_reference.heartbeat("building the inputs"):
        host = sdpa.inputs(
            batches,
            heads,
            kv_heads,
            sequence_length,
            sequence_length,
            head_dim,
            head_dim,
        )
    ttnn_reference.log("inputs built")
    card_index = ttnn_reference.visible_card()
    device = ttnn.open_device(device_id=0)
    try:
        grid = device.compute_with_storage_grid_size()
        report["device"] = {
            "arch": str(device.arch()),
            "compute_grid": [grid.x, grid.y],
        }
        with ttnn_reference.heartbeat("upload"):
            query_tensor, key_tensor, value_tensor = (
                upload(device, array) for array in host
            )
            ttnn.synchronize_device(device)
        ttnn_reference.log("uploaded")
        report["results"] = {}
        for route in arguments.routes:
            compute, program = configs(device, route, sequence_length)
            ttnn_reference.log(f"== {route}")

            def call(compute=compute, program=program):
                return ttnn.transformer.scaled_dot_product_attention(
                    query_tensor,
                    key_tensor,
                    value_tensor,
                    is_causal=causal,
                    program_config=program,
                    compute_kernel_config=compute,
                )

            measurement = ttnn_reference.measure(
                call,
                lambda: ttnn.synchronize_device(device),
                card_index,
                arguments.loops,
                arguments.target,
            )
            output = call()
            ttnn.synchronize_device(device)
            with ttnn_reference.heartbeat("check"):
                measurement["check"] = check(output, host, call, causal, arguments.rows)
            measurement["exact_exp"] = ROUTES[route][2]
            ttnn_reference.log(
                f"{route}: {measurement['microseconds_per_call']:.2f} us, "
                f"check {measurement['check']}"
            )
            report["results"][route] = measurement
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
    if arguments.json:
        Path(arguments.json).write_text(json.dumps(report))


if __name__ == "__main__":
    main()
