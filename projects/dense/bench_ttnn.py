"""TT-NN's time for the matrix product (`ttnn.matmul`) and SwiGLU on the inputs of
bench.py and swiglu.py: one preset per call, every route and compute config in one
device open, timed as ttnn_reference.measure does and checked on sampled rows
against float64. The method is written out in README.md."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from komorebi.bench import power  # noqa: E402
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
    "lofi": dict(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    ),
}
# The product's routes: automatic, the whole grid named, tuned as TT-NN's model configs,
# minimal_matmul, decode's DRAM-sharded form (1 or 2 readers a bank), operands in L1.
MATMUL_ROUTES = (
    "matmul",
    "core-grid",
    "tuned",
    "minimal",
    "dram-sharded",
    "dram-sharded-2",
    "l1",
)
# SwiGLU: silu as an op, inside the multiply, `ttnn.linear`'s activation, or
# `minimal_matmul` with W1 and W3 interleaved a tile pair at a time and SwiGLU fused.
SWIGLU_ROUTES = ("composed", "fused-silu", "linear-silu", "minimal-swiglu")
# The Qwen3-1.7B decode products (hidden 2048, MLP 6144, Q 2048 and K, V 1024 wide).
DECODE_SHAPES = {
    "dec-qkv": (32, 2048, 4096),
    "dec-o": (32, 2048, 2048),
    "dec-gate-up": (32, 2048, 12288),
    "dec-down": (32, 6144, 2048),
}
L1_BYTES = 24 << 20  # the "l1" route: A, W and Y together at most this
TRACE_BYTES = 32 << 20  # --trace: the device's trace region


def upload(device, array, memory=None):
    """`array` (float32, bf16 values) as a bf16 tile-layout tensor on the device (in
    DRAM, interleaved, unless `memory` says otherwise)."""
    host_tensor = torch.from_numpy(array).bfloat16()
    return ttnn.from_torch(
        host_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory,
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


def largest_divisor(n, most):
    """The largest divisor of n that is at most `most`."""
    return next(d for d in range(min(n, most), 0, -1) if n % d == 0)


def subblock(block_h, block_w, fp32):
    """(h, w) of the output subblock: w the largest divisor of block_w, then h the
    largest divisor of block_h, h x w at most 4 tiles with fp32 in DST, else 8."""
    most = 4 if fp32 else 8
    width = largest_divisor(block_w, most)
    return largest_divisor(block_h, most // width), width


def tuned_config(grid, m_tiles, k_tiles, n_tiles, fp32):
    """The program config of TT-NN's model configs (`matmul_1d_config`,
    `matmul_config`): a few row tiles take the 1D form with X multicast over N's
    cores, else the 2D form over the whole grid with blocks of at most 8 x 8 tiles."""
    if m_tiles <= 4:
        cores = grid.x * grid.y
        per_core_n = -(-n_tiles // cores)
        height, width = subblock(m_tiles, per_core_n, fp32)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=largest_divisor(k_tiles, 8),
            out_subblock_h=height,
            out_subblock_w=width,
            per_core_M=m_tiles,
            per_core_N=per_core_n,
            fuse_batch=True,
            mcast_in0=True,
        )
    per_core_m, per_core_n = -(-m_tiles // grid.y), -(-n_tiles // grid.x)
    block_h, block_w = largest_divisor(per_core_m, 8), largest_divisor(per_core_n, 8)
    height, width = subblock(block_h, block_w, fp32)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=largest_divisor(k_tiles, 4),
        out_subblock_h=height,
        out_subblock_w=width,
        out_block_h=block_h,
        out_block_w=block_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
    )


def shard_grid(k_tiles, grid):
    """The cores X is width-sharded over (TT-NN's `find_grid`): the count nearest 32
    that divides K's tiles, as rows x columns within the grid."""
    for cores in sorted(range(1, grid.x * grid.y + 1), key=lambda c: abs(c - 32)):
        if k_tiles % cores:
            continue
        for rows in range(1, grid.y + 1):
            if cores % rows == 0 and cores // rows <= grid.x:
                return ttnn.CoreGrid(y=rows, x=cores // rows)
    raise ValueError(f"no shard grid for {k_tiles} K tiles")


def decode_block_width(k_tiles, n_tiles, cores, banks):
    """in0_block_w of TT-NN's `dram_decode_in0_block_w` (one reader a bank)."""
    per_core_k = k_tiles // cores
    per_worker_n = -(-n_tiles // banks)
    base = largest_divisor(per_core_k, 8)
    for width in range(min(16, k_tiles), base, -1):
        if k_tiles % width or (per_core_k % width and width % per_core_k):
            continue
        if width * per_worker_n <= 100 and width * per_worker_n <= 800 * 1024 // 3264:
            return width
    return base


def dram_sharded(device, host, workers):
    """(X, W, program config) of decode's DRAM-sharded product: W width-sharded over
    the DRAM banks, X width-sharded in L1 (as TT-NN's model configs place them)."""
    a_array, w_array = host
    k_depth, n_columns = w_array.shape
    banks = device.dram_grid_size().x
    padded = math.ceil(n_columns / (TILE * banks)) * TILE * banks
    if padded != n_columns or (n_columns // TILE // banks) % workers:
        raise ValueError(f"N {n_columns} does not split over {banks} banks x {workers}")
    weight_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}
            ),
            (k_depth, padded // banks),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    grid = shard_grid(k_depth // TILE, device.compute_with_storage_grid_size())
    cores = grid.x * grid.y
    input_memory = ttnn.create_sharded_memory_config(
        (TILE, k_depth // cores),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    x_tensor = upload(device, a_array, input_memory)
    w_tensor = upload(device, w_array, weight_memory)
    program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=decode_block_width(
            k_depth // TILE, n_columns // TILE, cores, banks
        ),
        per_core_M=1,
        per_core_N=-(-n_columns // (TILE * cores)),
        num_workers_per_dram_bank=workers,
    )
    return x_tensor, w_tensor, program


def matmul_call(route, device, host, tensors, config, fp32):
    """One product by `route` (MATMUL_ROUTES); it may upload operands of its own."""
    a_tensor, w_tensor = tensors
    m_rows, k_depth = host[0].shape
    n_columns = host[1].shape[1]
    grid = device.compute_with_storage_grid_size()
    if route == "matmul":
        return lambda: ttnn.matmul(a_tensor, w_tensor, compute_kernel_config=config)
    if route == "core-grid":
        core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
        return lambda: ttnn.matmul(
            a_tensor, w_tensor, compute_kernel_config=config, core_grid=core_grid
        )
    if route == "tuned":
        program = tuned_config(
            grid, m_rows // TILE, k_depth // TILE, n_columns // TILE, fp32
        )
        return lambda: ttnn.matmul(
            a_tensor, w_tensor, compute_kernel_config=config, program_config=program
        )
    if route == "minimal":
        return lambda: ttnn.experimental.minimal_matmul(
            a_tensor, w_tensor, compute_kernel_config=config
        )
    if route.startswith("dram-sharded"):
        if m_rows != TILE:
            raise ValueError("decode only: one row tile")
        workers = 2 if route.endswith("-2") else 1
        x_tensor, w_sharded, program = dram_sharded(device, host, workers)
        return lambda: ttnn.matmul(
            x_tensor,
            w_sharded,
            compute_kernel_config=config,
            program_config=program,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
    if 2 * (m_rows * k_depth + k_depth * n_columns + m_rows * n_columns) > L1_BYTES:
        raise ValueError(f"A, W and Y exceed {L1_BYTES} B of L1")
    a_l1, w_l1 = (upload(device, array, ttnn.L1_MEMORY_CONFIG) for array in host)
    return lambda: ttnn.matmul(
        a_l1, w_l1, compute_kernel_config=config, memory_config=ttnn.L1_MEMORY_CONFIG
    )


def interleave_pairs(w1, w3):
    """[W1 | W3] with their column tiles interleaved a pair at a time (gate 2p, up
    2p + 1), the weight `minimal_matmul(fuse_swiglu=True)` takes."""
    depth, width = w1.shape
    pairs = np.stack(
        [
            w1.reshape(depth, width // TILE, TILE),
            w3.reshape(depth, width // TILE, TILE),
        ],
        axis=2,
    )
    return np.ascontiguousarray(pairs.reshape(depth, 2 * width))


def swiglu_call(route, device, host, tensors, config):
    """One layer, (silu(x W1) * (x W3)) W2; `composed` has silu and multiply as ops
    of their own, `fused-silu` puts silu on the multiply's first input,
    `linear-silu` on `ttnn.linear`'s output, `minimal-swiglu` makes the gate and up
    one `minimal_matmul` with SwiGLU fused."""
    x_tensor, w1, w3, w2 = tensors
    if route == "minimal-swiglu":
        w13 = upload(device, interleave_pairs(host[1], host[2]))

        def fused():
            hidden = ttnn.experimental.minimal_matmul(
                x_tensor, w13, compute_kernel_config=config, fuse_swiglu=True
            )
            return ttnn.experimental.minimal_matmul(
                hidden, w2, compute_kernel_config=config
            )

        return fused

    def layer():
        if route == "linear-silu":
            x_w1 = ttnn.linear(
                x_tensor, w1, compute_kernel_config=config, activation="silu"
            )
        else:
            x_w1 = ttnn.matmul(x_tensor, w1, compute_kernel_config=config)
        x_w3 = ttnn.matmul(x_tensor, w3, compute_kernel_config=config)
        if route == "composed":
            hidden = ttnn.multiply(ttnn.silu(x_w1), x_w3)
        elif route == "linear-silu":
            hidden = ttnn.multiply(x_w1, x_w3)
        else:
            hidden = ttnn.multiply(
                x_w1, x_w3, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
            )
        return ttnn.matmul(hidden, w2, compute_kernel_config=config)

    return layer


def measure_route(swiglu, route, name, device, operands, timing):
    """One route and compute config timed (ttnn_reference.measure), with its last
    output (on the device) under "output". `operands`: the host arrays and the
    device tensors; `timing`: loops, seconds a loop, replayed as a trace or not."""
    host, tensors = operands
    loops, target, trace = timing
    config = CONFIGS[name]
    fp32 = config is not None and config["fp32_dest_acc_en"]
    config = (
        None
        if config is None
        else ttnn.init_device_compute_kernel_config(device.arch(), None, **config)
    )
    if swiglu:
        call = swiglu_call(route, device, host, tensors, config)
    else:
        call = matmul_call(route, device, host, tensors, config, fp32)
    output = None
    if trace:
        call, output = traced(device, call)
    result = ttnn_reference.measure(
        call,
        lambda: ttnn.synchronize_device(device),
        ttnn_reference.visible_card(),
        loops,
        target,
    )
    result["output"] = call() if output is None else output
    ttnn.synchronize_device(device)
    return result


def traced(device, call):
    """`call` captured once as a trace (after a first call that compiles it): the
    replay, not blocking, and the output tensor every replay writes."""
    call()
    ttnn.synchronize_device(device)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    output = call()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    ttnn.synchronize_device(device)
    return (lambda: ttnn.execute_trace(device, trace, cq_id=0, blocking=False)), output


def check_route(output, rows, want):
    """The sampled rows of `output` (moved to DRAM, interleaved) against float64."""
    with ttnn_reference.heartbeat("check"):
        output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
        return {"check": ref.errors(read_rows(output, rows), want)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--operation", choices=("matmul", "swiglu"), required=True)
    parser.add_argument("--shape", required=True)
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=CONFIGS)
    parser.add_argument(
        "--routes",
        nargs="*",
        default=None,
        choices=MATMUL_ROUTES + SWIGLU_ROUTES,
        help="default: every route of the operation",
    )
    parser.add_argument("--loops", type=int, default=ttnn_reference.LOOPS)
    parser.add_argument("--target", type=float, default=ttnn_reference.TARGET_S)
    parser.add_argument(
        "--trace", action="store_true", help="each route captured once, replayed"
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    swiglu = args.operation == "swiglu"
    shapes = sp.SHAPES if swiglu else {**pl.SHAPES, **DECODE_SHAPES}
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
    report["trace"] = args.trace
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
    device = ttnn.open_device(
        device_id=0, trace_region_size=TRACE_BYTES if args.trace else 0
    )
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
        routes = args.routes or (SWIGLU_ROUTES if swiglu else MATMUL_ROUTES)
        report["results"], report["failed"] = {}, {}
        for route in routes:
            for name in args.configs:
                key = f"{route}/{name}" + ("+trace" if args.trace else "")
                ttnn_reference.log(f"== {key}")
                try:
                    result = measure_route(
                        swiglu,
                        route,
                        name,
                        device,
                        (host, tensors),
                        (args.loops, args.target, args.trace),
                    )
                    result.update(check_route(result.pop("output"), rows, want))
                except (RuntimeError, ValueError, TypeError) as error:  # named
                    report["failed"][key] = f"{type(error).__name__}: {error}"[:600]
                    ttnn_reference.log(f"{key}: failed, {report['failed'][key][:200]}")
                    continue
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
    print(json.dumps(power.clocked(report)), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report))


if __name__ == "__main__":
    main()
