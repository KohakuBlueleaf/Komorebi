"""Can compute read page k of a buffer it waited for in full, by index, with nothing
popped before it? One put from core (1, 0) into page k of core (0, 0). JSON line.

Example: python probe.py --slot 2 --other 1"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
RECV, SEND = (0, 0), (1, 0)
PAGES, TILE, WORDS = 4, 32, 16
T0 = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def cores(*cs):
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cs]
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--slot", type=int, default=2, help="page k the sender puts")
    p.add_argument("--other", type=int, default=1, help="a page the reader filled")
    args = p.parse_args()
    k, j = args.slot, args.other
    assert 0 <= k < PAGES and 0 <= j < PAGES and j != k
    # Tile t holds t * 32 + row: exact in bfloat16, every tile and row distinct.
    x = np.repeat(np.arange(PAGES * TILE, dtype=np.float32)[:, None], TILE, axis=1)
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    log("device open")
    try:
        out = run(device, x, k, j)
    finally:
        ttnn.close_mesh_device(device)
    print(json.dumps(out), flush=True)


def run(device, x, k, j):
    def tiles(a):
        return ttnn.from_torch(
            torch.from_numpy(a),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    inp = tiles(x)
    out = tiles(np.full((2 * TILE, TILE), -1.0, dtype=np.float32))
    both = cores(RECV, SEND)
    flags = ttnn.from_torch(
        torch.zeros((2, WORDS), dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            (1, WORDS),
            both,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )
    recv_xy = device.worker_core_from_logical_core(ttnn.CoreCoord(*RECV))
    fl = flags.buffer_address()
    in_ct = ttnn.TensorAccessorArgs(inp).get_compile_time_args()
    out_ct = ttnn.TensorAccessorArgs(out).get_compile_time_args()

    def rt(core, words):
        a = ttnn.RuntimeArgs()
        a[core[0]][core[1]] = words
        return a

    def dm(name, core, words, riscv, ct):
        proc, noc = (
            (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)
            if riscv == 0
            else (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)
        )
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores(core),
            compile_time_args=list(ct),
            runtime_args=rt(core, words),
            config=ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc),
        )

    def cb(index, pages, where):
        return ttnn.CBDescriptor(
            total_size=2048 * pages,
            core_ranges=where,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=index, data_format=ttnn.bfloat16, page_size=2048
                )
            ],
        )

    xy = recv_xy.x | recv_xy.y << 16
    kernels = [
        dm("send.cpp", SEND, [inp.buffer_address(), fl, xy, k], 1, in_ct),
        dm("recv_reader.cpp", RECV, [inp.buffer_address(), fl, k], 0, in_ct),
        dm("recv_writer.cpp", RECV, [out.buffer_address(), fl], 1, out_ct),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "compute.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores(RECV),
            compile_time_args=[],
            runtime_args=rt(RECV, [k, j]),
            config=ttnn.ComputeConfigDescriptor(),
        ),
    ]
    desc = ttnn.ProgramDescriptor(
        kernels=kernels,
        semaphores=[],
        cbs=[cb(0, PAGES, both), cb(16, 2, cores(RECV))],
    )
    log("launch")
    ttnn.generic_op([inp, flags, out], desc)
    ttnn.synchronize_device(device)
    log("kernels done")
    got = ttnn.to_torch(out).float().numpy().reshape(2, TILE, TILE)
    st = ttnn.to_torch(flags).numpy().astype(np.int64)
    log("read back")
    ref = x.reshape(PAGES, TILE, TILE)
    return {
        "slot": k,
        "other": j,
        "send": int(st[1, 4]),  # shard order follows the core set: RECV, SEND
        "recv_flag": int(st[0, 0]),
        "recv": int(st[0, 8]),
        "write": int(st[0, 12]),
        "page_k_exact": bool(np.array_equal(got[0], ref[k])),
        "page_j_exact": bool(np.array_equal(got[1], ref[j])),
        "got_rows0": [float(got[0, 0, 0]), float(got[1, 0, 0])],
    }


if __name__ == "__main__":
    main()
