"""Hop probe: items reserved by an atomic add and written to the receiver's DRAM queue,
read back by the receiver. Round trip and rate. JSON line.

Example: python hop.py --mode pair --n 32 --packets 1"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

SHARD_WORDS = 0x1100 // 4
SSTAT, RSTAT = 0x40 // 4, 0x80 // 4
CHUNK = 32
M32 = 0xFFFFFFFF
T0 = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def lcg(x):
    return (x * 1664525 + 1013904223) & M32


def payload(src, k, i):
    return ((src * 0x9E3779B1) & M32) ^ ((k * 0x85EBCA77) & M32) ^ i


def plan(mode, cores, packets, n):
    """Expected items and payload sum per receiver, replaying the senders."""
    count, total = [0] * cores, [0] * cores
    for me in range(cores if mode == "random" else 1):
        x = (me * 7919 + 1) & M32
        for k in range(packets):
            x = lcg(x)
            dest = 1 if mode == "pair" else (x >> 8) % cores
            count[dest] += n
            total[dest] += sum(payload(me, k, i) for i in range(n))
    return count, total


def kernel(name, cores_set, args, proc, noc):
    return ttnn.KernelDescriptor(
        kernel_source=str(__file__).replace("hop.py", name),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores_set,
        compile_time_args=[],
        runtime_args=args,
        config=ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=["pair", "random"], required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--packets", type=int, required=True)
    p.add_argument("--commit", action="store_true", help="first mark after a barrier")
    a = p.parse_args()
    assert 1 <= a.n <= 32
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    log("device open")
    try:
        g = device.compute_with_storage_grid_size()
        cores = 2 if a.mode == "pair" else g.x * g.y
        count, total = plan(a.mode, cores, a.packets, a.n)
        cap = CHUNK
        while cap < max(count):
            cap *= 2
        salt = random.getrandbits(32)
        logical = [(c % g.x, c // g.x) for c in range(cores)]
        table = []
        for x, y in logical:
            w = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            table.append(w.x | w.y << 16)
        grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))]
        )
        config = ttnn.create_sharded_memory_config(
            (1, SHARD_WORDS),
            grid,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        shard = ttnn.from_torch(
            torch.zeros((g.x * g.y, SHARD_WORDS), dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=config,
        )
        queue = ttnn.empty(
            (cores * cap // CHUNK, CHUNK * 16),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        log(f"cores {cores} cap {cap} queue MB {cores * cap * 64 / 2**20:.1f}")
        sh, qb = shard.buffer_address(), queue.buffer_address()
        sargs, rargs = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        mode = (0 if a.mode == "pair" else 1) | (256 if a.commit else 0)
        for c, (x, y) in enumerate(logical):
            sargs[x][y] = [sh, qb, cap, c, cores, mode, a.packets, a.n, salt, *table]
            rargs[x][y] = [sh, qb, cap, c, count[c], salt]
        used = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(*xy), ttnn.CoreCoord(*xy)) for xy in logical]
        )
        desc = ttnn.ProgramDescriptor(
            kernels=[
                kernel(
                    "hop_recv.cpp",
                    used,
                    rargs,
                    ttnn.DataMovementProcessor.RISCV_0,
                    ttnn.NOC.RISCV_0_default,
                ),
                kernel(
                    "hop_send.cpp",
                    used,
                    sargs,
                    ttnn.DataMovementProcessor.RISCV_1,
                    ttnn.NOC.RISCV_1_default,
                ),
            ],
            semaphores=[],
            cbs=[],
        )
        clock = Path("/sys/class/tenstorrent/tenstorrent!0/tt_aiclk")
        aiclk_before = clock.read_text().strip()
        log("launch")
        ttnn.generic_op([queue, shard], desc)
        ttnn.synchronize_device(device)
        log("kernel done")
        rows = ttnn.to_torch(shard).numpy().astype(np.int64) & M32
        log("read back")
        aiclk = clock.read_text().strip()
    finally:
        ttnn.close_mesh_device(device)
    s, r = rows[:cores, SSTAT : SSTAT + 16], rows[:cores, RSTAT : RSTAT + 16]
    senders = [c for c in range(cores) if s[c, 2] > 0]
    start = min(int(s[c, 3]) for c in senders)
    last = max(int(r[c, 8]) for c in range(cores) if count[c])
    items = int(sum(count))
    span = (last - start) & M32
    sums = [(int(r[c, 3]) | int(r[c, 4]) << 32) for c in range(cores)]
    out = {
        "mode": a.mode,
        "commit": a.commit,
        "n": a.n,
        "packets": a.packets,
        "cores": cores,
        "cap": cap,
        "items": items,
        "sent": int(s[:, 2].sum()),
        "got": int(r[:, 2].sum()),
        "exact": all(int(r[c, 2]) == count[c] for c in range(cores))
        and all(sums[c] == total[c] % 2**64 for c in range(cores)),
        "send_fail": sorted({int(v) for v in s[:, 1]}),
        "recv_fail": sorted({int(v) for v in r[:, 1]}),
        "done": bool((s[:, 0] == 1).all() and (r[:, 0] == 1).all()),
        "torn": int(r[:, 5].sum()),
        "short_reads": int(r[:, 6].sum()),
        "reads": int(r[:, 10].sum()),
        "bad_payload": int(r[:, 11].sum()),
        "gap_pos": [int(v) for v in r[:, 9] if v],
        "span_cycles": span,
        "items_per_mcycle": items / span * 1e6 if span else None,
        "atomic_cycles_mean": float(s[senders, 6].sum() / (a.packets * len(senders))),
        "atomic_cycles_max": int(s[senders, 5].max()),
        "aiclk_before": aiclk_before,
        "aiclk_after": aiclk,
    }
    if a.mode == "pair":
        out["first_after_start"] = (int(r[1, 7]) - start) & M32
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
