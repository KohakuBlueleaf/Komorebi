"""Reservation probe: do non-posted atomic adds return the value before the add, and do
concurrent adders get disjoint positions covering the range exactly. JSON line.

Example: python reservation.py --adders 2 --k 4"""

import argparse
import json
import time

import numpy as np
import torch
import ttnn

WORDS = 512  # per core
TARGET, READY, STAT, POS = 0, 4, 16, 32
X0 = 1000  # initial value of the reserved word
T0 = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def check(rows, adders, k):
    """Positions per adder against the expected increments; the summary dict."""
    intervals, stats = [], []
    for c in range(1, adders + 1):
        r = rows[c]
        st = [int(v) for v in r[STAT : STAT + 9]]
        stats.append(st)
        intervals.extend((int(r[POS + j]), 1 + (7 * c + j) % 32) for j in range(k))
    total = sum(n for _, n in intervals)
    intervals.sort()
    cover, at = True, X0
    for p, n in intervals:
        if p != at:
            cover = False
            break
        at += n
    cyc_sum = sum(s[6] | s[7] << 32 for s in stats)
    return {
        "adders": adders,
        "k": k,
        "total": total,
        "final": int(rows[0][TARGET]),
        "expected_final": X0 + total,
        "ready": int(rows[0][READY]),
        "disjoint_cover": cover and at == X0 + total,
        "first_bad": None if cover else [p, n, at],
        "done": all(s[0] == 1 for s in stats),
        "fail": {f: sum(s[1] == f for s in stats) for f in {s[1] for s in stats}},
        "late": sum(s[2] for s in stats),
        "ret_mask": sorted({s[3] for s in stats}),
        "ids_ok": all(s[8] == c for c, s in enumerate(stats, 1)),
        "cycles_min": min(s[4] for s in stats),
        "cycles_max": max(s[5] for s in stats),
        "cycles_mean": cyc_sum / (adders * k),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--adders", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    a = p.parse_args()
    assert a.k <= WORDS - POS
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    log("device open")
    try:
        g = device.compute_with_storage_grid_size()
        cores = g.x * g.y
        assert 1 <= a.adders < cores
        grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))]
        )
        config = ttnn.create_sharded_memory_config(
            (1, WORDS),
            grid,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        init = torch.zeros((cores, WORDS), dtype=torch.int32)
        init[0, TARGET] = X0
        out = ttnn.from_torch(
            init,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=config,
        )
        dummy = ttnn.from_torch(
            torch.zeros((1, 16), dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        t = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
        args = ttnn.RuntimeArgs()
        ranges = []
        for c in range(1, a.adders + 1):
            x, y = c % g.x, c // g.x
            args[x][y] = [out.buffer_address(), t.x | t.y << 16, a.adders, a.k, c]
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)))
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(__file__).replace("reservation.py", "reservation.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ttnn.CoreRangeSet(ranges),
            compile_time_args=[],
            runtime_args=args,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default,
            ),
        )
        desc = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[])
        log("launch")
        ttnn.generic_op([dummy, out], desc)
        ttnn.synchronize_device(device)
        log("kernel done")
        rows = ttnn.to_torch(out).numpy().astype(np.int64) & 0xFFFFFFFF
        log("read back")
    finally:
        ttnn.close_mesh_device(device)
    print(json.dumps(check(rows, a.adders, a.k)), flush=True)


if __name__ == "__main__":
    main()
