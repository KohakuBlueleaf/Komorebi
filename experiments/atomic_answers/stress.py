"""Atomic answers probe: every asking core reserves K positions with non-posted atomic
adds on random cores, one at a time, and waits (bounded) for each returned old value.
The tails are on the asking cores themselves, or on counter tiles that only answer.
One JSON line. python stress.py [--counters 1] [--fixed-return]"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

HERE = Path(__file__).resolve().parent
W = 1280  # stress.h: kWords
TAIL, STAT, SENT, LOSTN, TABLE, TAILARR = 0, 16, 112, 232, 480, 640
MHZ = 1350
T0 = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cores", type=int, default=119, help="asking cores 0 to cores - 1")
    p.add_argument("--k", type=int, default=20000, help="reservations per core")
    p.add_argument("--bound", type=int, default=3000000, help="answer wait, cycles")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--counters", type=int, default=0, help="tiles after the askers holding tails"
    )
    p.add_argument(
        "--fixed-return", action="store_true", help="one return line for every add"
    )
    return p.parse_args()


def main():
    a = parse()
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    log("device open")
    try:
        g = device.compute_with_storage_grid_size()
        total, n = g.x * g.y, a.cores
        if n < 1 or n + a.counters > total:
            raise SystemExit(f"{n} askers and {a.counters} counters: {total} cores")
        noc = []
        for c in range(total):
            xy = device.worker_core_from_logical_core(ttnn.CoreCoord(c % g.x, c // g.x))
            noc.append(xy.x | xy.y << 16)
        grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))]
        )
        config = ttnn.create_sharded_memory_config(
            (1, W),
            grid,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        init = np.zeros((total, W), dtype=np.uint32)
        init[:, TABLE : TABLE + total] = np.array(noc, dtype=np.uint32)
        blk = ttnn.from_torch(
            torch.from_numpy(init.view(np.int32)),
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
        rng = np.random.default_rng(a.seed)
        args, ranges = ttnn.RuntimeArgs(), []
        for c in range(n):
            x, y = c % g.x, c // g.x
            seed = int(rng.integers(1, 1 << 31))
            ctr = [n, a.counters, int(a.fixed_return)]
            args[x][y] = [blk.buffer_address(), n, a.k, c, seed, a.bound, *ctr]
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)))
        writer = ttnn.KernelDescriptor(
            kernel_source=str(HERE / "writer.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ttnn.CoreRangeSet(ranges),
            compile_time_args=[],
            defines=[],
            runtime_args=args,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1,
                noc=ttnn.NOC.RISCV_1_default,
            ),
            compiler_include_paths=[str(HERE)],
        )
        desc = ttnn.ProgramDescriptor(kernels=[writer], semaphores=[], cbs=[])
        log("launch")
        t = time.perf_counter()
        ttnn.generic_op([dummy, blk], desc)
        ttnn.synchronize_device(device)
        host_s = time.perf_counter() - t
        log(f"done in {host_s:.2f} s")
        rows = ttnn.to_torch(blk).numpy().view(np.uint32).astype(np.int64)
        log("read back")
    finally:
        ttnn.close_mesh_device(device)
    print(json.dumps(report(a, rows, host_s)), flush=True)


def report(a, rows, host_s):
    """Reservations, lost answers, whether every tail is exact, latency and rate."""
    n = a.cores
    st = rows[:n, STAT : STAT + 10]
    if a.counters:
        tails = np.array([rows[n + d % a.counters, TAILARR + 4 * d] for d in range(n)])
    else:
        tails = rows[:n, TAIL]
    added = rows[:n, SENT : SENT + n].sum(0) + rows[:n, LOSTN : LOSTN + n].sum(0)
    resv = int(st[:, 2].sum())
    cycles = int((st[:, 8] | st[:, 9] << 32).max())
    return {
        "askers": n,
        "counters": a.counters,
        "fixed_return": a.fixed_return,
        "done": int((st[:, 0] == 1).sum()),
        "fail": sorted({int(v) for v in st[:, 1] if v}),
        "reservations": resv,
        "lost": int(st[:, 3].sum()),
        "lost_cores": [int(c) for c in np.nonzero(st[:, 3])[0]],
        "older_answers": int(st[:, 7].sum()),
        "tails_exact": bool((tails == added).all()),
        "latency_mean": round(float((st[:, 5] | st[:, 6] << 32).sum()) / max(resv, 1)),
        "latency_max": int(st[:, 4].max()),
        "per_second_M": round(resv / max(cycles, 1) * MHZ, 1),
        "host_s": round(host_s, 2),
    }


if __name__ == "__main__":
    main()
