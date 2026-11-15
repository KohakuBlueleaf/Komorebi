"""Forward probe: messages reserved by an atomic add and written to the receiver's DRAM
queue, read back by the receiver. Round trip and rate. JSON line.

Example: python forward.py --mode pair --n 32 --packets 1"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

import komorebi as km

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
    """Expected messages and payload sum per receiver, replaying the senders."""
    count, total = [0] * cores, [0] * cores
    for me in range(cores if mode == "random" else 1):
        x = (me * 7919 + 1) & M32
        for k in range(packets):
            x = lcg(x)
            dest = 1 if mode == "pair" else (x >> 8) % cores
            count[dest] += n
            total[dest] += sum(payload(me, k, i) for i in range(n))
    return count, total


def kernel(launch, name, cores_set, args, riscv):
    """A data movement kernel on RISC-V `riscv` and its NoC; args [(x, y, words)]."""
    source = str(__file__).replace("forward.py", name)
    launch.kernel(source, cores_set, riscv, [], [], [], args)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=["pair", "random"], required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--packets", type=int, required=True)
    p.add_argument("--commit", action="store_true", help="first mark after a barrier")
    a = p.parse_args()
    assert 1 <= a.n <= 32
    with km.Session() as session:
        log("device open")
        gx, gy = session.grid
        cores = 2 if a.mode == "pair" else gx * gy
        count, total = plan(a.mode, cores, a.packets, a.n)
        cap = CHUNK
        while cap < max(count):
            cap *= 2
        salt = random.getrandbits(32)
        logical = [(c % gx, c // gx) for c in range(cores)]
        table = []
        for x, y in logical:
            wx, wy = session.device.worker_core(x, y)
            table.append(wx | wy << 16)
        shard = session.tensor(
            np.zeros((gx * gy, SHARD_WORDS), np.uint32), "uint32", shard_grid=(gx, gy)
        )
        queue = session.empty((cores * cap // CHUNK, CHUNK * 16), "uint32")
        log(f"cores {cores} cap {cap} queue MB {cores * cap * 64 / 2**20:.1f}")
        sh, qb = shard.address, queue.address
        sargs, rargs = [], []
        mode = (0 if a.mode == "pair" else 1) | (256 if a.commit else 0)
        for c, (x, y) in enumerate(logical):
            words = [sh, qb, cap, c, cores, mode, a.packets, a.n, salt, *table]
            sargs.append((x, y, words))
            rargs.append((x, y, [sh, qb, cap, c, count[c], salt]))
        used = [(x, y, x, y) for x, y in logical]
        launch = session.launch()
        kernel(launch, "forward_recv.cpp", used, rargs, 0)
        kernel(launch, "forward_send.cpp", used, sargs, 1)
        clock = Path("/sys/class/tenstorrent/tenstorrent!0/tt_aiclk")
        aiclk_before = clock.read_text().strip()
        log("launch")
        launch.run()
        session.device.finish()
        log("kernel done")
        rows = shard.read().astype(np.int64)
        log("read back")
        aiclk = clock.read_text().strip()
    s, r = rows[:cores, SSTAT : SSTAT + 16], rows[:cores, RSTAT : RSTAT + 16]
    senders = [c for c in range(cores) if s[c, 2] > 0]
    start = min(int(s[c, 3]) for c in senders)
    last = max(int(r[c, 8]) for c in range(cores) if count[c])
    messages = int(sum(count))
    span = (last - start) & M32
    sums = [(int(r[c, 3]) | int(r[c, 4]) << 32) for c in range(cores)]
    out = {
        "mode": a.mode,
        "commit": a.commit,
        "n": a.n,
        "packets": a.packets,
        "cores": cores,
        "cap": cap,
        "messages": messages,
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
        "gap_position": [int(v) for v in r[:, 9] if v],
        "span_cycles": span,
        "messages_per_mcycle": messages / span * 1e6 if span else None,
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
