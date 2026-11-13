"""Can compute read page k of a buffer it waited for in full, by index, with nothing
popped before it? One put from core (1, 0) into page k of core (0, 0). JSON line.

Example: python probe.py --slot 2 --other 1"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
RECV, SEND = (0, 0), (1, 0)
PAGES, TILE, WORDS = 4, 32, 16
T0 = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def cores(*cs):
    return [(x, y, x, y) for x, y in cs]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--slot", type=int, default=2, help="page k the sender puts")
    p.add_argument("--other", type=int, default=1, help="a page the reader filled")
    args = p.parse_args()
    k, j = args.slot, args.other
    assert 0 <= k < PAGES and 0 <= j < PAGES and j != k
    # Tile t holds t * 32 + row: exact in bfloat16, every tile and row distinct.
    x = np.repeat(np.arange(PAGES * TILE, dtype=np.float32)[:, None], TILE, axis=1)
    with km.Session() as session:
        log("device open")
        out = run(session, x, k, j)
    print(json.dumps(out), flush=True)


def run(session, x, k, j):
    inp = session.tensor(x, "bfloat16", "tile")
    out = session.tensor(
        np.full((2 * TILE, TILE), -1.0, np.float32), "bfloat16", "tile"
    )
    flags = session.tensor(np.zeros((2, WORDS), np.uint32), "uint32", shard_grid=(2, 1))
    rx, ry = session.device.worker_core(*RECV)
    fl = flags.address
    in_ct, out_ct = inp.accessor_args(), out.accessor_args()
    launch = session.launch()

    def dm(name, core, words, riscv, ct):
        launch.kernel(
            str(KERNELS / name), cores(core), riscv, ct, [], [], [(*core, words)]
        )

    xy = rx | ry << 16
    dm("send.cpp", SEND, [inp.address, fl, xy, k], 1, in_ct)
    dm("recv_reader.cpp", RECV, [inp.address, fl, k], 0, in_ct)
    dm("recv_writer.cpp", RECV, [out.address, fl], 1, out_ct)
    launch.kernel(
        str(KERNELS / "compute.cpp"), cores(RECV), 2, [], [], [], [(*RECV, [k, j])]
    )
    launch.cb(cores(RECV, SEND), 0, "bfloat16", 2048, 2048 * PAGES)
    launch.cb(cores(RECV), 16, "bfloat16", 2048, 2048 * 2)
    log("launch")
    launch.run()
    session.device.finish()
    log("kernels done")
    got = out.read().reshape(2, TILE, TILE)
    st = flags.read().astype(np.int64)
    log("read back")
    ref = x.reshape(PAGES, TILE, TILE)
    return {
        "slot": k,
        "other": j,
        "send": int(st[1, 4]),  # a row per core: RECV (0, 0), SEND (1, 0)
        "recv_flag": int(st[0, 0]),
        "recv": int(st[0, 8]),
        "write": int(st[0, 12]),
        "page_k_exact": bool(np.array_equal(got[0], ref[k])),
        "page_j_exact": bool(np.array_equal(got[1], ref[j])),
        "got_rows0": [float(got[0, 0, 0]), float(got[1, 0, 0])],
    }


if __name__ == "__main__":
    main()
