"""An L1 tensor sharded over every core written and read back through the host channel,
without a run and inside one, every byte compared; then a narrower tensor whose rows
go as one command each. One JSON line a measurement.
Example: TT_VISIBLE_DEVICES=1 python l1_tensor.py --bytes 131072"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.bench.power import visible_card

ROOT = Path(__file__).resolve().parents[3]
KERNELS = ROOT / "examples" / "01_mapcollect" / "kernels"
T0 = time.monotonic()


def aiclk():
    """The card's AICLK in MHz, as its driver reports it."""
    path = Path(f"/sys/class/tenstorrent/tenstorrent!{visible_card()}/tt_aiclk")
    return int(path.read_text().split()[0])


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def put(**fields):
    print(json.dumps(fields), flush=True)


def program():
    """The map/collect example's program: a run whose cores serve the transfers."""
    return km.Program(
        "mapcollect",
        kernels=KERNELS,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
        ],
        compile_args={"reader": [2, 2048], "writer": [2, 2048]},
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        handlers=KERNELS / "mapcollect.h",
    )


def timed(what, way, tensor, call):
    """Runs call() once; a JSON line with the host seconds and the cores' cycles."""
    log(f"{what} {way}: {tensor.nbytes} B on {tensor.pages} cores")
    start = time.monotonic()
    out = call()
    seconds = time.monotonic() - start
    put(
        what=what, way=way, bytes=tensor.nbytes, cores=tensor.pages, s=seconds,
        host_mb_s=tensor.nbytes / seconds / 1e6, cycles=session_cycles[0],
    )  # fmt: skip
    return out


session_cycles = [None]


def round_trip(session, tensor, data, way):
    """Writes `data` into the L1 tensor and reads it back, both by `way`; exact."""
    got = session._transfer_way([tensor], 0, None)
    if got != way:
        sys.exit(f"stop: the transfer goes {got}, not {way}")

    def write():
        session.write(tensor, data)
        session_cycles[0] = session.transfer_cycles

    def read():
        (pages,) = session.read(tensor)
        session_cycles[0] = session.transfer_cycles
        return pages

    timed("write", way, tensor, write)
    pages = timed("read", way, tensor, read)
    wrong = int(np.count_nonzero(pages != data))
    put(what="check", way=way, cores=tensor.pages, wrong_bytes=wrong)
    if wrong:
        sys.exit(f"stop: {wrong} bytes differ ({way})")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bytes", type=int, default=128 << 10, help="bytes a core")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    with km.Session(log=log) as session:
        if session.channel is None:
            sys.exit("stop: no host channel")
        grid = session.grid
        cores = grid[0] * grid[1]
        put(what="clock", aiclk_mhz=aiclk())
        data = rng.integers(0, 256, (cores, args.bytes), dtype=np.uint8)
        tensor = session.empty((cores, args.bytes), "uint8", shard_grid=grid)
        round_trip(session, tensor, data, "launch")
        narrow = session.empty((15, 4096), "uint8", shard_grid=(5, 3))
        small = rng.integers(0, 256, (15, 4096), dtype=np.uint8)
        log("loading the run")
        run = session.load(program())
        run.wait(run.submit(0, (0, 8)))
        round_trip(session, tensor, data[::-1].copy(), "run")
        round_trip(session, narrow, small, "run")
        run.stop()
        error = run.result()["error"]
        put(what="run", error=error)
        put(what="clock", aiclk_mhz=aiclk())
        if error:
            sys.exit(f"stop: the run ended with error {error}")
    log("done")


if __name__ == "__main__":
    main()
