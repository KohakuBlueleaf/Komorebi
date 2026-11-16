"""The framework's host transfers: a 20 MiB frame and a 94 MB tensor written and read
back through Session.tensor and Session.read_chunks, with and without the host channel,
and inside a live run. Needs TT_VISIBLE_DEVICES=1; one JSON line a measurement.
Example: python framework.py --way device (today's path) or --way channel"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import session as session_module
from komorebi.device import channel as channel_module

ROOT = Path(__file__).resolve().parents[3]
KERNELS = ROOT / "examples" / "01_mapcollect" / "kernels"
T0 = time.monotonic()
# name: (shape, dtype) of the data; a frame's page is one row of RGBA bytes
DATA = {
    "frame": ((2048, 2560 * 4), "uint8"),
    "tensor": ((5876, 4000), "float32"),
}
SIZES = (16 << 10, 64 << 10, 256 << 10, 1 << 20)  # the break-even's reads


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


def source(name, rng):
    shape, dtype = DATA[name]
    if dtype == "uint8":
        return rng.integers(0, 256, shape, dtype=np.uint8)
    return rng.standard_normal(shape, dtype=np.float32)


def read_all(session, tensor):
    """Every page through read_chunks, a log line a chunk; (pages, seconds)."""
    out = np.empty((tensor.pages, tensor.page_bytes), np.uint8)
    start = time.monotonic()
    for first, pages in session.read_chunks(tensor):
        out[first : first + len(pages)] = pages
        log(f"  pages {first} .. {first + len(pages) - 1} of {tensor.pages}")
    return out, time.monotonic() - start


def round_trip(session, name, data, reps, way):
    """Writes `data` to the card, reads it back `reps` times; a JSON line each."""
    start = time.monotonic()
    tensor = session.tensor(data, DATA[name][1], "row")
    put(what="write", data=name, way=way, bytes=data.nbytes, s=time.monotonic() - start)
    want = data.view(np.uint8).reshape(tensor.pages, tensor.page_bytes)
    for rep in range(reps):
        got, seconds = read_all(session, tensor)
        exact = bool(np.array_equal(got, want))
        put(what="read", data=name, way=way, rep=rep, bytes=data.nbytes, s=seconds)
        if not exact:
            sys.exit(f"stop: the {name} read back differs")
    return tensor


def break_even(session, tensor, launch):
    """Seconds of a read of SIZES bytes, by the device API or the transfer kernel."""
    page = tensor.aligned_page_bytes
    for size in SIZES:
        pages = max(size // page, 1)
        times = []
        for _ in range(3):
            start = time.monotonic()
            if launch:
                session._push(tensor, 0, pages, "launch")
            else:
                session_module.read_tensors(
                    session.device, session._module, [tensor], 0, pages
                )
            times.append(time.monotonic() - start)
        way = "launch" if launch else "device"
        put(what="small", way=way, bytes=pages * page, s=float(np.median(times)))


def in_run(session, tensors, datas, reps):
    """The same reads and writes while a run is loaded (its cores serve them)."""
    run = session.load(program())
    run.wait(run.submit(0, (0, 8)))
    for name, tensor in tensors.items():
        want = datas[name].view(np.uint8).reshape(tensor.pages, tensor.page_bytes)
        for rep in range(reps):
            start = time.monotonic()
            session.write(tensor, want[::-1].copy())
            put(what="write", data=name, way="run", rep=rep, s=time.monotonic() - start)
            got, seconds = read_all(session, tensor)
            put(what="read", data=name, way="run", rep=rep, s=seconds)
            if not np.array_equal(got, want[::-1]):
                sys.exit(f"stop: the {name} read back in the run differs")
    run.stop()
    error = run.result()["error"]
    put(what="run", error=error)
    if error:
        sys.exit(f"stop: the run ended with error {error}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--way", choices=("device", "channel"), required=True)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if os.environ.get("TT_VISIBLE_DEVICES") != "1":
        sys.exit("stop: run on card 1 (TT_VISIBLE_DEVICES=1)")
    rng = np.random.default_rng(args.seed)
    datas = {name: source(name, rng) for name in DATA}
    channel = 0 if args.way == "device" else channel_module.REGION_DEFAULT
    reps = 1 if args.way == "device" else args.reps
    with km.Session(log=log, channel=channel) as session:
        if args.way == "channel" and session.channel is None:
            sys.exit("stop: no host channel")
        launch = "device" if session.channel is None else "launch"
        tensors = {
            name: round_trip(session, name, data, reps, launch)
            for name, data in datas.items()
        }
        break_even(session, tensors["frame"], launch == "launch")
        if session.channel is not None:
            in_run(session, tensors, datas, reps)


if __name__ == "__main__":
    main()
