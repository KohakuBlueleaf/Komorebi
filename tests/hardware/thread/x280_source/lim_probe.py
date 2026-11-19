"""Card case (Blackhole): a Tensix core's NoC write into the work area of X280 cluster 1
(the KMD_WORK_* lines a Tensix dispatcher's events will land in) and its read back, by
the core and by the host; the line's old words are put back. Prints one JSON line."""

import json
import struct
import sys
from pathlib import Path

import komorebi as km
from komorebi.program import thread as thread_module

HERE = Path(__file__).resolve().parent
CLUSTER = 1
SEED = 0x5A3C0000


def main():
    program = km.Program(
        "lim_probe",
        kernels=HERE / "kernels",
        handlers=HERE / "kernels" / "handlers.h",
        buffers=[km.Buffer(index=0, format="uint32", page_bytes=128, pages=1)],
    )
    with km.Session() as session:
        run = session.load(program)
        port = run.backend.port
        info = port.work_info(CLUSTER)
        offset = (
            thread_module.layout()["KMX_LIM_BYTES"]
            + km.abi.headers()["KMD_WORK_EVENTS"]
        )
        address = info["lim_base"] + offset
        tile = info["virtual_x"] | info["virtual_y"] << 16
        before = bytes(port.work_read(CLUSTER, offset, 64))
        handle = run.submit(
            "probe", group=(0, 1), args=[tile, address, SEED], result=True
        )
        same = run.wait(handle, timeout=10.0).result
        host = struct.unpack("<16I", bytes(port.work_read(CLUSTER, offset, 64)))
        port.work_write(CLUSTER, offset, before)
        restored = bytes(port.work_read(CLUSTER, offset, 64)) == before
        run.stop()
    pattern = tuple(SEED + k for k in range(16))
    out = {
        "cluster": CLUSTER,
        "tile": f"{tile:#x}",
        "address": f"{address:#x}",
        "core_read_back": same,
        "host_sees_it": host == pattern,
        "restored": restored,
    }
    out["exact"] = same == 16 and host == pattern and restored
    print(json.dumps(out), flush=True)
    return 0 if out["exact"] else 1


sys.exit(main())
