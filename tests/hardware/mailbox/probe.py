"""Mailbox depth on a Tensix core: unpack writes k words before math (and pack) read
any, for k = 1 to 8; prints the write time and the words read, one JSON line per k."""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "feeder"))

import fed  # noqa: E402
import records as rec  # noqa: E402

import komorebi as km  # noqa: E402


def main():
    with km.Session() as session:
        mhz = fed.aiclk_mhz()
        delay = 20 * mhz  # 20 us
        for both in (0, 1):
            for k in range(1, 9):
                out = fed.int32_tensor(session, np.zeros((1, 16)), (1, 1))
                launch = session.launch()
                args = [(0, 0, [out.address, k, both, delay])]
                launch.kernel(
                    str(HERE / "probe.cpp"), [(0, 0, 0, 0)], 2, [], [],
                    [str(rec.INCLUDE)], args,
                )  # fmt: skip
                launch.run()
                session.device.finish()
                r = rec.as_uint32(fed.read_int32(out))[0]
                print(
                    json.dumps(
                        {
                            "k": k,
                            "to": "math+pack" if both else "math",
                            "write_us": round(int(r[0]) / mhz, 2),
                            "readers_done": [int(r[1]), int(r[15])],
                            "read": [int(v) for v in r[2 : 2 + k]],
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
