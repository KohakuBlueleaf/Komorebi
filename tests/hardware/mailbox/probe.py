"""Mailbox depth on a Tensix core: unpack writes k words before math (and pack) read
any, for k = 1 to 8; prints the write time and the words read, one JSON line per k."""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "feeder"))

import commands as feeder_commands  # noqa: E402
import fed  # noqa: E402

import komorebi as km  # noqa: E402


def main():
    with km.Session() as session:
        mhz = fed.aiclk_mhz()
        delay = 20 * mhz  # 20 us
        for both in (0, 1):
            for readers in range(1, 9):
                out = fed.int32_tensor(session, np.zeros((1, 16)), (1, 1))
                launch = session.launch()
                args = [(0, 0, [out.address, readers, both, delay])]
                launch.kernel(
                    str(HERE / "probe.cpp"), [(0, 0, 0, 0)], 2, [], [],
                    [str(feeder_commands.INCLUDE)], args,
                )  # fmt: skip
                launch.run()
                session.device.finish()
                row = feeder_commands.as_uint32(fed.read_int32(out))[0]
                print(
                    json.dumps(
                        {
                            "k": readers,
                            "to": "math+pack" if both else "math",
                            "write_microseconds": round(int(row[0]) / mhz, 2),
                            "readers_done": [int(row[1]), int(row[15])],
                            "read": [int(word) for word in row[2 : 2 + readers]],
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
