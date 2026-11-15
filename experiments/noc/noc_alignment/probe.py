"""Small NoC writes whose source and destination differ in address bits [3:0]: which
word of the target arrives, and whether a non-posted one is acknowledged. JSON line.

Example: python probe.py --set controls; python probe.py --set mismatched"""

import argparse
import json
import time

import numpy as np

import komorebi as km

WORDS = 128  # per core: 8 target slots of 16 words on B; on A the sources and results
SRC, RESULT = 32, 64
A, B = (0, 0), (1, 0)
SETS = {  # (slot, source word, destination word, posted)
    "controls": [(0, 0, 0, 1), (1, 1, 1, 1), (2, 3, 3, 1), (7, 1, 1, 0)],
    "mismatched": [
        (3, 1, 0, 1),
        (4, 2, 1, 1),
        (5, 3, 0, 1),
        (6, 0, 3, 1),
        (7, 1, 0, 0),
    ],
}
T0 = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--set", choices=list(SETS), required=True)
    tests = SETS[p.parse_args().set]
    with km.Session() as session:
        log("device open")
        out = session.tensor(
            np.zeros((2, WORDS), np.uint32), "uint32", shard_grid=(2, 1)
        )
        bx, by = session.device.worker_core(*B)
        words = [s | sw << 8 | dw << 16 | posted << 24 for s, sw, dw, posted in tests]
        args = [(*A, [out.address, bx | by << 16, len(tests), *words])]
        source = str(__file__).replace("probe.py", "probe.cpp")
        launch = session.launch()
        launch.kernel(source, [(*A, *A)], 0, [], [], [], args)
        log("launch")
        launch.run()
        session.device.finish()
        log("kernel done")
        rows = out.read().astype(np.int64)
        log("read back")
    a_row, b_row = rows[0], rows[1]  # a row per core: A (0, 0), then B (1, 0)
    results = []
    for t, (slot, sw, dw, posted) in enumerate(tests):
        slot_words = b_row[16 * slot : 16 * slot + 4]
        results.append(
            {
                "slot": slot,
                "src_word": sw,
                "dst_word": dw,
                "posted": bool(posted),
                "arrived": {
                    i: f"S[{int(v) & 0xF}]" if int(v) >> 16 == 0xA0A0 else hex(int(v))
                    for i, v in enumerate(slot_words)
                    if v
                },
                "done": {0: "no", 1: "yes", 2: "acked", 3: "no ack"}[
                    int(a_row[RESULT + t])
                ],
            }
        )
    sources_on_a = int(a_row[SRC]) == 0xA0A00000  # row 0 is A's shard
    print(json.dumps({"rows_ok": sources_on_a, "tests": results}), flush=True)


if __name__ == "__main__":
    main()
