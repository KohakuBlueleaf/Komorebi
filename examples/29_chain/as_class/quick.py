"""The README's quick start, as printed, then a check of every tensor against numpy
and the examples' JSON line."""

import json
import sys

import numpy as np
from steps_class import Steps  # examples/29_chain/as_class: the kernels

import komorebi as km

with km.Session() as session:
    x = session.tensor(np.ones((16, 1024), np.float32), "float32")  # 16 tiles in DRAM
    y, z, s = (
        session.tensor(np.zeros((16, 1024), np.float32), "float32") for _ in "yzs"
    )
    t = session.tensor(np.zeros((1, 1024), np.float32), "float32")
    run = session.load(Steps(a=x, b=x, out=y, s=s, t=t))  # compiled and loaded once
    cores = (0, 8)  # a command goes to a core group; each core takes its share

    def scale(a, out, factor):  # a kernel call, as a function
        return run.submit(Steps.scale, cores, args=(a, out, factor, 2), exclusive=True)

    # Commands queue on the card: submit as many as you like, wait once.
    run.wait([scale(x, y, 2.0), scale(y, z, 3.0), scale(z, s, 0.5)])

    # A fixed sequence as ONE command: every core goes from each kernel to the next by
    # itself, with no host between them. Stated once, submitted as often as needed.
    layer = run.chain(
        [
            Steps.scale(x, y, 2.0, 2),
            Steps.scale(y, z, 3.0, 2),
            Steps.add(z, km.past(y), s, 2),
            Steps.sum(s, t, 16),
        ],
        cores=cores,
    )
    for _ in range(100):
        run.wait(run.submit(layer))
    run.stop()  # the device reads a tensor once the run is stopped
    print(t.read()[0, :4])

    # The check: every tensor against numpy, and the JSON line the examples print.
    ones = np.ones((16, 1024), np.float32)
    want = {"y": 2 * ones, "z": 6 * ones, "s": 8 * ones, "t": 128 * ones[:1]}
    got = {"y": y.read(), "z": z.read(), "s": s.read(), "t": t.read()}
    wrong = {name: int(np.sum(got[name] != want[name])) for name in want}
    error = run.result()["error_name"]
report = {
    "wrong": wrong,
    "kernel": error,
    "ok": error == "none" and not any(wrong.values()),
}
print(json.dumps(report))
sys.exit(0 if report["ok"] else 1)
