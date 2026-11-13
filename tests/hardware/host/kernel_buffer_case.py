"""Card case: a session with a 128 KiB kernel buffer reports its L1 split from the
runtime and runs the map/collect example exact. Prints one JSON line with `ok`."""

import importlib.util
import json
from pathlib import Path

import numpy as np

import komorebi as km

ROOT = Path(__file__).resolve().parents[3]
KERNEL_BYTES = 128 << 10


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    mapcollect_module = load(
        "mapcollect", ROOT / "examples" / "01_mapcollect" / "mapcollect.py"
    )
    with km.Session(log=mapcollect_module.log, kernel_bytes=KERNEL_BYTES) as session:
        kernel, free = session.l1()
        args = mapcollect_module.parse(["--jobs", "20", "--count", "8"])
        out = mapcollect_module.run(session, args, np.random.default_rng(args.seed))
    out.update(kernel_buffer=kernel, l1_free=free)
    out["ok"] = out["ok"] and kernel == KERNEL_BYTES
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
