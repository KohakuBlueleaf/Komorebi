"""The converted cache: every placed weight as its arena's bytes, so a load is a copy.

python convert.py CKPT PLACEMENT_JSON CACHE [--islands 0,5] [--jobs 31] [--limit N]
(layout and resume: README.md, "The converted cache")
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ckpt  # noqa: E402


def _formats():
    """komorebi.formats by its file (pure numpy), without the package's native part."""
    root = os.environ.get("KOMOREBI_SRC", os.path.join(HERE, "..", "..", "..", "src"))
    path = os.path.join(root, "komorebi", "formats.py")
    spec = importlib.util.spec_from_file_location("km_formats", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


formats = _formats()


def arena_path(cache, island, dtype):
    return os.path.join(cache, f"i{island:02d}-{dtype}.bin")


def done_path(cache, island):
    return os.path.join(cache, f"i{island:02d}.done")


def read_done(cache, island):
    """{name: (crc, rel_rms)} of the weights an island's log holds."""
    out = {}
    try:
        with open(done_path(cache, island)) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    out[parts[0]] = (int(parts[1]), float(parts[2]))
    except FileNotFoundError:
        pass
    return out


def held(values, conv):
    """The held array from the released values [out, in] (or [N]) by `conv`."""
    if conv == "T":
        return values.T
    if conv == "groups T":
        out, inner = values.shape
        return values.reshape(8, out // 8, inner).transpose(0, 2, 1)
    if conv == "row":
        return values.reshape(1, -1)
    if conv == "=":
        return values
    if conv.startswith("T rows "):
        a, b = (int(x) for x in conv[len("T rows ") :].split(":"))
        return values[a:b].T
    raise ValueError(conv)


def entry_bytes(c, entry, check=False):
    """(the weight's arena bytes, its relative RMS against the released values or
    -1 when not checked)."""
    values = c.values(entry["source"])
    array = np.ascontiguousarray(held(values, entry["conv"]), np.float32)
    raw, shape = formats.to_bytes(array, entry["dtype"], "tile")
    if list(formats.padded_shape(array.shape, "tile")) != list(shape) or len(
        raw
    ) != entry["pages"] * formats.tile_bytes(entry["dtype"]):
        raise ValueError(f"{entry['name']}: {len(raw)} B for {entry['pages']} pages")
    rel = -1.0
    if check:
        back = formats.from_bytes(raw, shape, entry["dtype"], "tile")
        back = back[tuple(slice(0, n) for n in array.shape)]
        err = back.astype(np.float64) - array
        power = max(float(np.mean(array.astype(np.float64) ** 2)), 1e-300)
        rel = float(np.sqrt(np.mean(err**2) / power))
    return raw, rel


def sampled(entry):
    """Whether a weight's error is measured: every one but the routed experts; of
    those expert 0..2 of each layer."""
    name = entry["name"]
    if ".experts." not in name:
        return True
    return name.split(".experts.")[1].split(".")[0] in ("0", "1", "2", "31", "100")


def convert_island(args):
    root, placement_path, cache, island, limit = args
    with open(placement_path) as f:
        p = json.load(f)
    entries = [e for e in p["entries"] if e["island"] == island]
    sizes = {a["dtype"]: a["bytes"] for a in p["arenas"] if a["island"] == island}
    files = {}
    for dtype, size in sizes.items():
        path = arena_path(cache, island, dtype)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        if os.fstat(fd).st_size != size:
            os.ftruncate(fd, size)
        files[dtype] = fd
    done = read_done(cache, island)
    c = ckpt.Checkpoint(root)
    t0, n, nbytes = time.time(), 0, 0
    with open(done_path(cache, island), "a") as log:
        for entry in entries:
            if entry["name"] in done:
                continue
            if limit is not None and n >= limit:
                break
            raw, rel = entry_bytes(c, entry, sampled(entry))
            offset = entry["first"] * p["tile_bytes"][entry["dtype"]]
            os.pwrite(files[entry["dtype"]], raw, offset)
            os.fdatasync(files[entry["dtype"]])
            log.write(f"{entry['name']}\t{zlib.crc32(raw)}\t{rel:.6e}\n")
            log.flush()
            n += 1
            nbytes += len(raw)
    for fd in files.values():
        os.close(fd)
    left = sum(1 for e in entries if e["name"] not in read_done(cache, island))
    return island, n, nbytes, time.time() - t0, left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("placement")
    ap.add_argument("cache")
    ap.add_argument("--islands", default=None)
    ap.add_argument("--jobs", type=int, default=31)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    with open(a.placement) as f:
        count = len(json.load(f)["cards"])
    islands = (
        list(range(count))
        if a.islands is None
        else [int(x) for x in a.islands.split(",")]
    )
    os.makedirs(a.cache, exist_ok=True)
    t0 = time.time()
    jobs = [(a.ckpt, a.placement, a.cache, i, a.limit) for i in islands]
    total = 0
    with Pool(min(a.jobs, len(jobs))) as pool:
        for island, n, nbytes, sec, left in pool.imap_unordered(convert_island, jobs):
            total += nbytes
            print(
                f"island {island}: {n} weights, {nbytes:,} B in {sec:.1f} s "
                f"({nbytes / max(sec, 1e-9) / 1e6:.1f} MB/s); {left} left",
                flush=True,
            )
    sec = time.time() - t0
    print(f"all: {total:,} B in {sec:.1f} s ({total / sec / 1e9:.2f} GB/s)", flush=True)


if __name__ == "__main__":
    main()
