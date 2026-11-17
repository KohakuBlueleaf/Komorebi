"""The decode chain's reads grown one difference at a time: a product's cores read
equal runs (even), then its real W pages (w), then X beside W on AAAB (w_x) or AB
(w_x_ab), as a class's zip does. Per run the card's GB/s and when each core ends.
Example: python chain.py"""

import argparse
import json
import sys
import time

import numpy as np
import routes
from decode import TOKEN_GB, line_means
from routes import AREA, COUNTERS, HERE, LANDING, fed, km, rec

TILE = {"bfp8_b": 1088, "bfp4_b": 576}  # bytes a W tile
TOKEN = {"bfp8_b": TOKEN_GB, "bfp4_b": 14.4}  # GB a token (62 layers, products only)
X_PAGE = 2048  # X: one bfloat16 tile a page
W_TILES, WIDTH, CHUNK = 2, 14, 2  # W's tiles a page, a group's columns, K a block
SLOTS = 16  # W's landing pages
# A product: K and N in tiles, ranks a group (decode_cost.PLAN, one card)
PRODUCTS = {"gate": (168, 672, 2), "down": (672, 168, 10)}
STEPS = {  # a step: (W's own pages, X's pattern or None, W's pattern)
    "even": (False, None, "AB"),
    "w": (True, None, "AB"),
    "w_x": (True, "AAAB", "AB"),
    "w_x_ab": (True, "AB", "AB"),
}
RUNS = [("gate", f, s) for f in ("bfp4_b", "bfp8_b") for s in STEPS]
RUNS += [("down", "bfp4_b", s) for s in ("w", "w_x", "w_x_ab")]


def shares(product):
    """Each rank's (group, first K row, rows) of `product`, rank r at core r % 12,
    r // 12 (S consecutive ranks a group, K split as the class splits it)."""
    kt, nt, s = PRODUCTS[product]
    out = []
    for rank in range(nt // WIDTH * s):
        share = rank % s
        k0 = share * (kt // s) + min(share, kt % s)
        k1 = (share + 1) * (kt // s) + min(share + 1, kt % s)
        out.append((rank // s, k0, k1 - k0))
    return out


def takes_of(letters, mover):
    return sum(1 << b for b, letter in enumerate(letters) if letter == mover)


def one(session, shard, sources, index, run, mhz):
    """Launches `run`; its record: the card's rates and each core's end."""
    product, fmt, step = run
    own, x_pattern, w_pattern = STEPS[step]
    kt, nt, _ = PRODUCTS[product]
    page = TILE[fmt] * W_TILES
    ranks = shares(product)
    cores = [(r % 12, r // 12) for r in range(len(ranks))]
    lx, ly = session.device.worker_core(*cores[0])
    counter = shard.address + 4 * (COUNTERS + 16 * index + 15)
    landing = shard.address + 4 * LANDING
    launch = session.launch()
    for mover, (risc, noc) in (("A", (0, 0)), ("B", (1, 1))):
        args = []
        for rank, ((x, y), (group, k0, rows)) in enumerate(zip(cores, ranks)):
            count = rows * WIDTH // W_TILES
            first = (k0 * nt + group * WIDTH) // W_TILES if own else rank * count
            words = [
                shard.address + 4 * AREA * risc,
                landing,
                SLOTS,
                sources["w"].address,
                page,
                first,
                1,
                count,
                CHUNK * WIDTH // W_TILES,  # a block: CHUNK rows of W
                1,
                lx | ly << 16,
                counter,
                2 * len(cores),
                0xD7A00000 | x | y << 8 | risc << 16,
                len(w_pattern),
                takes_of(w_pattern, mover),
                0xFF,
                WIDTH // W_TILES if own else 0,  # a row's pages, rows N apart
                nt // W_TILES,
                sources["x"].address,
                X_PAGE,
                k0,
                CHUNK if x_pattern else 0,
                rows,
                landing + SLOTS * TILE["bfp8_b"] * W_TILES,
                len(x_pattern or "A"),
                takes_of(x_pattern or "A", mover),
            ]
            args.append((x, y, words))
        launch.kernel(
            str(HERE / "reader.cpp"),
            fed.core_set(cores),
            risc,
            [],
            [],
            [str(rec.INCLUDE)],
            args,
            noc=noc,
        )
    launch.build()  # found in the cache when prebuilt
    t0 = time.perf_counter()
    launch.run()
    session.device.finish()
    host_s = time.perf_counter() - t0
    kernels = []
    for r in rec.as_uint32(fed.read_int32(shard)).astype(np.int64):
        for risc in (0, 1):
            area = r[AREA * risc : AREA * (risc + 1)]
            tag = int(area[0])
            core = (tag & 0xFF, tag >> 8 & 0xFF)
            if tag >> 20 != 0xD7A or (tag >> 16 & 1) != risc or core not in cores:
                continue
            kernels.append(
                {
                    "core": core,
                    "start": int(area[2]) | int(area[3]) << 32,
                    "cycles": int(area[4]),
                    "w": int(area[5]),
                    "x": int(area[10]) if x_pattern else 0,
                    "ok": int(area[8]) == 1 and int(area[7]) == 0,
                }
            )
    expected = {
        c: (rows * WIDTH // W_TILES, rows) for c, (_, _, rows) in zip(cores, ranks)
    }
    record = summarize(
        f"{product}_{fmt}_{step}", kernels, expected, x_pattern, page, fmt, mhz
    )
    record["host_s"] = round(host_s, 3)
    return record


def summarize(name, kernels, expected, x_pattern, page, fmt, mhz):
    """W's GB/s and every byte's (to the last core's end), tok/s, each core's end
    (us after the first start) and its spread by row and column."""
    record = {"run": name, "kernels": len(kernels)}
    if len(kernels) != 2 * len(expected):
        record["ok"] = False
        return record
    t0 = min(k["start"] for k in kernels)
    ends, w, xs = {}, {}, {}
    for k in kernels:
        c = k["core"]
        ends[c] = max(ends.get(c, 0), (k["start"] + k["cycles"] - t0) / mhz)
        w[c] = w.get(c, 0) + k["w"]
        xs[c] = xs.get(c, 0) + k["x"]
    last = max(ends.values())
    w_bytes = sum(w.values()) * page
    x_bytes = sum(xs.values()) * X_PAGE
    median = float(np.median(list(ends.values())))
    record.update(
        {
            "ok": all(k["ok"] for k in kernels)
            and all(w[c] == p for c, (p, _) in expected.items())
            and all(xs[c] == (r if x_pattern else 0) for c, (_, r) in expected.items()),
            "w_gbs": round(w_bytes / (last * 1e3), 1),
            "all_gbs": round((w_bytes + x_bytes) / (last * 1e3), 1),
            "tok_s": round(w_bytes / (last * 1e3) / TOKEN[fmt], 2),
            "end_us": routes.stats(list(ends.values())),
            "last_over_median": round(last / median, 3),
            "end_by_row": line_means(ends, 1),
            "end_by_column": line_means(ends, 0),
            "latest": [
                [*c, round(e, 1)]
                for c, e in sorted(ends.items(), key=lambda item: -item[1])[:6]
            ],
        }
    )
    return record


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    with km.Session() as session:
        mhz = fed.aiclk_mhz()
        gx, gy = session.grid
        fed.log(f"device open, AICLK {mhz} MHz, grid {gx} x {gy}")
        most = max(kt * nt for kt, nt, _ in PRODUCTS.values()) * TILE["bfp8_b"]
        sources = {
            "w": session.empty((most // 1024 + 64, 256), "int32"),
            "x": session.empty((max(PRODUCTS["down"][0], 1024), X_PAGE // 4), "int32"),
        }
        words = LANDING + (SLOTS * TILE["bfp8_b"] * W_TILES + 2 * X_PAGE) // 4
        shard = fed.int32_tensor(session, np.zeros((gx * gy, words)), (gx, gy))
        records = []
        for index, run in enumerate(RUNS):
            record = one(session, shard, sources, index, run, mhz)
            fed.log(json.dumps(record))
            records.append(record)
    ok = all(r["ok"] for r in records)
    print(json.dumps({"runs": len(records), "ok": ok, "aiclk_mhz": mhz}), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
