"""The decode chain's reads grown one difference at a time: a product's cores read
equal runs (even), its real W pages (w), X beside W on AAAB (w_x) or AB (w_x_ab) as a
class's zip does, then a consumer that takes the blocks in order (--consumer: _c).
Per run the card's GB/s and when each core ends. Example: python chain.py"""

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
STEPS = {  # a step: (W's own pages, X's pattern or None, W's pattern, a consumer)
    "even": (False, None, "AB", False),
    "w": (True, None, "AB", False),
    "w_x": (True, "AAAB", "AB", False),
    "w_x_ab": (True, "AB", "AB", False),
    "w_x_c": (True, "AAAB", "AB", True),
    "w_x_ab_c": (True, "AB", "AB", True),
}
FIRST = ("even", "w", "w_x", "w_x_ab")
RUNS = [("gate", f, s) for f in ("bfp4_b", "bfp8_b") for s in FIRST]
RUNS += [("down", "bfp4_b", s) for s in ("w", "w_x", "w_x_ab")]
LATER = ("w_x", "w_x_c", "w_x_ab", "w_x_ab_c")
RUNS_CONSUMER = [("gate", f, s) for f in ("bfp4_b", "bfp8_b") for s in LATER]
PRODUCT_CYCLES = 126  # compute's cycles a W tile (both formats)
HALF_BLOCKS = 4  # a read pair's blocks a half
STAMPS = 2 * 64  # words of the consumer's stamps (64 blocks a kernel)


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


def ahead(w_pattern, x_pattern, mover):
    """Blocks a mover's next block may lead the consumer by: its half holds
    HALF_BLOCKS of its own blocks of each stream (the fewest of W's and X's)."""
    out = []
    for letters in (w_pattern, x_pattern):
        if letters and mover in letters:
            share = letters.count(mover) / len(letters)
            out.append(int(HALF_BLOCKS / share))
    return min(out)


def one(session, shard, sources, index, run, mhz):
    """Launches `run`; its record: the card's rates and each core's end."""
    product, fmt, step = run
    own, x_pattern, w_pattern, consumer = STEPS[step]
    kt, nt, _ = PRODUCTS[product]
    page = TILE[fmt] * W_TILES
    ranks = shares(product)
    cores = [(r % 12, r // 12) for r in range(len(ranks))]
    lx, ly = session.device.worker_core(*cores[0])
    counter = shard.address + 4 * (COUNTERS + 16 * index + 15)
    landing = shard.address + 4 * LANDING
    x_landing = landing + SLOTS * TILE["bfp8_b"] * W_TILES
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
                x_landing,
                len(x_pattern or "A"),
                takes_of(x_pattern or "A", mover),
                PRODUCT_CYCLES * CHUNK * WIDTH if consumer else 0,
                x_landing + 2 * X_PAGE,
                ahead(w_pattern, x_pattern, mover),
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--consumer", action="store_true", help="the consumer's runs")
    runs = RUNS_CONSUMER if parser.parse_args().consumer else RUNS
    with km.Session() as session:
        mhz = fed.aiclk_mhz()
        gx, gy = session.grid
        fed.log(f"device open, AICLK {mhz} MHz, grid {gx} x {gy}")
        most = max(kt * nt for kt, nt, _ in PRODUCTS.values()) * TILE["bfp8_b"]
        sources = {
            "w": session.empty((most // 1024 + 64, 256), "int32"),
            "x": session.empty((max(PRODUCTS["down"][0], 1024), X_PAGE // 4), "int32"),
        }
        words = LANDING + (SLOTS * TILE["bfp8_b"] * W_TILES + 2 * X_PAGE) // 4 + STAMPS
        shard = fed.int32_tensor(session, np.zeros((gx * gy, words)), (gx, gy))
        records = []
        for index, run in enumerate(runs):
            record = one(session, shard, sources, index, run, mhz)
            fed.log(json.dumps(record))
            records.append(record)
    ok = all(r["ok"] for r in records)
    print(json.dumps({"runs": len(records), "ok": ok, "aiclk_mhz": mhz}), flush=True)
    if not any(r["kernels"] for r in records):  # no kernel's record: every area zeros
        raise SystemExit("chain.py: no kernel's record came back (reads were zeros)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
