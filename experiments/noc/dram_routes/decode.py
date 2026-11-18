"""The decode's reads: every core reads its own run of bfp8_b tiles, its blocks shared
by the two movers by a pattern (AAAB: mover B every fourth), one mover on two banks, or
(--by-place) a pattern chosen by the core's row and column, on pages of one and two
tiles. Per run the card's GB/s and when each core ends. Example: python decode.py"""

import argparse
import json
import sys
import time

import numpy as np
import routes
from routes import AREA, COUNTERS, HERE, LANDING, SLOTS, fed, km, rec

PAGE = 1088  # bytes a page: a bfp8_b tile
COUNT = 1024  # pages a core
BLOCK = 16  # pages a block
TOKEN_GB = 27.2  # bytes a token of the bfp8_b decode (62 layers, products only)
MOVERS = {"A": (0, 0), "B": (1, 1)}  # RISC-V and NoC of each mover
SPAN = 16  # blocks a pattern by place
# A core's end (us) a row on A and a column on B: "alone" with each mover alone
# (pattern_A, pattern_B), "mixed" each mover's own end under AB on tile pages
ENDS = {
    "alone": (
        (321.3, 328.7, 309.7, 276.4, 254.7, 216.7, 181.9, 143.3, 119.5, 83.1),
        (501.2, 517.8, 528.4, 535.1, 539.2, 541.4, 385.4, 417.4, 447.3, 467.2)
        + (469.4, 478.8),
    ),
    "mixed": (
        (249.9, 251.3, 233.5, 206.2, 189.2, 171.5, 145.1, 118.7, 95.9, 78.6),
        (253.4, 259.9, 267.6, 273.4, 277.0, 279.2, 220.8, 232.9, 239.7, 242.1)
        + (240.7, 244.8),
    ),
}
RUNS_BY_PLACE = {  # --by-place's runs a table: (name, how a core's A share is set)
    "alone": (
        ("AB", None),
        ("AAAB", None),
        ("place", "own"),
        ("place_centred", "mean"),
    ),
    "mixed": (("AB", None), ("mixed", "own"), ("mixed_half", "half")),
}


def run_list():
    """Every run: (name, pattern letters, banks mask)."""
    out = [(f"pattern_{p}", p, 0xFF) for p in ("A", "B", "AB", "AAB", "AAAB")]
    out.append(("pattern_AAAAAAAB", "AAAAAAAB", 0xFF))
    for mover in "AB":
        out.append((f"banks04_{mover}", mover, 0x11))
        out.append((f"banks15_{mover}", mover, 0x22))
    return out


def share_a(x, y, ends, how):
    """Mover A's share of a core's blocks: A's rate at its row over A's and B's at its
    place; "mean": the same differences about one half, "half": half of them."""
    rows, columns = ends
    own = columns[x] / (rows[y] + columns[x])
    if how == "mean":
        return own - np.mean([b / (r + b) for r in rows for b in columns]) + 0.5
    return 0.5 + (own - 0.5) / 2 if how == "half" else own


def letters_of(share):
    """SPAN letters with round(share * SPAN) A's spread evenly over them."""
    count = int(round(share * SPAN))
    return "".join(
        "AB"[(i + 1) * count // SPAN == i * count // SPAN] for i in range(SPAN)
    )


def place_list(table):
    """The runs by place: (name, pattern a core or one for all, banks, page tiles)."""
    rows, columns = ENDS[table]
    out = []
    for tiles in (1, 2):
        for name, how in RUNS_BY_PLACE[table]:
            pattern = name
            if how:
                pattern = {
                    (x, y): letters_of(share_a(x, y, ENDS[table], how))
                    for y in range(len(rows))
                    for x in range(len(columns))
                }
            out.append((f"{name}_tiles{tiles}", pattern, 0xFF, tiles))
    return out


def one(session, shard, source, index, run, cores, mhz):
    """Launches `run` on `cores`; its record: the card's rate and each core's end."""
    name, pattern, banks, tiles = run if len(run) == 4 else (*run, 1)
    if not isinstance(pattern, dict):
        pattern = {core: pattern for core in cores}
    movers = sorted(set("".join(pattern.values())))
    page, count, block = PAGE * tiles, COUNT // tiles, BLOCK // tiles
    lx, ly = session.device.worker_core(*cores[0])
    counter = shard.address + 4 * (COUNTERS + 16 * index + 15)
    launch = session.launch()
    for mover in movers:
        risc, noc = MOVERS[mover]
        args = []
        for order, (x, y) in enumerate(cores):
            letters = pattern[(x, y)]
            takes = sum(1 << b for b, letter in enumerate(letters) if letter == mover)
            words = [
                shard.address + 4 * AREA * risc,
                shard.address + 4 * LANDING,
                SLOTS,
                source.address,
                page,
                order * count * 8 // bin(banks).count("1"),  # its own pages
                1,
                count,
                block,
                1,
                lx | ly << 16,
                counter,
                len(cores) * len(movers),
                0xD7A00000 | x | y << 8 | risc << 16,
                len(letters),
                takes,
                banks,
                *routes.NO_ROWS_NO_X,
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
    rows = rec.as_uint32(fed.read_int32(shard)).astype(np.int64)
    kernels = []
    for r in rows:
        for risc in {MOVERS[m][0] for m in movers}:
            area = r[AREA * risc : AREA * (risc + 1)]
            tag = int(area[0])
            core = (tag & 0xFF, tag >> 8 & 0xFF)
            if tag >> 20 != 0xD7A or (tag >> 16 & 1) != risc or core not in cores:
                continue
            kernels.append(
                {
                    "core": core,
                    "mover": "AB"[risc],
                    "start": int(area[2]) | int(area[3]) << 32,
                    "cycles": int(area[4]),
                    "done": int(area[5]),
                    "ok": int(area[8]) == 1 and int(area[7]) == 0,
                }
            )
    record = summarize(name, cores, movers, kernels, mhz, page, count)
    record["host_s"] = round(host_s, 3)
    return record


def summarize(name, cores, movers, kernels, mhz, page=PAGE, count=COUNT):
    """The card's GB/s (every byte over first start to last end), each core's end
    after the first start (us) and its own rate (GB/s), by row and by column."""
    record = {"run": name, "kernels": len(kernels)}
    if len(kernels) != len(cores) * len(movers):
        record["ok"] = False
        return record
    t0 = min(k["start"] for k in kernels)
    by_core = {}
    for k in kernels:
        held = by_core.setdefault(
            k["core"], {"start": k["start"], "end": 0, "pages": 0}
        )
        held["start"] = min(held["start"], k["start"])
        held["end"] = max(held["end"], k["start"] + k["cycles"])
        held["pages"] += k["done"]
    ends = {c: (h["end"] - t0) / mhz for c, h in by_core.items()}
    rates = {
        c: h["pages"] * page / ((h["end"] - h["start"]) / mhz) / 1e3
        for c, h in by_core.items()
    }
    total = sum(h["pages"] for h in by_core.values()) * page
    card = total / (max(ends.values()) * 1e3)
    median = float(np.median(list(ends.values())))
    record.update(
        {
            "ok": all(k["ok"] for k in kernels)
            and all(h["pages"] == count for h in by_core.values()),
            "card_gbs": round(card, 1),
            "tok_s": round(card / TOKEN_GB, 2),
            "skew_us": round((max(k["start"] for k in kernels) - t0) / mhz, 2),
            "end_us": routes.stats(list(ends.values())),
            "last_over_median": round(max(ends.values()) / median, 3),
            "core_gbs": routes.stats(list(rates.values())),
            "end_by_row": line_means(ends, 1),
            "end_by_column": line_means(ends, 0),
            **mover_ends(kernels, movers, t0, mhz),
            "latest": [
                [*c, round(e, 1)]
                for c, e in sorted(ends.items(), key=lambda item: -item[1])[:8]
            ],
        }
    )
    return record


def mover_ends(kernels, movers, t0, mhz):
    """Each mover's end (us) by row and by column: which mover a late line waits for."""
    out = {}
    for mover in movers:
        ends = {
            k["core"]: (k["start"] + k["cycles"] - t0) / mhz
            for k in kernels
            if k["mover"] == mover and k["done"]
        }
        out[f"{mover}_end_by_row"] = line_means(ends, 1)
        out[f"{mover}_end_by_column"] = line_means(ends, 0)
    return out


def line_means(values, axis):
    """Mean of `values` (by core) a row (axis 1) or a column (axis 0)."""
    lines = {}
    for core, value in values.items():
        lines.setdefault(core[axis], []).append(value)
    return {k: round(float(np.mean(v)), 1) for k, v in sorted(lines.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--by-place", nargs="?", const="alone", choices=ENDS, help="patterns by place"
    )
    by_place = parser.parse_args().by_place
    with km.Session() as session:
        mhz = fed.aiclk_mhz()
        gx, gy = session.grid
        cores = [(x, y) for y in range(gy) for x in range(gx)]
        fed.log(f"device open, AICLK {mhz} MHz, grid {gx} x {gy}")
        source = session.empty((len(cores) * COUNT * 4, PAGE // 4), "int32")
        shard = fed.int32_tensor(
            session, np.zeros((gx * gy, LANDING + SLOTS * PAGE // 2)), (gx, gy)
        )
        if by_place and (gx, gy) != tuple(len(e) for e in ENDS[by_place][::-1]):
            sys.exit(f"--by-place: the table is a 12 x 10 grid's, not {gx} x {gy}")
        records = []
        for index, run in enumerate(place_list(by_place) if by_place else run_list()):
            record = one(session, shard, source, index, run, cores, mhz)
            fed.log(json.dumps(record))
            records.append(record)
    ok = all(r["ok"] for r in records)
    print(json.dumps({"runs": len(records), "ok": ok, "aiclk_mhz": mhz}), flush=True)
    if not any(r["kernels"] for r in records):  # no kernel's record: every area zeros
        raise SystemExit("decode.py: no kernel's record came back (reads were zeros)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
