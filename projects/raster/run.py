"""The host-binned rasterizer on a card, one scene, checked against the reference (the
whole frame) or `raster.fill_tile` (--tiles). Options in `--help`.
`python -m projects.raster.run --scene tiny [--tiles 0 33] [--cores 4] [--out DIR]`"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from projects.raster import card, lists, pictures, stress
from projects.raster.reference import harness
from ref import native, raster, results, scenes

K = lists.K
SPLIT = 0.33  # a tile above this fraction of a core's mean load is split into bands
ARGS = ("recs", "offsets", "members", "tiles", "counts")  # check()'s list arguments
PROF = (
    "wait_in",
    "init",
    "tri",
    "rows",
    "drain",
    "out",
    "wait_out",
    "tiles",
    "tris",
    "row_count",
    "total",
)


def profile_rows(prof, workers):
    """Per core: the pack thread's, reader's and writer's counters (dict of arrays)."""
    p = prof[:workers].astype(np.int64)
    out = {name: p[:, k] for k, name in enumerate(PROF)}
    for k, name in enumerate(("wait", "read", "total", "pages")):
        out[f"reader_{name}"] = p[:, K["kProfReader"] + k]
    for k, name in enumerate(("wait", "write", "total", "items")):
        out[f"writer_{name}"] = p[:, K["kProfWriter"] + k]
    return out


def load(name):
    """The scene's arrays and its load time in seconds."""
    t0 = time.perf_counter()
    if name in stress.NAMES:
        return stress.make(name), time.perf_counter() - t0
    native.ensure_scene(name)
    scene = scenes.load_scene(native.data_dir() / f"{name}.npz")
    return scene, time.perf_counter() - t0


def prepare(scene, workers, mode="balanced", tiles=None, cores=0, split=SPLIT):
    """Host part, each step timed: setup, binning, records, assignment and lists; when a
    tile costs more than a core's mean, tiles above `split` times it go in bands.
    Returns recs, counts, offsets, members, tiles, tiles_of (items per core), pages,
    directory and host (seconds per step)."""
    t0 = time.perf_counter()
    recs = raster.setup(scene)
    t1 = time.perf_counter()
    counts, offsets, members = raster.bin_triangles(recs)
    t2 = time.perf_counter()
    cards = lists.pack_records(recs)
    t3 = time.perf_counter()
    used = cores or workers
    tiles = tiles if tiles is not None else list(range(lists.TILES))
    cost = lists.tile_cost(cards, offsets, members)
    mean = cost[tiles].sum() / used
    # Bands only when a tile is above a core's mean load: otherwise they only add work.
    if split > 0 and cost[tiles].max() > mean:
        target = split * mean
        items, item_cost = lists.split_tiles(
            cards, offsets, members, tiles, cost, target
        )
    else:
        items, item_cost = [(t, 0, 32) for t in tiles], cost[tiles]
    part = lists.assign(item_cost, used, mode)
    tiles_of = [[items[i] for i in t] for t in part]
    tiles_of += [[] for _ in range(workers - used)]
    pages, directory = lists.build_lists(cards, offsets, members, tiles_of)
    # Output items in run order: the frame holds only the tiles run.
    slot = {t: i for i, t in enumerate(tiles)}
    for c, ts in enumerate(tiles_of):
        ids = directory[c, K["kDirIds"] : K["kDirIds"] + len(ts)]
        rest = ids & ~np.uint32(K["kItemSlot"])
        ids[:] = np.array([slot[t[0]] for t in ts], dtype=np.uint32) | rest
    t4 = time.perf_counter()
    host = {"setup_s": t1 - t0, "bin_s": t2 - t1, "pack_s": t3 - t2, "lists_s": t4 - t3}
    return {
        "recs": recs,
        "counts": counts,
        "offsets": offsets,
        "members": members,
        "tiles": tiles,
        "tiles_of": tiles_of,
        "used": used,
        "pages": pages,
        "directory": directory,
        "host": host,
    }


def frame_stats(us):
    """Submit-to-acknowledgement time of the commands in microseconds."""
    us = np.asarray(us, dtype=np.float64)
    return {
        "first": float(us[0]),
        "median": float(np.median(us)),
        "min": float(us.min()),
        "max": float(us.max()),
    }


def check_ok(check):
    """True when the frame, or every tile of a subset, equals the reference."""
    if "tiles" in check:
        return all(t["depth"] == t["colour"] == t["tri"] == 0 for t in check["tiles"])
    return check["any_differs"] == 0 and bool(check["tile_counts_equal"])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", default="tiny")
    p.add_argument(
        "--tiles", type=int, nargs="*", default=None, help="only these tiles"
    )
    p.add_argument("--cores", type=int, default=0, help="0: every worker")
    p.add_argument("--assign", default="balanced")
    p.add_argument("--split", type=float, default=SPLIT, help="0: no bands")
    p.add_argument("--reps", type=int, default=1, help="FILL commands in the program")
    p.add_argument("--out", type=Path, default=None, help="results-layout folder")
    p.add_argument("--no-frame", action="store_true", help="time only: no read-back")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    out = {"scene": args.scene, "assign": args.assign, "reps": args.reps}
    out["split"] = args.split

    scene, out["load_s"] = load(args.scene)
    with km.Session(log=card.log) as session:
        workers = card.workers(session)
        h = prepare(scene, workers, args.assign, args.tiles, args.cores, args.split)
        tiles, tiles_of, used = h["tiles"], h["tiles_of"], h["used"]
        out["host"] = h["host"]
        out["list_pages"] = len(h["pages"])
        out["list_bytes"] = int(h["pages"].nbytes)
        frame, prof, out["run"], us, out["times"] = card.run_lists(
            session,
            K["kHandlerFill"],
            h["pages"],
            h["directory"],
            len(tiles) * K["kPlanes"],
            K["kPlanes"],
            reps=args.reps,
            frame=not args.no_frame,
        )
    out["frame_microseconds"] = frame_stats(us)
    pr = profile_rows(prof, workers)
    busy = pr["total"]
    out["busy_cycles"] = {
        "max": int(busy.max()),
        "mean": float(busy[:used].mean()),
        "min": int(busy[:used].min()),
    }
    out["profile_sum"] = {k: int(v.sum()) for k, v in pr.items()}
    ok = out["run"]["ok"]
    if frame is not None:
        out["check"] = check(args.scene, args.tiles, *(h[k] for k in ARGS), frame)
        ok = ok and check_ok(out["check"])
    if args.out and args.tiles is None and frame is not None:
        save_result(args.out, out, frame, h["counts"])
    out["ok"] = bool(ok)
    print(json.dumps(out), flush=True)
    if args.json:
        np.savez(
            args.json.replace(".json", "_prof.npz"),
            prof=prof[:workers],
            microseconds=us,
            tiles_of=np.array(
                [t[0] for ts in tiles_of for t in ts] or [0], dtype=np.int64
            ),
            owner=np.array([c for c, ts in enumerate(tiles_of) for _ in ts] or [0]),
        )
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    if not ok:
        sys.exit(1)


def check(name, subset, recs, offsets, members, tiles, counts, frame):
    """Tiles run against fill_tile (subset) or the whole frame against the reference."""
    planes = frame.reshape(len(tiles), K["kPlanes"], 1024)
    if subset is not None:
        bad = []
        for i, t in enumerate(tiles):
            ty, tx = divmod(t, raster.TILES_X)
            colour, depth, win = raster.fill_tile(
                recs, members[offsets[t] : offsets[t + 1]], tx, ty
            )
            want = {
                "depth": depth,
                "colour": colour,
                "tri": np.where(win >= 0, recs["tri"][np.maximum(win, 0)], -1),
            }
            got = planes[i][:, lists.TILE_ORDER].view(np.float32)
            g = {"depth": got[0], "colour": np.stack(got[1:4], axis=2), "tri": got[4]}
            diff = {
                "tile": t,
                "list": int(offsets[t + 1] - offsets[t]),
                "depth": int((g["depth"] != want["depth"]).sum()),
                "colour": int((g["colour"] != want["colour"]).any(axis=2).sum()),
                "tri": int((g["tri"] != want["tri"]).sum()),
            }
            bad.append(diff)
        return {"tiles": bad}
    got = lists.decode_frame(frame)
    got["tile_counts"] = counts.reshape(raster.TILES_Y, raster.TILES_X)
    stats = harness.compare(pictures.reference(name), got)
    stats.pop("mask")
    return stats


def save_result(folder, out, frame, counts, power=None):
    """<folder>/<scene>.npz and .json in the layout of the reference's scripts."""
    arrays = lists.decode_frame(frame)
    arrays["tile_counts"] = np.asarray(counts, np.int32).reshape(32, 32)
    host = out["host"]
    fill = out["frame_microseconds"]["median"] / 1e3
    stages = {
        "setup_ms": 1e3 * host["setup_s"],
        "bin_ms": 1e3 * host["bin_s"],
        "lists_ms": 1e3 * (host["pack_s"] + host["lists_s"]),
        "fill_ms": fill,
    }
    stages["total_ms"] = sum(stages.values())
    info = {
        "name": "Blackhole, host-binned",
        "scene": out["scene"],
        "seconds_per_frame": fill / 1e3,
        "timings_ms": stages,
    }
    if power:
        info["power"] = power
    results.save_result(folder, out["scene"], "raster", arrays, info)


if __name__ == "__main__":
    main()
