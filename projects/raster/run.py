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
from komorebi.bench import power
from projects.raster import card, lists, pictures, stress
from projects.raster.reference import harness
from reference import native, raster, results, scenes

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


def profile_rows(profile, workers):
    """Per core: the pack thread's, reader's and writer's counters (dict of arrays)."""
    counters = profile[:workers].astype(np.int64)
    rows = {name: counters[:, column] for column, name in enumerate(PROF)}
    for column, name in enumerate(("wait", "read", "total", "pages")):
        rows[f"reader_{name}"] = counters[:, K["kProfReader"] + column]
    for column, name in enumerate(("wait", "write", "total", "items")):
        rows[f"writer_{name}"] = counters[:, K["kProfWriter"] + column]
    if card.SPREAD:
        rows["landing_wait"] = counters[:, K["kProfLanding"]]
        names = ("cycles", "wait", "records", "writer_cycles", "writer_records")
        for column, name in enumerate(names):
            rows[f"spread_{name}"] = counters[:, K["kProfSpread"] + column]
    return rows


def load(name):
    """The scene's arrays and its load time in seconds."""
    started = time.perf_counter()
    if name in stress.NAMES:
        return stress.make(name), time.perf_counter() - started
    native.ensure_scene(name)
    scene = scenes.load_scene(native.data_dir() / f"{name}.npz")
    return scene, time.perf_counter() - started


def prepare(scene, workers, mode="balanced", tiles=None, cores=0, split=SPLIT):
    """Host part, each step timed: setup, binning, records, assignment and lists; when a
    tile costs more than a core's mean, tiles above `split` times it go in bands.
    Returns recs, counts, offsets, members, tiles, tiles_of (items per core), pages,
    directory and host (seconds per step)."""
    started = time.perf_counter()
    records = raster.setup(scene)
    setup_done = time.perf_counter()
    counts, offsets, members = raster.bin_triangles(records)
    bin_done = time.perf_counter()
    packed = lists.pack_records(records)
    pack_done = time.perf_counter()
    used = cores or workers
    tiles = tiles if tiles is not None else list(range(lists.TILES))
    cost = lists.tile_cost(packed, offsets, members)
    mean = cost[tiles].sum() / used
    # Bands only when a tile is above a core's mean load: otherwise they only add work.
    if split > 0 and cost[tiles].max() > mean:
        target = split * mean
        items, item_cost = lists.split_tiles(
            packed, offsets, members, tiles, cost, target
        )
    else:
        items, item_cost = [(tile, 0, 32) for tile in tiles], cost[tiles]
    partition = lists.assign(item_cost, used, mode)
    tiles_of = [[items[item] for item in own] for own in partition]
    tiles_of += [[] for _ in range(workers - used)]
    pages, directory = lists.build_lists(packed, offsets, members, tiles_of)
    # Output items in run order: the frame holds only the tiles run.
    slot_of_tile = {tile: slot for slot, tile in enumerate(tiles)}
    for core, own_items in enumerate(tiles_of):
        item_ids = directory[core, K["kDirIds"] : K["kDirIds"] + len(own_items)]
        other_bits = item_ids & ~np.uint32(K["kItemSlot"])
        item_ids[:] = (
            np.array([slot_of_tile[item[0]] for item in own_items], dtype=np.uint32)
            | other_bits
        )
    lists_done = time.perf_counter()
    host = {
        "setup_s": setup_done - started,
        "bin_s": bin_done - setup_done,
        "pack_s": pack_done - bin_done,
        "lists_s": lists_done - pack_done,
    }
    return {
        "recs": records,
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


def frame_stats(microseconds):
    """Submit-to-acknowledgement time of the commands in microseconds."""
    microseconds = np.asarray(microseconds, dtype=np.float64)
    return {
        "first": float(microseconds[0]),
        "median": float(np.median(microseconds)),
        "min": float(microseconds.min()),
        "max": float(microseconds.max()),
    }


def check_ok(check):
    """True when the frame, or every tile of a subset, equals the reference."""
    if "tiles" in check:
        return all(
            tile["depth"] == tile["colour"] == tile["tri"] == 0
            for tile in check["tiles"]
        )
    return check["any_differs"] == 0 and bool(check["tile_counts_equal"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="tiny")
    parser.add_argument(
        "--tiles", type=int, nargs="*", default=None, help="only these tiles"
    )
    parser.add_argument("--cores", type=int, default=0, help="0: every worker")
    parser.add_argument("--assign", default="balanced")
    parser.add_argument("--split", type=float, default=SPLIT, help="0: no bands")
    parser.add_argument(
        "--reps", type=int, default=1, help="FILL commands in the program"
    )
    parser.add_argument("--out", type=Path, default=None, help="results-layout folder")
    parser.add_argument(
        "--no-frame", action="store_true", help="time only: no read-back"
    )
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--cards", default="", help="e.g. 0,1: a session of islands, every flat rank"
    )
    parser.add_argument(
        "--far", action="store_true", help="with --cards: the frame on island 1 alone"
    )
    parser.add_argument("--save", default=None, help="the frame's pages to this .npy")
    parser.add_argument(
        "--spread",
        action="store_true",
        help="the reader spreads each record over the lanes (RASTER_SPREAD)",
    )
    args = parser.parse_args()
    card.SPREAD = args.spread
    report = {"scene": args.scene, "assign": args.assign, "reps": args.reps}
    report["split"] = args.split
    report["spread"] = args.spread

    scene, report["load_s"] = load(args.scene)
    cards = [int(n) for n in args.cards.split(",")] if args.cards else []
    opened = {"cards": cards, "links": 1} if len(cards) > 1 else {}
    report["cards"], report["far"] = cards, args.far
    with km.Session(log=card.log, **opened) as session:
        workers = card.workers(session)
        frame_placed = None
        if opened and args.far:
            graph = session.graph
            frame_placed = {"split": 0, "group": (graph.first(1), graph.count(1))}
        prepared = prepare(
            scene, workers, args.assign, args.tiles, args.cores, args.split
        )
        tiles, tiles_of, used = (
            prepared["tiles"],
            prepared["tiles_of"],
            prepared["used"],
        )
        report["host"] = prepared["host"]
        report["list_pages"] = len(prepared["pages"])
        report["list_bytes"] = int(prepared["pages"].nbytes)
        frame, profile, report["run"], microseconds, report["times"] = card.run_lists(
            session,
            K["kHandlerFill"],
            prepared["pages"],
            prepared["directory"],
            len(tiles) * K["kPlanes"],
            K["kPlanes"],
            repetitions=args.reps,
            frame=not args.no_frame,
            frame_placed=frame_placed,
        )
    if args.save and frame is not None:  # for a comparison on the host
        np.save(args.save, frame)
    report["frame_microseconds"] = frame_stats(microseconds)
    rows = profile_rows(profile, workers)
    busy = rows["total"]
    report["busy_cycles"] = {
        "max": int(busy.max()),
        "mean": float(busy[:used].mean()),
        "min": int(busy[:used].min()),
    }
    report["profile_sum"] = {name: int(counter.sum()) for name, counter in rows.items()}
    succeeded = report["run"]["ok"]
    if frame is not None:
        report["check"] = check(
            args.scene, args.tiles, *(prepared[key] for key in ARGS), frame
        )
        succeeded = succeeded and check_ok(report["check"])
    if args.out and args.tiles is None and frame is not None:
        save_result(args.out, report, frame, prepared["counts"])
    report["ok"] = bool(succeeded)
    print(json.dumps(power.clocked(report)), flush=True)
    if args.json:
        np.savez(
            args.json.replace(".json", "_prof.npz"),
            prof=profile[:workers],
            microseconds=microseconds,
            tiles_of=np.array(
                [item[0] for own_items in tiles_of for item in own_items] or [0],
                dtype=np.int64,
            ),
            owner=np.array(
                [core for core, own_items in enumerate(tiles_of) for _ in own_items]
                or [0]
            ),
        )
        with open(args.json, "w") as file:
            json.dump(report, file, indent=1)
    if not succeeded:
        sys.exit(1)


def check(name, subset, records, offsets, members, tiles, counts, frame):
    """Tiles run against fill_tile (subset) or the whole frame against the reference."""
    planes = frame.reshape(len(tiles), K["kPlanes"], 1024)
    if subset is not None:
        differences = []
        for position, tile in enumerate(tiles):
            tile_y, tile_x = divmod(tile, raster.TILES_X)
            colour, depth, winner = raster.fill_tile(
                records, members[offsets[tile] : offsets[tile + 1]], tile_x, tile_y
            )
            want = {
                "depth": depth,
                "colour": colour,
                "tri": np.where(winner >= 0, records["tri"][np.maximum(winner, 0)], -1),
            }
            got = planes[position][:, lists.TILE_ORDER].view(np.float32)
            decoded = {
                "depth": got[0],
                "colour": np.stack(got[1:4], axis=2),
                "tri": got[4],
            }
            difference = {
                "tile": tile,
                "list": int(offsets[tile + 1] - offsets[tile]),
                "depth": int((decoded["depth"] != want["depth"]).sum()),
                "colour": int((decoded["colour"] != want["colour"]).any(axis=2).sum()),
                "tri": int((decoded["tri"] != want["tri"]).sum()),
            }
            differences.append(difference)
        return {"tiles": differences}
    got = lists.decode_frame(frame)
    got["tile_counts"] = counts.reshape(raster.TILES_Y, raster.TILES_X)
    stats = harness.compare(pictures.reference(name), got)
    stats.pop("mask")
    return stats


def save_result(folder, report, frame, counts, power=None):
    """<folder>/<scene>.npz and .json in the layout of the reference's scripts."""
    arrays = lists.decode_frame(frame)
    arrays["tile_counts"] = np.asarray(counts, np.int32).reshape(32, 32)
    host = report["host"]
    fill = report["frame_microseconds"]["median"] / 1e3
    stages = {
        "setup_ms": 1e3 * host["setup_s"],
        "bin_ms": 1e3 * host["bin_s"],
        "lists_ms": 1e3 * (host["pack_s"] + host["lists_s"]),
        "fill_ms": fill,
    }
    stages["total_ms"] = sum(stages.values())
    info = {
        "name": "Blackhole, host-binned",
        "scene": report["scene"],
        "seconds_per_frame": fill / 1e3,
        "timings_ms": stages,
    }
    if power:
        info["power"] = power
    results.save_result(folder, report["scene"], "raster", arrays, info)


if __name__ == "__main__":
    main()
